"""Master V2 architecture comparison on the frozen Master V1 split.

Protocol
  * identical chronological, game-grouped split as Master V1 (asserted), identical tracked shots;
  * no hyperparameter search: tree candidates reuse Master V1's tuned hyperparameters (CatBoost from
    the `tracking_without_player` set, LightGBM from the V1 architecture comparison), so differences
    come from the player representation only;
  * every choice (calibrator, adjustment coefficients, the selected candidate) is made on the
    calibration partition (Oct 28-30) with two-fold game-grouped cross-fitting;
  * the test partition (Oct 31-Nov 2) is scored once at the end. It was already used once to
    evaluate Master V1, so this is its second use (no fresh games were available: none were played
    on Nov 3 and Nov 4 games had not finished);
  * primary metric log loss; Brier, AUC, calibration; game-clustered bootstrap CIs.

Candidates
  1  v1_control            Master V1 predictions (CatBoost, tracking_without_player)
  1r v1_refit              the same CatBoost refitted in this harness (harness check)
  2  cb_raw_hier           CatBoost + raw hierarchical FG priors
  3  cb_context_skill      CatBoost + context-adjusted hierarchical skill (+ reliability)
  3b cb_delta              CatBoost + one compact feature: the hierarchical logit effect
  4  cb_context_contest    CatBoost + context-adjusted skill + contest features
  5  adj_eb                logit(V1) + a + b * delta_EB             (base model + player adjustment)
  5c adj_eb_contest        logit(V1) + a + b * delta_EB + c * contest_dev + d * sens_x_open  (approach C)
  6  adj_glmm              logit(V1) + a + b * delta_GLMM           (hierarchical logistic)
  6j adj_glmm_joint        logit(V1) + a + b * delta_GLMM_joint     (joint model with contest, approach D)
  7  lgb_context_skill     LightGBM + context-adjusted skill
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pipeline.config import EXPERIMENTS_DIR, MASTER_DATASET_DIR, PROJECT_ROOT
from pipeline.evaluation.metrics import paired_bootstrap, per_shot_log_loss, prob_metrics
from pipeline.features.dataset import feature_list, load_feature_config
from pipeline.training import models as M
from pipeline.training.calibration import Calibrator, select_calibration
from pipeline.training.splits import chronological_split, game_halves
from pipeline.training.train import load_training_config
from pipeline.utils import get_logger, require, utc_now, write_json

log = get_logger("player_skill.experiments")

V1_RUN = EXPERIMENTS_DIR / "master_runs" / "20261104T210006"
EPS = 1e-6

CTX_SKILL = ["ps_oe_all", "ps_oe_type", "ps_oe_zone", "ps_oe_action", "ps_oe_zone_action", "ps_delta",
             "ps_rel_all", "ps_rel_zone_action", "ps_log_neff_all", "ps_has_history"]
RAW_HIER = ["ps_raw_fg_all", "ps_raw_fg_type", "ps_raw_fg_zone", "ps_raw_fg_action", "ps_raw_fg_zone_action",
            "ps_rel_all", "ps_log_neff_all", "ps_has_history"]
CONTEST = ["ps_contest_dev", "ps_contest_sens", "ps_contest_sens_x_open", "ps_glmm_joint_contest"]


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sig(x):
    return 1 / (1 + np.exp(-x))


def load_frame(version: str = "master_v2") -> pd.DataFrame:
    df = pd.read_parquet(MASTER_DATASET_DIR / version / "master_shots.parquet")
    df = df[df["in_blueprint"].astype(bool)].copy()
    df["ps_log_neff_all"] = np.log1p(df["ps_neff_all"])
    df["made"] = df["made"].astype(int)
    return df.sort_values(["game_date", "GAME_ID", "GAME_EVENT_ID"]).reset_index(drop=True)


class Adjustment:
    """logit(p) = logit(p_base) + a + sum_j b_j x_j, fitted by (lightly ridge-penalized) max likelihood."""

    def __init__(self, cols: list[str], ridge: float = 1.0):
        self.cols, self.ridge = cols, ridge

    def _X(self, df):
        return np.column_stack([np.nan_to_num(df[c].to_numpy(float), nan=0.0) for c in self.cols])

    def fit(self, df, base):
        X, off, y = self._X(df), _logit(base), df["made"].to_numpy(float)

        def f(t):
            eta = off + t[0] + X @ t[1:]
            p = _sig(eta)
            val = np.sum(np.logaddexp(0, eta) - y * eta) + 0.5 * self.ridge * np.sum(t[1:] ** 2)
            g = p - y
            return val, np.concatenate([[g.sum()], X.T @ g + self.ridge * t[1:]])

        self.theta = minimize(f, np.zeros(len(self.cols) + 1), jac=True, method="L-BFGS-B").x
        return self

    def predict(self, df, base):
        return _sig(_logit(base) + self.theta[0] + self._X(df) @ self.theta[1:])


def _crossfit(fit_pred, cal: pd.DataFrame, halves) -> np.ndarray:
    a, b = halves
    out = np.empty(len(cal))
    for fm, em in ((a, b), (b, a)):
        out[em] = fit_pred(cal[fm], cal[em])
    return out


def run(out_root=EXPERIMENTS_DIR / "master_v2_runs") -> dict:
    fcfg, tcfg = load_feature_config(), load_training_config()
    df = load_frame()
    s = tcfg["split"]
    split = chronological_split(df, s["test_fraction_games"], s["calibration_fraction_games"], s["rolling_folds"],
                                s["min_train_games_first_fold"])
    v1 = json.loads((V1_RUN / "training_summary.json").read_text())
    for part, dates in (("model_fit", split.fit_dates), ("calibration", split.calib_dates), ("test", split.test_dates)):
        require([dates[0], dates[-1]] == v1["split"][part]["dates"], f"{part} dates differ from Master V1")
    fit_df = df[df["game_date"].isin(split.fit_dates)].reset_index(drop=True)
    cal = df[df["game_date"].isin(split.calib_dates)].reset_index(drop=True)
    test = df[df["game_date"].isin(split.test_dates)].reset_index(drop=True)
    preds = pd.read_parquet(V1_RUN / "predictions.parquet")[["shot_uid", "master_xfg"]]
    cal = cal.merge(preds, on="shot_uid", how="left", validate="one_to_one")
    test = test.merge(preds, on="shot_uid", how="left", validate="one_to_one")
    require(cal["master_xfg"].notna().all() and test["master_xfg"].notna().all(), "V1 predictions missing")
    halves = game_halves(cal, tcfg["seed"])
    y_cal = cal["made"].to_numpy()

    base_feats = feature_list(fcfg, tcfg["feature_sets"]["tracking_without_player"])
    cb = v1["feature_sets"]["tracking_without_player"]["tuned"]
    lg = v1["architectures"]["lightgbm"]["tuned"]
    tree = {
        "v1_refit": ("catboost", cb, base_feats),
        "cb_raw_hier": ("catboost", cb, base_feats + RAW_HIER),
        "cb_context_skill": ("catboost", cb, base_feats + CTX_SKILL),
        "cb_delta": ("catboost", cb, base_feats + ["ps_delta"]),
        "cb_context_contest": ("catboost", cb, base_feats + CTX_SKILL + CONTEST),
        "lgb_context_skill": ("lightgbm", lg, base_feats + CTX_SKILL),
    }
    adjust = {
        "adj_eb": ["ps_delta"],
        "adj_eb_contest": ["ps_delta", "ps_contest_dev", "ps_contest_sens_x_open"],
        "adj_glmm": ["ps_glmm_delta"],
        "adj_glmm_joint": ["ps_glmm_joint_delta"],
    }
    results, cal_pred, test_pred = {}, {"v1_control": cal["master_xfg"].to_numpy()}, {"v1_control": test["master_xfg"].to_numpy()}
    results["v1_control"] = {"kind": "control", "cal_crossfit_log_loss": float(per_shot_log_loss(y_cal, cal_pred["v1_control"]).mean())}

    for name, (fam, tuned, feats) in tree.items():
        for c in feats:
            for d in (fit_df, cal, test):
                d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
        m = M.make_model(fam, feats, tuned["params"], tcfg["seed"], n_estimators=max(int(tuned["n_estimators"]), 10))
        m.fit(fit_df, fit_df["made"])
        p_cal, p_test = m.predict(cal), m.predict(test)
        method, scores = select_calibration(p_cal, y_cal, halves, tcfg["calibration_methods"])
        calib = Calibrator(method).fit(p_cal, y_cal)
        cal_pred[name], test_pred[name] = calib.transform(p_cal), calib.transform(p_test)
        results[name] = {"kind": fam, "features_added": [f for f in feats if f not in base_feats],
                         "calibration": method, "cal_crossfit_log_loss": scores[method], "cal_raw_log_loss": scores["none"]}
        log.info("%-20s calib=%s crossfit=%.5f", name, method, scores[method])

    for name, cols in adjust.items():
        cf = _crossfit(lambda a, b, cols=cols: Adjustment(cols).fit(a, a["master_xfg"]).predict(b, b["master_xfg"]), cal, halves)
        adj = Adjustment(cols).fit(cal, cal["master_xfg"])
        cal_pred[name] = cf
        test_pred[name] = adj.predict(test, test["master_xfg"])
        results[name] = {"kind": "adjustment", "columns": cols, "coef": adj.theta.tolist(),
                         "cal_crossfit_log_loss": float(per_shot_log_loss(y_cal, cf).mean())}
        log.info("%-20s crossfit=%.5f coef=%s", name, results[name]["cal_crossfit_log_loss"], np.round(adj.theta, 3))
    # fixed prior, nothing fitted on 2026-27: logit(V1) + delta_EB
    test_pred["prior_eb_fixed"] = _sig(_logit(test["master_xfg"]) + test["ps_delta"].to_numpy())
    cal_pred["prior_eb_fixed"] = _sig(_logit(cal["master_xfg"]) + cal["ps_delta"].to_numpy())
    results["prior_eb_fixed"] = {"kind": "fixed prior", "cal_crossfit_log_loss": float(per_shot_log_loss(y_cal, cal_pred["prior_eb_fixed"]).mean())}

    selected = min((k for k in results if k != "v1_control"), key=lambda k: results[k]["cal_crossfit_log_loss"])
    log.info("selected on calibration: %s", selected)

    # ---- single test evaluation
    y_test = test["made"].to_numpy()
    tdf = test[["shot_uid", "GAME_ID", "made", "nba_xfg", "PLAYER_ID", "PLAYER_NAME", "ps_has_history",
                "ps_rel_all", "is_three", "zone_group", "action_family"]].copy()
    for k, p in test_pred.items():
        tdf[f"p_{k}"] = p
        results[k]["test"] = prob_metrics(y_test, p)
    nba_ok = tdf["nba_xfg"].notna()
    results["nba_xfg"] = {"kind": "external benchmark", "test": prob_metrics(y_test[nba_ok], tdf.loc[nba_ok, "nba_xfg"])}
    cols = {k: f"p_{k}" for k in test_pred} | {"nba_xfg": "nba_xfg"}
    pairs = [(k, "v1_control") for k in test_pred if k != "v1_control"] + [(selected, "nba_xfg"), ("v1_control", "nba_xfg")]
    boot = paired_bootstrap(tdf, "made", cols, pairs, n_boot=2000, cluster_col="GAME_ID")
    groups = {}
    for gname, mask in (("has_history", tdf["ps_has_history"] == 1), ("no_history", tdf["ps_has_history"] == 0),
                        ("three", tdf["is_three"] == 1), ("two", tdf["is_three"] == 0)):
        sub = tdf[mask]
        groups[gname] = {"n": int(len(sub)), **{k: float(per_shot_log_loss(sub["made"], sub[f"p_{k}"]).mean())
                                               for k in ("v1_control", selected, "adj_eb", "adj_glmm")}}

    run_id = utc_now().replace(":", "").replace("-", "")[:15]
    out = out_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    cdf = cal[["shot_uid"]].copy()
    for k, p in cal_pred.items():
        cdf[f"p_{k}"] = p
    pd.concat([cdf.assign(partition="calibration_crossfit"), tdf.assign(partition="test")], ignore_index=True).to_parquet(
        out / "predictions.parquet", index=False)
    summary = {
        "run_id": run_id, "created_utc": utc_now(), "dataset_version": "master_v2", "v1_run": str(V1_RUN.relative_to(PROJECT_ROOT)),
        "split": {"model_fit": [split.fit_dates[0], split.fit_dates[-1], len(fit_df)],
                  "calibration": [split.calib_dates[0], split.calib_dates[-1], len(cal)],
                  "test": [split.test_dates[0], split.test_dates[-1], len(test)]},
        "test_use": "second use of the Oct 31-Nov 2 test partition (first: Master V1 evaluation); scored once",
        "selected_on_calibration": selected, "candidates": results,
        "bootstrap": boot.to_dict(orient="records"), "subgroups_test_log_loss": groups,
    }
    write_json(out / "comparison.json", summary)
    with open(out_root / "experiments.jsonl", "a") as fh:
        fh.write(json.dumps({"run_id": run_id, "selected": selected,
                             "test_log_loss": {k: v["test"]["log_loss"] for k, v in results.items() if "test" in v}}) + "\n")
    log.info("wrote %s", out)
    return summary


__all__ = ["Adjustment", "run"]
