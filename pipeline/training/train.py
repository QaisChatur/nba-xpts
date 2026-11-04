"""Master xFG training protocol (see configs/training.yaml and docs/methodology/MODEL_METHODOLOGY.md).

1. Chronological split on whole game dates: model-fit | calibration | test (newest, untouched).
2. For each candidate family (full feature set): Optuna search scored by rolling-origin CV log loss
   inside the model-fit partition; boosted models early-stop on each fold's validation block and the
   final number of rounds is the mean best iteration across folds.
3. Refit each tuned candidate on the whole model-fit partition (no early stopping data), score the
   calibration partition, choose a calibrator by game-grouped cross-fitting, and pick the
   architecture with the lowest calibrated calibration-partition log loss.
4. Feature-set ablations with the chosen architecture (re-tuned per set) chosen the same way.
5. Only then predict the test partition, once, for every model and benchmark.
"""

from __future__ import annotations

import json
import platform
import shutil
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline.config import CONFIGS_DIR, EXPERIMENTS_DIR, MASTER_DATASET_DIR, MASTER_MODELS_DIR
from pipeline.evaluation.metrics import per_shot_log_loss, prob_metrics
from pipeline.features.dataset import feature_list, load_feature_config
from pipeline.training import models as M
from pipeline.training.calibration import Calibrator, select_calibration
from pipeline.training.splits import Split, chronological_split, game_halves
from pipeline.utils import get_logger, require, utc_now, write_json

log = get_logger("train")

PACKAGES = ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "optuna", "shap", "scipy", "pyyaml"]


def load_training_config(path: Path = CONFIGS_DIR / "training.yaml") -> dict:
    return yaml.safe_load(path.read_text())


def package_versions() -> dict:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def load_dataset(version: str) -> tuple[pd.DataFrame, dict]:
    d = MASTER_DATASET_DIR / version
    df = pd.read_parquet(d / "master_shots.parquet")
    meta = json.loads((d / "dataset_meta.json").read_text())
    return df, meta


def prepare(df: pd.DataFrame, feats: list[str], forbidden: list[str]) -> pd.DataFrame:
    require(not set(feats) & set(forbidden), "forbidden feature requested")
    out = df[df["in_blueprint"].astype(bool)].copy()
    for c in feats:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    out["made"] = out["made"].astype(int)
    return out.sort_values(["game_date", "GAME_ID", "GAME_EVENT_ID"]).reset_index(drop=True)


# --------------------------------------------------------------------------- CV / tuning


def cv_score(name: str, params: dict, feats: list[str], df: pd.DataFrame, split: Split, tcfg: dict) -> dict:
    lls, ns, iters = [], [], []
    for tr_dates, va_dates in split.folds:
        tr = df[df["game_date"].isin(tr_dates)]
        va = df[df["game_date"].isin(va_dates)]
        m = M.make_model(name, feats, params, tcfg["seed"], tcfg["max_boosting_rounds"], tcfg["early_stopping_rounds"])
        m.fit(tr, tr["made"], va, va["made"])
        p = m.predict(va)
        lls.append(float(per_shot_log_loss(va["made"], p).mean()))
        ns.append(len(va))
        iters.append(m.best_iteration)
    return {"cv_log_loss": float(np.average(lls, weights=ns)), "fold_log_loss": lls,
            "best_iterations": iters, "n_estimators": int(np.mean(iters)) if iters[0] else None}


def tune(name: str, feats: list[str], df: pd.DataFrame, split: Split, tcfg: dict, n_trials: int) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=tcfg["seed"])
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial):
        params = M.suggest(trial, name)
        res = cv_score(name, params, feats, df, split, tcfg)
        trial.set_user_attr("n_estimators", res["n_estimators"])
        trial.set_user_attr("fold_log_loss", res["fold_log_loss"])
        return res["cv_log_loss"]

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=tcfg["optuna"]["timeout_s_per_model"])
    best = study.best_trial
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "duration"))
    return {"family": name, "params": best.params, "cv_log_loss": float(best.value),
            "n_estimators": best.user_attrs.get("n_estimators"), "fold_log_loss": best.user_attrs.get("fold_log_loss"),
            "n_trials": len(study.trials), "seconds": round(time.time() - t0, 1), "trials": trials}


def fit_final(name: str, tuned: dict, feats: list[str], fit_df: pd.DataFrame, tcfg: dict):
    n_est = tuned.get("n_estimators") or tcfg["max_boosting_rounds"]
    m = M.make_model(name, feats, tuned["params"], tcfg["seed"], n_estimators=max(int(n_est), 10))
    return m.fit(fit_df, fit_df["made"])


def calibration_eval(p_cal: np.ndarray, cal_df: pd.DataFrame, tcfg: dict) -> dict:
    halves = game_halves(cal_df, tcfg["seed"])
    best, scores = select_calibration(p_cal, cal_df["made"].to_numpy(), halves, tcfg["calibration_methods"])
    calib = Calibrator(best).fit(p_cal, cal_df["made"])
    return {"method": best, "crossfit_log_loss": scores, "best_crossfit_log_loss": scores[best],
            "raw": prob_metrics(cal_df["made"], p_cal), "calibrator": calib}


# --------------------------------------------------------------------------- main protocol


def run_training(dataset_version: str | None = None, quick: bool = False, model_version: str | None = None) -> Path:
    fcfg = load_feature_config()
    tcfg = load_training_config()
    dataset_version = dataset_version or fcfg["dataset_version"]
    raw, dmeta = load_dataset(dataset_version)
    all_feats = feature_list(fcfg)
    df = prepare(raw, all_feats, fcfg["forbidden"])
    s = tcfg["split"]
    split = chronological_split(df, s["test_fraction_games"], s["calibration_fraction_games"], s["rolling_folds"],
                                s["min_train_games_first_fold"])
    fit_df = df[df["game_date"].isin(split.fit_dates)]
    cal_df = df[df["game_date"].isin(split.calib_dates)]
    test_df = df[df["game_date"].isin(split.test_dates)]
    run_id = utc_now().replace(":", "").replace("-", "")[:15]
    run_dir = EXPERIMENTS_DIR / "master_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    split_info = split.describe(df)
    log.info("split: %s", json.dumps({k: v for k, v in split_info.items() if k != "rolling_folds"}))
    trials_n = {k: (4 if quick else v) for k, v in tcfg["optuna"]["n_trials"].items()}

    # ---- 2/3. architecture comparison on the full feature set
    full = feature_list(fcfg, tcfg["feature_sets"]["full"])
    arch = {}
    for name in tcfg["candidates"]:
        log.info("tuning %s (%d trials)", name, trials_n[name])
        t = tune(name, full, fit_df, split, tcfg, trials_n[name])
        t["trials"].to_csv(run_dir / f"trials_{name}_full.csv", index=False)
        m = fit_final(name, t, full, fit_df, tcfg)
        ce = calibration_eval(m.predict(cal_df), cal_df, tcfg)
        arch[name] = {"tuned": {k: v for k, v in t.items() if k != "trials"}, "model": m, "calib": ce}
        log.info("%s: cv=%.4f calib raw=%.4f calibrated(%s)=%.4f", name, t["cv_log_loss"], ce["raw"]["log_loss"],
                 ce["method"], ce["best_crossfit_log_loss"])
    chosen = min(arch, key=lambda k: arch[k]["calib"]["best_crossfit_log_loss"])
    log.info("chosen architecture: %s", chosen)

    # ---- 4. feature-set ablations with the chosen architecture
    sets = {}
    n_abl = 4 if quick else max(10, trials_n[chosen] // 2)
    for set_name, groups in tcfg["feature_sets"].items():
        feats = feature_list(fcfg, groups)
        if set_name == "full":
            t, m = arch[chosen]["tuned"], arch[chosen]["model"]
            ce = arch[chosen]["calib"]
        else:
            tt = tune(chosen, feats, fit_df, split, tcfg, n_abl)
            tt["trials"].to_csv(run_dir / f"trials_{chosen}_{set_name}.csv", index=False)
            t = {k: v for k, v in tt.items() if k != "trials"}
            m = fit_final(chosen, t, feats, fit_df, tcfg)
            ce = calibration_eval(m.predict(cal_df), cal_df, tcfg)
        sets[set_name] = {"features": feats, "tuned": t, "model": m, "calib": ce}
        log.info("feature set %-24s cv=%.4f calibrated=%.4f", set_name, t["cv_log_loss"], ce["best_crossfit_log_loss"])
    final_set = min(tcfg["final_feature_set_candidates"], key=lambda k: sets[k]["calib"]["best_crossfit_log_loss"])
    log.info("final feature set: %s", final_set)

    # ---- 5. predictions (calibration + test) for every model, benchmarks added in evaluate.py
    pred_rows = []
    for part, pdf in (("calibration", cal_df), ("test", test_df)):
        out = pdf[["shot_uid"]].copy()
        out["partition"] = part
        for name, a in arch.items():
            praw = a["model"].predict(pdf)
            out[f"arch_{name}_raw"] = praw
            out[f"arch_{name}"] = a["calib"]["calibrator"].transform(praw)
        for set_name, sd in sets.items():
            praw = sd["model"].predict(pdf)
            out[f"set_{set_name}_raw"] = praw
            out[f"set_{set_name}"] = sd["calib"]["calibrator"].transform(praw)
        pred_rows.append(out)
    preds = pd.concat(pred_rows, ignore_index=True)
    preds["master_xfg"] = preds[f"set_{final_set}"]
    preds.to_parquet(run_dir / "predictions.parquet", index=False)

    # ---- save the final artifact (never overwrites: new version directory)
    final = sets[final_set]
    model_version = model_version or (f"{dataset_version}_{run_id}" if dataset_version.startswith("master")
                                      else f"master_{dataset_version}_{run_id}")
    art = MASTER_MODELS_DIR / model_version
    require(not art.exists(), f"{art} already exists; refusing to overwrite a saved model")
    art.mkdir(parents=True)
    files = final["model"].save(art)
    calib = final["calib"]["calibrator"]
    spec = M.write_spec(art, final["model"], files, {
        "model_version": model_version, "feature_set": final_set, "feature_schema_version": f"{dataset_version}:{final_set}",
        "calibration": calib.to_dict(), "calibration_version": f"{final['calib']['method']}@{split.calib_dates[0]}..{split.calib_dates[-1]}",
    })
    write_json(art / "calibrator.json", calib.to_dict())
    shutil.copytree(MASTER_DATASET_DIR / dataset_version / "history_assets", art / "history_assets")
    from pipeline.inference import build_prior_state
    build_prior_state(raw, before_date=None).to_csv(art / "in_season_state.csv", index=False)

    fb = sets.get("context_player")
    if fb is not None:  # fallback for official shots without a Blueprint tracking record (labelled at export)
        fdir = art / "fallback_no_tracking"
        fdir.mkdir()
        ffiles = fb["model"].save(fdir)
        fcal = fb["calib"]["calibrator"]
        M.write_spec(fdir, fb["model"], ffiles, {
            "model_version": f"{model_version}:fallback_no_tracking", "feature_set": "context_player",
            "feature_schema_version": f"{dataset_version}:context_player", "calibration": fcal.to_dict(),
            "calibration_version": f"{fb['calib']['method']}@{split.calib_dates[0]}..{split.calib_dates[-1]}"})
        write_json(fdir / "calibrator.json", fcal.to_dict())
        write_json(fdir / "metadata.json", {"model_version": f"{model_version}:fallback_no_tracking",
                                            "dataset_version": dataset_version, "feature_set": "context_player",
                                            "feature_groups": tcfg["feature_sets"]["context_player"], "priors": fcfg["priors"],
                                            "family": chosen, "purpose": "official FGA without Blueprint tracking only"})

    def summarize(d):
        return {k: v for k, v in d.items() if k not in ("model", "calibrator")} | {
            "calib": {k: v for k, v in d["calib"].items() if k != "calibrator"}}

    summary = {
        "run_id": run_id, "created_utc": utc_now(), "quick": quick, "dataset_version": dataset_version,
        "dataset_meta": {k: dmeta[k] for k in ("rows", "rows_with_tracking", "games", "games_with_tracking", "date_range")},
        "split": split_info, "architectures": {k: summarize(v) for k, v in arch.items()}, "chosen_architecture": chosen,
        "feature_sets": {k: summarize(v) for k, v in sets.items()}, "final_feature_set": final_set,
        "model_version": model_version, "artifact": str(art.relative_to(MASTER_MODELS_DIR.parent.parent)),
        "packages": package_versions(), "configs": {"features": fcfg, "training": tcfg},
    }
    write_json(run_dir / "training_summary.json", summary)
    write_json(art / "training_summary.json", summary)
    write_json(art / "metadata.json", {
        "model_version": model_version, "dataset_version": dataset_version, "feature_set": final_set,
        "features": final["features"], "feature_groups": tcfg["feature_sets"][final_set], "priors": fcfg["priors"],
        "family": chosen, "calibration": spec["calibration_version"],
        "trained_on_dates": [split.fit_dates[0], split.fit_dates[-1]], "calibrated_on_dates": [split.calib_dates[0], split.calib_dates[-1]],
        "test_dates": [split.test_dates[0], split.test_dates[-1]], "packages": package_versions(), "run_id": run_id,
        "in_season_state_through": str(raw["game_date"].max()),
    })
    with (EXPERIMENTS_DIR / "master_runs" / "experiments.jsonl").open("a") as f:
        f.write(json.dumps({"run_id": run_id, "created_utc": summary["created_utc"], "model_version": model_version,
                            "chosen": chosen, "final_set": final_set, "quick": quick,
                            "calib_ll": final["calib"]["best_crossfit_log_loss"]}) + "\n")
    log.info("saved %s; run dir %s", art, run_dir)
    return run_dir
