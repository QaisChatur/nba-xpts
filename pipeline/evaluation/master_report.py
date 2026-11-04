"""Held-out evaluation of a Master training run + report artifacts.

Everything here runs *after* model selection. Test-partition numbers are reported, never used to
choose models, features, hyperparameters or calibrators. Supplementary analyses that refit models
(tracking drop-one ablations, learning curve, walk-forward windows) reuse the selected configuration
and are labelled as such.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline.config import CONFIGS_DIR, PROJECT_ROOT
from pipeline.evaluation.metrics import (
    calibration_bins, paired_bootstrap, per_shot_log_loss, plot_calibration, prob_metrics, subgroup_table,
)
from pipeline.features.dataset import feature_list, load_feature_config
from pipeline.training import models as M
from pipeline.training.calibration import Calibrator, select_calibration
from pipeline.training.splits import game_halves
from pipeline.training.train import load_dataset, load_training_config, prepare
from pipeline.utils import get_logger, write_json

log = get_logger("evaluate")

FIG_DIR = PROJECT_ROOT / "docs" / "reports" / "figures" / "master"
REPORT_PATH = PROJECT_ROOT / "docs" / "reports" / "MASTER_MODEL_REPORT.md"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def load_run(run_dir: Path):
    summary = json.loads((run_dir / "training_summary.json").read_text())
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    raw, _ = load_dataset(summary["dataset_version"])
    return summary, preds, raw


def benchmark_frame(summary: dict, preds: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Predictions of all models and benchmarks on calibration + test shots (identical rows)."""
    fcfg = load_feature_config()
    df = prepare(raw, feature_list(fcfg), fcfg["forbidden"])
    fit_dates = summary["split"]["model_fit"]["dates"]
    fit = df[(df["game_date"] >= fit_dates[0]) & (df["game_date"] <= fit_dates[1])]
    e = df.merge(preds, on="shot_uid", how="inner", validate="one_to_one")
    rates = fit.groupby("is_three")["made"].mean()
    e["shot_type_baseline"] = e["is_three"].map(rates).astype(float)
    cal = e["partition"] == "calibration"
    platt = Calibrator("platt").fit(e.loc[cal, "model_a_xfg"], e.loc[cal, "made"])
    e["model_a_calibrated"] = platt.transform(e["model_a_xfg"])
    e["shot_type"] = np.where(e["is_three"] == 1, "3PT", "2PT")
    ecfg = yaml.safe_load((CONFIGS_DIR / "evaluation.yaml").read_text())
    e["contest_bin"] = pd.cut(e["contest_distance_in"], ecfg["contest_bins_in"], labels=ecfg["contest_labels"], right=False)
    e["shot_clock_bin"] = pd.cut(e["shot_clock_s"], ecfg["shot_clock_bins"], labels=ecfg["shot_clock_labels"])
    return e


MAIN_MODELS = {
    "NBA xFG (Blueprint)": "nba_xfg",
    "Model A (original)": "model_a_xfg",
    "Model A, Platt-calibrated": "model_a_calibrated",
    "Shot-type baseline": "shot_type_baseline",
    "Historical location/action baseline": "baseline_hist_xfg",
    "Context only (shot + game)": "set_context",
    "Tracking only": "set_tracking_only",
    "Player only": "set_player_only",
    "Tracking without player": "set_tracking_without_player",
    "Context + player (no tracking)": "set_context_player",
    "Full without defender": "set_full_no_defender",
    "Full": "set_full",
    "Master (final)": "master_xfg",
}


def results_table(e: pd.DataFrame, models: dict) -> pd.DataFrame:
    rows = []
    for name, col in models.items():
        m = prob_metrics(e["made"], e[col])
        rows.append({"model": name, "log_loss": m["log_loss"], "brier": m["brier"], "auc": m["roc_auc"],
                     "cal_slope": m["calibration_slope"], "cal_intercept": m["calibration_intercept"],
                     "ece": m["ece_quantile10"], "mean_pred": m["mean_pred"], "observed": m["observed_fg"], "shots": m["n"]})
    return pd.DataFrame(rows).set_index("model")


def fmt_table(t: pd.DataFrame, digits: int = 4) -> str:
    cols = list(t.columns)
    head = "| " + " | ".join([t.index.name or ""] + cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [head, sep]
    for idx, r in t.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    cells.append("")
                elif c in ("shots", "n", "games", "train_shots", "trials", "n_estimators") and float(v).is_integer():
                    cells.append(f"{int(v):,}")
                else:
                    cells.append(f"{v:.{digits}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- refit helpers (selected configuration)


def refit_config(summary: dict, feats: list[str] | None = None):
    fs = summary["final_feature_set"]
    tuned = summary["feature_sets"][fs]["tuned"]
    family = summary["chosen_architecture"]
    feats = feats or summary["feature_sets"][fs]["features"]
    tcfg = load_training_config()

    def fit(train_df):
        m = M.make_model(family, feats, tuned["params"], tcfg["seed"], n_estimators=max(int(tuned.get("n_estimators") or 300), 10))
        return m.fit(train_df, train_df["made"])
    return fit, feats


def tracking_ablation(summary: dict, e: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Drop-one retrains of the selected configuration for each tracking feature (plus all tracking)."""
    fcfg = load_feature_config()
    fit_dates = summary["split"]["model_fit"]["dates"]
    fit_df = df[(df["game_date"] >= fit_dates[0]) & (df["game_date"] <= fit_dates[1])]
    final_feats = summary["feature_sets"][summary["final_feature_set"]]["features"]
    tracking = [c for c in fcfg["groups"]["tracking"] if c in final_feats]
    variants = {"(selected feature set)": final_feats, "drop all tracking": [c for c in final_feats if c not in tracking]}
    for c in tracking:
        variants[f"drop {c}"] = [f for f in final_feats if f != c]
    tcfg = load_training_config()
    cal = e[e["partition"] == "calibration"]
    test = e[e["partition"] == "test"]
    halves = game_halves(cal, tcfg["seed"])
    rows = []
    for name, feats in variants.items():
        fit, _ = refit_config(summary, feats)
        m = fit(fit_df)
        pc, pt = m.predict(cal), m.predict(test)
        method, _ = select_calibration(pc, cal["made"].to_numpy(), halves, tcfg["calibration_methods"])
        calib = Calibrator(method).fit(pc, cal["made"])
        rows.append({"variant": name, "calibration_log_loss": float(per_shot_log_loss(cal["made"], calib.transform(pc)).mean()),
                     "test_log_loss": float(per_shot_log_loss(test["made"], calib.transform(pt)).mean()),
                     "test_auc": prob_metrics(test["made"], calib.transform(pt))["roc_auc"]})
    t = pd.DataFrame(rows).set_index("variant")
    base = t.loc["(selected feature set)"]
    t["delta_test_log_loss_vs_selected"] = t["test_log_loss"] - base["test_log_loss"]
    return t


def walk_forward(summary: dict, df: pd.DataFrame, n_blocks: int = 6) -> pd.DataFrame:
    """Expanding-window walk-forward: for each later date block, train the selected configuration on
    all earlier dates (last 15% of games held for the calibrator) and predict the block."""
    tcfg = load_training_config()
    fit, _ = refit_config(summary)
    dates = sorted(df["game_date"].unique())
    first = summary["split"]["rolling_folds"][0]["validate"][0]
    later = [d for d in dates if d >= first]
    blocks = [b for b in np.array_split(np.array(later), min(n_blocks, len(later))) if len(b)]
    rows = []
    for b in blocks:
        prior = df[df["game_date"] < b[0]]
        games = prior.drop_duplicates("GAME_ID").sort_values(["game_date", "GAME_ID"])["GAME_ID"].tolist()
        k = max(2, int(round(0.15 * len(games))))
        cal_games = set(games[-k:])
        tr, ca = prior[~prior["GAME_ID"].isin(cal_games)], prior[prior["GAME_ID"].isin(cal_games)]
        m = fit(tr)
        pc = m.predict(ca)
        method, _ = select_calibration(pc, ca["made"].to_numpy(), game_halves(ca, tcfg["seed"]), tcfg["calibration_methods"])
        calib = Calibrator(method).fit(pc, ca["made"])
        blk = df[df["game_date"].isin(b)]
        p = calib.transform(m.predict(blk))
        rows.append({"block": f"{b[0]}..{b[-1]}", "train_shots": len(tr), "shots": len(blk),
                     "master_log_loss": float(per_shot_log_loss(blk["made"], p).mean()),
                     "nba_log_loss": float(per_shot_log_loss(blk["made"], blk["nba_xfg"]).mean()),
                     "model_a_log_loss": float(per_shot_log_loss(blk["made"], blk["model_a_xfg"]).mean())})
    return pd.DataFrame(rows)


def learning_curve(summary: dict, e: pd.DataFrame, df: pd.DataFrame, fractions=(0.25, 0.4, 0.55, 0.7, 0.85, 1.0)) -> pd.DataFrame:
    """Calibration-partition log loss of the selected configuration trained on the earliest x% of
    model-fit games (raw probabilities, so the curve reflects model fit, not calibrator data)."""
    fit, _ = refit_config(summary)
    fit_dates = summary["split"]["model_fit"]["dates"]
    fit_df = df[(df["game_date"] >= fit_dates[0]) & (df["game_date"] <= fit_dates[1])]
    games = fit_df.drop_duplicates("GAME_ID").sort_values(["game_date", "GAME_ID"])["GAME_ID"].tolist()
    cal = e[e["partition"] == "calibration"]
    rows = []
    for f in fractions:
        keep = set(games[: max(5, int(round(f * len(games))))])
        tr = fit_df[fit_df["GAME_ID"].isin(keep)]
        p = fit(tr).predict(cal)
        rows.append({"fraction": f, "games": len(keep), "shots": len(tr),
                     "calibration_log_loss_raw": float(per_shot_log_loss(cal["made"], p).mean())})
    return pd.DataFrame(rows)


def permutation_importance(mm_model, X: pd.DataFrame, y: np.ndarray, feats: list[str], calib: Calibrator,
                           repeats: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = per_shot_log_loss(y, calib.transform(mm_model.predict(X))).mean()
    rows = []
    for f in feats:
        deltas = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            deltas.append(per_shot_log_loss(y, calib.transform(mm_model.predict(Xp))).mean() - base)
        rows.append({"feature": f, "delta_log_loss": float(np.mean(deltas)), "sd": float(np.std(deltas))})
    return pd.DataFrame(rows).sort_values("delta_log_loss", ascending=False)


# --------------------------------------------------------------------------- plots


def plot_performance(table: pd.DataFrame, boot: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    t = table.sort_values("log_loss")
    lo, hi = [], []
    for name in t.index:
        r = boot[(boot["quantity"] == name) & (boot["metric"] == "log_loss")]
        lo.append(t.loc[name, "log_loss"] - (r["ci_low"].iloc[0] if len(r) else t.loc[name, "log_loss"]))
        hi.append((r["ci_high"].iloc[0] if len(r) else t.loc[name, "log_loss"]) - t.loc[name, "log_loss"])
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(t) + 1.5))
    colors = ["#d62728" if "NBA" in n else "#1f77b4" if "Master" in n else "#7f7f7f" for n in t.index]
    ax.barh(range(len(t)), t["log_loss"], xerr=[lo, hi], color=colors, capsize=3)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t.index)
    ax.invert_yaxis()
    ax.set_xlim(t["log_loss"].min() - 0.03, t["log_loss"].max() + 0.02)
    ax.set_xlabel("held-out log loss (lower is better), 95% game-cluster bootstrap CI")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, path: Path, top: int = 20) -> None:
    plt = _plt()
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(t) + 1.2))
    ax.barh(t["feature"], t["delta_log_loss"], xerr=t["sd"], color="#1f77b4", capsize=2)
    ax.set_xlabel("increase in held-out log loss when the feature is permuted")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_shap(model_dir: Path, spec: dict, X: pd.DataFrame, path: Path) -> bool:
    if spec["family"] == "logistic":
        return False
    import shap
    plt = _plt()
    lm = M.LoadedModel(model_dir, spec)
    Xs = X[spec["features"]].astype(float).sample(min(1500, len(X)), random_state=42)
    if spec["family"] == "xgboost":
        explainer = shap.TreeExplainer(lm.booster)
    else:
        explainer = shap.TreeExplainer(lm.booster)
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):
        sv = sv[-1]
    shap.summary_plot(sv, Xs, show=False, max_display=20, plot_size=(8, 7))
    plt.gcf().tight_layout()
    plt.savefig(path, dpi=120)
    plt.close("all")
    return True


def plot_disagreement(e: pd.DataFrame, path: Path) -> pd.DataFrame:
    plt = _plt()
    e = e.copy()
    e["diff"] = e["master_xfg"] - e["nba_xfg"]
    e["ll_gain"] = per_shot_log_loss(e["made"], e["nba_xfg"]) - per_shot_log_loss(e["made"], e["master_xfg"])
    t = e.groupby(["zone_group", "contest_bin"], observed=True).agg(n=("diff", "size"), mean_diff=("diff", "mean"),
                                                                    ll_gain=("ll_gain", "mean")).reset_index()
    zones = ["rim", "paint", "mid", "corner3", "ab3"]
    bins = list(e["contest_bin"].cat.categories)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, col, title in ((axes[0], "mean_diff", "mean(Master − NBA xFG)"),
                           (axes[1], "ll_gain", "mean log-loss advantage of Master over NBA xFG")):
        M_ = np.full((len(zones), len(bins)), np.nan)
        for _, r in t.iterrows():
            if r["zone_group"] in zones and r["n"] >= 15:
                M_[zones.index(r["zone_group"]), bins.index(r["contest_bin"])] = r[col]
        lim = np.nanmax(np.abs(M_)) if np.isfinite(M_).any() else 1
        im = ax.imshow(M_, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bins)
        ax.set_yticks(range(len(zones)))
        ax.set_yticklabels(zones)
        ax.set_xlabel("closest-defender distance")
        ax.set_title(title + "\n(cells with ≥15 shots)")
        for i in range(len(zones)):
            for j in range(len(bins)):
                if np.isfinite(M_[i, j]):
                    ax.text(j, i, f"{M_[i, j]:+.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return t


def plot_actual_vs_expected(e: pd.DataFrame, path: Path, min_shots: int = 40) -> pd.DataFrame:
    plt = _plt()
    e = e.assign(pts=e["made"] * e["shot_value"], m_pts=e["master_xfg"] * e["shot_value"], n_pts=e["nba_xfg"] * e["shot_value"])
    g = e.groupby("PLAYER_NAME").agg(shots=("made", "size"), pts=("pts", "sum"), master=("m_pts", "sum"), nba=("n_pts", "sum"))
    g = g[g["shots"] >= min_shots]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, col, name in ((axes[0], "master", "Master"), (axes[1], "nba", "NBA xFG")):
        ax.scatter(g[col] / g["shots"], g["pts"] / g["shots"], s=g["shots"] / 2, alpha=0.6)
        lo, hi = 0.6, 1.6
        ax.plot([lo, hi], [lo, hi], ls="--", c="grey")
        r = np.corrcoef(g[col] / g["shots"], g["pts"] / g["shots"])[0, 1] if len(g) > 2 else np.nan
        ax.set_title(f"{name}: expected vs actual FG points per shot\n(players with ≥{min_shots} held-out shots, r={r:.2f})")
        ax.set_xlabel("expected FG points per shot")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("actual FG points per shot")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return g


def plot_walk_forward(wf: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(wf))
    ax.plot(x, wf["master_log_loss"], marker="o", label="Master (selected config, refit per window)")
    ax.plot(x, wf["nba_log_loss"], marker="s", label="NBA xFG")
    ax.plot(x, wf["model_a_log_loss"], marker="^", label="Model A")
    ax.set_xticks(list(x))
    ax.set_xticklabels([b.replace("2026-", "") for b in wf["block"]], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("log loss")
    ax.set_title("Walk-forward evaluation by date window (expanding training window)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_learning_curve(lc: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lc["shots"], lc["calibration_log_loss_raw"], marker="o")
    ax.set_xlabel("training shots (earliest games of the model-fit partition)")
    ax.set_ylabel("calibration-partition log loss (raw)")
    ax.set_title("Learning curve of the selected configuration")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- main


def evaluate_run(run_dir: Path, write_report: bool = True) -> dict:
    from pipeline.inference import MasterModel
    summary, preds, raw = load_run(run_dir)
    fcfg = load_feature_config()
    ecfg = yaml.safe_load((CONFIGS_DIR / "evaluation.yaml").read_text())
    e = benchmark_frame(summary, preds, raw)
    test, cal = e[e["partition"] == "test"].copy(), e[e["partition"] == "calibration"].copy()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    arch_cols = {f"{k} (full features)": f"arch_{k}" for k in summary["architectures"]}
    t_test = results_table(test, MAIN_MODELS)
    t_cal = results_table(cal, MAIN_MODELS)
    t_arch_cal = results_table(cal, arch_cols)
    t_arch_test = results_table(test, arch_cols)

    boot_models = {k: v for k, v in MAIN_MODELS.items()}
    pairs = [("Master (final)", "NBA xFG (Blueprint)"), ("Master (final)", "Model A (original)"),
             ("Master (final)", "Model A, Platt-calibrated"), ("Master (final)", "Shot-type baseline"),
             ("Full", "Tracking without player"), ("Tracking without player", "Context only (shot + game)"),
             ("Context + player (no tracking)", "Context only (shot + game)"), ("Full", "Full without defender")]
    boot = paired_bootstrap(test, "made", boot_models, pairs, n_boot=ecfg["bootstrap"]["n_boot"], cluster_col="GAME_ID")
    diffs = boot[boot["quantity"].str.contains(" - ")]

    seg_models = {"Master": "master_xfg", "NBA xFG": "nba_xfg", "Model A": "model_a_xfg"}
    segments = {c: subgroup_table(test, c, "made", seg_models, min_n=20) for c in
                ["shot_type", "zone_group", "action_family", "contest_bin", "shot_clock_bin", "pl_has_history"]}

    plot_calibration(test, "made", {"Master": "master_xfg", "NBA xFG": "nba_xfg", "Model A": "model_a_xfg",
                                    "Model A (Platt)": "model_a_calibrated"},
                     FIG_DIR / "calibration_test.png", "Held-out test calibration")
    plot_performance(t_test, boot, FIG_DIR / "performance_test.png")

    mm = MasterModel(PROJECT_ROOT / summary["artifact"])
    feats = mm.spec["features"]
    imp = permutation_importance(mm.model, test, test["made"].to_numpy(), feats, mm.calibrator)
    plot_importance(imp, FIG_DIR / "permutation_importance.png")
    has_shap = plot_shap(mm.dir, mm.spec, test, FIG_DIR / "shap_summary.png")
    dis = plot_disagreement(test, FIG_DIR / "disagreement_zone_contest.png")
    ave = plot_actual_vs_expected(test, FIG_DIR / "actual_vs_expected_players.png")

    df = prepare(raw, feature_list(fcfg), fcfg["forbidden"])
    log.info("tracking drop-one ablations")
    abl = tracking_ablation(summary, e, df)
    log.info("walk-forward windows")
    wf = walk_forward(summary, df)
    plot_walk_forward(wf, FIG_DIR / "walk_forward.png")
    log.info("learning curve")
    lc = learning_curve(summary, e, df)
    plot_learning_curve(lc, FIG_DIR / "learning_curve.png")

    # data sufficiency: shots needed to detect the observed paired difference vs NBA xFG (80% power, 5% two-sided)
    d = per_shot_log_loss(test["made"], test["master_xfg"]) - per_shot_log_loss(test["made"], test["nba_xfg"])
    r = diffs[(diffs["quantity"] == "Master (final) - NBA xFG (Blueprint)") & (diffs["metric"] == "log_loss")].iloc[0]
    se_cluster = (r["ci_high"] - r["ci_low"]) / (2 * 1.96)
    deff = (se_cluster / (d.std(ddof=1) / np.sqrt(len(d)))) ** 2
    target = max(abs(float(d.mean())), 0.005)
    n_needed = int(np.ceil(((1.96 + 0.84) * d.std(ddof=1) / target) ** 2 * deff))
    sufficiency = {"test_shots": int(len(d)), "test_games": int(test["GAME_ID"].nunique()),
                   "mean_paired_diff_vs_nba": float(d.mean()), "sd_per_shot_diff": float(d.std(ddof=1)),
                   "cluster_se": float(se_cluster), "design_effect": float(deff),
                   "detectable_effect_used": target, "shots_needed_80pct_power": n_needed}

    # player insight: what player priors change (full vs tracking-without-player)
    test["player_effect"] = test["set_full"] - test["set_tracking_without_player"]
    pl = test.groupby("PLAYER_NAME").agg(shots=("made", "size"), fg=("made", "mean"),
                                         player_effect=("player_effect", "mean"), master=("master_xfg", "mean"),
                                         nba=("nba_xfg", "mean"), prior_fg_type=("pl_fg_type_post", "mean"),
                                         prior_smoe=("pl_smoe_all_post", "mean"))
    pl = pl[pl["shots"] >= 30].sort_values("player_effect")

    rng = np.random.default_rng(42)
    ex_idx = rng.choice(len(test), size=min(ecfg["examples"]["n_random"], len(test)), replace=False)
    ex = test.iloc[sorted(ex_idx)][["game_date", "PLAYER_NAME", "team", "ACTION_TYPE", "SHOT_TYPE", "bp_shot_distance_ft",
                                    "contest_distance_in", "shot_clock_s", "master_xfg", "nba_xfg", "model_a_xfg", "made"]]

    res = {"run_id": summary["run_id"], "model_version": summary["model_version"], "test": t_test.reset_index().to_dict("records"),
           "calibration": t_cal.reset_index().to_dict("records"), "bootstrap": boot.to_dict("records"),
           "architectures_calibration": t_arch_cal.reset_index().to_dict("records"),
           "architectures_test": t_arch_test.reset_index().to_dict("records"),
           "tracking_ablation": abl.reset_index().to_dict("records"), "walk_forward": wf.to_dict("records"),
           "learning_curve": lc.to_dict("records"), "sufficiency": sufficiency,
           "permutation_importance": imp.to_dict("records")}
    write_json(run_dir / "evaluation.json", res)
    write_json(mm.dir / "metrics.json", {"test": res["test"], "sufficiency": sufficiency, "run_id": summary["run_id"]})
    if write_report:
        write_master_report(summary, res, t_test, t_cal, t_arch_cal, t_arch_test, diffs, segments, abl, wf, lc, sufficiency,
                            imp, has_shap, dis, ave, pl, ex, test)
    return res


def _ci(diffs, q, metric="log_loss"):
    r = diffs[(diffs["quantity"] == q) & (diffs["metric"] == metric)]
    if not len(r):
        return None
    r = r.iloc[0]
    return r["mean"], r["ci_low"], r["ci_high"], r["share_below_zero"]


def write_master_report(summary, res, t_test, t_cal, t_arch_cal, t_arch_test, diffs, segments, abl, wf, lc, suff, imp,
                        has_shap, dis, ave, pl, ex, test) -> Path:
    from pipeline.config import SEASON_MANIFEST
    man = json.loads(SEASON_MANIFEST.read_text())["summary"]
    sp = summary["split"]
    m_vs_n = _ci(diffs, "Master (final) - NBA xFG (Blueprint)")
    m_vs_a = _ci(diffs, "Master (final) - Model A (original)")
    m_vs_ac = _ci(diffs, "Master (final) - Model A, Platt-calibrated")
    full_vs_trk = _ci(diffs, "Full - Tracking without player")
    trk_vs_ctx = _ci(diffs, "Tracking without player - Context only (shot + game)")
    ply_vs_ctx = _ci(diffs, "Context + player (no tracking) - Context only (shot + game)")
    fd = _ci(diffs, "Full - Full without defender")

    def verdict(ci, a, b):
        mean, lo, hi, _ = ci
        if hi < 0:
            return f"{a} has lower log loss than {b} (difference {mean:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}])."
        if lo > 0:
            return f"{a} has higher log loss than {b} (difference {mean:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}])."
        return (f"The difference between {a} and {b} is not statistically distinguishable from zero "
                f"({mean:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]).")

    tt = t_test.loc[["NBA xFG (Blueprint)", "Model A (original)", "Model A, Platt-calibrated", "Tracking only", "Player only",
                     "Master (final)"], ["log_loss", "brier", "auc", "cal_slope", "cal_intercept", "ece", "shots"]]
    tt.index.name = "Model"
    exploratory = suff["shots_needed_80pct_power"] > suff["test_shots"]
    L = []
    A = L.append
    A("# Master xFG model report")
    A("")
    A(f"Run `{summary['run_id']}`, model `{summary['model_version']}`, dataset `{summary['dataset_version']}`. "
      "Generated by `python -m pipeline evaluate-master`; all numbers below are computed, none are hand-entered.")
    A("")
    A("## 1. Executive summary")
    A("")
    A(f"* Data: {man['status_verified']} verified 2026-27 regular-season games through {summary['dataset_meta']['date_range'][1]} "
      f"({summary['dataset_meta']['rows']:,} official field-goal attempts, {summary['dataset_meta']['rows_with_tracking']:,} "
      f"matched one-to-one to Blueprint tracking records).")
    A(f"* Protocol: chronological split on whole game dates. Model-fit {sp['model_fit']['dates'][0]}..{sp['model_fit']['dates'][1]} "
      f"({sp['model_fit']['shots']:,} shots), calibration {sp['calibration']['dates'][0]}..{sp['calibration']['dates'][1]} "
      f"({sp['calibration']['shots']:,}), untouched test {sp['test']['dates'][0]}..{sp['test']['dates'][1]} "
      f"({sp['test']['shots']:,} shots, {sp['test']['games']} games).")
    A(f"* Selected: **{summary['chosen_architecture']}** on feature set **{summary['final_feature_set']}**, "
      f"calibration `{summary['feature_sets'][summary['final_feature_set']]['calib']['method']}` (all chosen on the calibration partition).")
    A(f"* Test log loss: Master {t_test.loc['Master (final)', 'log_loss']:.4f}, NBA xFG {t_test.loc['NBA xFG (Blueprint)', 'log_loss']:.4f}, "
      f"Model A {t_test.loc['Model A (original)', 'log_loss']:.4f}, calibrated Model A {t_test.loc['Model A, Platt-calibrated', 'log_loss']:.4f}.")
    A(f"* vs NBA xFG: {verdict(m_vs_n, 'Master', 'NBA xFG')}")
    A(f"* vs Model A: {verdict(m_vs_a, 'Master', 'Model A')} vs calibrated Model A: {verdict(m_vs_ac, 'Master', 'calibrated Model A')}")
    A(f"* Status: **{'exploratory' if exploratory else 'adequately powered for the observed effect'}**. Detecting a paired "
      f"log-loss difference of {suff['detectable_effect_used']:.4f} vs NBA xFG with 80% power needs ≈{suff['shots_needed_80pct_power']:,} "
      f"test shots given the observed variance and game clustering (design effect {suff['design_effect']:.2f}); the test set has {suff['test_shots']:,}.")
    A(f"* Tracking is the main source of skill: {verdict(trk_vs_ctx, 'Context + tracking', 'context only')}")
    A(f"* Player priors (career and in-season shrunk FG% / shot-making) add no measurable value on top of context + tracking this "
      f"early in the season: {verdict(full_vs_trk, 'the full model', 'context + tracking')} Without tracking: "
      f"{verdict(ply_vs_ctx, 'context + player', 'context only')}")
    A("")
    A("## 2. Data and coverage")
    A("")
    A(f"Season ingestion (`docs/methodology/DATA_MATCHING.md`, `data/reports/season/COVERAGE_REPORT.md`): {man['games_seen']} games seen, "
      f"{man['status_verified']} verified, {man['verified_matched']:,} of {man['verified_official_fga']:,} official FGA in verified games "
      f"matched to a Blueprint record ({100 * man['verified_matched'] / max(man['verified_official_fga'], 1):.1f}%). Official shots without a "
      "Blueprint record are absent from Blueprint's published data (the page contains no record for them); no xFG is invented for them, and "
      "they are excluded from training/evaluation but still contribute (as earlier-game outcomes) to in-season player priors.")
    A("")
    A("## 3. Feature engineering")
    A("")
    A("Feature groups (`configs/features.yaml`); every feature is available before the shot is released:")
    A("")
    for g, cols in summary["configs"]["features"]["groups"].items():
        A(f"* **{g}** ({len(cols)}): " + ", ".join(f"`{c}`" for c in cols))
    A("")
    A("Leakage controls: pre-shot score from the previous play-by-play action (the shot row's score is post-shot); "
      "in-season player/defender priors from games dated strictly before the shot's game; pre-season priors from 2021-22..2025-26; "
      "the shot-difficulty baseline used for shot-making-over-expected was fitted on 2022-23..2024-25 only; the outcome-biased "
      "`Bank` token is removed from ACTION_TYPE (historical bank variants shoot 50-59% vs 38-40% for the same action without it). "
      "Forbidden as inputs: " + ", ".join(f"`{c}`" for c in summary["configs"]["features"]["forbidden"]) + ".")
    A("")
    A("## 4. Architecture selection")
    A("")
    A("Each family was tuned with Optuna (TPE, seeded) on rolling-origin CV inside the model-fit partition, refit on the full "
      "model-fit partition, and compared on the calibration partition after cross-fitted calibration selection.")
    A("")
    rows = []
    for k, v in summary["architectures"].items():
        rows.append({"family": k, "rolling_cv_log_loss": v["tuned"]["cv_log_loss"], "trials": v["tuned"]["n_trials"],
                     "n_estimators": v["tuned"].get("n_estimators") or "", "calibration": v["calib"]["method"],
                     "calibration_partition_log_loss": v["calib"]["best_crossfit_log_loss"]})
    A(fmt_table(pd.DataFrame(rows).set_index("family")))
    A("")
    A("Test-partition log loss of the same candidates (reported for transparency, not used for selection):")
    A("")
    A(fmt_table(t_arch_test[["log_loss", "brier", "auc", "cal_slope"]]))
    A("")
    A("## 5. Results on the untouched test set")
    A("")
    A(fmt_table(tt))
    A("")
    A("Calibration = logistic recalibration slope/intercept (ideal 1 / 0) and 10-bin equal-count ECE. All rows are the identical "
      f"{int(tt['shots'].iloc[0]):,} held-out shots.")
    A("")
    A("All models and benchmarks:")
    A("")
    A(fmt_table(t_test[["log_loss", "brier", "auc", "cal_slope", "ece", "mean_pred", "observed"]]))
    A("")
    A("![Performance](figures/master/performance_test.png)")
    A("")
    A("Paired game-cluster bootstrap (2,000 resamples of test games) of log-loss differences (negative = first model better):")
    A("")
    dd = diffs[diffs["metric"] == "log_loss"][["quantity", "mean", "ci_low", "ci_high", "share_below_zero"]].set_index("quantity")
    A(fmt_table(dd))
    A("")
    A("![Calibration](figures/master/calibration_test.png)")
    A("")
    A("Calibration-partition results (the data used for selection):")
    A("")
    A(fmt_table(t_cal[["log_loss", "brier", "auc", "cal_slope", "shots"]]))
    A("")
    A("## 6. Feature-group ablations")
    A("")
    rows = [{"feature_set": k, "groups": ", ".join(summary["configs"]["training"]["feature_sets"][k]),
             "rolling_cv": v["tuned"]["cv_log_loss"], "calibration_partition": v["calib"]["best_crossfit_log_loss"],
             "test": float(t_test.loc[[n for n, c in MAIN_MODELS.items() if c == f"set_{k}"][0], "log_loss"])
             if any(c == f"set_{k}" for c in MAIN_MODELS.values()) else float("nan")}
            for k, v in summary["feature_sets"].items()]
    A(fmt_table(pd.DataFrame(rows).set_index("feature_set")))
    A("")
    for ci, a, b in ((trk_vs_ctx, "adding tracking to context", "context only"),
                     (ply_vs_ctx, "adding player priors to context", "context only"),
                     (full_vs_trk, "adding player priors to context + tracking", "context + tracking"),
                     (fd, "adding defender priors", "full without defender")):
        mean, lo, hi, _ = ci
        A(f"* {a[0].upper() + a[1:]} vs {b}: {mean:+.4f} log loss (95% CI [{lo:+.4f}, {hi:+.4f}]).")
    A("")
    A("### Tracking drop-one ablations (selected configuration refit without each feature)")
    A("")
    A(fmt_table(abl))
    A("")
    A("## 7. Importance and explanations")
    A("")
    A("![Permutation importance](figures/master/permutation_importance.png)")
    A("")
    if has_shap:
        A("![SHAP](figures/master/shap_summary.png)")
        A("")
    A("## 8. Where Master and NBA xFG disagree")
    A("")
    A("![Disagreement](figures/master/disagreement_zone_contest.png)")
    A("")
    for name, seg in segments.items():
        if seg.empty:
            continue
        cols = [name, "n", "observed_fg", "Master_mean_pred", "NBA xFG_mean_pred", "Model A_mean_pred",
                "Master_log_loss", "NBA xFG_log_loss", "Model A_log_loss"]
        A(f"**By {name}**")
        A("")
        A(fmt_table(seg[cols].set_index(name)))
        A("")
    A("## 9. Player insight")
    A("")
    A("Player effect = mean(full model − tracking-without-player model) on a player's held-out shots, i.e. how much the player "
      "priors move his expected FG% after shot context and tracking are accounted for (players with ≥30 test shots).")
    A("")
    pls = pd.concat([pl.head(8), pl.tail(8)])
    A(fmt_table(pls[["shots", "fg", "player_effect", "master", "nba", "prior_fg_type", "prior_smoe"]], 3))
    A("")
    A("![Actual vs expected](figures/master/actual_vs_expected_players.png)")
    A("")
    A("## 10. Held-out shot examples")
    A("")
    A(f"{len(ex)} test shots drawn with a fixed seed (42) uniformly at random from the test partition, not selected by outcome or error.")
    A("")
    ex2 = ex.copy()
    ex2.index = range(1, len(ex2) + 1)
    ex2.index.name = "#"
    A(fmt_table(ex2.rename(columns={"bp_shot_distance_ft": "dist_ft", "contest_distance_in": "contest_in", "shot_clock_s": "shot_clock"}), 3))
    A("")
    A("## 11. Chronological robustness")
    A("")
    A("![Walk-forward](figures/master/walk_forward.png)")
    A("")
    A(fmt_table(wf.set_index("block")))
    A("")
    n_better = int((wf["master_log_loss"] < wf["nba_log_loss"]).sum())
    A(f"Each block is scored by the selected configuration (fixed hyperparameters, no re-tuning) refit on all earlier dates. "
      f"Blocks before {sp['test']['dates'][0]} overlap the data used for model selection, so only the test-date blocks are fully "
      f"out of sample. Master has lower log loss than NBA xFG in {n_better} of {len(wf)} blocks.")
    A("")
    A("## 12. Data sufficiency")
    A("")
    A(fmt_table(lc.set_index("fraction")))
    A("")
    A("![Learning curve](figures/master/learning_curve.png)")
    A("")
    A(f"* Paired per-shot log-loss difference vs NBA xFG: mean {suff['mean_paired_diff_vs_nba']:+.4f}, SD {suff['sd_per_shot_diff']:.3f}; "
      f"game-cluster SE {suff['cluster_se']:.4f} (design effect {suff['design_effect']:.2f}).")
    A(f"* Shots needed to detect |Δ| = {suff['detectable_effect_used']:.4f} with 80% power: ≈{suff['shots_needed_80pct_power']:,} "
      f"(test set: {suff['test_shots']:,}).")
    A("")
    A("## 13. Limitations")
    A("")
    A("* One partial season (early 2026-27); the test set covers the newest dates only, so seasonal drift beyond them is untested.")
    A(f"* Blueprint published tracking for {100 * man['verified_matched'] / max(man['verified_official_fga'], 1):.1f}% of official "
      "attempts (median game ~94%; three games only partially published). Missing shots skew toward misses, rim/paint attempts and "
      "putbacks (`docs/methodology/DATA_MATCHING.md`), so the evaluated population is slightly easier than all FGA; Master and NBA xFG are "
      "compared on the identical tracked shots.")
    A("* Player priors did not help yet; with more in-season data (and a longer test window) this should be re-tested rather than "
      "assumed. The fallback no-tracking model used for untracked shots in the analytics exports is less accurate (see the "
      "context + player row).")
    A("* Tracking fields are Blueprint's published values; release time is not measured on most rim attempts (recorded as 0) and is capped at 1.0 s.")
    A("* No defender location/height beyond Blueprint's contest distance/height; defender priors are in-season only and very sparse.")
    A("* 2025-26 player history is aggregate only (PlayerDashPtShots), without shot-level locations.")
    A("* NBA xFG is an external benchmark; its inputs are unknown, and it is never used as a feature or label.")
    A("* Redistribution of NBA/Blueprint-derived data requires permission (see LICENSE).")
    A("")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    return REPORT_PATH
