"""Audit numbers, player diagnostics and figures behind docs/reports/PLAYER_SKILL_MODELING.md and
docs/reports/MASTER_V2_COMPARISON.md. Writes small CSV/JSON tables to docs/reports/player_skill/ and
figures to docs/reports/figures/player_skill/.

Diagnostic players are chosen by rules fixed before looking at model results (no cherry-picking):
  named           Luka Doncic, Shai Gilgeous-Alexander, Stephen Curry
  catch_shoot     >= 300 historical 3PA, >= 85% of 3PA from catch-and-shoot ("jump" family), 3PA >= 60% of FGA
  rim_finisher    >= 300 historical FGA, rim share >= 55%
  guard           listed "G" in 2026-27 box scores (starters only; heights unavailable -> no "small" filter)
  limited_history < 200 historical FGA (including none)
For each group: aggregate over all members, plus the 5 members with the most 2026-27 tracked attempts.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import DASH_CLOSEST_PATH, GAME_PAGES_DIR, MASTER_DATASET_DIR, PROJECT_ROOT
from pipeline.player_skill import audit as au
from pipeline.player_skill import contest as ct
from pipeline.player_skill import physical as ph
from pipeline.player_skill import taxonomy as tx
from pipeline.player_skill.history import load_history
from pipeline.utils import get_logger, write_json

log = get_logger("player_skill.report")

OUT_DIR = PROJECT_ROOT / "docs" / "reports" / "player_skill"
FIG_DIR = PROJECT_ROOT / "docs" / "reports" / "figures" / "player_skill"
V1_PRED = PROJECT_ROOT / "models" / "experiments" / "master_runs" / "20261104T210006" / "predictions.parquet"
NAMED = {1629029: "Luka Doncic", 1628983: "Shai Gilgeous-Alexander", 201939: "Stephen Curry"}


def _walk(o, found):
    if isinstance(o, dict):
        if "personId" in o and o.get("position"):
            found.append((int(o["personId"]), o["position"]))
        for v in o.values():
            _walk(v, found)
    elif isinstance(o, list):
        for v in o:
            _walk(v, found)


def box_positions() -> pd.Series:
    found = []
    for f in sorted(Path(GAME_PAGES_DIR).glob("*.next_data.json.gz")):
        with gzip.open(f, "rt") as fh:
            _walk(json.load(fh), found)
    if not found:
        return pd.Series(dtype=str)
    d = pd.DataFrame(found, columns=["PLAYER_ID", "pos"])
    return d.groupby("PLAYER_ID")["pos"].agg(lambda s: s.mode().iloc[0])


def audit_tables(h: pd.DataFrame, v2: pd.DataFrame) -> dict:
    out = {}
    sv = au.skill_variance(h, "xb")
    sv_in = au.skill_variance(h, "xb_insample")
    sv.to_csv(OUT_DIR / "skill_variance.csv", index=False)
    rel = pd.concat([au.season_reliability(h, "xb", 300, "all"), au.season_reliability(h, "xb", 150, "type"),
                     au.season_reliability(h, "xb", 150, "zone")])
    rel.to_csv(OUT_DIR / "season_reliability.csv", index=False)
    tr = v2[v2["in_blueprint"].astype(bool)]
    p = tr["baseline_hist_xfg"].to_numpy(float)
    tau_all = float(sv.loc[sv["level"] == "all", "tau_over_expected"].iloc[0])
    out["ceiling"] = {
        "tau_all_over_expected": tau_all,
        "tau_all_over_expected_insample_baseline": float(sv_in.loc[sv_in["level"] == "all", "tau_over_expected"].iloc[0]),
        "perfect_knowledge_general_level": au.ceiling(tau_all, p),
        "realistic_general_level_with_2026_reliability": au.ceiling(tau_all, p, tr["ps_rel_all"].to_numpy()),
    }
    cov = au.coverage(tr, h)
    out["coverage"] = {c: {"zero": float((cov[c] == 0).mean()), "lt50": float((cov[c] < 50).mean()),
                           "lt200": float((cov[c] < 200).mean()), "median": float(cov[c].median())} for c in cov}
    dash = pd.read_csv(DASH_CLOSEST_PATH)
    d26 = set(dash.loc[dash["SEASON"] == "2025-26", "PLAYER_ID"])
    nohist = tr[cov["hist_att_all"] == 0]
    out["coverage"]["no_history_shots"] = int(len(nohist))
    out["coverage"]["no_history_players"] = int(nohist["PLAYER_ID"].nunique())
    out["coverage"]["no_history_shots_with_dash_2025_26"] = float(nohist["PLAYER_ID"].isin(d26).mean())
    # contest compatibility (nominal vs quantile-mapped thresholds)
    fit = tr[tr["game_date"] <= "2026-10-27"]
    thr = ct.quantile_thresholds(dash, fit)
    ct.compatibility_audit(dash, tr).to_csv(OUT_DIR / "contest_compat_nominal.csv", index=False)
    ct.compatibility_audit(dash, tr, thresholds=thr).to_csv(OUT_DIR / "contest_compat_mapped.csv", index=False)
    out["contest_thresholds_in"] = thr
    tunes = {}
    for target, hist_seasons in (("2025-26", ["2021-22", "2022-23", "2023-24", "2024-25"]),
                                 ("2024-25", ["2021-22", "2022-23", "2023-24"])):
        r = ct.tune_contest(dash, target, hist_seasons, 730.0)
        tunes[target] = {"dev": r["dev"], "sens": r["sens"], "cells": r["cells"]}
    out["contest_tuning_binomial_delta"] = tunes
    # archetypes
    arch = ph.fit_archetypes(ph.diet_matrix(h, min_att=200))
    summ = ph.archetype_summary(h, arch["labels"])
    summ.to_csv(OUT_DIR / "archetypes.csv")
    out["archetypes"] = {"k": arch["k"], "silhouette": arch["silhouette"],
                         "between_archetype_sd_over_expected": float(np.sqrt(np.average(
                             (summ["over_expected"] - np.average(summ["over_expected"], weights=summ["shots"])) ** 2,
                             weights=summ["shots"])))}
    # V1 prior and V2 feature signal on top of the V1 tracking model: calibration partition only
    from pipeline.training.splits import game_halves
    pr = pd.read_parquet(V1_PRED)
    m = pr[pr["partition"] == "calibration"][["shot_uid", "master_xfg"]].merge(v2, on="shot_uid")
    m = m.sort_values(["game_date", "GAME_ID", "GAME_EVENT_ID"]).reset_index(drop=True)
    a, b = game_halves(m, 42)
    feats = [c for c in v2.columns if c.startswith(("pl_", "def_"))] + [
        "ps_oe_all", "ps_oe_action", "ps_oe_zone_action", "ps_delta", "ps_delta_all", "ps_delta_action", "ps_glmm_delta",
        "ps_glmm_joint_delta", "ps_raw_fg_zone_action", "ps_contest_dev", "ps_contest_sens_x_open", "ps_release_vertical"]
    r1, r2 = au.prior_signal(m, "master_xfg", feats, a, b), au.prior_signal(m, "master_xfg", feats, b, a)
    sig = r1[["feature", "corr_with_base_logit"]].assign(
        coef_per_sd=(r1["coef_per_sd"] + r2["coef_per_sd"]) / 2,
        crossfit_delta=(r1["eval_delta"] * b.sum() + r2["eval_delta"] * a.sum()) / len(m))
    sig.to_csv(OUT_DIR / "feature_signal_calibration.csv", index=False)
    corr = tr[[c for c in v2.columns if c.startswith("pl_")]].corr()
    corr.round(3).to_csv(OUT_DIR / "v1_player_feature_correlations.csv")
    out["v1_player_features_high_corr_pairs"] = int(((corr.abs() > 0.8).sum().sum() - len(corr)) / 2)
    return out


def _hist_player(h: pd.DataFrame) -> pd.DataFrame:
    zone = tx.skill_zone(h["zone_group"], h["shot_distance_ft"])
    g = h.assign(rim=(zone == "rim"), three=h["is_three"] == 1,
                 cs3=(h["is_three"] == 1) & (h["action_family"] == "jump")).groupby("PLAYER_ID")
    out = pd.DataFrame({"hist_att": g.size(), "hist_fg": g["made"].mean(), "hist_xfg": g["xb"].mean(),
                        "rim_share": g["rim"].mean(), "three_share": g["three"].mean(), "cs3": g["cs3"].sum(),
                        "three_att": g["three"].sum()})
    out["hist_resid"] = out["hist_fg"] - out["hist_xfg"]
    out["cs_share_of_3pa"] = out["cs3"] / out["three_att"].where(out["three_att"] > 0)
    return out


def player_diagnostics(h: pd.DataFrame, v2: pd.DataFrame, comparison_dir: Path) -> pd.DataFrame:
    preds = pd.read_parquet(comparison_dir / "predictions.parquet")
    cols = ["p_v1_control", "p_adj_eb", "p_prior_eb_fixed", "p_adj_glmm"]
    pp = preds[["shot_uid", "partition"] + cols]
    tr = v2[v2["in_blueprint"].astype(bool)]
    d = pp.merge(tr, on="shot_uid", how="inner")
    hp = _hist_player(h)
    pos = box_positions()
    last = tr.sort_values("game_date").groupby("PLAYER_ID").tail(1).set_index("PLAYER_ID")
    last3 = tr[tr["is_three"] == 1].sort_values("game_date").groupby("PLAYER_ID").tail(1).set_index("PLAYER_ID")
    last2 = tr[tr["is_three"] == 0].sort_values("game_date").groupby("PLAYER_ID").tail(1).set_index("PLAYER_ID")
    g = d.groupby("PLAYER_ID")
    eval_ = pd.DataFrame({"name": g["PLAYER_NAME"].first(), "eval_shots": g.size(), "eval_fg": g["made"].mean(),
                          "v1_mean": g["p_v1_control"].mean(), "v2_adj_eb_mean": g["p_adj_eb"].mean(),
                          "nba_xfg_mean": g["nba_xfg"].mean()})
    for c in ("p_v1_control", "p_adj_eb"):
        eval_[f"ll_{c[2:]}"] = d.assign(ll=-(d["made"] * np.log(d[c]) + (1 - d["made"]) * np.log(1 - d[c]))).groupby("PLAYER_ID")["ll"].mean()
    df = eval_.join(hp, how="left")
    df["hist_att"] = df["hist_att"].fillna(0)
    df["root_prior"] = last["ps_has_history"].reindex(df.index).map({1: "veteran", 0: "newcomer"})
    df["theta_all"] = last["ps_oe_all"].reindex(df.index)
    df["neff_all"] = last["ps_neff_all"].reindex(df.index)
    df["rel_all"] = last["ps_rel_all"].reindex(df.index)
    df["theta_three"] = last3["ps_oe_type"].reindex(df.index)
    df["theta_two"] = last2["ps_oe_type"].reindex(df.index)
    df["position"] = pos.reindex(df.index)
    groups = {
        "named": df.index.isin(list(NAMED)),
        "catch_shoot": (df["three_att"] >= 300) & (df["cs_share_of_3pa"] >= 0.85) & (df["three_share"] >= 0.60),
        "rim_finisher": (df["hist_att"] >= 300) & (df["rim_share"] >= 0.55),
        "guard": df["position"].eq("G"),
        "limited_history": df["hist_att"] < 200,
    }
    rows = []
    for gname, mask in groups.items():
        sub = df[mask]
        top = sub if gname == "named" else sub.sort_values("eval_shots", ascending=False).head(5)
        for pid, r in top.iterrows():
            rows.append({"group": gname, "PLAYER_ID": pid, **r.to_dict()})
        w = sub["eval_shots"]
        if w.sum() > 0:
            rows.append({"group": gname, "PLAYER_ID": -1, "name": f"ALL {gname} ({len(sub)} players)",
                         "eval_shots": int(w.sum()), "eval_fg": np.average(sub["eval_fg"], weights=w),
                         "v1_mean": np.average(sub["v1_mean"], weights=w), "v2_adj_eb_mean": np.average(sub["v2_adj_eb_mean"], weights=w),
                         "ll_v1_control": np.average(sub["ll_v1_control"], weights=w), "ll_adj_eb": np.average(sub["ll_adj_eb"], weights=w)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "player_diagnostics.csv", index=False)
    return out


def figures(h: pd.DataFrame, comparison: dict, tuning: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # season-to-season stability of over-expected shooting
    r = (h["made"] - h["xb"]).rename("r")
    g = h.assign(r=r).groupby(["PLAYER_ID", "SEASON"]).agg(n=("r", "size"), r=("r", "mean")).reset_index()
    a = g[g["SEASON"] == "2023-24"].set_index("PLAYER_ID")
    b = g[g["SEASON"] == "2024-25"].set_index("PLAYER_ID")
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    j = j[(j["n_a"] >= 300) & (j["n_b"] >= 300)]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(j["r_a"], j["r_b"], s=10, alpha=0.6)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("2023-24 FG% over expected")
    ax.set_ylabel("2024-25 FG% over expected")
    ax.set_title(f"Season-to-season stability\nr = {np.corrcoef(j['r_a'], j['r_b'])[0, 1]:.2f}, "
                 f"{len(j)} players with >= 300 FGA in both seasons", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "season_stability.png", dpi=130)
    plt.close(fig)
    # validation gain by hierarchy level
    gains = tuning["best_gains"]
    lv = ["delta_all", "delta_type", "delta_zone", "delta_action", "delta_zone_action"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar([x.replace("delta_", "") for x in lv], [-gains[x] * 1000 for x in lv])
    ax.set_ylabel("log-loss reduction (x1000)")
    ax.set_title("2024-25 out-of-sample gain over the player-agnostic baseline")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hierarchy_level_gains.png", dpi=130)
    plt.close(fig)
    # test deltas vs V1 with game-clustered CIs
    bs = pd.DataFrame(comparison["bootstrap"])
    bs = bs[(bs["metric"] == "log_loss") & bs["quantity"].str.endswith("- v1_control") & ~bs["quantity"].str.startswith("v1_refit")]
    bs = bs.assign(name=bs["quantity"].str.replace(" - v1_control", "", regex=False)).sort_values("mean")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = np.arange(len(bs))
    ax.errorbar(bs["mean"] * 1000, y, xerr=[(bs["mean"] - bs["ci_low"]) * 1000, (bs["ci_high"] - bs["mean"]) * 1000],
                fmt="o", capsize=3)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(bs["name"])
    ax.set_xlabel("test log loss minus Master V1 (x1000), 95% game-clustered CI")
    ax.set_title("Master V2 candidates vs V1 (Oct 31-Nov 2, second use)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v2_test_deltas.png", dpi=130)
    plt.close(fig)


def build(comparison_dir: Path, tuning_path: Path) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    h = load_history()
    v2 = pd.read_parquet(MASTER_DATASET_DIR / "master_v2" / "master_shots.parquet")
    comparison = json.loads((comparison_dir / "comparison.json").read_text())
    tuning = json.loads(tuning_path.read_text())
    out = audit_tables(h, v2)
    diag = player_diagnostics(h, v2, comparison_dir)
    figures(h, comparison, tuning)
    write_json(OUT_DIR / "audit.json", out)
    log.info("wrote %s (%d diagnostic rows)", OUT_DIR, len(diag))
    return out


__all__ = ["audit_tables", "box_positions", "build", "figures", "player_diagnostics"]
