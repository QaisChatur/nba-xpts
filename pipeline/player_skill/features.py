"""As-of-date player-skill features for every 2026-27 shot -> master_v2 dataset.

For each game date D, only evidence dated strictly before D is used:
  * historical shots 2022-25 (out-of-fold expected makes),
  * official 2026-27 shots of earlier dates (expected makes from the pre-season baseline),
  * PlayerDashPtShots 2021-26 for contest priors / the joint model.
Master V1 columns are copied unchanged; new columns are prefixed `ps_`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline.config import CONFIGS_DIR, DASH_CLOSEST_PATH, MASTER_DATASET_DIR
from pipeline.player_skill import contest as ct
from pipeline.player_skill import glmm as gm
from pipeline.player_skill import hierarchy as hi
from pipeline.player_skill import physical as ph
from pipeline.player_skill import taxonomy as tx
from pipeline.player_skill.history import HISTORY_PATH, load_history
from pipeline.utils import get_logger, sha256_file, utc_now, write_json

log = get_logger("player_skill.features")

MAIN_GROUPS = ["player", "type", "zone", "action", "zone_action"]


def load_config(path: Path = CONFIGS_DIR / "player_skill.yaml") -> dict:
    return yaml.safe_load(path.read_text())


def skill_params(cfg: dict) -> hi.SkillParams:
    c = cfg["hierarchy"]
    return hi.SkillParams(k=dict(c["k"]), k_raw=dict(c["k_raw"]), half_life_days=c["half_life_days"],
                          mu_veteran=c["mu_veteran"], mu_newcomer=c["mu_newcomer"], w_dash=c["w_dash"])


def _season_evidence(u: pd.DataFrame) -> pd.DataFrame:
    return hi.evidence_from_shots(u.assign(made=u["made"].astype(float)), "baseline_hist_xfg")


def build_features(u: pd.DataFrame, h: pd.DataFrame, dash: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    params = skill_params(cfg)
    league = hi.League.from_history(h)
    ev_hist = hi.evidence_from_shots(h, "xb")
    ev_season = _season_evidence(u)
    ev = pd.concat([ev_hist, ev_season], ignore_index=True)

    # contest: quantile mapping from model-fit games, Dash tables as of the season start
    cc = cfg["contest"]
    tracked = u["in_blueprint"].astype(bool)
    thr = ct.quantile_thresholds(dash, u[tracked & (u["game_date"] <= cc["threshold_partition_end"])])
    dev, sens, share = ct.contest_tables(dash, cfg["preseason_cutoff"], cc["dash_seasons"], cc["half_life_days"],
                                         cc["k_dev"], cc["k_sens"])
    rng = ct.mapped_range(u, thr)
    typ = np.where(u["is_three"].astype(int) == 1, "three", "two")
    u_contest = pd.Series(np.where(tracked & (rng != "unknown"), pd.Series(typ, index=u.index) + "|" + rng, None),
                          index=u.index)

    # joint-model rows: shots (+ contest key for tracked in-season shots) and Dash binomial rows
    gc = cfg["glmm"]
    ev_joint = pd.concat([ev_hist.assign(cat_contest=None), ev_season.assign(cat_contest=u_contest.to_numpy()),
                          gm.dash_rows(dash, cc["dash_seasons"], gc["joint_dash_weight"])], ignore_index=True)

    # archetypes from historical diet; refreshed per date with in-season diet for every player
    diet_hist = ph.diet_matrix(h, min_att=200)
    arch = ph.fit_archetypes(diet_hist)
    log.info("archetypes: k=%d silhouette=%s", arch["k"], {k: round(v, 3) for k, v in arch["silhouette"].items()})

    frames = []
    for day in sorted(u["game_date"].unique()):
        rows = u.index[u["game_date"] == day]
        shots = u.loc[rows]
        f = hi.skill_as_of(ev, shots, day, params, league, preseason_cutoff=cfg["preseason_cutoff"])
        out = pd.DataFrame(index=rows)
        for lvl in hi.LEVELS:
            out[f"ps_oe_{lvl}"] = f[f"theta_{lvl}"]
            out[f"ps_rel_{lvl}"] = f[f"rel_{lvl}"]
            out[f"ps_neff_{lvl}"] = f[f"neff_{lvl}"]
            out[f"ps_att_{lvl}"] = f[f"att_{lvl}"]
            out[f"ps_raw_fg_{lvl}"] = f[f"raw_fg_{lvl}"]
            out[f"ps_raw_oe_{lvl}"] = f[f"raw_theta_{lvl}"]
        out["ps_delta"] = f["delta_zone_action"]
        out["ps_delta_all"] = f["delta_all"]
        out["ps_delta_action"] = f["delta_action"]
        out["ps_has_history"] = f["has_history"]

        cats = tx.categories(shots)
        main = gm.fit(gm.shot_rows(ev, day, gc["half_life_days"]), gc["lambda"], groups=MAIN_GROUPS)
        kshots = cats.assign(PLAYER_ID=shots["PLAYER_ID"].to_numpy())
        kshots.columns = [c if c == "PLAYER_ID" else f"cat_{c}" for c in kshots.columns]
        out["ps_glmm_delta"] = main.intercept + main.effect(kshots, MAIN_GROUPS)
        joint = gm.fit(gm.shot_rows(ev_joint, day, gc["half_life_days"]), gc["lambda"], groups=gm.GROUPS)
        kj = kshots.assign(cat_contest=u_contest.loc[rows].to_numpy())
        out["ps_glmm_joint_delta"] = joint.intercept + joint.effect(kj, gm.GROUPS)
        out["ps_glmm_joint_contest"] = joint.effect(kj, ["contest"])

        prior_season = u[u["game_date"] < day]
        diet_now = ph.diet_matrix(pd.concat([h[["PLAYER_ID", "zone_group", "shot_distance_ft", "action_family"]],
                                             prior_season[["PLAYER_ID", "zone_group", "shot_distance_ft", "action_family"]]]))
        lab = ph.assign(arch, diet_now)
        out["ps_archetype"] = shots["PLAYER_ID"].astype(int).map(lab).fillna(-1).astype(int).to_numpy()
        frames.append(out)
        log.info("%s: %d shots", day, len(rows))
    feats = pd.concat(frames).loc[u.index]
    feats = feats.join(ct.contest_features(u, dev, sens, share, thr))
    feats["ps_release_vertical"] = ph.release_vertical_prior(u)
    meta = {"contest_thresholds_in": thr, "archetypes": {"k": arch["k"], "silhouette": arch["silhouette"]}}
    return feats, meta


def build_master_v2(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    src = MASTER_DATASET_DIR / cfg["base_dataset_version"]
    u = pd.read_parquet(src / "master_shots.parquet")
    h = load_history()
    dash = pd.read_csv(DASH_CLOSEST_PATH)
    feats, fmeta = build_features(u, h, dash, cfg)
    out = u.join(feats)
    d = MASTER_DATASET_DIR / cfg["dataset_version"]
    d.mkdir(parents=True, exist_ok=True)
    path = d / "master_shots.parquet"
    out.to_parquet(path, index=False)
    base_meta = json.loads((src / "dataset_meta.json").read_text())
    meta = {
        "dataset_version": cfg["dataset_version"], "built_utc": utc_now(), "base_dataset": cfg["base_dataset_version"],
        "base_dataset_sha256": sha256_file(src / "master_shots.parquet"), "rows": int(len(out)),
        "history_sha256": sha256_file(HISTORY_PATH), "player_skill_config": cfg, "new_columns": list(feats.columns),
        "date_range": base_meta["date_range"], **fmeta,
    }
    write_json(d / "dataset_meta.json", meta)
    log.info("wrote %s (%d rows, %d new columns)", path, len(out), feats.shape[1])
    return path


__all__ = ["build_features", "build_master_v2", "load_config", "skill_params"]
