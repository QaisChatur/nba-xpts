"""Chronological tuning of the hierarchical skill parameters on historical seasons (never on 2026-27).

Validation target: 2024-25 shots, scored month by month with evidence dated before each month
(2022-23, 2023-24 and the earlier 2024-25 months), exactly like in-season use. Objective: log loss of
sigmoid(logit(xb_oof) + delta), where xb_oof is the out-of-fold shot-difficulty expectation.

  1. half-life grid; for each, coordinate search of k per level
  2. veteran / newcomer root means (players with vs without evidence before the season)
  3. weight of last-season PlayerDashPtShots aggregates, in the production-analog setup:
     shot-level history ends one season before the target season and the gap season is known only
     through Dash aggregates (2022-23 shots + 2023-24 Dash -> 2024-25, as 2022-25 shots + 2025-26
     Dash -> 2026-27)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.player_skill import hierarchy as hi
from pipeline.utils import get_logger

log = get_logger("player_skill.tuning")

TARGET_SEASON = "2024-25"
TARGET_START = "2024-10-01"


def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def level_gains(prep: dict, params: hi.SkillParams, league: hi.League) -> dict:
    """Validation log loss of the baseline alone and with the skill effect truncated at each level."""
    xb = np.clip(prep["xb"], 1e-4, 1 - 1e-4)
    base = _ll(prep["y"], xb)
    out = {"baseline_oof": base}
    for lvl in hi.LEVELS:
        out[f"delta_{lvl}"] = hi.validation_loss(prep, params, league, lvl) - base
    return out


def root_means(h: pd.DataFrame) -> tuple[float, float]:
    """Mean over-expected rate in 2024-25 of players with / without shots in 2022-24."""
    prior = set(h.loc[h["SEASON"] != TARGET_SEASON, "PLAYER_ID"])
    t = h[h["SEASON"] == TARGET_SEASON]
    vet = t["PLAYER_ID"].isin(prior)
    r = t["made"] - t["xb"]
    return float(r[vet].mean()), float(r[~vet].mean())


def tune_all(h: pd.DataFrame, dash: pd.DataFrame, half_lives=(365.0, 730.0, 1460.0, None)) -> dict:
    league = hi.League.from_history(h)
    ev = hi.evidence_from_shots(h, "xb")
    mu_vet, mu_new = root_means(h)
    results = {"root_means": {"veteran": mu_vet, "newcomer": mu_new}, "half_life": {}}
    best = None
    for hl in half_lives:
        prep = hi.prepare_validation(h, ev, TARGET_SEASON, hl, TARGET_START)
        start = hi.SkillParams(half_life_days=hl, mu_veteran=mu_vet, mu_newcomer=mu_new)
        p = hi.tune(prep, league, start)
        g = level_gains(prep, p, league)
        log.info("half-life %s: k=%s gains=%s", hl, p.k, {k: round(v, 5) for k, v in g.items()})
        results["half_life"][str(hl)] = {"k": p.k, "gains": g}
        if best is None or g["delta_zone_action"] < best[1]["delta_zone_action"]:
            best = (p, g, prep)
    params, gains, prep = best
    # root means: compare tuned values with plain 0
    zero = hi.SkillParams.from_dict({**params.to_dict(), "mu_veteran": 0.0, "mu_newcomer": 0.0})
    results["root_mean_effect"] = hi.validation_loss(prep, params, league) - hi.validation_loss(prep, zero, league)
    # Dash weight, production-analog setup
    analog = h[h["SEASON"] != "2023-24"]
    ev_a = hi.evidence_from_shots(analog, "xb")
    prep_cache = {}
    dash_res = {}
    for w in (0.0, 0.25, 0.5, 1.0):
        e = ev_a if w == 0 else pd.concat([ev_a, hi.evidence_from_dash(dash, "2023-24", "2024-06-30", w)],
                                          ignore_index=True)
        prep_cache[w] = hi.prepare_validation(analog, e, TARGET_SEASON, params.half_life_days, TARGET_START)
        dash_res[w] = level_gains(prep_cache[w], params, league)["delta_zone_action"]
    results["w_dash_analog"] = dash_res
    params.w_dash = float(min(dash_res, key=dash_res.get))
    # raw hierarchy k: method of moments reused, reported for transparency
    results["params"] = params.to_dict()
    results["best_gains"] = gains
    results["validation"] = {"target_season": TARGET_SEASON, "shots": int(len(prep["y"])),
                             "blocks": [str(b[0].date()) for b in hi.validation_blocks(h, TARGET_SEASON)]}
    return results


def tune_glmm(h: pd.DataFrame, half_life_days: float | None, multipliers=(0.5, 1.0, 2.0, 4.0),
              block_ids=(1, 3, 5)) -> dict:
    """Common multiplier of the GLMM ridge penalties, on three 2024-25 monthly blocks."""
    from pipeline.player_skill import glmm as gm
    groups = ["player", "type", "zone", "action", "zone_action"]
    ev = hi.evidence_from_shots(h, "xb")
    blocks = [hi.validation_blocks(h, TARGET_SEASON)[i] for i in block_ids]
    out = {}
    for mult in multipliers:
        lam = {k: v * mult for k, v in gm.DEFAULT_LAMBDA.items()}
        tot = base = n = 0.0
        for start, idx in blocks:
            f = gm.fit(gm.shot_rows(ev, start, half_life_days), lam, groups=groups)
            eff = f.intercept + f.effect(ev.iloc[idx], groups)
            xb = np.clip(h["xb"].to_numpy()[idx], 1e-4, 1 - 1e-4)
            y = h["made"].to_numpy(float)[idx]
            tot += _ll(y, 1 / (1 + np.exp(-(np.log(xb / (1 - xb)) + eff)))) * len(y)
            base += _ll(y, xb) * len(y)
            n += len(y)
        out[str(mult)] = (tot - base) / n
        log.info("glmm multiplier %s: delta=%.5f", mult, out[str(mult)])
    return {"delta_by_multiplier": out, "blocks": [str(hi.validation_blocks(h, TARGET_SEASON)[i][0].date()) for i in block_ids]}


__all__ = ["level_gains", "root_means", "tune_all", "tune_glmm"]
