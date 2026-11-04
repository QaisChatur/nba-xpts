"""Event matching between Blueprint shots and official play-by-play / ShotChartDetail shots.

Algorithm (per game):
  1. Event-ID hypothesis. Pair Blueprint `ngss_event_id` with official `GAME_EVENT_ID` and accept a
     pair only if it is independently corroborated: same period, same shooter, same result,
     |clock difference| <= clock_tol and court-location difference <= coord_tol_ft. Equality of the
     identifiers is therefore *tested*, never assumed. Per-game hypothesis statistics are reported.
  2. Attribute fallback for Blueprint rows not verified in step 1: candidates are unclaimed official
     shots with the same period, shooter and result within the clock/location tolerances.
     A unique candidate is accepted; with several candidates the best is accepted only if its
     distance score beats the runner-up by `margin`; otherwise the row is marked ambiguous.
  3. One-to-one enforcement: an official shot claimed by more than one Blueprint row in step 2
     leaves all claimants ambiguous (no arbitrary choice).

Shot value (2 vs 3) disagreement does not block a corroborated match but is reported as a
classification discrepancy. A Blueprint row whose event ID exists officially but fails
corroboration (e.g. the sources disagree on make/miss) and has no fallback candidate gets
`rejected_id_conflict`; the official record remains in the unified table as `official_only`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline.config import MATCH_CLOCK_TOL_S, MATCH_COORD_TOL_FT
from pipeline.utils import require, require_columns

BP_REQUIRED = ["shot_uid", "game_id", "ngss_event_id", "period", "clock_remaining_s", "shooter_id", "made",
               "shot_value", "bp_x", "bp_y"]
OFFICIAL_REQUIRED = ["GAME_ID", "GAME_EVENT_ID", "PERIOD", "PLAYER_ID", "SHOT_MADE_FLAG", "SHOT_TYPE",
                     "LOC_X", "LOC_Y", "clock_remaining_s"]


@dataclass
class MatchResult:
    bp_status: pd.DataFrame            # one row per Blueprint shot
    unmatched_official: pd.DataFrame   # official shots with no Blueprint match
    hypothesis: pd.DataFrame           # per-game event-ID hypothesis statistics
    params: dict = field(default_factory=dict)


def _pair_metrics(bp: pd.DataFrame, off: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=bp.index)
    out["player_ok"] = bp["shooter_id"].to_numpy() == off["PLAYER_ID"].to_numpy()
    out["period_ok"] = bp["period"].to_numpy() == off["PERIOD"].to_numpy()
    out["made_ok"] = bp["made"].to_numpy() == off["SHOT_MADE_FLAG"].to_numpy()
    off_value = np.where(off["SHOT_TYPE"].astype(str).str.startswith("3"), 3, 2)
    out["value_ok"] = bp["shot_value"].to_numpy() == off_value
    out["clock_diff_s"] = np.abs(bp["clock_remaining_s"].to_numpy() - off["clock_remaining_s"].to_numpy())
    out["coord_diff_ft"] = np.hypot(bp["bp_x"].to_numpy() - off["LOC_X"].to_numpy(),
                                    bp["bp_y"].to_numpy() - off["LOC_Y"].to_numpy()) / 10.0
    return out


def match_shots(bp: pd.DataFrame, official: pd.DataFrame, clock_tol: float = MATCH_CLOCK_TOL_S,
                coord_tol_ft: float = MATCH_COORD_TOL_FT, margin: float = 0.5) -> MatchResult:
    require_columns(bp, BP_REQUIRED, "Blueprint shots")
    require_columns(official, OFFICIAL_REQUIRED, "official shots")
    require(not bp["shot_uid"].duplicated().any(), "Blueprint shot_uid not unique")
    require(not official.duplicated(["GAME_ID", "GAME_EVENT_ID"]).any(), "official (GAME_ID, GAME_EVENT_ID) not unique")

    status_frames, unmatched_frames, hyp_rows = [], [], []
    for gid in sorted(set(bp["game_id"]) | set(official["GAME_ID"])):
        b = bp[bp["game_id"] == gid].copy()
        o = official[official["GAME_ID"] == gid].copy()
        st, um, hyp = _match_game(gid, b, o, clock_tol, coord_tol_ft, margin)
        status_frames.append(st)
        unmatched_frames.append(um)
        hyp_rows.append(hyp)
    return MatchResult(
        bp_status=pd.concat(status_frames, ignore_index=True),
        unmatched_official=pd.concat(unmatched_frames, ignore_index=True),
        hypothesis=pd.DataFrame(hyp_rows),
        params={"clock_tol_s": clock_tol, "coord_tol_ft": coord_tol_ft, "ambiguity_margin": margin},
    )


def _match_game(gid, b, o, clock_tol, coord_tol_ft, margin):
    status = pd.DataFrame({
        "shot_uid": b["shot_uid"].to_numpy(), "game_id": gid, "ngss_event_id": b["ngss_event_id"].to_numpy(),
        "official_event_id": pd.array([pd.NA] * len(b), dtype="Int64"), "match_status": "unmatched",
        "match_method": None, "n_candidates": 0, "clock_diff_s": np.nan, "coord_diff_ft": np.nan,
        "value_agree": pd.array([pd.NA] * len(b), dtype="boolean"), "id_pair_rejection": None,
    }, index=b.index)

    # 1. Event-ID hypothesis.
    o_by_id = o.set_index("GAME_EVENT_ID", drop=False)
    has_id = b["ngss_event_id"].isin(o_by_id.index)
    n_id_verified = 0
    if has_id.any():
        bb = b[has_id]
        oo = o_by_id.loc[bb["ngss_event_id"].to_numpy()]
        pm = _pair_metrics(bb, oo.set_index(bb.index))
        ok = pm["player_ok"] & pm["period_ok"] & pm["made_ok"] & (pm["clock_diff_s"] <= clock_tol) & (pm["coord_diff_ft"] <= coord_tol_ft)
        n_id_verified = int(ok.sum())
        idx = ok[ok].index
        status.loc[idx, "official_event_id"] = bb.loc[idx, "ngss_event_id"].astype("Int64")
        status.loc[idx, "match_status"] = "matched"
        status.loc[idx, "match_method"] = "event_id_verified"
        status.loc[idx, "n_candidates"] = 1
        for c in ["clock_diff_s", "coord_diff_ft"]:
            status.loc[idx, c] = pm.loc[idx, c]
        status.loc[idx, "value_agree"] = pm.loc[idx, "value_ok"]
        bad = ok[~ok].index
        reasons = pm.loc[bad].apply(
            lambda r: ",".join(k for k, v in [("player", not r.player_ok), ("period", not r.period_ok), ("result", not r.made_ok),
                                              ("clock", r.clock_diff_s > clock_tol), ("location", r.coord_diff_ft > coord_tol_ft)] if v),
            axis=1) if len(bad) else pd.Series(dtype=object)
        status.loc[bad, "id_pair_rejection"] = reasons

    # 2. Attribute fallback.
    claimed = set(status.loc[status["match_status"] == "matched", "official_event_id"].dropna().astype(int))
    pool = o[~o["GAME_EVENT_ID"].isin(claimed)]
    proposals: dict[int, list] = {}
    for i in status.index[status["match_status"] != "matched"]:
        r = b.loc[i]
        cand = pool[(pool["PERIOD"] == r["period"]) & (pool["PLAYER_ID"] == r["shooter_id"]) & (pool["SHOT_MADE_FLAG"] == r["made"])]
        if cand.empty:
            continue
        pm = _pair_metrics(pd.DataFrame([r] * len(cand), index=cand.index), cand)
        pm = pm[(pm["clock_diff_s"] <= clock_tol) & (pm["coord_diff_ft"] <= coord_tol_ft)]
        status.loc[i, "n_candidates"] = len(pm)
        if pm.empty:
            continue
        score = pm["clock_diff_s"] / clock_tol + pm["coord_diff_ft"] / coord_tol_ft
        order = score.sort_values()
        if len(order) == 1:
            method = "attributes_unique"
        elif order.iloc[1] - order.iloc[0] >= margin:
            method = "attributes_best_by_margin"
        else:
            status.loc[i, "match_status"] = "ambiguous"
            continue
        j = order.index[0]
        proposals.setdefault(int(cand.loc[j, "GAME_EVENT_ID"]), []).append((i, method, pm.loc[j]))

    # 3. One-to-one enforcement.
    for eid, claims in proposals.items():
        if len(claims) > 1:
            for i, _, _ in claims:
                status.loc[i, "match_status"] = "ambiguous"
            continue
        i, method, m = claims[0]
        status.loc[i, ["match_status", "match_method"]] = ["matched", method]
        status.loc[i, "official_event_id"] = eid
        status.loc[i, ["clock_diff_s", "coord_diff_ft"]] = [m["clock_diff_s"], m["coord_diff_ft"]]
        status.loc[i, "value_agree"] = bool(m["value_ok"])

    conflict = (status["match_status"] == "unmatched") & status["id_pair_rejection"].notna()
    status.loc[conflict, "match_status"] = "rejected_id_conflict"

    matched_ids = set(status.loc[status["match_status"] == "matched", "official_event_id"].dropna().astype(int))
    require(len(matched_ids) == int((status["match_status"] == "matched").sum()), f"{gid}: one-to-one violated")
    unmatched = o[~o["GAME_EVENT_ID"].isin(matched_ids)].copy()

    hyp = {
        "game_id": gid, "n_blueprint": len(b), "n_official": len(o),
        "bp_ids_present_in_official": int(has_id.sum()), "id_pairs_corroborated": n_id_verified,
        "id_hypothesis_support": n_id_verified / max(int(has_id.sum()), 1),
        "matched": int((status["match_status"] == "matched").sum()),
        "matched_by_fallback": int(status["match_method"].astype(str).str.startswith("attributes").sum()),
        "ambiguous": int((status["match_status"] == "ambiguous").sum()),
        "rejected_id_conflict": int((status["match_status"] == "rejected_id_conflict").sum()),
        "unmatched_blueprint": int((status["match_status"] == "unmatched").sum()),
        "unmatched_official": len(unmatched),
    }
    return status.reset_index(drop=True), unmatched, hyp


def build_unified(official: pd.DataFrame, bp: pd.DataFrame, result: MatchResult) -> pd.DataFrame:
    """One row per official field-goal attempt, joined to its Blueprint record when matched.

    Blueprint rows that could not be matched are appended with `in_official = False` so no source
    row is silently dropped. `shot_uid` = <game_id>_<official event id> (or `_bp<ngss id>` for
    Blueprint-only rows).
    """
    st = result.bp_status
    matched = st[st["match_status"] == "matched"][["shot_uid", "official_event_id", "match_method", "clock_diff_s", "coord_diff_ft", "value_agree"]]
    bpm = bp.drop(columns=["game_id"]).rename(columns={"shot_uid": "bp_shot_uid"}).merge(
        matched.rename(columns={"shot_uid": "bp_shot_uid"}), on="bp_shot_uid", how="inner", validate="one_to_one")
    off = official.copy()
    off["official_event_id"] = off["GAME_EVENT_ID"].astype("Int64")
    overlap = sorted((set(off.columns) & set(bpm.columns)) - {"official_event_id"})
    bpm = bpm.rename(columns={c: f"bp_{c}" for c in overlap})
    u = _merge_by_game(off, bpm, bp)
    u["shot_uid"] = u["GAME_ID"] + "_" + u["GAME_EVENT_ID"].astype(int).astype(str).str.zfill(4)
    u["in_official"] = True
    u["in_blueprint"] = u["bp_shot_uid"].notna()
    u["match_status"] = np.where(u["in_blueprint"], "matched", "official_only")

    leftover = st[st["match_status"] != "matched"]
    if len(leftover):
        extra = bp[bp["shot_uid"].isin(leftover["shot_uid"])].merge(
            leftover[["shot_uid", "match_status"]], on="shot_uid").rename(columns={"shot_uid": "bp_shot_uid"})
        extra["GAME_ID"] = extra["game_id"]
        extra["shot_uid"] = extra["game_id"] + "_bp" + extra["ngss_event_id"].astype(str).str.zfill(4)
        extra["in_official"] = False
        extra["in_blueprint"] = True
        for c in overlap:
            if c in extra.columns:
                extra[f"bp_{c}"] = extra[c]
        extra = extra.drop(columns=["game_id"])
        u = pd.concat([u, extra], ignore_index=True)
    for c in ["GAME_EVENT_ID", "official_event_id", "PLAYER_ID", "TEAM_ID", "PERIOD", "SHOT_MADE_FLAG", "SHOT_DISTANCE",
              "LOC_X", "LOC_Y", "ngss_event_id", "shooter_id", "defender_id", "period", "made", "shot_value"]:
        if c in u.columns:
            u[c] = u[c].astype("Int64")
    require(not u["shot_uid"].duplicated().any(), "unified shot_uid not unique")
    return u


def _merge_by_game(off: pd.DataFrame, bpm: pd.DataFrame, bp: pd.DataFrame) -> pd.DataFrame:
    gid_map = dict(zip(bp["shot_uid"], bp["game_id"]))
    bpm = bpm.copy()
    bpm["GAME_ID"] = bpm["bp_shot_uid"].map(gid_map)
    return off.merge(bpm, on=["GAME_ID", "official_event_id"], how="left", validate="one_to_one")
