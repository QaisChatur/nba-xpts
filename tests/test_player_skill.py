"""Player-skill (Master V2 research track): shrinkage, as-of correctness, contest mapping, GLMM, adjustment."""

import json

import numpy as np
import pandas as pd
import pytest

from pipeline import config
from pipeline.player_skill import contest as ct
from pipeline.player_skill import glmm as gm
from pipeline.player_skill import hierarchy as hi
from pipeline.player_skill import taxonomy as tx
from pipeline.player_skill.experiments import Adjustment


def _shots(n, pid, date, made, xb=0.5, three=0, zone="rim", dist=2.0, fam="layup"):
    return pd.DataFrame({"PLAYER_ID": pid, "game_date": pd.Timestamp(date), "made": made, "xb": xb,
                         "is_three": three, "zone_group": zone, "shot_distance_ft": dist, "action_family": fam},
                        index=range(n))


def _league():
    rate = {lvl: pd.Series(dtype=float) for lvl in hi.LEVELS}
    pq = {lvl: pd.Series(dtype=float) for lvl in hi.LEVELS}
    return hi.League(rate=rate, pq=pq)


def test_taxonomy_splits_midrange_and_keys_actions_by_type():
    s = pd.DataFrame({"is_three": [0, 0, 1, 0], "zone_group": ["mid", "mid", "ab3", "rim"],
                      "shot_distance_ft": [12.0, 19.0, 25.0, 1.0], "action_family": ["jump", "pullup", "jump", "dunk"]})
    c = tx.categories(s)
    assert c["zone"].tolist() == ["short_mid", "long_mid", "ab3", "rim"]
    assert c["action"].tolist() == ["two|jump", "two|pullup", "three|jump", "two|dunk"]
    assert c["zone_action"].iloc[1] == "long_mid|pullup"


def test_no_evidence_falls_back_to_root_prior_not_zero():
    ev = hi.evidence_from_shots(_shots(50, 1, "2024-01-01", 1), "xb")
    query = _shots(2, 2, "2024-06-01", 0)  # player 2 has no evidence
    p = hi.SkillParams(mu_veteran=0.01, mu_newcomer=-0.04)
    f = hi.skill_as_of(ev, query, "2024-06-01", p, _league())
    assert np.allclose(f["theta_zone_action"], -0.04)
    assert (f["rel_all"] == 0).all() and (f["has_history"] == 0).all()


def test_shrinkage_moves_toward_data_with_more_attempts():
    league = _league()
    p = hi.SkillParams(half_life_days=None, mu_veteran=0.0, mu_newcomer=0.0)
    query = _shots(1, 1, "2024-06-01", 0)
    thetas = []
    for n in (20, 200, 2000):
        ev = hi.evidence_from_shots(_shots(n, 1, "2024-01-01", np.r_[np.ones(int(n * 0.6)), np.zeros(n - int(n * 0.6))]), "xb")
        f = hi.skill_as_of(ev, query, "2024-06-01", p, league)
        thetas.append(float(f["theta_all"].iloc[0]))
        assert f["neff_all"].iloc[0] == pytest.approx(n)  # equal weights -> n_eff = n
    assert thetas[0] < thetas[1] < thetas[2] < 0.1 + 1e-9  # raw over-expected = 0.6 - 0.5


def test_evidence_on_or_after_as_of_date_is_ignored():
    league = _league()
    p = hi.SkillParams(half_life_days=None)
    before = _shots(100, 1, "2024-01-01", 1)
    same_day = _shots(100, 1, "2024-06-01", 0)
    query = _shots(1, 1, "2024-06-01", 0)
    a = hi.skill_as_of(hi.evidence_from_shots(before, "xb"), query, "2024-06-01", p, league)
    b = hi.skill_as_of(hi.evidence_from_shots(pd.concat([before, same_day], ignore_index=True), "xb"), query,
                       "2024-06-01", p, league)
    pd.testing.assert_frame_equal(a, b)


def test_recency_half_life_halves_weight():
    ev = hi.evidence_from_shots(pd.concat([_shots(1, 1, "2023-01-01", 1), _shots(1, 1, "2024-01-01", 1)],
                                          ignore_index=True), "xb")
    st = hi.node_stats(ev, "2024-01-02", half_life_days=365.0)
    sw = st["all"].loc[(1, "all"), "Sw"]
    assert sw == pytest.approx(0.5 ** (1 / 365) + 0.5 ** (366 / 365), rel=1e-6)


def test_quantile_thresholds_reproduce_dash_shares():
    dash = pd.DataFrame({"PLAYER_ID": [1] * 4, "SEASON": "2025-26",
                         "CLOSE_DEF_DIST_RANGE": list(ct.DASH_RANGES),
                         "FG2M": [1, 1, 1, 1], "FG2A": [10, 20, 30, 40], "FG3M": [1, 1, 1, 1], "FG3A": [40, 30, 20, 10]})
    rng = np.random.default_rng(0)
    shots = pd.DataFrame({"is_three": np.r_[np.zeros(4000), np.ones(4000)].astype(int),
                          "contest_distance_in": rng.uniform(0, 100, 8000), "made": 0})
    thr = ct.quantile_thresholds(dash, shots)
    mapped = ct.mapped_range(shots, thr)
    two = mapped[shots["is_three"] == 0].value_counts(normalize=True)
    three = mapped[shots["is_three"] == 1].value_counts(normalize=True)
    assert two["wide"] == pytest.approx(0.4, abs=0.01) and three["tight"] == pytest.approx(0.4, abs=0.01)


def test_contest_features_are_nan_without_contest_distance():
    shots = pd.DataFrame({"PLAYER_ID": [1, 1], "is_three": [1, 1], "contest_distance_in": [30.0, np.nan]})
    dev = pd.Series([0.02], index=pd.MultiIndex.from_tuples([(1, "three", "open")]))
    sens = pd.Series([0.01], index=pd.MultiIndex.from_tuples([(1, "three")]))
    f = ct.contest_features(shots, dev, sens, pd.Series({"three": 0.9, "two": 0.3}),
                            {"two": [10, 20, 40], "three": [10, 20, 40]})
    assert f["ps_contest_dev"].iloc[0] == pytest.approx(0.02) and np.isnan(f["ps_contest_dev"].iloc[1])


def test_glmm_recovers_planted_player_effect_and_unseen_player_is_zero():
    rng = np.random.default_rng(1)
    n = 6000
    pid = rng.integers(1, 4, n)
    true = {1: 0.5, 2: 0.0, 3: -0.5}
    eta = np.array([true[p] for p in pid])
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    rows = pd.DataFrame({"PLAYER_ID": pid, "cat_type": "two", "offset": 0.0, "n": 1.0, "made": y, "w": 1.0})
    f = gm.fit(rows, {g: 1.0 for g in gm.GROUPS}, groups=["player"])
    eff = f.intercept + f.effect(pd.DataFrame({"PLAYER_ID": [1, 3, 99]}), ["player"])
    assert eff[0] > 0.3 and eff[1] < -0.3 and eff[2] == pytest.approx(f.intercept)


def test_adjustment_recovers_coefficient():
    rng = np.random.default_rng(2)
    n = 20000
    base = rng.uniform(0.3, 0.7, n)
    x = rng.normal(0, 0.3, n)
    p = 1 / (1 + np.exp(-(np.log(base / (1 - base)) + 0.7 * x)))
    df = pd.DataFrame({"made": (rng.uniform(size=n) < p).astype(int), "x": x})
    adj = Adjustment(["x"], ridge=0.0).fit(df, base)
    assert adj.theta[1] == pytest.approx(0.7, abs=0.1) and abs(adj.theta[0]) < 0.05


def test_listing_fetched_on_game_day_is_refreshed(monkeypatch, tmp_path):
    from pipeline.ingestion import season
    calls = []

    def fake_listing(day, allow_network=False, refresh=False):
        calls.append((day, refresh))
        return []

    monkeypatch.setattr(season, "fetch_games_listing", fake_listing)
    monkeypatch.setattr(season, "GAMES_LISTING_DIR", tmp_path)
    (tmp_path / "2026-11-03.json").write_text("[]")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"season": "2026-27", "games": {}, "days": {"2026-11-03": {"listing_links": 0,
                                                                                            "fetched_on": "2026-11-03"}}}))
    monkeypatch.setattr(season.time, "sleep", lambda s: None)
    season.ingest_season("2026-11-03", "2026-11-03", allow_network=True, manifest_path=manifest)
    assert calls == [("2026-11-03", True)]


# --------------------------------------------------------------------------- built artifacts (skip if absent)

V2 = config.MASTER_DATASET_DIR / "master_v2" / "master_shots.parquet"
V1 = config.MASTER_DATASET_DIR / "master_v1" / "master_shots.parquet"


@pytest.mark.skipif(not (V1.exists() and V2.exists()), reason="master datasets not built")
def test_master_v2_keeps_v1_columns_unchanged_and_has_no_missing_skill():
    v1, v2 = pd.read_parquet(V1), pd.read_parquet(V2)
    assert len(v1) == len(v2)
    pd.testing.assert_frame_equal(v1, v2[v1.columns], check_dtype=False)
    for c in ("ps_oe_all", "ps_oe_zone_action", "ps_delta", "ps_rel_all", "ps_glmm_delta"):
        assert v2[c].notna().all(), c
    assert v2.loc[v2["ps_has_history"] == 0, "ps_rel_all"].lt(1).all()


def test_player_skill_config_is_not_tuned_on_2026_27():
    import yaml
    cfg = yaml.safe_load((config.CONFIGS_DIR / "player_skill.yaml").read_text())
    assert cfg["hierarchy"]["w_dash"] == 0.0 and cfg["dataset_version"] == "master_v2"
    serving = yaml.safe_load((config.CONFIGS_DIR / "serving.yaml").read_text())
    assert serving["active_master_model"] == "master_master_v1_20261104T210006"
