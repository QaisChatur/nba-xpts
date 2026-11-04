"""Shot-difficulty baseline fitted on historical ShotChartDetail shots (2022-23 .. 2024-25).

Inputs are location, shot type, normalized action and game time only (no player identity, no
tracking). Its role is to turn raw makes into *shot-making over expected* (made - baseline) when
building player and defender priors, so that a player's prior is not just a proxy for the shot
diet he takes. It is fitted once, before the 2026-27 season, so it cannot see 2026-27 outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from pipeline.config import RANDOM_SEED
from pipeline.features.shot_context import ACTION_FLAGS, ZONE_GROUP_ORDER

FAMILIES = ["jump", "pullup", "stepback", "post_fade", "floater", "hook", "layup", "dunk", "tip_putback", "alley_oop"]
BASELINE_FEATURES = (
    ["is_three", "shot_distance_ft", "loc_x", "loc_y", "abs_loc_x", "shot_angle_deg", "is_corner3", "is_backcourt",
     "period", "period_seconds_remaining", "end_of_period_heave_window", "zone_code", "family_code"]
    + list(ACTION_FLAGS)
)


def design(feats: pd.DataFrame) -> pd.DataFrame:
    X = feats.copy()
    X["zone_code"] = X["zone_group"].map({z: i for i, z in enumerate(ZONE_GROUP_ORDER)}).fillna(-1).astype(int)
    X["family_code"] = X["action_family"].map({f: i for i, f in enumerate(FAMILIES)}).fillna(-1).astype(int)
    return X[BASELINE_FEATURES].astype(float)


def fit_baseline(feats: pd.DataFrame, y: pd.Series) -> HistGradientBoostingClassifier:
    X = design(feats)
    cat = [BASELINE_FEATURES.index("zone_code"), BASELINE_FEATURES.index("family_code")]
    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
        categorical_features=cat, early_stopping=True, validation_fraction=0.1, random_state=RANDOM_SEED,
    )
    model.fit(X, y.astype(int))
    return model


def predict_baseline(model, feats: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(design(feats))[:, 1]
