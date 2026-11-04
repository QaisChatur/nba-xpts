"""Post-hoc probability calibrators, serializable as plain JSON."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from pipeline.evaluation.metrics import per_shot_log_loss

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Calibrator:
    def __init__(self, method: str = "none", params: dict | None = None):
        self.method = method
        self.params = params or {}

    def fit(self, p, y) -> "Calibrator":
        y = np.asarray(y, int)
        if self.method == "platt":
            lr = LogisticRegression(C=1e6, solver="newton-cholesky").fit(_logit(p).reshape(-1, 1), y)
            self.params = {"a": float(lr.intercept_[0]), "b": float(lr.coef_[0, 0])}
        elif self.method == "isotonic":
            iso = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(np.asarray(p, float), y)
            self.params = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
        return self

    def transform(self, p) -> np.ndarray:
        p = np.asarray(p, float)
        if self.method == "platt":
            z = self.params["a"] + self.params["b"] * _logit(p)
            return 1 / (1 + np.exp(-z))
        if self.method == "isotonic":
            return np.clip(np.interp(p, self.params["x"], self.params["y"]), EPS, 1 - EPS)
        return p

    def to_dict(self) -> dict:
        return {"method": self.method, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(d["method"], d.get("params"))


def select_calibration(p, y, halves, methods=("none", "platt", "isotonic")) -> tuple[str, dict]:
    """Cross-fitted log loss of each method over two game-grouped halves; returns (best, scores)."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    a, b = halves
    if a.sum() < 50 or b.sum() < 50:  # too little data to compare calibrators honestly
        return "none", {"none": float(per_shot_log_loss(y, p).mean())}
    scores = {}
    for m in methods:
        out = np.empty_like(p)
        for fit_mask, ev_mask in ((a, b), (b, a)):
            out[ev_mask] = Calibrator(m).fit(p[fit_mask], y[fit_mask]).transform(p[ev_mask])
        scores[m] = float(per_shot_log_loss(y, out).mean())
    return min(scores, key=scores.get), scores
