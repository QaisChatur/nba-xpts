"""Reproduce a saved Master model's held-out benchmark from its artifact alone.

Re-scores the run's test partition through the public `predict_shots` path (with in-season state
rebuilt per game date, so no look-ahead), then checks the predictions against the run's stored
predictions and the recomputed metrics against the artifact's `metrics.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import EXPERIMENTS_DIR, MASTER_DATASET_DIR, PROJECT_ROOT
from pipeline.evaluation.metrics import prob_metrics
from pipeline.utils import get_logger, require

log = get_logger("verify")


def verify_master(model_dir: str | Path, pred_tol: float = 1e-6, metric_tol: float = 1e-6) -> dict:
    from pipeline.inference import MasterModel, build_prior_state, predict_shots
    mm = MasterModel(Path(model_dir))
    summary = json.loads((mm.dir / "training_summary.json").read_text())
    run_dir = EXPERIMENTS_DIR / "master_runs" / summary["run_id"]
    stored = pd.read_parquet(run_dir / "predictions.parquet")
    stored = stored[stored["partition"] == "test"].set_index("shot_uid")["master_xfg"]
    raw = pd.read_parquet(MASTER_DATASET_DIR / mm.meta["dataset_version"] / "master_shots.parquet")
    test = raw[raw["shot_uid"].isin(stored.index)]
    require(len(test) == len(stored), f"test shots missing from dataset: {len(stored) - len(test)}")

    parts = []
    for day in sorted(test["game_date"].unique()):
        d = test[test["game_date"] == day]
        p = predict_shots(d, model=mm, state=build_prior_state(raw, before_date=day))
        parts.append(pd.Series(p["master_xfg"].to_numpy(), index=d["shot_uid"].to_numpy()))
    rep = pd.concat(parts).reindex(stored.index)
    max_diff = float(np.abs(rep - stored).max())

    y = test.set_index("shot_uid").loc[stored.index, "made"].astype(int)
    m = prob_metrics(y, rep)
    m["auc"] = m["roc_auc"]
    rec = {r["model"]: r for r in json.loads((mm.dir / "metrics.json").read_text())["test"]}["Master (final)"]
    diffs = {k: abs(m[k] - rec[k]) for k in ("log_loss", "brier", "auc")}
    out = {
        "model_version": mm.meta["model_version"], "run_id": summary["run_id"], "test_shots": int(len(rep)),
        "prediction_max_abs_diff": max_diff, "recomputed": {k: m[k] for k in diffs}, "recorded": {k: rec[k] for k in diffs},
        "metric_max_abs_diff": max(diffs.values()),
        "status": "passed" if max_diff <= pred_tol and max(diffs.values()) <= metric_tol else "failed",
        "model_dir": str(mm.dir.relative_to(PROJECT_ROOT)),
    }
    log.info("verify %s: %s (pred diff %.2e, metric diff %.2e)", out["model_version"], out["status"], max_diff,
             out["metric_max_abs_diff"])
    return out
