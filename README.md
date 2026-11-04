# xPoints: NBA expected field-goal points

xPoints estimates the probability that an NBA field-goal attempt goes in (xFG) from information
available before the shot. It turns that into expected points for games, teams, players and
standings.

* **Master model:** trained on 2026-27 tracking data (NBA.com Blueprint), official play-by-play and
  shrunk player priors. It is evaluated against NBA's own xFG, which is used only as an external
  benchmark.
* **Model A:** the original player-specific model trained on 2022-25 ShotChartDetail. Kept unchanged
  as a legacy benchmark.

**Start with [`PROJECT_STATUS.md`](PROJECT_STATUS.md)** for current results, limitations and next
steps. Results: [`docs/reports/MASTER_MODEL_REPORT.md`](docs/reports/MASTER_MODEL_REPORT.md).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate     # Python 3.12+ (developed on 3.14)
python -m pip install -r requirements.txt               # macOS: brew install libomp (else a fallback is used)
python -m pytest -m "not slow"                          # runs on the committed samples only
```

Raw and processed data are **not** in Git (third-party NBA data, about 1.2 GB). To restore or rebuild
them, see [`docs/architecture/DATA_FILES.md`](docs/architecture/DATA_FILES.md). A different data root can be used with
`XPTS_DATA_DIR=/path/to/data`.

## Pipeline commands

```bash
python -m pipeline audit                                  # which inputs are present
python -m pipeline ingest-season --allow-network          # incremental: discover, verify, fetch new games (rate-limited)
python -m pipeline coverage-report [--reverify]           # data/reports/season/COVERAGE_REPORT.md
python -m pipeline build-dataset                          # data/processed/master/master_v1/master_shots.parquet
python -m pipeline train-master [--quick]                 # tune/select/calibrate, save models/master/<version>/, evaluate, write report
python -m pipeline evaluate-master --run <run_dir>        # re-run evaluation + report for an existing run
python -m pipeline export-predictions                     # season predictions (no look-ahead priors)
python -m pipeline update-analytics                       # Game Center, standings, explorer, highlights
python -m pipeline verify-master                          # re-score the active Master model's test set; must match exactly
python -m pipeline verify-model-a                         # Model A loads safely and reproduces 2024-25 results
python -m pipeline player-skill all                       # Master V2 research: hierarchical player skill + comparison vs V1
python -m pipeline run --date 2026-10-28                  # legacy single-night Model A vs NBA xFG pipeline
```

Scoring new shots from Python:

```python
from pipeline.inference import predict_shots
preds = predict_shots(shots_df)          # validates inputs; returns master_xfg, master_xpts_fg, versions
```

## Layout

| Path | Contents |
| --- | --- |
| `pipeline/ingestion/` | Blueprint + NBA.com page fetching/parsing, season manifest and verification |
| `pipeline/matching/` | court/clock normalization and Blueprint ↔ official event matching |
| `pipeline/features/` | shot context, play-by-play context, historical baseline, player/defender priors, dataset assembly |
| `pipeline/training/` | chronological splits, model families, Optuna tuning, calibration |
| `pipeline/evaluation/` | metrics, bootstrap, Master evaluation report, artifact verification |
| `pipeline/player_skill/` | Master V2 research: OOF historical residuals, hierarchical EB / penalized hierarchical logistic player skill, contest audit, archetypes, V2 comparison |
| `pipeline/inference.py`, `pipeline/analytics/season_exports.py` | `predict_shots`, season predictions, analytics exports |
| `pipeline/analytics/` | Game Center timelines, expected standings, shot explorer, rankings |
| `pipeline/legacy/`, `scripts/model_a/`, `scripts/legacy/` | Model A loader, single-night pipeline, original Model A training scripts, legacy backend/DB tools |
| `configs/` | feature groups and forbidden columns, training and evaluation settings, active served model (`serving.yaml`) |
| `models/master/<version>/` | versioned Master artifacts (never overwritten; binaries ignored by Git, metadata committed) |
| `models/legacy/` | Model A metadata (the pickle stays outside Git), exploratory Model B metadata |
| `data/samples/` | small committed fixture (one game) used by tests |
| `docs/methodology/`, `docs/architecture/`, `docs/reports/`, `docs/legacy/` | how the models and data work; data layout, contracts, roadmap, migration manifest; generated reports; legacy notes |
| `notebooks/legacy/model_a_original.ipynb` | the original Model A notebook |

## Docs

* [`docs/methodology/MODEL_METHODOLOGY.md`](docs/methodology/MODEL_METHODOLOGY.md): features, leakage controls, training
  and evaluation protocol.
* [`docs/reports/PLAYER_SKILL_MODELING.md`](docs/reports/PLAYER_SKILL_MODELING.md) and
  [`docs/reports/MASTER_V2_COMPARISON.md`](docs/reports/MASTER_V2_COMPARISON.md): player-aware skill
  research and the V2-vs-V1 comparison. V1 remains active.
* [`docs/methodology/DATA_MATCHING.md`](docs/methodology/DATA_MATCHING.md): sources, season ingestion, matching, coverage
  and missing-shot analysis.
* [`docs/architecture/DATA_FILES.md`](docs/architecture/DATA_FILES.md): data layout, storage policy, restore steps.
* [`docs/architecture/PRODUCT_ROADMAP.md`](docs/architecture/PRODUCT_ROADMAP.md): analytics outputs and the path to a product.
* [`docs/architecture/FRONTEND_DATA_CONTRACT.md`](docs/architecture/FRONTEND_DATA_CONTRACT.md): JSON schemas for frontend
  consumers.
* [`docs/architecture/MIGRATION_MANIFEST.md`](docs/architecture/MIGRATION_MANIFEST.md): what moved where in the
  repository cleanup.

## Data and rights

NBA statistics, Blueprint tracking data and video belong to their owners. The repository holds code
and small samples only. Network access respects rate limits and stops on access refusal. Public
hosting, redistribution or video use requires a separate licensing review. See [`LICENSE`](LICENSE).
