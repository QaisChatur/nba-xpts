# Project status: xPoints

Last updated 2026-11-04. Data through 2026-11-02 (107 games). Full results:
[`docs/reports/MASTER_MODEL_REPORT.md`](docs/reports/MASTER_MODEL_REPORT.md).

## Headline

On the untouched test set (26 games, 2026-10-31..11-02, 4,091 tracked shots), the **Master xFG model
is statistically indistinguishable from NBA's own xFG** and far better than the original Model A.

| | Master | NBA xFG | Model A | Model A, Platt-calibrated | Shot-type baseline |
| --- | --- | --- | --- | --- | --- |
| Log loss | 0.6152 | **0.6138** | 0.6852 | 0.6691 | 0.6713 |
| Brier | 0.2142 | **0.2138** | 0.2409 | 0.2378 | 0.2392 |
| ROC AUC | 0.6944 | **0.6970** | 0.6141 | 0.6141 | 0.5999 |
| Calibration slope | 0.965 | 0.992 | 0.502 | 1.040 | 0.896 |
| ECE | **0.013** | 0.019 | 0.065 | 0.050 | 0.011 |

* **Master vs NBA xFG:** log-loss difference +0.0012, 95% game-clustered CI [−0.0063, +0.0073].
  This is a statistical tie. The project does **not** claim to beat NBA xFG.
* **Master vs Model A:** −0.070 [−0.083, −0.058]. Master vs calibrated Model A: −0.054 [−0.065, −0.043].
* **Tracking is the source of skill.** Adding Blueprint tracking to shot and game context improves log
  loss by 0.025 [0.019, 0.031]. Contest distance and contest/shot-distance ratio are the strongest
  features.
* **Player priors do not help yet.** Shrunk career and in-season FG% and shot-making over expected
  change log loss by +0.003 [−0.001, +0.007] on top of context and tracking. So the selected model is
  CatBoost on context + tracking, uncalibrated, since calibration "none" won on the calibration
  partition.
* **Status: exploratory.** Detecting a 0.005 log-loss difference against NBA xFG at 80% power needs
  about 15,600 test shots (design effect 1.42); the test set has 4,091.

## Master V2 research track: player-aware skill (2026-11-04)

Reports: [`docs/reports/PLAYER_SKILL_MODELING.md`](docs/reports/PLAYER_SKILL_MODELING.md) and
[`docs/reports/MASTER_V2_COMPARISON.md`](docs/reports/MASTER_V2_COMPARISON.md).

**Decision: Master V1 remains the active model** (`configs/serving.yaml`).

* **Player shooting skill is real.** The between-player SD of FG% over expected is 0.031, and the
  season-to-season correlation is 0.65.
* **Historical validation gain.** A hierarchical empirical-Bayes skill estimate improves
  out-of-sample log loss on 2024-25 by **0.0023** over a player-agnostic difficulty model. It uses
  out-of-fold difficulty residuals, tuned shrinkage per level, n_eff and reliability, newcomer
  priors and as-of-date updating.
* **2026-27 gain on top of V1.** It is −0.0009 log loss [−0.0026, +0.0008] when skill is added as
  one logit offset (`adj_eb`), and −0.0005 [−0.0029, +0.0019] for the fixed prior chosen on the
  calibration partition. Both are consistent with calibration but not statistically resolvable. The
  Oct 31–Nov 2 test set was used a second time, once; no newer games were available.
* **Why V1's player features failed.** The main cause was integration: 20 correlated raw features
  handed to CatBoost on 9k shots. Other causes:
  - raw FG confounded with difficulty;
  - Dash defender ranges mapped to Blueprint contest distance on mismatched scales;
  - in-sample residuals;
  - untuned shrinkage;
  - zero-filled newcomers.
* **Contest and physical signals.** Contest-specific ability, physical proxies and archetypes add
  no detectable signal. No licensed height data was reachable (NBA.com players index: HTTP 403).
* **Biggest data gap.** There is no shot-level 2025-26 data, so second-year players are treated as
  newcomers.
* **Next step.** Re-run `python -m pipeline player-skill build` and `compare` on games after Nov 4.
  Promote `adj_eb` (V1 + fixed skill offset, no tree retraining) only if its CI excludes zero.

## Mission phases

| Phase | Status | Evidence |
| --- | --- | --- |
| 1 Repo cleanup and GitHub preparation | done; pushed to `origin/main` (public repository, with approval); reorganized again later, see `docs/architecture/MIGRATION_MANIFEST.md` | `.gitignore`, `docs/architecture/DATA_FILES.md`, secrets scan below |
| 2 Season-wide ingestion | done: 107/107 games verified | `data/reports/season/COVERAGE_REPORT.md`, `docs/methodology/DATA_MATCHING.md` |
| 3 Master shot dataset | done: 18,852 FGA, 17,389 tracked | `data/processed/master/master_v1/`, `configs/features.yaml` |
| 4 Model development | done: 4 families × 8 feature sets | `models/experiments/master_runs/20261104T210006/` |
| 5 Evaluation | done | `docs/reports/MASTER_MODEL_REPORT.md`, `docs/reports/figures/master/` |
| 6 Data sufficiency | assessed: exploratory | report §12 |
| 7 Final model + inference | done | `models/master/master_master_v1_20261104T210006/`, `pipeline.inference.predict_shots` |
| 8 Report | done | `docs/reports/MASTER_MODEL_REPORT.md` |
| 9 Product analytics | done (files only) | `data/processed/analytics/season/<version>/`, `docs/architecture/PRODUCT_ROADMAP.md` |
| 10 Handoff | done | tests pass; clean-environment install passes; docs below |
| V2 Player-aware skill | research done; V1 kept active | `pipeline/player_skill/`, `configs/player_skill.yaml`, `data/processed/master/master_v2/`, `docs/reports/player_skill/comparison_20261104T221133.json` |

## Data

* **Coverage:** 107 games, 2026-10-20 to 2026-11-02, discovered from the NBA.com games listings and
  dated from each official game page. All are Final, with play-by-play equal to the box score.
* **Matching:** 17,389 of 18,852 official FGA (92.2%) matched one-to-one to a Blueprint record, all by
  corroborated event ID. There were 35 conflicting ID pairs; these were rejected rather than forced.
* **Partial games:** three games are only partially published on Blueprint (15.7%, 49.7%, 56.9%).
* **Missing shots are not random:** 24% made against 46% overall, with more rim, paint, putback and
  4th-quarter attempts. Models and NBA xFG are compared on identical tracked shots, and no NBA xFG is
  invented for untracked shots.
* **Play-by-play formats:** three games use NBA's live play-by-play format. It is normalized to the
  ShotChartDetail vocabulary (`docs/methodology/DATA_MATCHING.md`).
* **Leakage controls** (`docs/methodology/MODEL_METHODOLOGY.md`):
  - pre-shot score only;
  - in-season priors only from earlier dates;
  - historical baseline fitted on 2022-25;
  - the outcome-biased "Bank" token removed;
  - a forbidden-feature list enforced in code and tests.

## Repository and GitHub

* **Secrets scan:** no secrets, tokens or credentials in the working tree or in Git history (re-run
  before every push). `.env.example` contains placeholders only.
  - Local absolute paths in the legacy notebook's saved outputs were replaced by `<project_root>`
    (they remain in earlier commits; no credentials were ever present).
* **What gets committed:** code, configs, docs, report figures and a one-game sample, about 2.5 MB in
  total. Raw and processed NBA data, model binaries and experiment runs are git-ignored (third-party
  data, about 1.2 GB; restore steps in `docs/architecture/DATA_FILES.md`). The Model A pickle stays local and
  untouched.
* **Remote:** `origin` = `https://github.com/QaisChatur/nba-xpts.git` (public). Pushed with your approval;
  normal pushes only, never force.

## Limitations

* Early, partial season. The test window is the 3 newest dates; drift later in the season is untested.
* Player-skill value should be re-tested as in-season data accumulates. Re-run `player-skill build`
  and `compare` on new games, and `train-master` monthly, under the same protocol.
* No shot-level 2025-26 history (`stats.nba.com` unreachable from here) and 2022-23 ShotChartDetail
  stops at Jan 6. No licensed roster heights are available.
* The game listing for 2026-11-03 had been cached before that day's games. `ingest-season` now
  re-fetches any listing fetched on or before its game day. NBA.com lists no games for Nov 3.
* The fallback no-tracking model, used only for untracked shots in exports, is weaker (test log loss
  about 0.64).
* Tracking quirks: release time is 0 (not measured) on rim attempts and capped at 1.0 s.
* NBA xFG's inputs are unknown. It is a benchmark only.
* NBA and Blueprint data and video require a licensing review before any public hosting or
  redistribution.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m pipeline ingest-season --allow-network      # incremental; ~3.5 s between requests; stops on 403/429
python -m pipeline coverage-report --reverify
python -m pipeline build-dataset
python -m pipeline train-master                       # ~10 min tuning + ~10 min evaluation on a laptop
python -m pipeline export-predictions && python -m pipeline update-analytics
python -m pipeline player-skill all                   # V2 research track: history, tune, build, compare, report (~6 min)
python -m pytest
```

## Legacy (Model A era)

* The original single-night comparison (2026-10-28, 4 games) is in `data/reports/2026-10-28/`.
* Model A's verified artifact is reproduced exactly by `python -m pipeline verify-model-a`.
* The exploratory Model B metadata is in `models/legacy/model_b/`.
