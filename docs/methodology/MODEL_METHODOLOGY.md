# Master xFG model: methodology

The Master model estimates P(make) for an NBA field-goal attempt from information available before the
shot is released, and expected field-goal points = P(make) × shot value. Results live in
[`docs/reports/MASTER_MODEL_REPORT.md`](reports/MASTER_MODEL_REPORT.md); this document describes the
method. Code: `pipeline/features/`, `pipeline/training/`, `pipeline/inference.py`; configs:
`configs/features.yaml`, `configs/training.yaml`, `configs/evaluation.yaml`.

## 1. Population and label

* Population: official 2026-27 regular-season field-goal attempts (play-by-play, equal to the box score)
  of verified games, restricted for training and evaluation to attempts matched one-to-one to a Blueprint
  tracking record (`in_blueprint`). Matching rules: [`DATA_MATCHING.md`](DATA_MATCHING.md).
* Label: official `SHOT_MADE_FLAG` (`shotResult`), never inferred from scores. Blueprint `success` must
  agree on every matched row (the build fails otherwise).
* NBA xFG (Blueprint `expFieldGoalPct`) is an external benchmark. It is never a feature, a label, or a
  target for distillation.

## 2. Features (all known before release)

| Group | Source | Examples |
| --- | --- | --- |
| shot_context | official location + action | distance, x/y, angle, zone, corner 3, action family and modifier flags |
| game_context | earlier play-by-play actions | period, clock, pre-shot score margin, possession start type (defensive rebound, offensive rebound, opponent turnover, opponent score), seconds since possession start, after-timeout, shooter's prior attempts this game |
| tracking | Blueprint | shot clock, closest-defender distance and height, shooter vertical, shooter speed, release time (+ not-measured and cap flags), tracking shot distance, contest distance / shot distance |
| player | history + earlier 2026-27 games | shrunk FG% overall / by 2PT-3PT / zone / action family, shrunk shot-making over expected (SMOE), volume, shot diet, season-to-season SMOE variability |
| player_tracking | PlayerDashPtShots × current contest | player's shrunk FG% in the shot's closest-defender range |
| defender | earlier 2026-27 Blueprint shots | defender's shrunk SMOE allowed and contest count |

### Leakage controls

1. **Score context** comes from the last play-by-play action *before* the shot (the shot row carries
   the post-shot score).
2. **Player and defender in-season priors** use only games dated strictly *before* the shot's game date
   (same-day games excluded), see `in_season_priors` and `tests/test_master.py`.
3. **Pre-season priors** use 2022-23..2024-25 ShotChartDetail (recency weights 0.5 / 0.75 / 1.0) and
   2021-22..2025-26 PlayerDashPtShots, all completed before 2026-27.
4. **Shot-making over expected** uses a shot-difficulty baseline (location, action, time) fitted once on
   2022-23..2024-25; it never sees 2026-27 outcomes.
5. **Action vocabulary:** the `Bank` token is removed. Historically, bank variants shoot 50–59% against
   38–40% for the same action without it, which shows that scorers tag "bank" mostly on makes.
6. **Forbidden columns** (`configs/features.yaml: forbidden`) are rejected by `feature_list` and at
   training time. They include `made`, NBA xFG, Model A output, post-shot scores, the description
   (it contains "MISS" and assists), and Blueprint `featured`.
7. Prior makes in the current game (`shooter_fgm_before_game`) are available before the shot but are
   excluded from the model and kept for analysis only.

### Shrinkage

For player p and category c: `fg_post = (makes_w + k·league_c) / (attempts_w + k)` and
`smoe_post = Σ_w(made − baseline) / (attempts_w + k_smoe)`. k values are in `configs/features.yaml`;
new players fall back smoothly to league rates and to SMOE = 0.

Master V2 research track: `pipeline/player_skill/` replaces these flat priors with a hierarchical,
difficulty-adjusted estimator, documented in `docs/reports/PLAYER_SKILL_MODELING.md`. It uses:

- out-of-fold difficulty residuals;
- shrinkage per level tuned on 2024-25;
- n_eff and reliability;
- veteran and newcomer root priors;
- as-of-date updating.

It is applied as a single logit offset on top of V1. V1 is unchanged and remains active.

### Suspicious tracking values

* `release_time_s == 0` occurs on 99% of attempts within 6 ft. It means "not measured" (rim attempts)
  rather than an instant release. Within distance bands it shows no consistent make/miss gap, so it is
  kept with an indicator.
* `release_time_s` is capped at 1.0 s (about 25% of tracked shots), flagged `release_time_capped`.
* A shot clock of 0 / missing is flagged; contest distances up to 27 ft are plausible (uncontested).

## 3. Model families

* **Regularized logistic GAM-style model:** cubic B-splines (quantile knots) on continuous inputs,
  one-hot codes, and an L2 penalty.
* **XGBoost, LightGBM, CatBoost:** gradient-boosted trees with early stopping.
* Hierarchical player effects enter through the empirical-Bayes priors above, not as random effects.
* Deep learning is not used. With about 15–20k labelled shots and tabular inputs it is not justified.

## 4. Training and selection protocol

1. **Chronological split on whole game dates.** The newest ~20% of games are the untouched **test**
   set. The preceding ~15% form the **calibration** partition. Everything earlier is the
   **model-fit** partition. No game is split across partitions.
2. **Hyperparameter search.** Optuna TPE (seed 42) per family, scored by game-grouped rolling-origin
   (expanding-window) CV log loss inside the model-fit partition. Boosted models early-stop on each
   fold's validation block; the final number of rounds is the mean best iteration across folds.
3. **Refit** each tuned family on the full model-fit partition, with no early-stopping data from later
   partitions.
4. **Calibration.** On the calibration partition, `none`, Platt and isotonic are compared by
   game-grouped two-fold cross-fitting. The best is refit on the whole calibration partition.
5. **Architecture choice:** lowest calibrated calibration-partition log loss.
6. **Feature-set choice.** Ablation sets are each re-tuned with the chosen family. The final set is
   chosen among the candidates in `configs/training.yaml` by calibration-partition log loss.
7. **Test** predictions are made once, for every model and benchmark, after all choices are fixed.
   Supplementary analyses that refit models (tracking drop-one, learning curve, walk-forward windows)
   reuse the fixed configuration and are labelled as such.

Every run writes Optuna trials, predictions, a training summary and the evaluation to
`models/experiments/master_runs/<run_id>/`, and appends to `experiments.jsonl`.

## 5. Evaluation

* **Primary metric:** log loss. Also Brier score, ROC AUC, logistic calibration slope/intercept,
  10-bin ECE and reliability diagrams.
* **Identical held-out shots for every model and benchmark:**
  - NBA xFG, Model A and Model A Platt-calibrated on the calibration partition;
  - a shot-type baseline (model-fit FG% by 2PT/3PT) and the historical location/action baseline;
  - context-only, tracking-only, player-only, tracking-without-player, context+player, full without
    defender, full, and the final Master model.
* **Uncertainty:** game-cluster bootstrap (2,000 resamples of test games) for each metric and for
  paired differences.
* **Data sufficiency:**
  - a learning curve on the calibration partition;
  - the test shots needed to detect the observed paired difference against NBA xFG at 80% power,
    inflated by the measured design effect from game clustering.

  If the test set is smaller than that, results are labelled exploratory.

## 6. Artifact and inference

`models/master/<model_version>/` (never overwritten) contains:

* `model.xgb.json` / `model.lgb.txt` / `model.cbm`, a native format with no pickle (or, for logistic,
  a sha256-pinned `model.joblib`);
* `model_spec.json` (family, features, params, number of rounds, file hashes, model / feature-schema /
  calibration versions), `calibrator.json` and `metadata.json`;
* `history_assets/` (baseline + pre-season tables) and `in_season_state.csv`;
* `fallback_no_tracking/`, the context + player model used only for official FGA without tracking,
  labelled at export;
* `training_summary.json` and `metrics.json`.

`pipeline.inference.predict_shots(shots_df)`:

* Validates required columns, types, ranges and uniqueness, then featurizes with the same code as
  training.
* Applies the model and calibrator. It returns `master_xfg`, `master_xpts_fg`, the IDs and the
  model / feature-schema / calibration versions.
* For scoring within the season without look-ahead, pass
  `state=build_prior_state(dataset, before_date=D)`. `tests` verify that inference reproduces the
  training pipeline's predictions exactly.
