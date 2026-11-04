# Model B (tracking-aware) — exploratory experiment, 2026-10-28

**Status: exploratory. Four games from one night cannot establish whether tracking features improve make-probability predictions. No improvement claim is made from these results.**

## Data availability

* Per-shot tracking (contest distance/height, shot vertical, shooter speed, release time) exists only in the Blueprint scrape: 649 matched shots from 4 games. No historical per-shot tracking exists in the repository (the 2022-25 inputs are ShotChartDetail plus aggregate PlayerDashPtShots splits).
* Defender identity is available per shot but there is no historical defender representation to encode it validly; it is not used.

## Field audit (units as named)

| index | n | missing | zeros | min | p01 | median | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contest_distance_in | 650.00 | 0.00 | 0.00 | 3.90 | 6.65 | 32.70 | 181.10 | 230.90 |
| contest_height_in | 650.00 | 0.00 | 0.00 | 1.50 | 32.10 | 93.30 | 120.46 | 125.90 |
| shot_vertical_in | 650.00 | 0.00 | 8.00 | 0.00 | 0.00 | 11.60 | 31.21 | 35.50 |
| shooter_speed_mph | 650.00 | 0.00 | 0.00 | 0.20 | 0.40 | 2.90 | 16.25 | 19.00 |
| release_time_s | 650 | 0 | 285 | 0.00 | 0.00 | 0.70 | 1.00 | 1.00 |
| shot_clock_s | 650.00 | 0.00 | 10.00 | 0.00 | 0.00 | 12.00 | 24.00 | 24.00 |
| bp_shot_distance_ft | 650.00 | 0.00 | 0.00 | 0.40 | 0.60 | 12.85 | 30.61 | 43.00 |

* `release_time_s` is exactly 0 for 100% of shots within 6 ft and 2.5% beyond 10 ft, takes values [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] and is capped at 1.0 (23% of shots) → zero treated as *not measured* (missing + indicator), cap flagged.
* `shot_clock_s` = 0 kept with an indicator (expiry or clock off); `shot_vertical_in` = 0 kept with an indicator.
* `contest_distance_in` has no zeros (min 3.9 in) and is converted to feet.

## Design

* B0 = Model A recalibrated (logistic on logit(Model A)); B1 = B0 + tracking block; B2 = court geometry + tracking, no Model A.
* B0 is the control: incremental tracking value is B1 vs B0 (any model fit on 2026-27 data can correct calibration drift).
* Nested leave-one-game-out: each game is predicted by a model trained on the other games; L2 strength is chosen by leave-one-game-out CV inside the training games. Imputation and scaling are fitted within training folds.
* Excluded inputs: NBA xFG, outcomes, scores, defender identity. Same-night games are used for training, i.e. the split is game-grouped but not strictly chronological.

## Out-of-fold results (identical shots for every model)

| model | n | mean_pred | mean_signed_error | log_loss | brier | roc_auc | ece_quantile10 | calibration_slope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model A (raw) | 649 | 0.4980 | 0.0080 | 0.6859 | 0.2385 | 0.6326 | 0.0721 | 0.4954 |
| B0_recalibrated_A | 649 | 0.4889 | -0.0011 | 0.6723 | 0.2382 | 0.6310 | 0.0635 | 0.9369 |
| B1_A_plus_tracking | 649 | 0.4892 | -0.0008 | 0.6427 | 0.2261 | 0.6601 | 0.0696 | 1.0185 |
| B2_geometry_plus_tracking | 649 | 0.4912 | 0.0012 | 0.6396 | 0.2246 | 0.6716 | 0.0563 | 1.0342 |
| NBA xFG | 649 | 0.4935 | 0.0035 | 0.6064 | 0.2116 | 0.6997 | 0.0438 | 1.0745 |

| quantity | metric | mean | ci_low | ci_high | share_below_zero |
| --- | --- | --- | --- | --- | --- |
| B1_A_plus_tracking - B0_recalibrated_A | log_loss | -0.0298 | -0.0503 | -0.0078 | 0.9995 |
| B1_A_plus_tracking - B0_recalibrated_A | brier | -0.0120 | -0.0207 | -0.0031 | 0.9970 |
| B1_A_plus_tracking - B0_recalibrated_A | roc_auc | 0.0293 | -0.0101 | 0.0701 | 0.0750 |
| B0_recalibrated_A - Model A (raw) | log_loss | -0.0138 | -0.0345 | 0.0051 | 0.9220 |
| B0_recalibrated_A - Model A (raw) | brier | -0.0004 | -0.0067 | 0.0053 | 0.5430 |
| B0_recalibrated_A - Model A (raw) | roc_auc | -0.0015 | -0.0096 | 0.0060 | 0.6330 |
| B2_geometry_plus_tracking - B0_recalibrated_A | log_loss | -0.0330 | -0.0571 | -0.0087 | 0.9965 |
| B2_geometry_plus_tracking - B0_recalibrated_A | brier | -0.0136 | -0.0239 | -0.0030 | 0.9935 |
| B2_geometry_plus_tracking - B0_recalibrated_A | roc_auc | 0.0412 | -0.0041 | 0.0855 | 0.0345 |
| B1_A_plus_tracking - NBA xFG | log_loss | 0.0364 | 0.0193 | 0.0537 | 0.0000 |
| B1_A_plus_tracking - NBA xFG | brier | 0.0146 | 0.0073 | 0.0217 | 0.0000 |
| B1_A_plus_tracking - NBA xFG | roc_auc | -0.0397 | -0.0693 | -0.0093 | 0.9945 |

How to read this: the intervals resample individual shots and so ignore within-game correlation; with four games the effective number of independent units is closer to four than to 649. The per-game table is the more honest view: B1 beats B0 in 4 of 4 held-out games. B1 and B2 also train on tracking data from the same night they are tested on (other games), which a production model could not do. Treat any gap as a hypothesis for a larger, chronologically held-out evaluation, not as evidence of improvement.

### Per held-out game log loss

| GAME_ID | n | Model A (raw) | B0_recalibrated_A | B1_A_plus_tracking | B2_geometry_plus_tracking | NBA xFG |
| --- | --- | --- | --- | --- | --- | --- |
| 0022600132 | 181 | 0.6529 | 0.6615 | 0.6354 | 0.6406 | 0.6133 |
| 0022600133 | 168 | 0.6728 | 0.6606 | 0.6235 | 0.6244 | 0.5746 |
| 0022600134 | 142 | 0.6797 | 0.6663 | 0.6661 | 0.6513 | 0.6058 |
| 0022600136 | 158 | 0.7432 | 0.7025 | 0.6503 | 0.6441 | 0.6329 |

### Chosen regularization per fold

| model | held_out_game | n_train | n_test | chosen_C |
| --- | --- | --- | --- | --- |
| B0_recalibrated_A | 0022600132 | 468 | 181 | 0.100 |
| B0_recalibrated_A | 0022600133 | 481 | 168 | 0.100 |
| B0_recalibrated_A | 0022600134 | 507 | 142 | 0.100 |
| B0_recalibrated_A | 0022600136 | 491 | 158 | 3.000 |
| B1_A_plus_tracking | 0022600132 | 468 | 181 | 0.100 |
| B1_A_plus_tracking | 0022600133 | 481 | 168 | 0.030 |
| B1_A_plus_tracking | 0022600134 | 507 | 142 | 0.100 |
| B1_A_plus_tracking | 0022600136 | 491 | 158 | 0.030 |
| B2_geometry_plus_tracking | 0022600132 | 468 | 181 | 0.030 |
| B2_geometry_plus_tracking | 0022600133 | 481 | 168 | 0.030 |
| B2_geometry_plus_tracking | 0022600134 | 507 | 142 | 0.030 |
| B2_geometry_plus_tracking | 0022600136 | 491 | 158 | 0.030 |

## How much data is needed

The per-shot log-loss difference between B1 and B0 has SD ≈ 0.2591; the observed mean improvement of B1 over B0 is +0.0296. Shots required for 80% power (two-sided α = 0.05, independent shots) to detect a given mean improvement:

| log-loss improvement | shots needed |
| --- | --- |
| 0.002 | 131684 |
| 0.005 | 21070 |
| 0.01 | 5268 |
| 0.02 | 1317 |

At roughly 170 FGA per game (≈ 160 with Blueprint coverage), detecting a 0.005 improvement needs on the order of 132 games before accounting for within-game correlation, which inflates the requirement further. A defensible evaluation also needs a chronologically later held-out block (e.g. train on the first half of the season, test on the following month) and tracking-aware player priors built only from earlier games.

The B1 model refit on all available games is saved for inspection at `models/model_b/exploratory_2026-10-28` with metadata; it is not a production model.
