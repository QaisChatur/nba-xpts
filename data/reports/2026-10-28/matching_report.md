# Matching-quality report — 2026-10-28

Generated 2026-11-04T20:15:43+00:00 by `python -m pipeline run --date 2026-10-28`.

## Sources

* **Blueprint** (NBA xFG + tracking): `NBA_xFG_Pipeline_Starter/nba_xfg_2026_27/games/*.csv`, cross-checked row by row against `raw/*.json`.
* **Official play-by-play + box score**: NBA.com game page `__NEXT_DATA__` payload, cached in `data/raw/nba_game_pages/`. ShotChartDetail (stats.nba.com) and cdn.nba.com liveData were not reachable from this environment (timeout / HTTP 403). The play-by-play supplies the ShotChartDetail fields directly (event number, player, team, period, clock, LOC_X/LOC_Y as xLegacy/yLegacy, ACTION_TYPE as subType, result); SHOT_DISTANCE and SHOT_ZONE_BASIC are derived with rules validated on 540,627 historical ShotChartDetail shots (100% and 99.98% agreement).

## Schedule validation (cached `schedule_days/2026-10-28.json`)

| game_id | away | home | official_date_et | status | flags |
| --- | --- | --- | --- | --- | --- |
| 0022600132 | MIA | ATL | 2026-10-28 | verified |  |
| 0022600133 | WAS | CLE | 2026-10-28 | verified |  |
| 0022600134 | CHA | DET | 2026-10-28 | verified |  |
| 0022600135 | PHI | IND | – | unverified |  |
| 0022600136 | ORL | TOR | 2026-10-28 | verified |  |
| 0022600137 | NYK | CHI | – | unverified |  |
| 0022600138 | HOU | MIL | – | unverified |  |
| 0022600139 | GSW | MIN | – | unverified |  |
| 0022600140 | DEN | NOP | – | unverified |  |
| 0022600141 | SAS | UTA | – | unverified |  |
| 0022600142 | OKC | DAL | – | unverified |  |
| 0022600143 | LAL | LAC | – | unverified |  |
| 0022600158 | OKC | CLE | 2026-11-02 | wrong_date | official game date is 2026-11-02 |

Games without a cached official page remain `unverified`; they were not fetched because no Blueprint data exists for them.

## Coverage against the official box score

Official play-by-play FGA/FGM/3PA equal the box score for every team (see `box_score_validation.csv`).

| game_id | team | box_fga | pbp_fga | blueprint_shots | matched | blueprint_coverage_of_box_fga | matched_coverage_of_box_fga |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0022600132 | ATL | 96 | 96 | 91 | 91 | 0.948 | 0.948 |
| 0022600132 | MIA | 96 | 96 | 90 | 90 | 0.938 | 0.938 |
| 0022600133 | CLE | 91 | 91 | 87 | 87 | 0.956 | 0.956 |
| 0022600133 | WAS | 83 | 83 | 82 | 81 | 0.988 | 0.976 |
| 0022600134 | DET | 76 | 76 | 73 | 73 | 0.961 | 0.961 |
| 0022600134 | CHA | 71 | 71 | 69 | 69 | 0.972 | 0.972 |
| 0022600136 | TOR | 88 | 88 | 82 | 82 | 0.932 | 0.932 |
| 0022600136 | ORL | 80 | 80 | 76 | 76 | 0.950 | 0.950 |

**Blueprint covers 650 of 681 official FGA (95.4%); 649 are matched.** The scraper manifest marked every game `unverified` (box-score fetch failed), so this gap was previously undetected.

## Event-identifier hypothesis

Hypothesis: Blueprint `ngss_event_id` equals the official play-by-play `actionNumber` (= ShotChartDetail `GAME_EVENT_ID`). A pair is counted as corroborating only if period, shooter and result agree, the clock differs by at most 2 s and the court location by at most 3 ft.

| game_id | n_blueprint | n_official | bp_ids_present_in_official | id_pairs_corroborated | id_hypothesis_support | matched | matched_by_fallback | ambiguous | rejected_id_conflict | unmatched_blueprint | unmatched_official |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0022600132 | 181 | 192 | 181 | 181 | 1.0000 | 181 | 0 | 0 | 0 | 0 | 11 |
| 0022600133 | 169 | 174 | 169 | 168 | 0.9941 | 168 | 0 | 0 | 1 | 0 | 6 |
| 0022600134 | 142 | 147 | 142 | 142 | 1.0000 | 142 | 0 | 0 | 0 | 0 | 5 |
| 0022600136 | 158 | 168 | 158 | 158 | 1.0000 | 158 | 0 | 0 | 0 | 0 | 10 |

Every Blueprint ID existed in the official play-by-play; all but the conflicts listed below were corroborated, so the attribute fallback was never needed on these games (it is exercised by unit tests).

## Match quality

* Status counts: {'matched': 649, 'rejected_id_conflict': 1}
* Clock difference on matched pairs (s): mean 0.039, max 0.90 (Blueprint clocks are whole seconds; official clocks have tenths in the last minute).
* Court-location difference on matched pairs (ft): max 0.00 (identical coordinates).
* Blueprint `shot_distance_ft` (tracking) minus official coordinate distance: median 0.40 ft, 5th–95th pct -1.93 to 3.83 ft. The two distances are measured differently; Model A uses the official one.
* 2PT/3PT classification disagreements on matched pairs: 2

| shot_uid | PLAYER_NAME | SHOT_TYPE | bp_value | bp_shot_distance_ft | LOC_X | LOC_Y | pbp_description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0022600133_0411 | James Harden | 3PT Field Goal | 2 | 24.20 | -175 | 188 | MISS Harden 26' 3PT Jump Shot |
| 0022600134_0007 | John Collins | 3PT Field Goal | 2 | 22.20 | 217 | 134 | MISS Collins 26' 3PT Jump Shot |

Official classification is used for shot value and Model A; NBA xFG is kept as published.

## Blueprint rows not matched

| shot_uid | match_status | id_pair_rejection | period | clock_raw | shooter_name | made | home_score_after | away_score_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0022600133_0203 | rejected_id_conflict | result | 2 | 9:31 | Deandre Ayton | 0 | 41 | 50 |

Investigated: for `0022600133_0203` Blueprint records Deandre Ayton's tip-in as a miss while its own running score rises 48→50 on that shot and the official play-by-play records a make. This is a Blueprint data error; the row is excluded from the comparison rather than force-matched, and the official record is kept.

## Official FGA absent from Blueprint

32 official FGA have no Blueprint record. They are not a random sample: 75% are misses (all official FGA: 52%) and 19% are tip/putback attempts (all official FGA: 4%). Any NBA-xFG evaluation therefore covers a subset that under-represents misses and tip-ins.

| GAME_ID | GAME_EVENT_ID | PERIOD | pbp_clock | PLAYER_NAME | ACTION_TYPE | SHOT_TYPE | SHOT_DISTANCE | SHOT_MADE_FLAG |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0022600132 | 87 | 1 | PT04M33.00S | Luguentz Dort | Fadeaway Jump Shot | 2PT Field Goal | 12 | 0 |
| 0022600132 | 131 | 1 | PT01M33.00S | Bobby Portis Jr. | Running Jump Shot | 3PT Field Goal | 25 | 0 |
| 0022600132 | 194 | 1 | PT00M00.80S | Nickeil Alexander-Walker | Driving Layup Shot | 2PT Field Goal | 3 | 0 |
| 0022600132 | 180 | 2 | PT10M17.00S | Luguentz Dort | Driving Layup Shot | 2PT Field Goal | 1 | 0 |
| 0022600132 | 344 | 2 | PT09M30.00S | Giannis Antetokounmpo | Layup Shot | 2PT Field Goal | 2 | 0 |
| 0022600132 | 353 | 2 | PT09M28.00S | Giannis Antetokounmpo | Tip Layup Shot | 2PT Field Goal | 0 | 0 |
| 0022600132 | 289 | 2 | PT03M13.00S | Bam Adebayo | Driving Reverse Layup Shot | 2PT Field Goal | 0 | 0 |
| 0022600132 | 377 | 3 | PT10M21.00S | Bam Adebayo | Cutting Layup Shot | 2PT Field Goal | 1 | 0 |
| 0022600132 | 515 | 3 | PT01M35.00S | Aaron Wiggins | Driving Layup Shot | 2PT Field Goal | 0 | 0 |
| 0022600132 | 557 | 4 | PT10M14.00S | Pelle Larsson | Pullup Jump shot | 3PT Field Goal | 25 | 1 |
| 0022600132 | 649 | 4 | PT03M37.00S | Onyeka Okongwu | Running Jump Shot | 3PT Field Goal | 25 | 0 |
| 0022600133 | 67 | 1 | PT05M19.00S | Anthony Davis | Step Back Jump shot | 2PT Field Goal | 15 | 0 |
| 0022600133 | 148 | 1 | PT04M57.00S | Evan Mobley | Tip Layup Shot | 2PT Field Goal | 0 | 0 |
| 0022600133 | 105 | 1 | PT03M43.00S | James Harden | Jump Shot | 3PT Field Goal | 24 | 1 |
| 0022600133 | 203 | 2 | PT09M31.00S | Deandre Ayton | Tip Layup Shot | 2PT Field Goal | 0 | 1 |
| 0022600133 | 289 | 2 | PT03M29.00S | Peyton Watson | Reverse Layup Shot | 2PT Field Goal | 1 | 0 |
| 0022600133 | 433 | 3 | PT07M23.00S | Peyton Watson | Tip Dunk Shot | 2PT Field Goal | 1 | 0 |
| 0022600134 | 156 | 1 | PT02M12.00S | Javonte Green | Layup Shot | 2PT Field Goal | 3 | 0 |
| 0022600134 | 395 | 2 | PT00M53.00S | Brandon Miller | Pullup Jump shot | 2PT Field Goal | 9 | 0 |
| 0022600134 | 420 | 3 | PT10M43.00S | Cade Cunningham | Pullup Jump shot | 3PT Field Goal | 24 | 0 |
| 0022600134 | 422 | 3 | PT10M25.00S | Dennis Schröder | Driving Layup Shot | 2PT Field Goal | 5 | 0 |
| 0022600134 | 471 | 3 | PT06M39.00S | Duncan Robinson | Jump Shot | 3PT Field Goal | 25 | 1 |
| 0022600136 | 29 | 1 | PT09M44.00S | Scottie Barnes | Tip Layup Shot | 2PT Field Goal | 0 | 1 |
| 0022600136 | 102 | 1 | PT05M52.00S | Desmond Bane | Driving Floating Jump Shot | 2PT Field Goal | 11 | 0 |
| 0022600136 | 128 | 1 | PT04M19.00S | Ja'Kobe Walter | Jump Shot | 3PT Field Goal | 23 | 0 |
| 0022600136 | 200 | 1 | PT00M34.20S | RJ Barrett | Jump Shot | 3PT Field Goal | 27 | 1 |
| 0022600136 | 262 | 2 | PT09M44.00S | Jamal Shead | Jump Shot | 3PT Field Goal | 27 | 0 |
| 0022600136 | 340 | 2 | PT05M23.00S | Desmond Bane | Driving Floating Bank Jump Shot | 2PT Field Goal | 1 | 0 |
| 0022600136 | 446 | 3 | PT09M41.00S | Jase Richardson | Jump Shot | 3PT Field Goal | 23 | 1 |
| 0022600136 | 506 | 3 | PT04M34.00S | RJ Barrett | Running Reverse Layup Shot | 2PT Field Goal | 0 | 1 |
| 0022600136 | 566 | 3 | PT00M00.40S | Immanuel Quickley | Step Back Jump shot | 3PT Field Goal | 29 | 0 |
| 0022600136 | 759 | 4 | PT01M18.00S | Jamal Cain | Tip Layup Shot | 2PT Field Goal | 4 | 0 |

## Blueprint internal consistency

| game_id | n_shots | event_id_order_is_chronological | negative_score_steps | made_with_insufficient_score_rise | last_home_score | last_away_score |
| --- | --- | --- | --- | --- | --- | --- |
| 0022600132 | 181 | True | 0 | 0 | 117 | 129 |
| 0022600133 | 169 | True | 0 | 0 | 121 | 111 |
| 0022600134 | 142 | True | 0 | 0 | 97 | 89 |
| 0022600136 | 158 | True | 0 | 0 | 129 | 99 |

`residual points` between consecutive Blueprint shots (free throws or missing made shots) per game: 0022600132: {0.0: 174, 1.0: 7}; 0022600133: {0.0: 162, 1.0: 3, 2.0: 3, 3.0: 1}; 0022600134: {0.0: 138, 1.0: 3, 2.0: 1}; 0022600136: {0.0: 147, 1.0: 8, 2.0: 3}

## Missing values in the unified dataset

Rows: 682 (649 matched, 32 official-only, 1 Blueprint-only). 85 columns are missing only because of row origin (Blueprint fields on official-only rows, official fields on the Blueprint-only row). Columns with other gaps:

| column | missing | missing_on_matched_rows | missing_on_official_only_rows | missing_on_blueprint_only_rows |
| --- | --- | --- | --- | --- |
| SHOT_ZONE_AREA | 682 | 649 | 32 | 1 |

`SHOT_ZONE_AREA` is not available from the play-by-play; `model_b_xfg_exploratory` exists only for compared shots.
