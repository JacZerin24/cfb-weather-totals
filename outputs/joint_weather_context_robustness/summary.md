# Joint-Core Robustness and Ablation Research

**Status: retrospective research only. No production, live-board, weekly-pick, orientation-shadow, or prospective-ledger effect.**

## Purpose

Stress-test the previously observed `joint_core` signal without changing the original evidence gate. This study asks which feature groups matter, whether the signal depends on any one test season, and whether it survives different historical training windows.

The original joint-weather result remains `NOT_PROVEN` unless its original predeclared gate is met. This robustness study cannot retroactively move that goalpost.

## Predeclared robustness checks

1. Grouped and component ablations from `joint_core`.
2. Leave-one-test-season-out sensitivity.
3. Expanding-history, rolling-six-season, and rolling-four-season walk-forward training windows.
4. The same paired MAE, game bootstrap, season-cluster uncertainty, and 70% season-stability framework used in the parent study.

For component ablations, a positive `mae_penalty_when_removed` means the full `joint_core` model performed better after that component was restored. Component support requires both uncertainty intervals to remain above zero and the full model to win in at least 70% of seasons.

The secondary `ROBUSTNESS_STRENGTHENED` label requires all training-window mean deltas to favor `joint_core`, every leave-one-season-out mean delta to remain favorable, every window to improve at least 70% of test seasons, and at least two training-window season-cluster intervals to be entirely below zero. It is not a production-promotion label.

Current robustness status: **ROBUSTNESS_MIXED**

## Component ablations

| removed_component   | ablation_model               |   paired_games |   test_seasons |   mae_penalty_when_removed |   component_game_ci_low |   component_game_ci_high |   component_season_ci_low |   component_season_ci_high |   test_seasons_joint_core_better |   seasons_required_for_component_support | component_evidence_status   |
|:--------------------|:-----------------------------|---------------:|---------------:|---------------------------:|------------------------:|-------------------------:|--------------------------:|---------------------------:|---------------------------------:|-----------------------------------------:|:----------------------------|
| orientation_vector  | joint_no_orientation         |           6521 |             10 |                  0.0415369 |             -0.0144779  |                0.0977687 |               -0.0178231  |                  0.0828132 |                                9 |                                        7 | COMPONENT_NOT_ISOLATED      |
| climate_context     | joint_no_climate             |           6521 |             10 |                  0.0415558 |             -0.0220341  |                0.106521  |                0.0133819  |                  0.0751906 |                                8 |                                        7 | COMPONENT_NOT_ISOLATED      |
| crosswind           | joint_no_crosswind           |           6521 |             10 |                  0.0310773 |             -0.0259483  |                0.0876359 |               -0.0347104  |                  0.0700112 |                                7 |                                        7 | COMPONENT_NOT_ISOLATED      |
| along_field_wind    | joint_no_alongwind           |           6521 |             10 |                  0.061717  |              0.00645231 |                0.115583  |                0.00772179 |                  0.105602  |                                7 |                                        7 | COMPONENT_SUPPORTED         |
| temperature_context | joint_no_temperature_context |           6521 |             10 |                  0.0394784 |             -0.0217336  |                0.101552  |               -0.036962   |                  0.113944  |                                7 |                                        7 | COMPONENT_NOT_ISOLATED      |
| local_wind_context  | joint_no_local_wind_context  |           6521 |             10 |                  0.0547707 |             -0.0035949  |                0.11388   |                0.0131018  |                  0.0960342 |                                8 |                                        7 | COMPONENT_NOT_ISOLATED      |
| latitude_context    | joint_no_latitude_context    |           6521 |             10 |                  0.0218977 |             -0.035474   |                0.0796936 |               -0.0149787  |                  0.0683306 |                                6 |                                        7 | COMPONENT_NOT_ISOLATED      |

## Alternate temporal training windows

| training_window   |   paired_games |   test_seasons |   mean_mae_delta_challenger_minus_reference |   game_bootstrap_ci_low |   game_bootstrap_ci_high |   season_cluster_ci_low |   season_cluster_ci_high |   test_seasons_improved |   seasons_required_for_support | evidence_status   |
|:------------------|---------------:|---------------:|--------------------------------------------:|------------------------:|-------------------------:|------------------------:|-------------------------:|------------------------:|-------------------------------:|:------------------|
| expanding         |           6521 |             10 |                                  -0.0585156 |               -0.125516 |               0.00670278 |               -0.11419  |               -0.0151388 |                       8 |                              7 | NOT_PROVEN        |
| rolling_6_seasons |           6521 |             10 |                                  -0.0433362 |               -0.117191 |               0.0295151  |               -0.107896 |                0.0127142 |                       6 |                              7 | NOT_PROVEN        |
| rolling_4_seasons |           6521 |             10 |                                  -0.0695252 |               -0.147724 |               0.00721304 |               -0.131718 |               -0.0208976 |                       8 |                              7 | NOT_PROVEN        |

## Leave-one-test-season-out sensitivity

|   omitted_test_season |   paired_games |   test_seasons |   mean_mae_delta_challenger_minus_reference |   game_bootstrap_ci_low |   game_bootstrap_ci_high |   season_cluster_ci_low |   season_cluster_ci_high |   test_seasons_improved |   seasons_required_for_support | evidence_status           |
|----------------------:|---------------:|---------------:|--------------------------------------------:|------------------------:|-------------------------:|------------------------:|-------------------------:|------------------------:|-------------------------------:|:--------------------------|
|                  2016 |           5873 |              9 |                                  -0.0591544 |               -0.127904 |              0.00960156  |              -0.11952   |              -0.0102068  |                       7 |                              7 | NOT_PROVEN                |
|                  2017 |           5842 |              9 |                                  -0.0689335 |               -0.13806  |              0.000476746 |              -0.125334  |              -0.0244475  |                       8 |                              7 | NOT_PROVEN                |
|                  2018 |           5865 |              9 |                                  -0.0377718 |               -0.106756 |              0.0322397   |              -0.0750327 |              -0.0037766  |                       7 |                              7 | NOT_PROVEN                |
|                  2019 |           5853 |              9 |                                  -0.0545339 |               -0.127304 |              0.0180157   |              -0.117571  |              -0.00762242 |                       7 |                              7 | NOT_PROVEN                |
|                  2020 |           6123 |              9 |                                  -0.0557556 |               -0.125074 |              0.0111227   |              -0.11479   |              -0.00645592 |                       7 |                              7 | NOT_PROVEN                |
|                  2021 |           5852 |              9 |                                  -0.0562203 |               -0.126257 |              0.0145038   |              -0.117215  |              -0.00940796 |                       7 |                              7 | NOT_PROVEN                |
|                  2022 |           5821 |              9 |                                  -0.0727736 |               -0.144094 |             -0.00146576  |              -0.127673  |              -0.0311258  |                       8 |                              7 | SUPPORTED_RETROSPECTIVELY |
|                  2023 |           5813 |              9 |                                  -0.0542454 |               -0.126228 |              0.0175222   |              -0.11553   |              -0.00665587 |                       7 |                              7 | NOT_PROVEN                |
|                  2024 |           5830 |              9 |                                  -0.062619  |               -0.136177 |              0.00829337  |              -0.121597  |              -0.0132461  |                       7 |                              7 | NOT_PROVEN                |
|                  2025 |           5817 |              9 |                                  -0.0634305 |               -0.139043 |              0.01073     |              -0.122062  |              -0.0157754  |                       7 |                              7 | NOT_PROVEN                |

## Robustness indicators

| robustness_status   | all_window_mean_deltas_negative   | all_loso_mean_deltas_negative   | all_windows_improve_70pct_seasons   |   windows_with_season_ci_below_zero |   windows_with_full_evidence_support |   worst_loso_mean_delta |
|:--------------------|:----------------------------------|:--------------------------------|:------------------------------------|------------------------------------:|-------------------------------------:|------------------------:|
| ROBUSTNESS_MIXED    | True                              | True                            | False                               |                                   2 |                                    0 |              -0.0377718 |

## Guardrails

- These tests are retrospective and do not alter the frozen 2026 live protocol.
- A component can be useful jointly without being individually identifiable because HGB features can be redundant and nonlinear.
- Rolling-window sensitivity tests whether the result depends on older training history; it does not select a new production training window.
- Leave-one-test-season-out sensitivity checks whether one evaluation season carries the result; it does not refit after removing that season from historical training.
- Even a strengthened robustness result only supports considering a separately frozen prospective shadow challenger.