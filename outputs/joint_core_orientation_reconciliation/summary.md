# Final Joint-Core Orientation Reconciliation

**Status: retrospective research only. No production, live-board, weekly-pick, shadow, or prospective-ledger effect.**

## Fixed question

Does the real venue-axis identity explain the previously supported along-field-wind contribution inside `joint_core`, or can randomized venue axes provide the same conditional benefit?

## Predeclared primary test

The reference model is `joint_core_no_along`: the frozen GENERAL baseline plus all climate-context features plus the REAL crosswind feature. The real challenger adds REAL `alongwind_mph`. Each of 39 placebo challengers is otherwise identical but replaces only `alongwind_mph` with an along-wind magnitude calculated from a deranged venue-to-axis assignment.

This deliberately holds raw wind, climate context, real crosswind, model hyperparameters, training folds, and the scored game cohort fixed. It isolates the exact along-field component that was supported in the earlier ablation.

Reconciliation as real conditional geometry requires BOTH: (1) the real challenger passes the same paired-MAE/game-bootstrap/season-cluster/70%-season evidence gate, and (2) the real venue axes beat the 39-placebo distribution at randomization p <= 0.05 and below its 5th percentile.

Current reconciliation status: **CONDITIONAL_REAL_GEOMETRY_RECONCILED**

## Real conditional component test

| challenger      | reference           |   paired_games |   test_seasons |   mean_mae_delta_challenger_minus_reference |   mae_penalty_when_real_along_removed |   game_bootstrap_ci_low |   game_bootstrap_ci_high |   season_cluster_ci_low |   season_cluster_ci_high |   test_seasons_improved |   seasons_required_for_support | evidence_status           |
|:----------------|:--------------------|---------------:|---------------:|--------------------------------------------:|--------------------------------------:|------------------------:|-------------------------:|------------------------:|-------------------------:|------------------------:|-------------------------------:|:--------------------------|
| joint_core_real | joint_core_no_along |           6521 |             10 |                                   -0.061717 |                              0.061717 |               -0.117065 |              -0.00678067 |               -0.106381 |              -0.00720549 |                       7 |                              7 | SUPPORTED_RETROSPECTIVELY |

## Conditional venue-axis placebo test

|   real_joint_core_delta_vs_no_along |   real_mae_penalty_when_along_removed |   placebo_count |   placebo_mean_delta_vs_no_along |   placebo_median_delta_vs_no_along |   placebo_05_quantile_delta_vs_no_along |   placebos_as_good_or_better_than_real |   randomization_p_value | real_better_than_95pct_placebos   | conditional_geometry_status                |
|------------------------------------:|--------------------------------------:|----------------:|---------------------------------:|-----------------------------------:|----------------------------------------:|---------------------------------------:|------------------------:|:----------------------------------|:-------------------------------------------|
|                           -0.061717 |                              0.061717 |              39 |                       -0.0130322 |                         -0.0111479 |                              -0.0431021 |                                      0 |                   0.025 | True                              | REAL_ALONG_DISTINGUISHED_WITHIN_JOINT_CORE |

## Real vs placebo by season

|   test_season |   paired_games |   real_delta_vs_no_along |   placebo_mean_delta |   placebo_median_delta |   placebo_05_quantile_delta |   placebos_as_good_or_better_than_real |   placebo_count |   real_percentile_among_placebos_lower_is_better |
|--------------:|---------------:|-------------------------:|---------------------:|-----------------------:|----------------------------:|---------------------------------------:|----------------:|-------------------------------------------------:|
|          2016 |            648 |              -0.0695722  |          -0.0358797  |            -0.023286   |                 -0.178672   |                                     10 |              39 |                                            0.275 |
|          2017 |            679 |              -0.183594   |          -0.130586   |            -0.119456   |                 -0.23474    |                                     12 |              39 |                                            0.325 |
|          2018 |            656 |              -0.160673   |           0.0267998  |             0.0353784  |                 -0.0756241  |                                      0 |              39 |                                            0.025 |
|          2019 |            668 |              -0.117288   |          -0.059956   |            -0.0673339  |                 -0.133476   |                                      6 |              39 |                                            0.175 |
|          2020 |            398 |               0.0833169  |           0.0833274  |             0.0693452  |                 -0.0334449  |                                     23 |              39 |                                            0.6   |
|          2021 |            669 |              -0.0935502  |          -0.038704   |            -0.0436854  |                 -0.137165   |                                      7 |              39 |                                            0.2   |
|          2022 |            700 |              -0.00865958 |           0.0281684  |             0.0167442  |                 -0.0506515  |                                      8 |              39 |                                            0.225 |
|          2023 |            708 |               0.0226014  |           0.0528854  |             0.0469499  |                 -0.0196811  |                                     10 |              39 |                                            0.275 |
|          2024 |            691 |              -0.0437834  |          -0.023431   |            -0.0172409  |                 -0.114206   |                                     11 |              39 |                                            0.3   |
|          2025 |            704 |               0.00110101 |           0.00165353 |             0.00223721 |                 -0.00920281 |                                     14 |              39 |                                            0.375 |

## Placebo distribution

|   placebo_id |   paired_games |   test_seasons |   mean_mae_delta_placebo_vs_no_along |   mae_penalty_when_placebo_along_removed |   test_seasons_improved_vs_no_along |
|-------------:|---------------:|---------------:|-------------------------------------:|-----------------------------------------:|------------------------------------:|
|            0 |           6521 |             10 |                         -0.0124792   |                              0.0124792   |                                   4 |
|            1 |           6521 |             10 |                         -0.0201187   |                              0.0201187   |                                   5 |
|            2 |           6521 |             10 |                          0.0022421   |                             -0.0022421   |                                   4 |
|            3 |           6521 |             10 |                         -0.0154952   |                              0.0154952   |                                   5 |
|            4 |           6521 |             10 |                          0.0211406   |                             -0.0211406   |                                   4 |
|            5 |           6521 |             10 |                         -0.0608416   |                              0.0608416   |                                   6 |
|            6 |           6521 |             10 |                         -0.00713148  |                              0.00713148  |                                   5 |
|            7 |           6521 |             10 |                         -0.0111479   |                              0.0111479   |                                   4 |
|            8 |           6521 |             10 |                         -0.0164632   |                              0.0164632   |                                   3 |
|            9 |           6521 |             10 |                          0.0364486   |                             -0.0364486   |                                   4 |
|           10 |           6521 |             10 |                         -0.0215333   |                              0.0215333   |                                   5 |
|           11 |           6521 |             10 |                          0.011865    |                             -0.011865    |                                   4 |
|           12 |           6521 |             10 |                         -0.0393723   |                              0.0393723   |                                   7 |
|           13 |           6521 |             10 |                          0.0064494   |                             -0.0064494   |                                   5 |
|           14 |           6521 |             10 |                          0.00102112  |                             -0.00102112  |                                   4 |
|           15 |           6521 |             10 |                         -0.00780678  |                              0.00780678  |                                   4 |
|           16 |           6521 |             10 |                         -0.0416296   |                              0.0416296   |                                   7 |
|           17 |           6521 |             10 |                         -0.0296547   |                              0.0296547   |                                   5 |
|           18 |           6521 |             10 |                         -0.0202117   |                              0.0202117   |                                   4 |
|           19 |           6521 |             10 |                         -0.0107077   |                              0.0107077   |                                   6 |
|           20 |           6521 |             10 |                         -0.0563546   |                              0.0563546   |                                   7 |
|           21 |           6521 |             10 |                         -0.0260155   |                              0.0260155   |                                   5 |
|           22 |           6521 |             10 |                         -0.00949416  |                              0.00949416  |                                   5 |
|           23 |           6521 |             10 |                          0.00623648  |                             -0.00623648  |                                   4 |
|           24 |           6521 |             10 |                          0.00811212  |                             -0.00811212  |                                   4 |
|           25 |           6521 |             10 |                         -0.00686255  |                              0.00686255  |                                   7 |
|           26 |           6521 |             10 |                         -0.037038    |                              0.037038    |                                   6 |
|           27 |           6521 |             10 |                         -0.00936077  |                              0.00936077  |                                   5 |
|           28 |           6521 |             10 |                          0.0196374   |                             -0.0196374   |                                   3 |
|           29 |           6521 |             10 |                         -0.000161415 |                              0.000161415 |                                   5 |
|           30 |           6521 |             10 |                          0.000902801 |                             -0.000902801 |                                   5 |
|           31 |           6521 |             10 |                         -0.0123511   |                              0.0123511   |                                   6 |
|           32 |           6521 |             10 |                         -0.0141426   |                              0.0141426   |                                   6 |
|           33 |           6521 |             10 |                         -0.00364948  |                              0.00364948  |                                   5 |
|           34 |           6521 |             10 |                         -0.0401662   |                              0.0401662   |                                   6 |
|           35 |           6521 |             10 |                         -0.0280984   |                              0.0280984   |                                   5 |
|           36 |           6521 |             10 |                         -0.00955479  |                              0.00955479  |                                   5 |
|           37 |           6521 |             10 |                         -0.0273692   |                              0.0273692   |                                   8 |
|           38 |           6521 |             10 |                         -0.0270988   |                              0.0270988   |                                   6 |

## Decision rule after this study

- `CONDITIONAL_REAL_GEOMETRY_RECONCILED`: real venue axes are uniquely informative inside joint_core. This still does not change production; it could only justify a separately frozen prospective shadow.
- `ORIENTATION_PROXY_NOT_DISTINGUISHED`: the real axes are not special even though an along-wind-like feature may help conditionally. Retire field orientation from further retrospective tuning and do not create an orientation shadow.
- `CONDITIONAL_ALONG_SIGNAL_NOT_REPRODUCED`: the prior along-field ablation itself did not reproduce. Retire that signal.

## Guardrails

- No new features, thresholds, training windows, or post-hoc season exclusions are tested here.
- The 39 permutations and seed are fixed before observing this run's outcome.
- The prior failed standalone geometry confirmation is not overwritten by this test.
- Production remains frozen regardless of the retrospective result.