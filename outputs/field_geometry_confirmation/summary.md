# Confirmatory Field-Geometry Research

**Status: retrospective research only. No production, live-board, weekly-pick, orientation-shadow, or prospective-ledger effect.**

## Purpose

Confirm or falsify the previously isolated along-field-wind signal without expanding the production feature set or changing the parent evidence gate.

## Predeclared primary confirmation

The real-axis `along_magnitude` challenger must satisfy the same four-part retrospective evidence gate used previously and must also beat a 39-permutation venue-axis placebo test at randomization p <= 0.05. Field axes are permuted across venues while preserving the real axis distribution and orientation coverage.

Alignment-only models are mechanism diagnostics: because baseline already contains wind speed, adding cos(field-relative angle) tests whether actual relative geometry adds information beyond wind magnitude alone.

Current confirmation status: **GEOMETRY_NOT_CONFIRMED**

## Real-axis model comparisons

| challenger       | reference   |   paired_games |   test_seasons |   mean_mae_delta_challenger_minus_reference |   game_bootstrap_ci_low |   game_bootstrap_ci_high |   season_cluster_ci_low |   season_cluster_ci_high |   test_seasons_improved |   seasons_required_for_support | evidence_status   |
|:-----------------|:------------|---------------:|---------------:|--------------------------------------------:|------------------------:|-------------------------:|------------------------:|-------------------------:|------------------------:|-------------------------------:|:------------------|
| along_magnitude  | baseline    |           6521 |             10 |                                  -0.0211844 |              -0.0758589 |               0.0326752  |              -0.0882511 |                0.0417961 |                       6 |                              7 | NOT_PROVEN        |
| cross_magnitude  | baseline    |           6521 |             10 |                                  -0.0281751 |              -0.0841051 |               0.0286086  |              -0.0752202 |                0.026978  |                       7 |                              7 | NOT_PROVEN        |
| vector_magnitude | baseline    |           6521 |             10 |                                  -0.0169598 |              -0.0745885 |               0.0411538  |              -0.0610545 |                0.0282835 |                       7 |                              7 | NOT_PROVEN        |
| along_alignment  | baseline    |           6521 |             10 |                                   0.0325215 |              -0.0229205 |               0.0887019  |              -0.035991  |                0.106153  |                       4 |                              7 | NOT_PROVEN        |
| alignment_pair   | baseline    |           6521 |             10 |                                  -0.0105502 |              -0.0664081 |               0.0452915  |              -0.0871878 |                0.050228  |                       6 |                              7 | NOT_PROVEN        |
| joint_core       | baseline    |           6521 |             10 |                                  -0.0585156 |              -0.12729   |               0.00752342 |              -0.112397  |               -0.0143002 |                       8 |                              7 | NOT_PROVEN        |

## Venue-axis permutation placebo

|   real_along_mean_mae_delta_vs_baseline |   placebo_count |   placebo_mean_delta |   placebo_median_delta |   placebo_05_quantile_delta |   placebos_as_good_or_better_than_real |   randomization_p_value | real_better_than_95pct_placebos   | placebo_geometry_status     |
|----------------------------------------:|----------------:|---------------------:|-----------------------:|----------------------------:|---------------------------------------:|------------------------:|:----------------------------------|:----------------------------|
|                              -0.0211844 |              39 |          -0.00268257 |            -0.00413064 |                  -0.0420874 |                                      8 |                   0.225 | False                             | REAL_AXIS_NOT_DISTINGUISHED |

## Results by test season

|   test_season | model            |   paired_games |     mae |   mae_delta_vs_baseline |    rmse |   signed_projection_bias |
|--------------:|:-----------------|---------------:|--------:|------------------------:|--------:|-------------------------:|
|          2016 | baseline         |            648 | 13.6698 |              0          | 17.6423 |               -0.590378  |
|          2016 | along_magnitude  |            648 | 13.5919 |             -0.0778768  | 17.6074 |               -0.56499   |
|          2016 | cross_magnitude  |            648 | 13.4648 |             -0.205053   | 17.4999 |               -0.765552  |
|          2016 | vector_magnitude |            648 | 13.5907 |             -0.0791362  | 17.5937 |               -0.664983  |
|          2016 | along_alignment  |            648 | 13.6147 |             -0.0551323  | 17.6547 |               -0.407239  |
|          2016 | alignment_pair   |            648 | 13.5862 |             -0.0835982  | 17.6661 |               -0.782874  |
|          2016 | joint_core       |            648 | 13.6171 |             -0.0527261  | 17.6795 |               -0.703494  |
|          2017 | baseline         |            679 | 14.0606 |              0          | 17.6704 |                0.621949  |
|          2017 | along_magnitude  |            679 | 14.1706 |              0.109936   | 17.6994 |                0.788089  |
|          2017 | cross_magnitude  |            679 | 14.087  |              0.0263313  | 17.7382 |                0.758902  |
|          2017 | vector_magnitude |            679 | 14.1871 |              0.126479   | 17.8117 |                0.514486  |
|          2017 | along_alignment  |            679 | 14.2741 |              0.213428   | 17.8296 |                0.831494  |
|          2017 | alignment_pair   |            679 | 14.2036 |              0.142912   | 17.7355 |                0.951491  |
|          2017 | joint_core       |            679 | 14.0918 |              0.0311181  | 17.7341 |                0.860141  |
|          2018 | baseline         |            656 | 13.5037 |              0          | 17.2376 |               -0.506767  |
|          2018 | along_magnitude  |            656 | 13.3289 |             -0.174723   | 17.1464 |               -0.719679  |
|          2018 | cross_magnitude  |            656 | 13.4121 |             -0.091565   | 17.19   |               -0.784068  |
|          2018 | vector_magnitude |            656 | 13.3598 |             -0.143838   | 17.1471 |               -0.21221   |
|          2018 | along_alignment  |            656 | 13.4037 |             -0.0999146  | 17.2449 |               -0.515695  |
|          2018 | alignment_pair   |            656 | 13.409  |             -0.094607   | 17.2371 |               -0.536267  |
|          2018 | joint_core       |            656 | 13.2597 |             -0.243977   | 17.0718 |               -0.478751  |
|          2019 | baseline         |            668 | 13.1563 |              0          | 16.5705 |                0.481441  |
|          2019 | along_magnitude  |            668 | 13.0204 |             -0.135865   | 16.4604 |                0.316452  |
|          2019 | cross_magnitude  |            668 | 13.145  |             -0.0112695  | 16.5199 |                0.683695  |
|          2019 | vector_magnitude |            668 | 13.0964 |             -0.0598903  | 16.5592 |                0.374005  |
|          2019 | along_alignment  |            668 | 13.2308 |              0.0744726  | 16.6609 |                0.518785  |
|          2019 | alignment_pair   |            668 | 13.0244 |             -0.13192    | 16.4058 |                0.531347  |
|          2019 | joint_core       |            668 | 13.0629 |             -0.093403   | 16.5137 |               -0.0362482 |
|          2020 | baseline         |            398 | 14.3639 |              0          | 18.0218 |               -0.610224  |
|          2020 | along_magnitude  |            398 | 14.2486 |             -0.115302   | 17.8432 |               -0.326789  |
|          2020 | cross_magnitude  |            398 | 14.4806 |              0.116702   | 18.1408 |               -0.459978  |
|          2020 | vector_magnitude |            398 | 14.3522 |             -0.0117571  | 17.9741 |               -0.187764  |
|          2020 | along_alignment  |            398 | 14.3415 |             -0.0224423  | 17.9272 |               -0.321982  |
|          2020 | alignment_pair   |            398 | 14.1696 |             -0.194384   | 17.7952 |               -0.563822  |
|          2020 | joint_core       |            398 | 14.263  |             -0.100977   | 18.0428 |               -0.250381  |
|          2021 | baseline         |            669 | 12.6486 |              0          | 16.1113 |               -1.31216   |
|          2021 | along_magnitude  |            669 | 12.8059 |              0.15728    | 16.2404 |               -1.76866   |
|          2021 | cross_magnitude  |            669 | 12.7107 |              0.0621233  | 16.0861 |               -0.989664  |
|          2021 | vector_magnitude |            669 | 12.6864 |              0.0378237  | 16.0697 |               -1.32976   |
|          2021 | along_alignment  |            669 | 12.9092 |              0.260563   | 16.3838 |               -2.5656    |
|          2021 | alignment_pair   |            669 | 12.8001 |              0.151509   | 16.316  |               -1.97799   |
|          2021 | joint_core       |            669 | 12.57   |             -0.0785934  | 15.9864 |               -0.821677  |
|          2022 | baseline         |            700 | 12.161  |              0          | 15.5351 |                0.247084  |
|          2022 | along_magnitude  |            700 | 12.2375 |              0.0765559  | 15.5809 |                0.0562733 |
|          2022 | cross_magnitude  |            700 | 12.1489 |             -0.0120813  | 15.467  |                0.0451742 |
|          2022 | vector_magnitude |            700 | 12.2019 |              0.0408899  | 15.5486 |                0.0661144 |
|          2022 | along_alignment  |            700 | 12.1809 |              0.019891   | 15.4599 |                0.0062408 |
|          2022 | alignment_pair   |            700 | 12.2467 |              0.0856807  | 15.5571 |                0.192786  |
|          2022 | joint_core       |            700 | 12.221  |              0.0600501  | 15.5089 |                0.310278  |
|          2023 | baseline         |            708 | 13.0901 |              0          | 16.3298 |               -1.10954   |
|          2023 | along_magnitude  |            708 | 13.05   |             -0.040034   | 16.2757 |               -0.927337  |
|          2023 | cross_magnitude  |            708 | 13.0272 |             -0.0628719  | 16.1971 |               -1.01181   |
|          2023 | vector_magnitude |            708 | 13.0192 |             -0.0708282  | 16.1857 |               -1.08903   |
|          2023 | along_alignment  |            708 | 12.9843 |             -0.105758   | 16.2291 |               -1.24943   |
|          2023 | alignment_pair   |            708 | 13.0751 |             -0.014991   | 16.3384 |               -1.11915   |
|          2023 | joint_core       |            708 | 12.9965 |             -0.0935755  | 16.1782 |               -1.04861   |
|          2024 | baseline         |            691 | 12.8281 |              0          | 16.3765 |               -0.733278  |
|          2024 | along_magnitude  |            691 | 12.7605 |             -0.0676002  | 16.3093 |               -0.634212  |
|          2024 | cross_magnitude  |            691 | 12.7929 |             -0.0351668  | 16.3653 |               -0.639741  |
|          2024 | vector_magnitude |            691 | 12.8239 |             -0.00417724 | 16.4326 |               -0.815321  |
|          2024 | along_alignment  |            691 | 12.8337 |              0.00556271 | 16.375  |               -0.708444  |
|          2024 | alignment_pair   |            691 | 12.7626 |             -0.0654672  | 16.3054 |               -0.625686  |
|          2024 | joint_core       |            691 | 12.8042 |             -0.0238952  | 16.4005 |               -0.959072  |
|          2025 | baseline         |            704 | 12.3969 |              0          | 15.4122 |                0.104045  |
|          2025 | along_magnitude  |            704 | 12.4043 |              0.00736777 | 15.4252 |                0.0970138 |
|          2025 | cross_magnitude  |            704 | 12.3801 |             -0.016871   | 15.3697 |                0.105298  |
|          2025 | vector_magnitude |            704 | 12.3869 |             -0.0100064  | 15.3805 |                0.102873  |
|          2025 | along_alignment  |            704 | 12.4117 |              0.0147749  | 15.4293 |                0.0885598 |
|          2025 | alignment_pair   |            704 | 12.4117 |              0.0147749  | 15.4293 |                0.0885598 |
|          2025 | joint_core       |            704 | 12.379  |             -0.0179052  | 15.3674 |                0.147205  |

## Season context diagnostics

|   test_season |   games |   unique_venues |   pct_venues_seen_in_prior_train |   mean_wind_mph |   mean_alongwind_mph |   mean_crosswind_mph |   mean_field_angle_deg |   mean_along_alignment |   mean_temperature_f |   mean_temperature_anomaly_f |   mean_local_wind_percentile |   mean_closing_total |   baseline_mae |   along_mae |   along_mae_delta_vs_baseline |   joint_core_mae |   joint_core_mae_delta_vs_baseline |
|--------------:|--------:|----------------:|---------------------------------:|----------------:|---------------------:|---------------------:|-----------------------:|-----------------------:|---------------------:|-----------------------------:|-----------------------------:|---------------------:|---------------:|------------:|------------------------------:|-----------------:|-----------------------------------:|
|          2016 |     648 |             122 |                         1        |         8.37562 |              5.25957 |              5.3783  |                45.1781 |               0.632587 |              67.2378 |                     3.4685   |                     0.529395 |              57.2593 |        13.6698 |     13.5919 |                   -0.0778768  |          13.6171 |                         -0.0527261 |
|          2017 |     679 |             126 |                         0.979381 |         8.34934 |              5.55321 |              4.99581 |                43.5185 |               0.652232 |              67.8221 |                     2.05148  |                     0.533667 |              55.9853 |        14.0606 |     14.1706 |                    0.109936   |          14.0918 |                          0.0311181 |
|          2018 |     656 |             126 |                         0.995427 |         8.43308 |              5.52609 |              5.21922 |                44.0425 |               0.648367 |              64.3091 |                    -2.16241  |                     0.490921 |              56.5244 |        13.5037 |     13.3289 |                   -0.174723   |          13.2597 |                         -0.243977  |
|          2019 |     668 |             127 |                         1        |         8.20135 |              5.22477 |              5.14677 |                43.8051 |               0.646085 |              64.4507 |                    -1.09389  |                     0.473229 |              55.5479 |        13.1563 |     13.0204 |                   -0.135865   |          13.0629 |                         -0.093403  |
|          2020 |     398 |             117 |                         0.969849 |         8.37261 |              5.8694  |              4.6964  |                39.9562 |               0.691539 |              60.7658 |                    -0.532326 |                     0.498578 |              56.7186 |        14.3639 |     14.2486 |                   -0.115302   |          14.263  |                         -0.100977  |
|          2021 |     669 |             127 |                         0.995516 |         7.46039 |              4.84839 |              4.62679 |                44.0185 |               0.64499  |              65.2873 |                     0.107042 |                     0.419508 |              55.8498 |        12.6486 |     12.8059 |                    0.15728    |          12.57   |                         -0.0785934 |
|          2022 |     700 |             128 |                         0.984286 |         8.06443 |              5.2018  |              5.07206 |                43.1317 |               0.654837 |              65.1226 |                    -0.206458 |                     0.485992 |              54.3679 |        12.161  |     12.2375 |                    0.0765559  |          12.221  |                          0.0600501 |
|          2023 |     708 |             130 |                         1        |         7.42415 |              4.69404 |              4.72621 |                42.9491 |               0.653289 |              66.8475 |                     0.6569   |                     0.471913 |              52.4336 |        13.0901 |     13.05   |                   -0.040034   |          12.9965 |                         -0.0935755 |
|          2024 |     691 |             126 |                         1        |         7.51346 |              4.71219 |              4.84005 |                43.2032 |               0.649019 |              67.7731 |                     2.23464  |                     0.488348 |              52.521  |        12.8281 |     12.7605 |                   -0.0676002  |          12.8042 |                         -0.0238952 |
|          2025 |     704 |             128 |                         1        |         7.09815 |              4.45353 |              4.59465 |                42.7107 |               0.661015 |              67.8037 |                     1.85259  |                     0.454213 |              52.2429 |        12.3969 |     12.4043 |                    0.00736777 |          12.379  |                         -0.0179052 |

## 2022 standardized context shift

These are descriptive standardized mean differences, not a post-hoc promotion test.

| feature               |   season_2022_mean |   other_seasons_mean |   standardized_mean_difference_2022_minus_others |   absolute_standardized_shift |
|:----------------------|-------------------:|---------------------:|-------------------------------------------------:|------------------------------:|
| temperature_anomaly_f |          -0.206458 |             0.801318 |                                      -0.103414   |                    0.103414   |
| closing_total         |          54.3679   |            54.8764   |                                      -0.0627782  |                    0.0627782  |
| temperature_f         |          65.1226   |            66.0739   |                                      -0.0614661  |                    0.0614661  |
| wind_mph              |           8.06443  |             7.88107  |                                       0.0364183  |                    0.0364183  |
| crosswind_mph         |           5.07206  |             4.91821  |                                       0.0357389  |                    0.0357389  |
| alongwind_mph         |           5.2018   |             5.08306  |                                       0.0272464  |                    0.0272464  |
| along_alignment       |           0.654837 |             0.651569 |                                       0.0105935  |                    0.0105935  |
| wind_field_angle_deg  |          43.1317   |            43.4047   |                                      -0.0102899  |                    0.0102899  |
| wind_local_percentile |           0.485992 |             0.483419 |                                       0.00876604 |                    0.00876604 |

## 2022 geometry-regime diagnostics

| grouping        | value           | period             |   games |   baseline_mae |   along_mae |   along_mae_delta_vs_baseline |
|:----------------|:----------------|:-------------------|--------:|---------------:|------------:|------------------------------:|
| field_angle_bin | 0-22.5_parallel | 2022               |     204 |        12.3842 |     12.4306 |                    0.0463694  |
| field_angle_bin | 0-22.5_parallel | other_test_seasons |    1680 |        13.2166 |     13.1866 |                   -0.0300233  |
| field_angle_bin | 67.5-90_cross   | 2022               |     168 |        10.9081 |     11.1637 |                    0.255556   |
| field_angle_bin | 67.5-90_cross   | other_test_seasons |    1441 |        13.4973 |     13.4944 |                   -0.00291651 |
| field_angle_bin | 22.5-45         | 2022               |     173 |        12.9669 |     12.8691 |                   -0.0978014  |
| field_angle_bin | 22.5-45         | other_test_seasons |    1351 |        13.013  |     12.9846 |                   -0.0284718  |
| field_angle_bin | 45-67.5         | 2022               |     155 |        12.3256 |     12.4425 |                    0.116877   |
| field_angle_bin | 45-67.5         | other_test_seasons |    1349 |        13.2366 |     13.1634 |                   -0.0731102  |
| wind_speed_bin  | 10-15           | 2022               |     156 |        12.609  |     12.7916 |                    0.182535   |
| wind_speed_bin  | 10-15           | other_test_seasons |    1214 |        12.7787 |     12.7337 |                   -0.0450665  |
| wind_speed_bin  | 0-5             | 2022               |     216 |        13.1404 |     13.0968 |                   -0.0436273  |
| wind_speed_bin  | 0-5             | other_test_seasons |    1756 |        13.4162 |     13.3113 |                   -0.104936   |
| wind_speed_bin  | 5-10            | 2022               |     259 |        11.1742 |     11.2192 |                    0.0449652  |
| wind_speed_bin  | 5-10            | other_test_seasons |    2358 |        13.2808 |     13.2934 |                    0.0126378  |
| wind_speed_bin  | 15+             | 2022               |      69 |        11.7858 |     12.1175 |                    0.331755   |
| wind_speed_bin  | 15+             | other_test_seasons |     493 |        13.594  |     13.6294 |                    0.0353872  |

## Guardrails

- The primary hypothesis is fixed before this run: real along-field geometry must beat baseline and randomized venue axes.
- Crosswind, vector, and alignment models are secondary mechanism checks and cannot substitute for failure of the primary confirmation.
- The 2022 diagnostics are explanatory only. No season may be removed to manufacture significance.
- Even retrospective confirmation would only justify considering a separately frozen prospective shadow challenger. It would not alter the live 2026 model.