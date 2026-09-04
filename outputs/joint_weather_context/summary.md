# Joint Weather-Context Research

**Status: retrospective research only. No production, live-board, weekly-pick, orientation-shadow, or prospective-ledger effect.**

## Research question

Does combining the existing GENERAL HGB weather/matchup model with field-relative wind, local climate context, latitude, and physically motivated interaction terms improve prediction of `market_residual = actual_total_points - closing_total` out of sample?

## Isolation boundary

- This script reads the historical modeling artifact and reference tables only.
- It writes only to `outputs/joint_weather_context/`.
- It does not call the live weekly runner, change production model features, modify thresholds, write the official 2026 prospective ledger, or change the existing orientation shadow.
- Retrospective evidence can only justify considering a separately versioned future challenger. Nothing here promotes itself.

## Paired-data coverage

Joint orientation-and-climate-ready FBS games: **7,823**

| scope   |   games |   outdoor_games |   fbs_vs_fbs_games |   context_ready_games |   orientation_ready_games |   joint_ready_games |   joint_ready_pct_of_fbs |   unique_joint_ready_venues |
|:--------|--------:|----------------:|-------------------:|----------------------:|--------------------------:|--------------------:|-------------------------:|----------------------------:|
| overall |   11757 |           11362 |               8605 |                 11137 |                      8916 |                7823 |                 0.909123 |                         137 |
| 2014    |     721 |             697 |                721 |                   689 |                       643 |                 643 |                 0.891817 |                         122 |
| 2015    |     724 |             701 |                724 |                   694 |                       659 |                 659 |                 0.910221 |                         123 |
| 2016    |     719 |             697 |                719 |                   691 |                       648 |                 648 |                 0.901252 |                         122 |
| 2017    |     745 |             719 |                736 |                   712 |                       687 |                 679 |                 0.922554 |                         126 |
| 2018    |     812 |             791 |                733 |                   778 |                       728 |                 656 |                 0.894952 |                         126 |
| 2019    |     841 |             818 |                734 |                   806 |                       768 |                 668 |                 0.910082 |                         127 |
| 2020    |     541 |             522 |                508 |                   514 |                       426 |                 398 |                 0.783465 |                         117 |
| 2021    |     849 |             825 |                732 |                   792 |                       776 |                 669 |                 0.913934 |                         127 |
| 2022    |    1413 |            1357 |                734 |                  1334 |                       896 |                 700 |                 0.953678 |                         128 |
| 2023    |    1345 |            1299 |                750 |                  1277 |                       890 |                 708 |                 0.944    |                         130 |
| 2024    |    1503 |            1447 |                752 |                  1404 |                       881 |                 691 |                 0.918883 |                         126 |
| 2025    |    1544 |            1489 |                762 |                  1446 |                       914 |                 704 |                 0.923885 |                         128 |

## Predeclared model ladder

1. `baseline`: existing GENERAL HGB features.
2. `orientation_crosswind`: baseline + field-relative crosswind.
3. `orientation_vector`: crosswind + along-field wind.
4. `climate_full`: latitude + leak-safe local temperature/wind context and latitude interactions.
5. `joint_core`: orientation vector + climate context together.
6. `joint_interactions`: joint core + a limited set of physically motivated orientation × climate/market interactions.

All models are evaluated on the exact same joint-ready games in each test season. Every test season is predicted using only prior seasons, and local weather baselines are fitted only from those prior seasons.

## Evidence gate

A challenger is labeled `SUPPORTED_RETROSPECTIVELY` only when all of the following hold against its reference:

- mean paired MAE delta is negative;
- the 95% game-level paired bootstrap interval is entirely below zero;
- the 95% season-cluster interval is entirely below zero;
- it improves in at least 70% of evaluated test seasons.

Qualifier hit rate and ROI are secondary context and cannot satisfy this gate. Even a supported retrospective result still requires a new frozen prospective shadow before any production discussion.

Current full-joint retrospective evidence status: **NOT_PROVEN**

## Model comparison vs baseline

| model                 |   paired_games |   test_seasons |     mae |    rmse |   signed_projection_bias |   mae_delta_vs_baseline |   game_bootstrap_ci_low |   game_bootstrap_ci_high |   season_cluster_ci_low |   season_cluster_ci_high |   test_seasons_improved_vs_baseline |   seasons_required_for_support | evidence_status   |   qualifiers |   qualifier_wins |   qualifier_losses |   qualifier_pushes |   qualifier_hit_rate |   qualifier_roi_per_1u |
|:----------------------|---------------:|---------------:|--------:|--------:|-------------------------:|------------------------:|------------------------:|-------------------------:|------------------------:|-------------------------:|------------------------------------:|-------------------------------:|:------------------|-------------:|-----------------:|-------------------:|-------------------:|---------------------:|-----------------------:|
| baseline              |           6521 |             10 | 13.1273 | 16.6384 |                -0.32784  |               0         |               0         |               0          |               0         |                0         |                                   0 |                              7 | REFERENCE         |          609 |              356 |                246 |                  7 |             0.591362 |              0.127482  |
| orientation_crosswind |           6521 |             10 | 13.0991 | 16.5999 |                -0.296925 |              -0.0281751 |              -0.0847592 |               0.0282242  |              -0.0762269 |                0.0264018 |                                   7 |                              7 | NOT_PROVEN        |          599 |              335 |                257 |                  7 |             0.565878 |              0.0793747 |
| orientation_vector    |           6521 |             10 | 13.1103 | 16.6198 |                -0.329858 |              -0.0169598 |              -0.0735358 |               0.0405225  |              -0.061211  |                0.0278044 |                                   7 |                              7 | NOT_PROVEN        |          605 |              345 |                252 |                  8 |             0.577889 |              0.101878  |
| climate_full          |           6521 |             10 | 13.1103 | 16.615  |                -0.206462 |              -0.0169787 |              -0.0812487 |               0.0477155  |              -0.0915177 |                0.0308958 |                                   6 |                              7 | NOT_PROVEN        |          576 |              344 |                225 |                  7 |             0.604569 |              0.152304  |
| joint_core            |           6521 |             10 | 13.0687 | 16.5947 |                -0.298078 |              -0.0585156 |              -0.123441  |               0.00815375 |              -0.113497  |               -0.0125806 |                                   8 |                              7 | NOT_PROVEN        |          582 |              347 |                228 |                  7 |             0.603478 |              0.150266  |
| joint_interactions    |           6521 |             10 | 13.1014 | 16.613  |                -0.251144 |              -0.0259026 |              -0.100622  |               0.0470019  |              -0.0862337 |                0.0274529 |                                   5 |                              7 | NOT_PROVEN        |          607 |              333 |                266 |                  8 |             0.555927 |              0.0605062 |

## Incremental comparisons

| challenger            | reference             |   paired_games |   mae_delta_challenger_minus_reference |   game_bootstrap_ci_low |   game_bootstrap_ci_high |   season_cluster_ci_low |   season_cluster_ci_high |   test_seasons_improved |   test_seasons |   seasons_required_for_support | incremental_evidence_status   |
|:----------------------|:----------------------|---------------:|---------------------------------------:|------------------------:|-------------------------:|------------------------:|-------------------------:|------------------------:|---------------:|-------------------------------:|:------------------------------|
| orientation_crosswind | baseline              |           6521 |                             -0.0281751 |              -0.0840289 |                0.0273198 |             -0.076568   |                0.027213  |                       7 |             10 |                              7 | NOT_PROVEN                    |
| orientation_vector    | orientation_crosswind |           6521 |                              0.0112153 |              -0.0430352 |                0.0655054 |             -0.0379294  |                0.0503755 |                       5 |             10 |                              7 | NOT_PROVEN                    |
| climate_full          | baseline              |           6521 |                             -0.0169787 |              -0.0825475 |                0.0471883 |             -0.0932977  |                0.0307312 |                       6 |             10 |                              7 | NOT_PROVEN                    |
| joint_core            | orientation_vector    |           6521 |                             -0.0415558 |              -0.107038  |                0.0228916 |             -0.0744037  |               -0.0139472 |                       8 |             10 |                              7 | NOT_PROVEN                    |
| joint_core            | climate_full          |           6521 |                             -0.0415369 |              -0.0986553 |                0.0150406 |             -0.0833863  |                0.0172211 |                       9 |             10 |                              7 | NOT_PROVEN                    |
| joint_interactions    | joint_core            |           6521 |                              0.0326129 |              -0.0319812 |                0.0971197 |             -0.00150216 |                0.0637932 |                       3 |             10 |                              7 | NOT_PROVEN                    |
| joint_interactions    | baseline              |           6521 |                             -0.0259026 |              -0.100304  |                0.0472199 |             -0.0859626  |                0.0254172 |                       5 |             10 |                              7 | NOT_PROVEN                    |

## Results by test season

|   test_season | model                 |   paired_games |     mae |   mae_delta_vs_baseline |    rmse |   signed_projection_bias |
|--------------:|:----------------------|---------------:|--------:|------------------------:|--------:|-------------------------:|
|          2016 | baseline              |            648 | 13.6698 |              0          | 17.6423 |               -0.590378  |
|          2016 | orientation_crosswind |            648 | 13.4648 |             -0.205053   | 17.4999 |               -0.765552  |
|          2016 | orientation_vector    |            648 | 13.5907 |             -0.0791362  | 17.5937 |               -0.664983  |
|          2016 | climate_full          |            648 | 13.6455 |             -0.024353   | 17.6693 |               -0.571032  |
|          2016 | joint_core            |            648 | 13.6171 |             -0.0527261  | 17.6795 |               -0.703494  |
|          2016 | joint_interactions    |            648 | 13.6894 |              0.0196109  | 17.6506 |               -0.455499  |
|          2017 | baseline              |            679 | 14.0606 |              0          | 17.6704 |                0.621949  |
|          2017 | orientation_crosswind |            679 | 14.087  |              0.0263313  | 17.7382 |                0.758902  |
|          2017 | orientation_vector    |            679 | 14.1871 |              0.126479   | 17.8117 |                0.514486  |
|          2017 | climate_full          |            679 | 14.1383 |              0.0776649  | 17.7527 |                1.32992   |
|          2017 | joint_core            |            679 | 14.0918 |              0.0311181  | 17.7341 |                0.860141  |
|          2017 | joint_interactions    |            679 | 14.1584 |              0.0977238  | 17.7574 |                0.871345  |
|          2018 | baseline              |            656 | 13.5037 |              0          | 17.2376 |               -0.506767  |
|          2018 | orientation_crosswind |            656 | 13.4121 |             -0.091565   | 17.19   |               -0.784068  |
|          2018 | orientation_vector    |            656 | 13.3598 |             -0.143838   | 17.1471 |               -0.21221   |
|          2018 | climate_full          |            656 | 13.4495 |             -0.0541238  | 17.1641 |               -0.599024  |
|          2018 | joint_core            |            656 | 13.2597 |             -0.243977   | 17.0718 |               -0.478751  |
|          2018 | joint_interactions    |            656 | 13.3436 |             -0.16005    | 17.2197 |               -0.609116  |
|          2019 | baseline              |            668 | 13.1563 |              0          | 16.5705 |                0.481441  |
|          2019 | orientation_crosswind |            668 | 13.145  |             -0.0112695  | 16.5199 |                0.683695  |
|          2019 | orientation_vector    |            668 | 13.0964 |             -0.0598903  | 16.5592 |                0.374005  |
|          2019 | climate_full          |            668 | 13.0819 |             -0.0743948  | 16.5213 |                0.541995  |
|          2019 | joint_core            |            668 | 13.0629 |             -0.093403   | 16.5137 |               -0.0362482 |
|          2019 | joint_interactions    |            668 | 13.0636 |             -0.092706   | 16.432  |               -0.149091  |
|          2020 | baseline              |            398 | 14.3639 |              0          | 18.0218 |               -0.610224  |
|          2020 | orientation_crosswind |            398 | 14.4806 |              0.116702   | 18.1408 |               -0.459978  |
|          2020 | orientation_vector    |            398 | 14.3522 |             -0.0117571  | 17.9741 |               -0.187764  |
|          2020 | climate_full          |            398 | 14.1113 |             -0.252606   | 17.8632 |               -0.698019  |
|          2020 | joint_core            |            398 | 14.263  |             -0.100977   | 18.0428 |               -0.250381  |
|          2020 | joint_interactions    |            398 | 14.238  |             -0.125911   | 17.903  |               -0.313048  |
|          2021 | baseline              |            669 | 12.6486 |              0          | 16.1113 |               -1.31216   |
|          2021 | orientation_crosswind |            669 | 12.7107 |              0.0621233  | 16.0861 |               -0.989664  |
|          2021 | orientation_vector    |            669 | 12.6864 |              0.0378237  | 16.0697 |               -1.32976   |
|          2021 | climate_full          |            669 | 12.6548 |              0.00615875 | 16.0743 |               -0.620148  |
|          2021 | joint_core            |            669 | 12.57   |             -0.0785934  | 15.9864 |               -0.821677  |
|          2021 | joint_interactions    |            669 | 12.6924 |              0.0437862  | 16.0748 |               -0.294046  |
|          2022 | baseline              |            700 | 12.161  |              0          | 15.5351 |                0.247084  |
|          2022 | orientation_crosswind |            700 | 12.1489 |             -0.0120813  | 15.467  |                0.0451742 |
|          2022 | orientation_vector    |            700 | 12.2019 |              0.0408899  | 15.5486 |                0.0661144 |
|          2022 | climate_full          |            700 | 12.2906 |              0.129587   | 15.6009 |                0.0796805 |
|          2022 | joint_core            |            700 | 12.221  |              0.0600501  | 15.5089 |                0.310278  |
|          2022 | joint_interactions    |            700 | 12.2486 |              0.0876686  | 15.5638 |                0.362336  |
|          2023 | baseline              |            708 | 13.0901 |              0          | 16.3298 |               -1.10954   |
|          2023 | orientation_crosswind |            708 | 13.0272 |             -0.0628719  | 16.1971 |               -1.01181   |
|          2023 | orientation_vector    |            708 | 13.0192 |             -0.0708282  | 16.1857 |               -1.08903   |
|          2023 | climate_full          |            708 | 12.9982 |             -0.0918271  | 16.2002 |               -1.10988   |
|          2023 | joint_core            |            708 | 12.9965 |             -0.0935755  | 16.1782 |               -1.04861   |
|          2023 | joint_interactions    |            708 | 12.9462 |             -0.143813   | 16.1979 |               -1.2096    |
|          2024 | baseline              |            691 | 12.8281 |              0          | 16.3765 |               -0.733278  |
|          2024 | orientation_crosswind |            691 | 12.7929 |             -0.0351668  | 16.3653 |               -0.639741  |
|          2024 | orientation_vector    |            691 | 12.8239 |             -0.00417724 | 16.4326 |               -0.815321  |
|          2024 | climate_full          |            691 | 12.8491 |              0.0209581  | 16.3854 |               -0.762194  |
|          2024 | joint_core            |            691 | 12.8042 |             -0.0238952  | 16.4005 |               -0.959072  |
|          2024 | joint_interactions    |            691 | 12.8373 |              0.00924451 | 16.4526 |               -0.915953  |
|          2025 | baseline              |            704 | 12.3969 |              0          | 15.4122 |                0.104045  |
|          2025 | orientation_crosswind |            704 | 12.3801 |             -0.016871   | 15.3697 |                0.105298  |
|          2025 | orientation_vector    |            704 | 12.3869 |             -0.0100064  | 15.3805 |                0.102873  |
|          2025 | climate_full          |            704 | 12.3881 |             -0.00885384 | 15.3892 |                0.143421  |
|          2025 | joint_core            |            704 | 12.379  |             -0.0179052  | 15.3674 |                0.147205  |
|          2025 | joint_interactions    |            704 | 12.3598 |             -0.0371727  | 15.3586 |                0.173266  |

## Regime stability for the full joint challenger

| challenger         | grouping                  | value   |   games |   seasons |   baseline_mae |   challenger_mae |   mae_delta_vs_baseline |   bootstrap_ci_low |   bootstrap_ci_high |
|:-------------------|:--------------------------|:--------|--------:|----------:|---------------:|-----------------:|------------------------:|-------------------:|--------------------:|
| joint_interactions | latitude_band             | <30N    |     566 |        10 |        13.3681 |          13.3618 |             -0.00624275 |         -0.289368  |          0.264311   |
| joint_interactions | latitude_band             | 30-35N  |    1723 |        10 |        13.3698 |          13.2319 |             -0.137825   |         -0.273553  |          0.00168542 |
| joint_interactions | latitude_band             | 35-40N  |    2172 |        10 |        13.155  |          13.1495 |             -0.00553446 |         -0.130789  |          0.124826   |
| joint_interactions | latitude_band             | 40N+    |    2060 |        10 |        12.829  |          12.8698 |              0.0408324  |         -0.0871007 |          0.170712   |
| joint_interactions | crosswind_bin             | 0-5     |    3893 |        10 |        13.1776 |          13.1352 |             -0.0423473  |         -0.131411  |          0.0458355  |
| joint_interactions | crosswind_bin             | 5-10    |    1860 |        10 |        13.1288 |          13.0685 |             -0.0602541  |         -0.199179  |          0.0804711  |
| joint_interactions | crosswind_bin             | 10-15   |     580 |        10 |        12.6836 |          12.8505 |              0.166925   |         -0.111298  |          0.440481   |
| joint_interactions | crosswind_bin             | 15-20   |     133 |        10 |        13.529  |          13.6406 |              0.111661   |         -0.515959  |          0.728738   |
| joint_interactions | crosswind_bin             | 20+     |      55 |        10 |        13.2214 |          13.1551 |             -0.0663252  |         -1.07095   |          0.91148    |
| joint_interactions | temperature_anomaly_bin   | <=-10F  |     833 |        10 |        12.9126 |          12.9469 |              0.0343548  |         -0.171087  |          0.235328   |
| joint_interactions | temperature_anomaly_bin   | -10-0F  |    2221 |        10 |        13.0985 |          12.9289 |             -0.169581   |         -0.296277  |         -0.0372517  |
| joint_interactions | temperature_anomaly_bin   | 0-10F   |    2401 |        10 |        12.9996 |          13.037  |              0.0373766  |         -0.0810653 |          0.15654    |
| joint_interactions | temperature_anomaly_bin   | 10F+    |    1066 |        10 |        13.6426 |          13.7264 |              0.083836   |         -0.100707  |          0.277832   |
| joint_interactions | local_wind_percentile_bin | <=25th  |    1808 |        10 |        13.0944 |          13.0067 |             -0.0877491  |         -0.22263   |          0.0437852  |
| joint_interactions | local_wind_percentile_bin | 25-50th |    1665 |        10 |        13.2234 |          13.2454 |              0.0219558  |         -0.117884  |          0.160302   |
| joint_interactions | local_wind_percentile_bin | 50-75th |    1581 |        10 |        13.0322 |          12.9809 |             -0.0513124  |         -0.20213   |          0.0911677  |
| joint_interactions | local_wind_percentile_bin | 75th+   |    1467 |        10 |        13.161  |          13.1844 |              0.0233862  |         -0.147249  |          0.195382   |
| joint_interactions | closing_total_bin         | <48     |    1280 |        10 |        11.9748 |          11.9236 |             -0.0512111  |         -0.203517  |          0.10415    |
| joint_interactions | closing_total_bin         | 48-56   |    2477 |        10 |        12.9871 |          12.9586 |             -0.0285105  |         -0.142005  |          0.0877665  |
| joint_interactions | closing_total_bin         | 56-64   |    1868 |        10 |        13.2923 |          13.1858 |             -0.106466   |         -0.25045   |          0.0374471  |
| joint_interactions | closing_total_bin         | 64+     |     896 |        10 |        14.8171 |          15.0025 |              0.185421   |         -0.0378543 |          0.407475   |

## Interpretation guardrails

- HistGradientBoosting can learn nonlinear interactions without manually enumerating every combination. The explicit interaction list is intentionally small to reduce overfit risk.
- Latitude is treated as context, not a causal mechanism.
- `temperature_anomaly_f` and `wind_local_percentile` are historical football-game weather context, not official NOAA climate normals.
- Field orientation is an undirected 0-180 degree axis; crosswind and along-field wind are magnitudes.
- A narrow betting-strategy result is not enough. Prediction error, uncertainty, year-to-year stability, and regime stability are primary.
- No retrospective result may alter the frozen 2026 live protocol or existing prospective records.