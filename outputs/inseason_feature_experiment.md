# In-Season Team Feature Experiment

Purpose: compare the current production HGB prior-season team controls with leakage-safe current-season advanced team features.

Every current-season snapshot uses only weeks completed before the game being predicted.
The production decision screen is UNDER residual <= -4.0 with market total >= 56.

## Overall comparison

| variant        | test_season   | screen                |   games |   graded |   wins |   losses |   pushes |   hit_rate |   net_units_1u_each |   roi_per_1u |   avg_pred_edge |   avg_actual_residual |     mae |   test_games |   inseason_ready_games |
|:---------------|:--------------|:----------------------|--------:|---------:|-------:|---------:|---------:|-----------:|--------------------:|-------------:|----------------:|----------------------:|--------:|-------------:|-----------------------:|
| baseline_prior | ALL           | under_edge_4_total_56 |     651 |      643 |    376 |      267 |        8 |   0.584759 |             74.8182 |    0.116358  |         6.36769 |              -1.55453 | 13.0446 |        10312 |                   7197 |
| inseason_only  | ALL           | under_edge_4_total_56 |     522 |      517 |    286 |      231 |        5 |   0.553191 |             29      |    0.0560928 |         6.25539 |              -1.17912 | 13.078  |        10312 |                   7197 |
| hybrid         | ALL           | under_edge_4_total_56 |     570 |      561 |    309 |      252 |        9 |   0.550802 |             28.9091 |    0.0515314 |         6.29267 |              -1.18158 | 13.0391 |        10312 |                   7197 |

## By-season production screen

| variant        |   test_season | screen                |   games |   graded |   wins |   losses |   pushes |   hit_rate |   net_units_1u_each |   roi_per_1u |   avg_pred_edge |   avg_actual_residual |
|:---------------|--------------:|:----------------------|--------:|---------:|-------:|---------:|---------:|-----------:|--------------------:|-------------:|----------------:|----------------------:|
| baseline_prior |          2016 | under_edge_4_total_56 |      90 |       88 |     49 |       39 |        2 |   0.556818 |           5.54545   |   0.0630165  |         6.43471 |            -0.955556  |
| baseline_prior |          2017 | under_edge_4_total_56 |      76 |       76 |     42 |       34 |        0 |   0.552632 |           4.18182   |   0.0550239  |         6.65491 |            -1.01316   |
| baseline_prior |          2018 | under_edge_4_total_56 |     101 |       99 |     54 |       45 |        2 |   0.545455 |           4.09091   |   0.0413223  |         7.27014 |             0.633663  |
| baseline_prior |          2019 | under_edge_4_total_56 |      54 |       53 |     30 |       23 |        1 |   0.566038 |           4.27273   |   0.0806175  |         6.00455 |            -1.36111   |
| baseline_prior |          2020 | under_edge_4_total_56 |      52 |       51 |     27 |       24 |        1 |   0.529412 |           0.545455  |   0.0106952  |         6.84935 |             1.32692   |
| baseline_prior |          2021 | under_edge_4_total_56 |     104 |      104 |     68 |       36 |        0 |   0.653846 |          25.8182    |   0.248252   |         6.33432 |            -3.8125    |
| baseline_prior |          2022 | under_edge_4_total_56 |      71 |       71 |     42 |       29 |        0 |   0.591549 |           9.18182   |   0.129321   |         5.89239 |            -3.8662    |
| baseline_prior |          2023 | under_edge_4_total_56 |      36 |       35 |     23 |       12 |        1 |   0.657143 |           8.90909   |   0.254545   |         5.62625 |            -1.25      |
| baseline_prior |          2024 | under_edge_4_total_56 |      38 |       37 |     25 |       12 |        1 |   0.675676 |          10.7273    |   0.289926   |         5.81871 |            -3.55263   |
| baseline_prior |          2025 | under_edge_4_total_56 |      29 |       29 |     16 |       13 |        0 |   0.551724 |           1.54545   |   0.0532915  |         4.99958 |            -1.98276   |
| inseason_only  |          2016 | under_edge_4_total_56 |      84 |       84 |     42 |       42 |        0 |   0.5      |          -3.81818   |  -0.0454545  |         6.63582 |             0.321429  |
| inseason_only  |          2017 | under_edge_4_total_56 |      68 |       68 |     37 |       31 |        0 |   0.544118 |           2.63636   |   0.0387701  |         6.5593  |            -0.507353  |
| inseason_only  |          2018 | under_edge_4_total_56 |      83 |       82 |     48 |       34 |        1 |   0.585366 |           9.63636   |   0.117517   |         6.44556 |            -2.15663   |
| inseason_only  |          2019 | under_edge_4_total_56 |      48 |       47 |     32 |       15 |        1 |   0.680851 |          14.0909    |   0.299807   |         6.3148  |            -6.1875    |
| inseason_only  |          2020 | under_edge_4_total_56 |      42 |       41 |     20 |       21 |        1 |   0.487805 |          -2.81818   |  -0.0687361  |         6.76092 |            -1.75      |
| inseason_only  |          2021 | under_edge_4_total_56 |      58 |       56 |     30 |       26 |        2 |   0.535714 |           1.27273   |   0.0227273  |         6.06034 |             0.448276  |
| inseason_only  |          2022 | under_edge_4_total_56 |      66 |       66 |     36 |       30 |        0 |   0.545455 |           2.72727   |   0.0413223  |         5.82657 |             0.143939  |
| inseason_only  |          2023 | under_edge_4_total_56 |      29 |       29 |     14 |       15 |        0 |   0.482759 |          -2.27273   |  -0.0783699  |         5.30511 |            -0.137931  |
| inseason_only  |          2024 | under_edge_4_total_56 |      25 |       25 |     17 |        8 |        0 |   0.68     |           7.45455   |   0.298182   |         5.84884 |            -3.06      |
| inseason_only  |          2025 | under_edge_4_total_56 |      19 |       19 |     10 |        9 |        0 |   0.526316 |           0.0909091 |   0.00478469 |         5.45785 |            -0.710526  |
| hybrid         |          2016 | under_edge_4_total_56 |      78 |       78 |     38 |       40 |        0 |   0.487179 |          -5.45455   |  -0.0699301  |         6.59842 |            -1.73718   |
| hybrid         |          2017 | under_edge_4_total_56 |      56 |       56 |     33 |       23 |        0 |   0.589286 |           7         |   0.125      |         6.69692 |            -1.86607   |
| hybrid         |          2018 | under_edge_4_total_56 |      85 |       84 |     45 |       39 |        1 |   0.535714 |           1.90909   |   0.0227273  |         7.04797 |             0.211765  |
| hybrid         |          2019 | under_edge_4_total_56 |      51 |       49 |     25 |       24 |        2 |   0.510204 |          -1.27273   |  -0.025974   |         6.07308 |            -0.0196078 |
| hybrid         |          2020 | under_edge_4_total_56 |      51 |       50 |     28 |       22 |        1 |   0.56     |           3.45455   |   0.0690909  |         6.39365 |            -2.30392   |
| hybrid         |          2021 | under_edge_4_total_56 |      87 |       84 |     50 |       34 |        3 |   0.595238 |          11.4545    |   0.136364   |         6.4999  |            -2.60345   |
| hybrid         |          2022 | under_edge_4_total_56 |      55 |       55 |     28 |       27 |        0 |   0.509091 |          -1.54545   |  -0.0280992  |         5.72127 |            -0.772727  |
| hybrid         |          2023 | under_edge_4_total_56 |      43 |       42 |     25 |       17 |        1 |   0.595238 |           5.72727   |   0.136364   |         5.53958 |             1.94186   |
| hybrid         |          2024 | under_edge_4_total_56 |      34 |       33 |     24 |        9 |        1 |   0.727273 |          12.8182    |   0.38843    |         5.34118 |            -4.33824   |
| hybrid         |          2025 | under_edge_4_total_56 |      30 |       30 |     13 |       17 |        0 |   0.433333 |          -5.18182   |  -0.172727   |         5.4092  |             0         |

## Walk-forward diagnostics

| variant        |   test_season |   train_games |   test_games |   numeric_features |   categorical_features |   prior_feature_count |   inseason_feature_count |     mae |
|:---------------|--------------:|--------------:|-------------:|-------------------:|-----------------------:|----------------------:|-------------------------:|--------:|
| baseline_prior |          2016 |          1445 |          719 |                136 |                     10 |                   126 |                        0 | 13.6661 |
| baseline_prior |          2017 |          2164 |          745 |                136 |                     10 |                   126 |                        0 | 14.0581 |
| baseline_prior |          2018 |          2909 |          812 |                136 |                     10 |                   126 |                        0 | 13.2007 |
| baseline_prior |          2019 |          3721 |          841 |                136 |                     10 |                   126 |                        0 | 13.2465 |
| baseline_prior |          2020 |          4562 |          541 |                136 |                     10 |                   126 |                        0 | 14.2443 |
| baseline_prior |          2021 |          5103 |          849 |                136 |                     10 |                   126 |                        0 | 12.6281 |
| baseline_prior |          2022 |          5952 |         1413 |                136 |                     10 |                   126 |                        0 | 12.6133 |
| baseline_prior |          2023 |          7365 |         1345 |                136 |                     10 |                   126 |                        0 | 12.8767 |
| baseline_prior |          2024 |          8710 |         1503 |                136 |                     10 |                   126 |                        0 | 13.0423 |
| baseline_prior |          2025 |         10213 |         1544 |                136 |                     10 |                   126 |                        0 | 12.4257 |
| inseason_only  |          2016 |          1445 |          719 |                 80 |                     10 |                     0 |                       70 | 13.9645 |
| inseason_only  |          2017 |          2164 |          745 |                 80 |                     10 |                     0 |                       70 | 14.4135 |
| inseason_only  |          2018 |          2909 |          812 |                 80 |                     10 |                     0 |                       70 | 13.2126 |
| inseason_only  |          2019 |          3721 |          841 |                 80 |                     10 |                     0 |                       70 | 13.5476 |
| inseason_only  |          2020 |          4562 |          541 |                 80 |                     10 |                     0 |                       70 | 14.1071 |
| inseason_only  |          2021 |          5103 |          849 |                 80 |                     10 |                     0 |                       70 | 12.6879 |
| inseason_only  |          2022 |          5952 |         1413 |                 80 |                     10 |                     0 |                       70 | 12.6471 |
| inseason_only  |          2023 |          7365 |         1345 |                 80 |                     10 |                     0 |                       70 | 12.7352 |
| inseason_only  |          2024 |          8710 |         1503 |                 80 |                     10 |                     0 |                       70 | 13.0299 |
| inseason_only  |          2025 |         10213 |         1544 |                 80 |                     10 |                     0 |                       70 | 12.2879 |
| hybrid         |          2016 |          1445 |          719 |                206 |                     10 |                   126 |                       70 | 13.815  |
| hybrid         |          2017 |          2164 |          745 |                206 |                     10 |                   126 |                       70 | 14.0751 |
| hybrid         |          2018 |          2909 |          812 |                206 |                     10 |                   126 |                       70 | 13.1868 |
| hybrid         |          2019 |          3721 |          841 |                206 |                     10 |                   126 |                       70 | 13.4089 |
| hybrid         |          2020 |          4562 |          541 |                206 |                     10 |                   126 |                       70 | 14.0216 |
| hybrid         |          2021 |          5103 |          849 |                206 |                     10 |                   126 |                       70 | 12.6368 |
| hybrid         |          2022 |          5952 |         1413 |                206 |                     10 |                   126 |                       70 | 12.5438 |
| hybrid         |          2023 |          7365 |         1345 |                206 |                     10 |                   126 |                       70 | 12.8917 |
| hybrid         |          2024 |          8710 |         1503 |                206 |                     10 |                   126 |                       70 | 13.0015 |
| hybrid         |          2025 |         10213 |         1544 |                206 |                     10 |                   126 |                       70 | 12.3943 |

Promotion guardrail: do not replace the operational feature set based on intuition alone. Prefer a challenger that improves the exact production-screen out-of-sample hit rate/ROI without a material MAE regression and without depending on one isolated test season.