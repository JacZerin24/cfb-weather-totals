# FCS Weather Totals Research

This track uses **FCS-vs-FCS games only** and predicts `actual_total_points - closing_total`.

Historical FCS games with usable totals/results: **2,345**.
Walk-forward test seasons: **2023-2025**.

## Current live candidate screen

- Dedicated FCS HistGradientBoosting model
- Predicted UNDER edge: **7.5+ points**
- Market total: **58+**
- Walk-forward record: **30-26** (53.6%)
- Paper ROI at -110: **2.3% per graded play**
- 95% Wilson interval for hit rate: **40.7% to 66.0%**

### Current screen by season

|   season |   games |   graded |   wins |   losses |   pushes |   hit_rate |   roi_per_1u |
|---------:|--------:|---------:|-------:|---------:|---------:|-----------:|-------------:|
|     2023 |      14 |       14 |      7 |        7 |        0 |        0.5 |   -0.0454545 |
|     2024 |      22 |       22 |     11 |       11 |        0 |        0.5 |   -0.0454545 |
|     2025 |      20 |       20 |     12 |        8 |        0 |        0.6 |    0.145455  |

## Leading candidate screens by season

These are shown explicitly so a rule is not selected only because it tops an aggregate ROI table.

| candidate         |   under_edge_threshold |   minimum_total |   season |   games |   graded |   wins |   losses |   pushes |   hit_rate |   roi_per_1u |
|:------------------|-----------------------:|----------------:|---------:|--------:|---------:|-------:|---------:|---------:|-----------:|-------------:|
| edge_1.5_total_60 |                    1.5 |              60 |     2023 |      31 |       30 |     21 |        9 |        1 |   0.7      |    0.336364  |
| edge_1.5_total_60 |                    1.5 |              60 |     2024 |      15 |       15 |      5 |       10 |        0 |   0.333333 |   -0.363636  |
| edge_1.5_total_60 |                    1.5 |              60 |     2025 |      35 |       35 |     19 |       16 |        0 |   0.542857 |    0.0363636 |
| edge_5.0_total_58 |                    5   |              58 |     2023 |      22 |       22 |     15 |        7 |        0 |   0.681818 |    0.301653  |
| edge_5.0_total_58 |                    5   |              58 |     2024 |      31 |       31 |     15 |       16 |        0 |   0.483871 |   -0.0762463 |
| edge_5.0_total_58 |                    5   |              58 |     2025 |      36 |       36 |     20 |       16 |        0 |   0.555556 |    0.0606061 |
| edge_7.5_total_56 |                    7.5 |              56 |     2023 |      15 |       15 |      8 |        7 |        0 |   0.533333 |    0.0181818 |
| edge_7.5_total_56 |                    7.5 |              56 |     2024 |      31 |       31 |     17 |       14 |        0 |   0.548387 |    0.0469208 |
| edge_7.5_total_56 |                    7.5 |              56 |     2025 |      27 |       27 |     16 |       11 |        0 |   0.592593 |    0.131313  |
| edge_5.0_total_60 |                    5   |              60 |     2023 |      21 |       21 |     14 |        7 |        0 |   0.666667 |    0.272727  |
| edge_5.0_total_60 |                    5   |              60 |     2024 |      10 |       10 |      3 |        7 |        0 |   0.3      |   -0.427273  |
| edge_5.0_total_60 |                    5   |              60 |     2025 |      26 |       26 |     15 |       11 |        0 |   0.576923 |    0.101399  |
| edge_7.5_total_58 |                    7.5 |              58 |     2023 |      14 |       14 |      7 |        7 |        0 |   0.5      |   -0.0454545 |
| edge_7.5_total_58 |                    7.5 |              58 |     2024 |      22 |       22 |     11 |       11 |        0 |   0.5      |   -0.0454545 |
| edge_7.5_total_58 |                    7.5 |              58 |     2025 |      20 |       20 |     12 |        8 |        0 |   0.6      |    0.145455  |

## Model diagnostics

|   test_season |   train_games |   test_games |   model_mae |   zero_residual_baseline_mae |
|--------------:|--------------:|-------------:|------------:|-----------------------------:|
|          2023 |           563 |          480 |     14.1168 |                      12.574  |
|          2024 |          1043 |          646 |     14.5712 |                      13.3204 |
|          2025 |          1689 |          656 |     13.0563 |                      12.2134 |

## Threshold sensitivity

|   under_edge_threshold |   minimum_total |   games |   graded |   wins |   losses |   pushes |   hit_rate |   roi_per_1u |
|-----------------------:|----------------:|--------:|---------:|-------:|---------:|---------:|-----------:|-------------:|
|                    1.5 |              60 |      81 |       80 |     45 |       35 |        1 |   0.5625   |   0.0738636  |
|                    5   |              58 |      89 |       89 |     50 |       39 |        0 |   0.561798 |   0.072523   |
|                    7.5 |              56 |      73 |       73 |     41 |       32 |        0 |   0.561644 |   0.0722291  |
|                    5   |              60 |      57 |       57 |     32 |       25 |        0 |   0.561404 |   0.0717703  |
|                    6   |              58 |      81 |       81 |     45 |       36 |        0 |   0.555556 |   0.0606061  |
|                    3.5 |              60 |      65 |       65 |     36 |       29 |        0 |   0.553846 |   0.0573427  |
|                    6   |              60 |      51 |       51 |     28 |       23 |        0 |   0.54902  |   0.0481283  |
|                    6   |              56 |     106 |      106 |     58 |       48 |        0 |   0.54717  |   0.0445969  |
|                    5   |              56 |     116 |      116 |     63 |       53 |        0 |   0.543103 |   0.0368339  |
|                    1.5 |              58 |     145 |      144 |     78 |       66 |        1 |   0.541667 |   0.0340909  |
|                    2.5 |              60 |      72 |       72 |     39 |       33 |        0 |   0.541667 |   0.0340909  |
|                    3.5 |              58 |     108 |      108 |     58 |       50 |        0 |   0.537037 |   0.0252525  |
|                    7.5 |              54 |      95 |       95 |     51 |       44 |        0 |   0.536842 |   0.0248804  |
|                    7.5 |              58 |      56 |       56 |     30 |       26 |        0 |   0.535714 |   0.0227273  |
|                    1.5 |              56 |     199 |      197 |    104 |       93 |        2 |   0.527919 |   0.00784495 |
|                    7.5 |              52 |     115 |      115 |     60 |       55 |        0 |   0.521739 |  -0.00395257 |
|                    5   |              54 |     161 |      161 |     84 |       77 |        0 |   0.521739 |  -0.00395257 |
|                    2.5 |              58 |     129 |      129 |     67 |       62 |        0 |   0.51938  |  -0.00845666 |
|                    3.5 |              56 |     143 |      143 |     74 |       69 |        0 |   0.517483 |  -0.0120788  |
|                    7.5 |              49 |     138 |      138 |     71 |       67 |        0 |   0.514493 |  -0.0177866  |

## Guardrails

- The threshold grid is a research screen, so the selected rule is not treated as a guaranteed edge.
- FCS market data in the current historical dataset begins in 2022, which limits the number of independent test seasons.
- Candidate screens are compared season-by-season before changing the live QUALIFIES rule.
- The FCS model MAE is worse than a zero-residual baseline in each walk-forward season; selective subset performance must therefore be treated cautiously.
- Keep the FCS track in paper-tracking mode while 2026 live forecasts, closing totals, and results accumulate.
- No FCS over strategy is promoted from this analysis.