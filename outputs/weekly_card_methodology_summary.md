# Weekly Card Backtest Methodology Summary

This test is a more realistic bridge between all-combination research and a weekly live card.

Instead of grading every possible same-week combination, it selects only the top-ranked combinations per week by model edge. When multiple weekly combinations are tested, the non-overlap setting prevents the same game from being reused within that weekly card.

## Weekly card summary

| card_strategy                                 |   legs |   cards |   graded |   wins |   losses |   pushes |   hit_rate |   breakeven_hit_rate |   net_units |   roi_per_card |   avg_cards_per_week |   max_drawdown_units |   avg_min_abs_pred_edge |   avg_market_residual |
|:----------------------------------------------|-------:|--------:|---------:|-------:|---------:|---------:|-----------:|---------------------:|------------:|---------------:|---------------------:|---------------------:|------------------------:|----------------------:|
| hgb_under_3p5_high_total_2leg_top1_nonoverlap |      2 |     120 |      120 |     43 |       77 |        0 |   0.358333 |             0.274376 |     36.719  |       0.305992 |              1       |             -6       |                 6.78701 |             -1.55833  |
| hgb_under_3p5_high_total_2leg_top2_nonoverlap |      2 |     211 |      211 |     74 |      137 |        0 |   0.350711 |             0.274376 |     58.7025 |       0.278211 |              1.75833 |             -9.77686 |                 6.25513 |             -1.23697  |
| hgb_under_5p0_high_total_2leg_top1_nonoverlap |      2 |     103 |      103 |     36 |       67 |        0 |   0.349515 |             0.274376 |     28.2066 |       0.273851 |              1       |             -8       |                 7.22609 |             -0.847087 |
| hgb_under_3p5_high_total_3leg_top1_nonoverlap |      3 |     103 |      103 |     18 |       85 |        0 |   0.174757 |             0.143721 |     22.2427 |       0.215948 |              1       |            -15       |                 6.14716 |             -1.33172  |
| hgb_under_3p5_all_2leg_top1_nonoverlap        |      2 |     126 |      126 |     42 |       84 |        0 |   0.333333 |             0.274376 |     25.3388 |       0.201102 |              1       |            -11.843   |                 7.33563 |             -1.17659  |
| hgb_under_3p5_consensus_2leg_top1_nonoverlap  |      2 |      86 |       86 |     28 |       58 |        0 |   0.325581 |             0.274376 |     16.0496 |       0.186623 |              1       |            -11.843   |                 7.57403 |             -1.07558  |
| hgb_under_5p0_consensus_2leg_top1_nonoverlap  |      2 |      81 |       81 |     26 |       55 |        0 |   0.320988 |             0.274376 |     13.7603 |       0.169881 |              1       |            -12.843   |                 7.75307 |             -0.759259 |

## Straight-leg equivalent summary

| card_strategy                                 |   unique_straight_legs |   graded |   wins |   losses |   hit_rate |   net_units_flat_1u_each |   roi_per_leg |   avg_abs_pred_edge |   avg_market_residual |
|:----------------------------------------------|-----------------------:|---------:|-------:|---------:|-----------:|-------------------------:|--------------:|--------------------:|----------------------:|
| hgb_under_3p5_high_total_2leg_top1_nonoverlap |                    240 |      237 |    139 |       98 |   0.586498 |                  28.3636 |     0.119678  |             7.56983 |             -1.55833  |
| hgb_under_3p5_high_total_3leg_top1_nonoverlap |                    309 |      307 |    178 |      129 |   0.579805 |                  32.8182 |     0.1069    |             7.29593 |             -1.33172  |
| hgb_under_3p5_all_2leg_top1_nonoverlap        |                    252 |      249 |    143 |      106 |   0.574297 |                  24      |     0.0963855 |             8.06757 |             -1.17659  |
| hgb_under_5p0_high_total_2leg_top1_nonoverlap |                    206 |      203 |    116 |       87 |   0.571429 |                  18.4545 |     0.0909091 |             7.97085 |             -0.847087 |
| hgb_under_3p5_consensus_2leg_top1_nonoverlap  |                    172 |      171 |     97 |       74 |   0.567251 |                  14.1818 |     0.0829346 |             8.32164 |             -1.07558  |
| hgb_under_3p5_high_total_2leg_top2_nonoverlap |                    422 |      419 |    237 |      182 |   0.565632 |                  33.4545 |     0.0798438 |             6.88728 |             -1.23697  |
| hgb_under_5p0_consensus_2leg_top1_nonoverlap  |                    162 |      161 |     90 |       71 |   0.559006 |                  10.8182 |     0.0671937 |             8.52204 |             -0.759259 |

## Season-by-season summary

| card_strategy                                 |   season |   cards |   graded |   wins |   losses |   hit_rate |   net_units |   roi_per_card |   max_drawdown_units |
|:----------------------------------------------|---------:|--------:|---------:|-------:|---------:|-----------:|------------:|---------------:|---------------------:|
| hgb_under_3p5_all_2leg_top1_nonoverlap        |     2016 |      13 |       13 |      3 |       10 |  0.230769  |  -2.06612   |    -0.158932   |             -8       |
| hgb_under_3p5_all_2leg_top1_nonoverlap        |     2017 |      14 |       14 |      4 |       10 |  0.285714  |   0.578512  |     0.0413223  |             -8       |
| hgb_under_3p5_all_2leg_top1_nonoverlap        |     2018 |      14 |       14 |      6 |        8 |  0.428571  |   7.86777   |     0.561983   |             -6       |
| hgb_under_3p5_all_2leg_top1_nonoverlap        |     2019 |      15 |       15 |      3 |       12 |  0.2       |  -4.06612   |    -0.271074   |             -8.35537 |
| hgb_under_3p5_all_2leg_top1_nonoverlap        |     2020 |      14 |       14 |      4 |       10 |  0.285714  |   0.578512  |     0.0413223  |             -5       |
| hgb_under_3p5_all_2leg_top1_nonoverlap        |     2021 |      14 |       14 |      2 |       12 |  0.142857  |  -6.71074   |    -0.479339   |             -9       |
| hgb_under_3p5_all_2leg_top1_nonoverlap        |     2022 |      15 |       15 |      8 |        7 |  0.533333  |  14.157     |     0.943802   |             -4       |
| hgb_under_3p5_all_2leg_top1_nonoverlap        |     2023 |      14 |       14 |      7 |        7 |  0.5       |   9.77686   |     0.698347   |             -5       |
| hgb_under_3p5_all_2leg_top1_nonoverlap        |     2024 |      13 |       13 |      5 |        8 |  0.384615  |   5.22314   |     0.40178    |             -3       |
| hgb_under_3p5_consensus_2leg_top1_nonoverlap  |     2017 |      14 |       14 |      4 |       10 |  0.285714  |   0.578512  |     0.0413223  |             -8       |
| hgb_under_3p5_consensus_2leg_top1_nonoverlap  |     2018 |      14 |       14 |      7 |        7 |  0.5       |  11.5124    |     0.822314   |             -4       |
| hgb_under_3p5_consensus_2leg_top1_nonoverlap  |     2019 |      15 |       15 |      3 |       12 |  0.2       |  -4.06612   |    -0.271074   |             -8.35537 |
| hgb_under_3p5_consensus_2leg_top1_nonoverlap  |     2020 |      14 |       14 |      4 |       10 |  0.285714  |   0.578512  |     0.0413223  |             -5       |
| hgb_under_3p5_consensus_2leg_top1_nonoverlap  |     2021 |      14 |       14 |      2 |       12 |  0.142857  |  -6.71074   |    -0.479339   |             -9       |
| hgb_under_3p5_consensus_2leg_top1_nonoverlap  |     2022 |      14 |       14 |      8 |        6 |  0.571429  |  15.157     |     1.08264    |             -3       |
| hgb_under_3p5_consensus_2leg_top1_nonoverlap  |     2023 |       1 |        1 |      0 |        1 |  0         |  -1         |    -1          |              0       |
| hgb_under_3p5_high_total_2leg_top1_nonoverlap |     2016 |      13 |       13 |      4 |        9 |  0.307692  |   1.57851   |     0.121424   |             -5       |
| hgb_under_3p5_high_total_2leg_top1_nonoverlap |     2017 |      14 |       14 |      5 |        9 |  0.357143  |   4.22314   |     0.301653   |             -4.35537 |
| hgb_under_3p5_high_total_2leg_top1_nonoverlap |     2018 |      13 |       13 |      6 |        7 |  0.461538  |   8.86777   |     0.682136   |             -6       |
| hgb_under_3p5_high_total_2leg_top1_nonoverlap |     2019 |      15 |       15 |      5 |       10 |  0.333333  |   3.22314   |     0.214876   |             -5.35537 |
| hgb_under_3p5_high_total_2leg_top1_nonoverlap |     2020 |      12 |       12 |      3 |        9 |  0.25      |  -1.06612   |    -0.088843   |             -4       |
| hgb_under_3p5_high_total_2leg_top1_nonoverlap |     2021 |      14 |       14 |      5 |        9 |  0.357143  |   4.22314   |     0.301653   |             -6       |
| hgb_under_3p5_high_total_2leg_top1_nonoverlap |     2022 |      15 |       15 |      6 |        9 |  0.4       |   6.86777   |     0.457851   |             -4       |
| hgb_under_3p5_high_total_2leg_top1_nonoverlap |     2023 |      11 |       11 |      4 |        7 |  0.363636  |   3.57851   |     0.325319   |             -5       |
| hgb_under_3p5_high_total_2leg_top1_nonoverlap |     2024 |      13 |       13 |      5 |        8 |  0.384615  |   5.22314   |     0.40178    |             -3       |
| hgb_under_3p5_high_total_2leg_top2_nonoverlap |     2016 |      26 |       26 |      7 |       19 |  0.269231  |  -0.487603  |    -0.018754   |             -9.35537 |
| hgb_under_3p5_high_total_2leg_top2_nonoverlap |     2017 |      26 |       26 |      9 |       17 |  0.346154  |   6.80165   |     0.261602   |             -8       |
| hgb_under_3p5_high_total_2leg_top2_nonoverlap |     2018 |      26 |       26 |      8 |       18 |  0.307692  |   3.15702   |     0.121424   |             -8.35537 |
| hgb_under_3p5_high_total_2leg_top2_nonoverlap |     2019 |      23 |       23 |      8 |       15 |  0.347826  |   6.15702   |     0.267697   |             -5.35537 |
| hgb_under_3p5_high_total_2leg_top2_nonoverlap |     2020 |      19 |       19 |      5 |       14 |  0.263158  |  -0.77686   |    -0.0408873  |             -5       |
| hgb_under_3p5_high_total_2leg_top2_nonoverlap |     2021 |      28 |       28 |     13 |       15 |  0.464286  |  19.3802    |     0.692149   |             -4.71074 |
| hgb_under_3p5_high_total_2leg_top2_nonoverlap |     2022 |      28 |       28 |     10 |       18 |  0.357143  |   8.44628   |     0.301653   |             -6       |
| hgb_under_3p5_high_total_2leg_top2_nonoverlap |     2023 |      16 |       16 |      6 |       10 |  0.375     |   5.86777   |     0.366736   |             -6       |
| hgb_under_3p5_high_total_2leg_top2_nonoverlap |     2024 |      19 |       19 |      8 |       11 |  0.421053  |  10.157     |     0.53458    |             -4       |
| hgb_under_3p5_high_total_3leg_top1_nonoverlap |     2016 |      13 |       13 |      1 |       12 |  0.0769231 |  -6.04207   |    -0.464775   |             -8       |
| hgb_under_3p5_high_total_3leg_top1_nonoverlap |     2017 |      13 |       13 |      1 |       12 |  0.0769231 |  -6.04207   |    -0.464775   |             -8       |
| hgb_under_3p5_high_total_3leg_top1_nonoverlap |     2018 |      13 |       13 |      3 |       10 |  0.230769  |   7.87378   |     0.605675   |             -8       |
| hgb_under_3p5_high_total_3leg_top1_nonoverlap |     2019 |      11 |       11 |      1 |       10 |  0.0909091 |  -4.04207   |    -0.367461   |            -10       |
| hgb_under_3p5_high_total_3leg_top1_nonoverlap |     2020 |       8 |        8 |      1 |        7 |  0.125     |  -1.04207   |    -0.130259   |             -4       |
| hgb_under_3p5_high_total_3leg_top1_nonoverlap |     2021 |      14 |       14 |      3 |       11 |  0.214286  |   6.87378   |     0.490984   |             -9       |
| hgb_under_3p5_high_total_3leg_top1_nonoverlap |     2022 |      13 |       13 |      2 |       11 |  0.153846  |   0.915853  |     0.0704502  |             -4       |
| hgb_under_3p5_high_total_3leg_top1_nonoverlap |     2023 |       7 |        7 |      2 |        5 |  0.285714  |   6.91585   |     0.987979   |             -3       |
| hgb_under_3p5_high_total_3leg_top1_nonoverlap |     2024 |      11 |       11 |      4 |        7 |  0.363636  |  16.8317    |     1.53016    |             -2       |
| hgb_under_5p0_consensus_2leg_top1_nonoverlap  |     2017 |      13 |       13 |      4 |        9 |  0.307692  |   1.57851   |     0.121424   |             -7       |
| hgb_under_5p0_consensus_2leg_top1_nonoverlap  |     2018 |      14 |       14 |      7 |        7 |  0.5       |  11.5124    |     0.822314   |             -4       |
| hgb_under_5p0_consensus_2leg_top1_nonoverlap  |     2019 |      14 |       14 |      2 |       12 |  0.142857  |  -6.71074   |    -0.479339   |             -8.35537 |
| hgb_under_5p0_consensus_2leg_top1_nonoverlap  |     2020 |      14 |       14 |      4 |       10 |  0.285714  |   0.578512  |     0.0413223  |             -5       |
| hgb_under_5p0_consensus_2leg_top1_nonoverlap  |     2021 |      14 |       14 |      2 |       12 |  0.142857  |  -6.71074   |    -0.479339   |             -9       |
| hgb_under_5p0_consensus_2leg_top1_nonoverlap  |     2022 |      12 |       12 |      7 |        5 |  0.583333  |  13.5124    |     1.12603    |             -2       |
| hgb_under_5p0_high_total_2leg_top1_nonoverlap |     2016 |      11 |       11 |      4 |        7 |  0.363636  |   3.57851   |     0.325319   |             -4       |
| hgb_under_5p0_high_total_2leg_top1_nonoverlap |     2017 |      12 |       12 |      5 |        7 |  0.416667  |   6.22314   |     0.518595   |             -3       |
| hgb_under_5p0_high_total_2leg_top1_nonoverlap |     2018 |      13 |       13 |      6 |        7 |  0.461538  |   8.86777   |     0.682136   |             -6       |
| hgb_under_5p0_high_total_2leg_top1_nonoverlap |     2019 |      12 |       12 |      3 |        9 |  0.25      |  -1.06612   |    -0.088843   |             -7       |
| hgb_under_5p0_high_total_2leg_top1_nonoverlap |     2020 |      11 |       11 |      3 |        8 |  0.272727  |  -0.0661157 |    -0.00601052 |             -3       |
| hgb_under_5p0_high_total_2leg_top1_nonoverlap |     2021 |      14 |       14 |      5 |        9 |  0.357143  |   4.22314   |     0.301653   |             -6       |
| hgb_under_5p0_high_total_2leg_top1_nonoverlap |     2022 |      13 |       13 |      6 |        7 |  0.461538  |   8.86777   |     0.682136   |             -2       |
| hgb_under_5p0_high_total_2leg_top1_nonoverlap |     2023 |       7 |        7 |      1 |        6 |  0.142857  |  -3.35537   |    -0.479339   |             -5       |
| hgb_under_5p0_high_total_2leg_top1_nonoverlap |     2024 |      10 |       10 |      3 |        7 |  0.3       |   0.933884  |     0.0933884  |             -4       |

## Interpretation notes

- Prefer strategies that remain positive season-by-season and do not depend on one outlier year.
- Compare the weekly combination result against the straight-leg equivalent.
- A strong historical card still needs live tracking with current lines and forecast weather.
- This is the most relevant historical test for what an actual weekly process could look like.