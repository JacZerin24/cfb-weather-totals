# Edge Refinement Summary

This module starts with the HGB under signal and tests whether additional filters improve the profile without relying only on raw ROI.

## Ranking logic

Rows are ranked by a stability-first score that considers overall ROI, recent ROI, positive-season rate, sample size, and drawdown. A high ROI with a tiny sample is flagged rather than treated as a production rule.

## Top shortlist candidates

| filter_name                              | category            |   graded |   hit_rate |   roi_per_1u |   recent_2022_plus_roi |   positive_season_rate |   max_drawdown_units | overfit_flags      |   score |
|:-----------------------------------------|:--------------------|---------:|-----------:|-------------:|-----------------------:|-----------------------:|---------------------:|:-------------------|--------:|
| hgb_under_3p5_wind_ge_12                 | weather             |      261 |   0.590038 |    0.126437  |              0.196364  |               0.888889 |            -10.7273  | none               | 44.7606 |
| hgb_under_3p5_pass_rate_high_total       | weather_team_style  |      231 |   0.593074 |    0.132231  |              0.193182  |               0.777778 |             -7.36364 | none               | 43.4916 |
| hgb_under_3p5_total_60_63                | high_total_bucket   |      143 |   0.608392 |    0.161475  |              0.0909091 |               0.888889 |             -7.36364 | none               | 38.1408 |
| hgb_under_3p5_high_total_cold50          | weather_total_combo |      129 |   0.534884 |    0.0211416 |              0.301653  |               0.666667 |            -10.5455  | thin_recent_sample | 36.7173 |
| hgb_under_3p5_total_ge_56                | high_total_bucket   |      802 |   0.55611  |    0.061664  |              0.117758  |               0.888889 |            -17.1818  | none               | 35.484  |
| hgb_under_3p5_total_ge_58                | high_total_bucket   |      697 |   0.558106 |    0.0654754 |              0.123636  |               0.777778 |            -18.1818  | none               | 34.4688 |
| hgb_under_3p5_total_ge_60                | high_total_bucket   |      595 |   0.563025 |    0.0748663 |              0.101399  |               0.777778 |            -16.8182  | none               | 33.9015 |
| hgb_under_5p0_baseline                   | baseline            |      752 |   0.550532 |    0.0510155 |              0.134774  |               0.666667 |            -16.8182  | none               | 32.5199 |
| hgb_under_3p5_no_model_strong_opposition | model_consensus     |      928 |   0.543103 |    0.0368339 |              0.117871  |               0.777778 |            -22.2727  | none               | 30.3252 |
| hgb_under_3p5_total_56_60                | high_total_bucket   |      207 |   0.536232 |    0.0237154 |              0.186732  |               0.666667 |            -12.5455  | none               | 28.941  |
| hgb_under_3p5_all_models_under           | model_consensus     |      577 |   0.538995 |    0.0289901 |              0.11094   |               0.666667 |            -17.3636  | none               | 28.3015 |
| hgb_under_3p5_provider_draftkings        | provider            |      127 |   0.559055 |    0.067287  |              0.067287  |               1        |            -11.5455  | none               | 27.3426 |
| hgb_under_3p5_provider_count_ge_3        | line_market_proxy   |     1048 |   0.54771  |    0.045628  |              0.044289  |               0.875    |            -24.9091  | none               | 26.2491 |
| hgb_under_3p5_total_ge_52                | high_total_bucket   |      976 |   0.538934 |    0.0288748 |              0.0782828 |               0.666667 |            -20.1818  | none               | 25.1137 |
| hgb_under_3p5_two_plus_models_under      | model_consensus     |      810 |   0.533333 |    0.0181818 |              0.0818182 |               0.666667 |            -24.4545  | none               | 23.4727 |
| hgb_under_3p5_baseline                   | baseline            |     1318 |   0.53566  |    0.0226238 |              0.0475207 |               0.777778 |            -23.0909  | none               | 23.1125 |
| hgb_under_3p5_provider_consensus         | provider            |      966 |   0.541408 |    0.0335968 |              0.0132867 |               0.833333 |            -22       | none               | 22.5226 |
| hgb_under_3p5_outdoor_only               | weather             |     1267 |   0.531965 |    0.0155701 |              0.0340909 |               0.777778 |            -27       | none               | 20.5509 |

## Top rows by ROI before guardrails

| filter_name                                          | category            |   graded |   hit_rate |   roi_per_1u |   recent_2022_plus_roi |   positive_season_rate |   max_drawdown_units | overfit_flags                                        |
|:-----------------------------------------------------|:--------------------|---------:|-----------:|-------------:|-----------------------:|-----------------------:|---------------------:|:-----------------------------------------------------|
| hgb_under_3p5_provider_caesars_sportsbook_(colorado) | provider            |        2 |   1        |    0.909091  |             0.909091   |               1        |              0       | small_sample,thin_recent_sample,high_roi_thin_sample |
| hgb_under_3p5_selected_total_above_median            | line_market_proxy   |       50 |   0.62     |    0.183636  |             0.248252   |               0.666667 |             -4.18182 | small_sample,thin_recent_sample                      |
| hgb_under_3p5_total_60_63                            | high_total_bucket   |      143 |   0.608392 |    0.161475  |             0.0909091  |               0.888889 |             -7.36364 | none                                                 |
| hgb_under_3p5_provider_william_hill_(new_jersey)     | provider            |       15 |   0.6      |    0.145455  |             0.174825   |               0.666667 |             -3.18182 | small_sample,thin_recent_sample                      |
| hgb_under_3p5_pass_rate_high_total                   | weather_team_style  |      231 |   0.593074 |    0.132231  |             0.193182   |               0.777778 |             -7.36364 | none                                                 |
| hgb_under_3p5_wind_ge_12                             | weather             |      261 |   0.590038 |    0.126437  |             0.196364   |               0.888889 |            -10.7273  | none                                                 |
| hgb_under_3p5_high_total_precip                      | weather_total_combo |       80 |   0.575    |    0.0977273 |            -0.0454545  |               0.666667 |             -4.27273 | small_sample,thin_recent_sample,negative_recent_roi  |
| hgb_under_3p5_high_pass_rate_top30                   | team_style          |      339 |   0.569322 |    0.0868866 |             0.186732   |               0.555556 |            -11.4545  | weak_season_stability                                |
| hgb_under_3p5_wind_ge_10                             | weather             |      451 |   0.567627 |    0.0836525 |             0.119122   |               0.555556 |            -10.5455  | weak_season_stability                                |
| hgb_under_3p5_total_ge_60                            | high_total_bucket   |      595 |   0.563025 |    0.0748663 |             0.101399   |               0.777778 |            -16.8182  | none                                                 |
| hgb_under_3p5_provider_draftkings                    | provider            |      127 |   0.559055 |    0.067287  |             0.067287   |               1        |            -11.5455  | none                                                 |
| hgb_under_3p5_total_ge_58                            | high_total_bucket   |      697 |   0.558106 |    0.0654754 |             0.123636   |               0.777778 |            -18.1818  | none                                                 |
| hgb_under_3p5_total_ge_56                            | high_total_bucket   |      802 |   0.55611  |    0.061664  |             0.117758   |               0.888889 |            -17.1818  | none                                                 |
| hgb_under_3p5_total_ge_66                            | high_total_bucket   |      321 |   0.551402 |    0.0526763 |             0.196364   |               0.555556 |            -15.1818  | weak_season_stability                                |
| hgb_under_3p5_total_66_plus                          | high_total_bucket   |      321 |   0.551402 |    0.0526763 |             0.196364   |               0.555556 |            -15.1818  | weak_season_stability                                |
| hgb_under_5p0_baseline                               | baseline            |      752 |   0.550532 |    0.0510155 |             0.134774   |               0.666667 |            -16.8182  | none                                                 |
| hgb_under_3p5_total_ge_63                            | high_total_bucket   |      452 |   0.548673 |    0.0474658 |             0.105263   |               0.555556 |            -17.2727  | weak_season_stability                                |
| hgb_under_3p5_provider_count_ge_3                    | line_market_proxy   |     1048 |   0.54771  |    0.045628  |             0.044289   |               0.875    |            -24.9091  | none                                                 |
| hgb_under_3p5_provider_bovada                        | provider            |       22 |   0.545455 |    0.0413223 |             0.00478469 |               0.666667 |             -4       | small_sample,thin_recent_sample                      |
| hgb_under_3p5_no_model_strong_opposition             | model_consensus     |      928 |   0.543103 |    0.0368339 |             0.117871   |               0.777778 |            -22.2727  | none                                                 |
| hgb_under_3p5_total_63_66                            | high_total_bucket   |      131 |   0.541985 |    0.0346981 |            -0.0699301  |               0.777778 |            -12.1818  | negative_recent_roi                                  |
| hgb_under_3p5_provider_consensus                     | provider            |      966 |   0.541408 |    0.0335968 |             0.0132867  |               0.833333 |            -22       | none                                                 |
| hgb_under_3p5_high_total_wind_or_precip              | weather_total_combo |      137 |   0.540146 |    0.0311878 |             0.0991736  |               0.555556 |            -11.9091  | weak_season_stability                                |
| hgb_under_3p5_all_models_under                       | model_consensus     |      577 |   0.538995 |    0.0289901 |             0.11094    |               0.666667 |            -17.3636  | none                                                 |
| hgb_under_3p5_total_ge_52                            | high_total_bucket   |      976 |   0.538934 |    0.0288748 |             0.0782828  |               0.666667 |            -20.1818  | none                                                 |

## CLV / line movement note

The historical dataset does not yet contain a true timestamped current line and final closing line for each betting decision. This module adds provider-dispersion proxies where available, but true CLV must be tracked live: decision-line total, closing total, and whether the strategy beat the close.

## Production guardrails

- Do not use a filter as a production rule only because it clears 10% historical ROI.
- Prefer rows with at least 100 graded plays, positive recent ROI, and positive seasons across most years.
- Treat tiny high-ROI filters as research leads, not plays.
- Continue using live paper tracking before staking real money.