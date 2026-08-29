# Three-Leg Parlay Research — 2026-08-29

## Disposition

**DO NOT PROMOTE THE THREE-LEG CARD TO PRODUCTION.**

The dedicated production-aligned historical study finds a positive aggregate result at idealized -110-per-leg pricing, but the result does not meet the evidentiary standard required for operational use. The primary three-leg result has wide season-block uncertainty crossing zero, strong era instability, no monotonic improvement when the edge threshold is raised, and no evidence that selecting the three strongest model edges adds value. Most importantly, on the exact same 100 eligible weeks, adding the third leg materially reduced return relative to stopping at the strongest two.

This report is research-only. No live card or production rule was changed.

## Provenance

- GitHub Actions research run: `33230470973`
- Branch: `three-leg-parlay-research`
- Head SHA: `21d86d7d83eb154687b65b94697b445ed3670bd0`
- Source training artifact: live-training artifact from workflow run `33229390276`
- Research output artifact: `three-leg-parlay-research-output`, artifact ID `9708335375`
- Output artifact SHA-256: `fea7aaeb023927a2e79305274330593ccbfa773b2ecabeacc4aad045b4be59da`
- Python: 3.11
- pandas: 3.0.5
- numpy: 2.4.6
- scikit-learn: 1.9.0
- The research self-test, full study, matched-week follow-up, and artifact upload all passed.

## Primary hypothesis fixed before the rerun

The primary three-leg rule was fixed before the dedicated study:

1. Use the current broad/general HGB model.
2. Exclude pure FCS-vs-FCS games, because live FCS uses a separate model and FCS parlays have not been validated.
3. UNDER only.
4. Predicted under edge >= 3.5 points.
5. Historical market total >= 56.0, using a numeric cutoff rather than the older categorical total bin.
6. One card per season/week only when at least three eligible games exist.
7. Select the three strongest absolute model edges deterministically.
8. No forced card.
9. One unit staked per card.
10. Base payout assumption: each leg priced at -110 and parlay decimal odds obtained by multiplication. Stress tests also use -115 and -120 per leg.

The HGB predictions are season walk-forward: each test season is predicted from models fit only on earlier seasons. The final 3.5/56 underlying leg rule itself was developed in prior historical research, so the three-leg historical test is **not** a pristine untouched confirmatory experiment.

## Primary three-leg result

Across the current production-aligned general scope:

- 100 weeks produced a three-leg card.
- 98 cards had no total push on any leg.
- Full three-leg record: **16 wins, 82 losses**.
- Two additional cards were push-affected, but both also contained a losing leg and therefore lost the parlay.
- Full-card hit rate: **16.33%**.
- Theoretical three-leg breakeven at -110 each: **14.37%**.
- Wilson 95% interval for the hit rate: **10.31% to 24.89%**.
- Nominal one-sided binomial p-value versus the -110 three-leg breakeven rate: **0.3315**.
- Net at -110 per leg: **+11.33 units** on 100 units staked.
- ROI per staked card: **+11.33%**.
- Season-block bootstrap 95% ROI interval: **-21.23% to +47.34%**.
- Maximum historical drawdown: **-16.0 units**.
- Longest losing streak: **16 cards**.
- Average number of qualifying candidates in weeks when a card was available: 6.62.
- Average selected absolute model edge: 7.40 points.
- Average weakest selected leg edge: 6.23 points.
- Average selected market total: 64.87.

The aggregate +11.3% ROI is therefore **not statistically or temporally robust enough to promote**. The block-bootstrap interval includes substantial negative performance.

## Price sensitivity

The historical result is highly sensitive to effective pricing:

| Effective price per leg | Three-leg ROI |
|---|---:|
| -110 | +11.33% |
| -115 | +4.55% |
| -120 | **-1.41%** |

The observed full-card hit rate of 16.33% is approximately the breakeven hit rate of an effective per-leg price near -120.5. This leaves little margin for worse-than-assumed pricing.

## Season stability

Primary three-leg card by season:

| Season | Cards | Wins | Losses | ROI/card |
|---:|---:|---:|---:|---:|
| 2016 | 13 | 1 | 12 | -46.48% |
| 2017 | 13 | 1 | 12 | -46.48% |
| 2018 | 13 | 3 | 9* | +60.57% |
| 2019 | 12 | 1 | 11 | -42.02% |
| 2020 | 8 | 1 | 7 | -13.03% |
| 2021 | 14 | 3 | 11 | +49.10% |
| 2022 | 13 | 2 | 11 | +7.05% |
| 2023 | 7 | 2 | 5 | +98.80% |
| 2024 | 7 | 2 | 4* | +98.80% |

`*` One card in 2018 and one in 2024 contained a total push plus at least one losing leg, so the parlay still lost; the full-leg W/L column excludes those push-affected cards when estimating hit probability.

There were cards in nine historical walk-forward seasons. Only **five of nine seasons** were profitable. No primary three-leg card was available in the 2025 OOF season.

Era split:

- 2016-2020: 59 cards, **-17.45% ROI/card**.
- 2021-2025 window: 41 cards, **+52.73% ROI/card**; the primary rule generated no card in 2025, so this positive block is effectively 2021-2024.

This is a large regime/era instability and argues strongly against treating the aggregate ROI as stationary.

## Straight-leg equivalent

The 300 selected primary legs, if played straight instead of parlayed:

- 298 graded legs.
- 171 wins, 127 losses, 2 pushes.
- Hit rate: **57.38%**.
- Wilson 95% interval: **51.71% to 62.87%**.
- Nominal one-sided binomial p-value versus -110 straight breakeven: **0.0471**.
- Straight ROI per unit staked: **+9.48%**.

The parlay does not create predictive information; it simply transforms the same leg-level signal into a higher-variance payoff structure.

## Leg-rank behavior and dependence

On the 98 primary cards with all three legs graded:

| Rank by model edge | Wins | Hit rate | Avg abs edge |
|---:|---:|---:|---:|
| 1 | 60/98 | 61.22% | 8.72 |
| 2 | 57/98 | 58.16% | 7.28 |
| 3 | 53/98 | **54.08%** | 6.23 |

Pairwise same-card win correlations:

- Rank 1 vs 2: +0.047
- Rank 1 vs 3: -0.061
- Rank 2 vs 3: **-0.200**

Using the rank-specific hit rates and assuming independence would imply a three-leg hit rate of **19.26%**. The actual observed rate was only **16.33%**, 2.93 percentage points lower. The pooled 57.82% selected-leg hit rate cubed similarly implies 19.33%.

This demonstrates why multiplying marginal leg hit rates is not a sufficient validation of a multi-leg structure.

## Strongest-three ranking test

A 20,000-run simulation kept the same qualifying weekly pool and randomly chose three eligible legs in each of the 100 card weeks. This tests the incremental value of choosing the three largest model edges while holding the underlying weekly candidate pool fixed.

- Observed top-three ROI: **+11.33%**.
- Mean random-three ROI: **+31.09%**.
- Median random-three ROI: **+32.20%**.
- Random-three 2.5th to 97.5th percentile: **-9.22% to +73.95%**.
- **80.93%** of random-selection simulations equaled or exceeded the observed strongest-three ROI.
- The observed strongest-three strategy sat at only the **19.07th percentile** of the random-three distribution.

Therefore there is **no evidence that ranking by the largest model edge improves the three-leg card**. The positive aggregate result appears to come from the qualifying pool rather than from the specific strongest-three construction.

## Stronger-edge sensitivity

Raising the minimum model edge from 3.5 to 5.0 did **not** improve the three-leg result:

- 71 cards.
- 10-59 on fully graded cards, plus 2 push-affected cards.
- Full-card hit rate: 14.49% versus 14.37% theoretical breakeven.
- ROI at -110: **-2.00%**.
- Season-block bootstrap ROI interval: **-48.46% to +39.16%**.
- Only 2 of 8 seasons with cards were profitable.

The lack of monotonic improvement as the minimum edge is raised is another negative robustness signal for the three-leg construction.

## FBS-only sensitivity

Restricting to FBS-vs-FBS only:

- 100 cards.
- 15-83 on fully graded cards, plus 2 push-affected cards.
- ROI at -110: **+4.37%**.
- Season-block bootstrap interval: **-24.22% to +37.71%**.
- Max drawdown: **-21.21 units**.

The production-aligned positive aggregate result is therefore not particularly strong when narrowed to FBS-vs-FBS only.

## Weather/provider ablations

The three-leg result is extremely sensitive to model feature specification:

- Current model: **+11.33% ROI**.
- Remove line-provider feature: **+8.08% ROI**.
- Remove dynamic weather: **+61.37% ROI**.
- Remove dynamic weather and line provider: **+40.82% ROI**.

The no-weather result is much stronger than the current weather-inclusive result, including a positive season-block bootstrap interval. That is **not a reason to retune the live model toward a no-weather parlay**. It is evidence that the historical three-leg payoff is highly specification-sensitive and that the current historical weather information set is not demonstrably the mechanism producing parlay value. Choosing the best ablation after seeing these outcomes would be classic post-selection bias.

## Same-pool two-leg comparator

Using the same current general HGB leg rule but taking the strongest two games whenever at least two are available produced:

- 117 cards.
- 41-73 on fully graded cards, plus 3 push-affected cards.
- Full-card hit rate: **35.96%** versus 27.44% theoretical two-leg breakeven at -110.
- ROI at -110: **+27.72%**.
- Season-block bootstrap ROI interval: **+5.42% to +54.82%**.
- Maximum drawdown: **-9 units**.
- Longest losing streak: 9.
- Six of nine seasons with cards were profitable.

This comparator is materially stronger than the primary three-leg result, but it includes 17 weeks where only two qualifying games were available, so a matched-week follow-up was added.

## Post-hoc matched-week two-versus-three diagnostic

This diagnostic was added **after the first dedicated run** to make the two-leg/three-leg comparison apples-to-apples. It does not redefine the primary hypothesis and must be labeled post-hoc.

On the exact same 100 weeks that had at least three candidates:

- Stop after strongest two: **+31.21 units, +31.21% ROI/card**.
- Add the third-ranked leg: **+11.33 units, +11.33% ROI/card**.
- Marginal ROI of adding the third leg: **-19.88 percentage points**.
- Season-block bootstrap 95% interval for the marginal third-leg effect: **-36.45 to -3.07 percentage points**.
- Only **0.92%** of season-block bootstrap replicates had the three-leg card outperforming the same-week two-leg card.
- Three-leg ROI was better in only **1 of 9 seasons** and worse in **8 of 9**.
- Two-sided sign-test p-value across the nine season-level differences: **0.0391**.

Season-by-season marginal effect of adding the third leg:

| Season | Top-2 ROI | Top-3 ROI | Top-3 minus Top-2 |
|---:|---:|---:|---:|
| 2016 | -15.89% | -46.48% | -30.58 pp |
| 2017 | +12.14% | -46.48% | -58.62 pp |
| 2018 | +68.21% | +60.57% | -7.65 pp |
| 2019 | -39.26% | -42.02% | -2.76 pp |
| 2020 | +36.67% | -13.03% | -49.70 pp |
| 2021 | +30.17% | +49.10% | +18.93 pp |
| 2022 | +40.18% | +7.05% | -33.13 pp |
| 2023 | +108.26% | +98.80% | -9.47 pp |
| 2024 | +108.26% | +98.80% | -9.47 pp |

Among the 36 matched weeks in which the first two legs both won, the third-ranked leg went only **16-20 (44.44%)**. That conditional result explains why the third leg frequently converted a winning two-leg card into a losing three-leg card.

## Card availability

Across 153 season/weeks in the OOF production-aligned scope:

- At least one primary qualifying candidate: 127 weeks.
- At least two candidates: 117 weeks (76.47% of scope weeks).
- At least three candidates: 100 weeks (65.36% of scope weeks).

Thus the three-leg requirement also produces fewer opportunities than the two-leg card.

## Model-level guardrail

The current HGB remains worse than the zero-residual market baseline as an unconditional point predictor in the season walk-forward sample. The weighted current-model MAE is about 13.02 points versus about 12.82 for predicting no market residual. The research question is therefore a selective-tail hypothesis, not a claim that the model is globally better than the market.

## Why the older exploratory three-leg result was not sufficient

The repository already contained generic parlay code that generated many same-week 2- and 3-leg combinations and an older weekly-card three-leg output. Those analyses were useful exploratory work but were not adequate for operational promotion because they could include overlapping combinations, a mixed-classification broad-model universe, categorical total-bin behavior that is not identical to numeric total >=56, and older/unpinned research outputs.

The dedicated study deliberately used one deterministic card per week, the pinned current environment, exact numeric thresholds, the current non-FCS production scope, dependence diagnostics, block resampling, feature ablations, and matched-week comparisons.

The legacy mixed-classification diagnostic from the new study produced 104 three-leg cards with 18 full-card wins and 84 full-card losses plus two push-affected losses, returning +20.43% per card at -110. That is close enough to the older exploratory result to provide a useful reproduction check, while the production-aligned primary scope falls to +11.33% and fails the robustness tests above.

## Final scientific conclusion

There is **some historical evidence that the underlying high-total UNDER qualifying pool contains signal**, but there is **not robust evidence that packaging the three strongest qualifying games into a three-leg parlay adds value**.

The strongest evidence against operational promotion is:

1. Primary three-leg season-block ROI CI crosses zero widely (-21.2% to +47.3%).
2. Nominal full-card test versus three-leg breakeven is weak (p=0.3315).
3. Price stress turns the result negative by -120 per leg.
4. Only 5/9 seasons were profitable and the early era was materially negative.
5. Edge >=5 sensitivity is slightly negative rather than stronger.
6. FBS-only sensitivity is only +4.4% with a wide negative-to-positive block interval.
7. Dynamic-weather ablation radically changes the result, demonstrating specification sensitivity.
8. The strongest-three ranking is worse than 80.9% of random-three simulations from the same qualifying weekly pool.
9. On matched weeks, the same strongest two legs outperform the three-leg structure by 19.9 percentage points of ROI, with the third leg reducing ROI in 8/9 seasons.
10. When the first two legs won, the third leg was only 16-20 historically.

### Operational recommendation

**Keep the existing live card unchanged. Do not add a three-leg card, do not replace the two-leg card, and do not promote the no-weather ablation.**

If continued research is desired, the scientifically defensible next step is only a **frozen 2026 shadow three-leg track** inside the existing immutable prospective ledger, using the already-declared primary rule (non-FCS general HGB UNDER >=3.5, total >=56, top three by model edge, no forced card). It should be recorded for research without being surfaced as an operational recommendation and without retuning from 2026 results until the predeclared review point.
