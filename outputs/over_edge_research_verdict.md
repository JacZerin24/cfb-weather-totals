# OVER Edge Research Verdict

## Status

**Do not promote an OVER strategy operationally.**

This study was intentionally run as isolated research on branch `over-edge-research` / draft PR #17. It used the latest live training artifact (workflow run `33229390276`) and the pinned production environment (`pandas 3.0.5`, `numpy 2.4.6`, `scikit-learn 1.9.0`). The research workflow was run `33239704916`; artifact SHA-256: `4b2f35260099a845cf9a8885d86fe21d8593c288012067cd69702c3eff928c23`.

No production thresholds, live statuses, weekly card rules, or prospective protocol were changed.

## Predeclared research design

- Season walk-forward HGB predictions only: each test season is predicted using prior seasons.
- General live scope excludes pure FCS-vs-FCS games, because those are handled by the dedicated FCS model.
- Separate FBS-vs-FBS cross-check.
- 220 finite, interpretable OVER candidates per scope using predeclared model-edge thresholds, market-total ranges, fixed weather thresholds, indoor/outdoor and limited total-weather interactions.
- Benjamini-Hochberg FDR and Bonferroni multiple-testing correction across the full candidate grid.
- Season-by-season stability, recent-era performance, Wilson intervals, drawdown, and exact one-sided tests versus -110 breakeven.
- Prior-season-only nested rule selection: select a candidate from earlier out-of-fold seasons and grade it on the next untouched season.
- Season-block bootstrap on broad predeclared OVER rules.
- Weather/provider ablations.
- Separate FCS OVER audit.

## Broad general OVER results

| HGB OVER edge | Graded | Record | Hit rate | Paper ROI at -110 | Positive seasons | 2022+ ROI |
|---:|---:|---:|---:|---:|---:|---:|
| 2.5+ | 1,792 | 900-892 | 50.22% | -4.12% | 3/9 | +0.42% |
| 3.5+ | 1,254 | 646-608 | 51.52% | -1.65% | 3/9 | -0.14% |
| 5.0+ | 719 | 381-338 | 52.99% | +1.16% | 4/9 | -0.57% |
| 6.0+ | 458 | 248-210 | 54.15% | +3.37% | 4/9 | -2.10% |
| 7.5+ | 242 | 133-109 | 54.96% | +4.92% | 5/9 | +7.69% |

The higher model-edge bands do improve directionally, which is the most encouraging feature of the OVER results. However, the 7.5+ rule still has a nominal one-sided p-value of about 0.23, a 95% Wilson hit-rate interval of about 48.7%-61.1%, only 5/9 profitable seasons, and a season-block bootstrap ROI interval that still crosses zero.

For 7.5+ OVER, the season-block bootstrap 95% ROI interval was approximately **-0.7% to +13.5%**. That is interesting enough to keep as a research observation, but not enough to call a validated edge.

## Discovery grid result

**0 of 220 general-live-scope candidates passed the predeclared exploratory robustness screen.**

**0 of 220 FBS-vs-FBS candidates passed.**

The smallest raw general-scope p-value was about 0.027, but it came from only 9 graded games. After Benjamini-Hochberg correction, the minimum q-value across the general candidate grid was about **0.988**; the Bonferroni-adjusted p-values bottomed out at 1.0.

The strongest-looking adequately sampled general candidate was `OVER edge >= 5 + temperature <= 55F`:
- 155 graded
- 93-62
- 60.0% hit
- +14.55% historical paper ROI
- 7/9 profitable seasons

But it failed the anti-selection tests:
- raw p about 0.034 became BH q about 0.988
- its 2022+ sample was 33 graded and slightly **negative** (-1.65% ROI)
- the nested prior-season-only test later lost money overall

A second exploratory lead, `OVER edge >= 6 + total >= 52`, was 150-115 (56.6%, +8.1% ROI) with 6/9 positive seasons and positive recent performance, but it also failed multiple-testing correction and was not independently validated.

These are research leads, not operational rules.

## Prior-season-only nested selection

This is the most important anti-overfitting result.

The selector was allowed to choose an OVER rule using only earlier out-of-fold seasons, then that rule was graded on the next untouched season.

### General live scope
- 122 graded
- 60-62
- 49.18% hit rate
- **-6.11% ROI**
- one-sided p about 0.79

### FBS-vs-FBS
- 97 graded
- 50-47
- 51.55% hit rate
- **-1.59% ROI**
- one-sided p about 0.61

So the attractive post-hoc OVER subsets did **not** translate into profitable prior-only rule selection.

## Edge monotonicity

The general-live-scope HGB OVER bands were:

| Edge band | Hit rate | ROI |
|---|---:|---:|
| 2.5-3.5 | 47.21% | -9.87% |
| 3.5-5.0 | 49.53% | -5.44% |
| 5.0-6.0 | 50.96% | -2.72% |
| 6.0-7.5 | 53.24% | +1.64% |
| 7.5+ | 54.96% | +4.92% |

That monotonic direction is scientifically interesting and is the main reason the OVER hypothesis is not dismissed as completely random. But the upper bands remain too uncertain and unstable to promote.

## Feature-ablation result

The apparent high-edge OVER performance is specification-sensitive.

General live scope:

- Current model, 5.0+ OVER: +1.16%
- No dynamic weather, 5.0+ OVER: **-3.76%**
- No line-provider feature, 5.0+ OVER: +5.16%
- No weather or provider, 5.0+ OVER: **-5.07%**

At 7.5+:
- Current: +4.92%
- No dynamic weather: **-1.18%**
- No line provider: +1.12%
- No weather or provider: **-0.29%**

This does not prove weather creates an OVER edge. It shows the result changes materially with feature specification, another reason not to operationalize it yet.

## FCS OVER result

The dedicated FCS model does not support an OVER strategy.

Broad FCS OVER results included:
- 3.5+ OVER: 276-326, **-12.47% ROI**
- 5.0+ OVER: 229-244, **-7.57% ROI**
- 7.5+ OVER: 142-153, **-8.10% ROI**

A few high-total FCS subsets were positive in very small samples (for example 3.5+ with total >=60 was 24-18, +9.1%), but none survived multiple-testing correction and the season stability was weak.

**No FCS OVER should be promoted.**

## Scientific takeaway

The research does **not** support adding OVERs to the live qualifier pool or the operational 2-leg card.

The useful takeaway is more specific than “overs never work”:

1. Broad 2.5-3.5 OVER signals are clearly poor.
2. Stronger HGB OVER edges improve monotonically and 7.5+ is historically positive, so the model may contain some directional information at the extreme upper tail.
3. No OVER candidate survived the full robustness screen.
4. Prior-season-only nested rule selection was negative.
5. FCS OVERs are clearly unsupported.
6. Therefore the current operational decision to promote only validated UNDERs remains scientifically justified.

A future prospective shadow test of a *predeclared* extreme general OVER rule could be justified as research, but it should not be mixed into the operational weekly card unless it first survives genuinely forward data.
