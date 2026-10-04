# NFL Long-History Football-Context Discovery — Preregistered Phase 2

**Branch:** `research/nfl-team-context-v2`  
**Frozen production/paper baseline:** `nfl_totals_paper_v1` (must remain unchanged)  
**Purpose:** Use pre-2018 NFL history to discover a compact football-context subset without selecting features on the 2018-2025 forecast-native evaluation period.

## Why this phase exists

Phase 1 showed that a broad football-context feature set did not beat the frozen v1 qualifier rule. A fixed 50/50 blend produced only a very small improvement in raw residual MAE, with uncertainty intervals crossing zero.

This phase asks a narrower question:

> Is there a small, repeatable subset of pregame football information that adds residual information beyond the market/context baseline before forecast weather is considered?

The study is intentionally designed so feature-group selection happens only on **pre-2018** games.

## Data chronology

- nflverse regular-season play-by-play: 2005-2025.
- 2005 is warmup history only.
- Long-history modeled seasons: 2006-2017 for feature discovery/confirmation.
- Discovery walk-forward block: 2009-2013.
- Internal confirmation block: 2014-2017.
- Forecast-native evaluation: 2018-2025.
- True decision-time market check: 2025 archived 24-hour market snapshots.

All team metrics are calculated from the previous eight completed games and shifted by one game. At least four prior games are required for both teams.

## Important holdout caveat

The 2018-2025 outcomes have already been inspected in Phase 1, so they are **not a pristine human-blind holdout anymore**. Phase 2 feature selection itself is algorithmic and may use only 2006-2017 results, which prevents direct outcome-driven selection on 2018-2025. Still, the strongest confirmation of any Phase 2 result must come from prospective 2026 tracking.

## Fixed candidate feature groups

All candidates use the same market/context variables as the context-only comparator. No weather is used for pre-2018 selection.

### efficiency_core
- two-team mean and mismatch for EPA/play and success rate
- two-team mean defensive EPA/play and success rate allowed
- offense-vs-opposing-defense EPA and success matchup means

### passing_core
- two-team mean and mismatch for pass EPA
- two-team mean pass-defense EPA allowed
- offense-vs-opposing-defense pass EPA matchup mean
- sack rate, defensive sack rate allowed, pass rate, pass-rate mismatch

### rushing_core
- two-team mean and mismatch for rush EPA
- two-team mean rush-defense EPA allowed
- offense-vs-opposing-defense rush EPA matchup mean

### explosive_turnover
- explosive-play rate and mismatch
- explosive-play rate allowed
- turnover rate and mismatch
- turnover rate allowed

### early_down
- early-down EPA mean and mismatch
- early-down EPA allowed

### efficiency_plus_passing
Exact union of `efficiency_core` and `passing_core`.

### compact_all
Exact union of all five primitive groups above.

No new feature group, rolling window, transformation, or model family may be added after pre-2018 results are inspected without creating a new research phase.

## Model architecture

To isolate the feature question, every candidate uses the existing fixed Random Forest architecture:

- regression: `reg_rf_leaf25`
- classifier: `cls_rf_leaf25`
- target: market residual
- no hyperparameter search

The comparator is a context-only model using the same architecture and market/context variables.

## Discovery procedure

Chronological expanding-window predictions are created for 2009-2017. For every test season the model trains only on earlier seasons.

For each candidate and the context-only comparator report:

- residual MAE
- Brier score
- fixed v1 QUALIFIES performance
- fixed v1 STRONG performance
- season-by-season results

## Fixed selection rule

For each candidate calculate, separately for 2009-2013 and 2014-2017:

- relative MAE gain = (baseline MAE - candidate MAE) / baseline MAE
- relative Brier gain = (baseline Brier - candidate Brier) / baseline Brier
- composite gain = 0.5 * relative MAE gain + 0.5 * relative Brier gain

A candidate is eligible for the Phase 2 reduced challenger only if:

1. discovery composite gain > 0;
2. confirmation composite gain > 0;
3. confirmation candidate MAE is not worse than baseline by more than 0.05 points;
4. confirmation candidate Brier score is not worse than baseline by more than 0.002;
5. its per-season composite gain is positive in at least 2 of the 4 confirmation seasons.

If multiple candidates qualify, select the one with the **highest confirmation composite gain**. Ties within 0.0001 are broken by fewer football features, then alphabetically.

If no candidate passes all gates, Phase 2 selects **no reduced football challenger**. Do not lower the gates after seeing the results.

## Post-selection evaluation

Only the automatically selected pre-2018 candidate is carried forward as the primary reduced challenger.

Using the fixed selected feature group, compare on the forecast-native period:

1. frozen-style v1 baseline — market/context + JMA forecast weather;
2. reduced football — market/context + selected football group;
3. reduced combined — market/context + JMA forecast weather + selected football group;
4. fixed 50/50 blend — v1 baseline and reduced-football predictions.

The football feature group and 50/50 blend weight are frozen before any Phase 2 forecast-native results are read.

Evaluation is chronological expanding-window from 2018 through 2025.

## Betting thresholds

Thresholds are inherited unchanged:

- QUALIFIES: residual >= +3.0 and P(Over) >= 0.60
- STRONG: residual >= +4.0 and P(Over) >= 0.60

No threshold optimization is allowed in this phase.

## Robustness

Report:

- season-by-season MAE/Brier;
- qualifier hit rate and flat -110 ROI;
- Wilson 95% intervals;
- paired bootstrap intervals for MAE/Brier improvement versus v1;
- agreement subsets where v1 and the reduced challenger independently qualify;
- the 2025 archived 24-hour market/price test.

## Interpretation

A historical edge is not enough to replace v1.

A reduced challenger is considered worth prospective tracking only if it:

- passes the pre-2018 selection gates exactly as written;
- improves or is effectively neutral in raw prediction quality over the forecast-native period;
- does not depend on a single season;
- retains useful qualifier volume;
- remains directionally credible in the 2025 decision-time test.

Even then it remains research-only. Prospective 2026 evidence is required before any production/paper-protocol replacement.
