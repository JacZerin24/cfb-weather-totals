# Final Joint-Core Orientation Reconciliation

## Scope

This is a retrospective research-only study. It does not modify the frozen 2026 production model, live picks, thresholds, weekly board, orientation shadow, or official prospective ledger.

This is intentionally the final retrospective field-orientation reconciliation test. No new weather features, thresholds, alternate training windows, or post-hoc season exclusions are introduced.

## Why this test exists

The earlier joint-core ablation found that removing `alongwind_mph` from the full joint weather-context model caused a statistically supported loss in MAE. A later standalone field-geometry confirmation did not show that real stadium axes were special: real-axis along-wind failed the primary model gate and did not beat randomized venue-axis assignments.

Those results can coexist if `alongwind_mph` is useful as a nonlinear representation or proxy inside the larger HGB model without the real stadium orientation being uniquely informative.

This study directly tests that possibility.

## Fixed primary comparison

All models use the same expanding-history walk-forward folds, the same `HistGradientBoostingRegressor`, and the same scored `joint_ready` games.

### Reference: `joint_core_no_along`

- frozen GENERAL baseline numeric/categorical features
- all existing climate-context features
- REAL `crosswind_mph`
- no along-field feature

### Real challenger: `joint_core_real`

Everything in the reference plus:

- REAL `alongwind_mph`

### Placebo challengers

There are 39 fixed venue-axis permutations using the same seed as the prior geometry confirmation. Each placebo challenger contains exactly the same features as the real challenger except:

- REAL `crosswind_mph` is retained
- all climate-context features are retained
- `alongwind_mph` is replaced by `placebo_alongwind_mph`, calculated from a deranged venue-to-axis assignment

This asymmetry is deliberate. The previous supported ablation isolated the conditional contribution of along-field wind while real crosswind and climate context remained in the model. This test reproduces that exact conditional question.

## Evidence gate for the real component

The real challenger must beat `joint_core_no_along` on all four pre-existing requirements:

1. paired mean MAE delta < 0
2. game-level bootstrap 95% interval entirely < 0
3. season-cluster 95% interval entirely < 0
4. improvement in at least 70% of evaluated test seasons

No requirement is changed after seeing results.

## Randomization gate

The real conditional along-wind contribution is compared with the 39 placebo contributions.

Real geometry is distinguished only if:

- the one-sided randomization p-value is <= 0.05, and
- the real MAE delta is below the 5th percentile of the placebo distribution.

With 39 placebos, the finite-sample p-value is `(placebos as good or better + 1) / 40`.

## Final interpretation

### `CONDITIONAL_REAL_GEOMETRY_RECONCILED`

The real along-field component passes the retrospective component gate and real venue axes beat the placebo distribution. This would indicate that real field geometry becomes uniquely informative only inside the fixed joint climate/context model.

It still would not alter production. The strongest permissible next step would be a separately frozen prospective shadow challenger.

### `ORIENTATION_PROXY_NOT_DISTINGUISHED`

The real along-field component still helps conditionally, but randomized venue-axis along-wind features perform similarly. The along-wind feature should then be interpreted as a proxy/nonlinear representation rather than evidence that actual field orientation matters.

Under this result, field orientation should be retired from further retrospective tuning and no orientation-specific prospective shadow should be created.

### `CONDITIONAL_ALONG_SIGNAL_NOT_REPRODUCED`

The earlier supported along-field ablation does not reproduce under this fixed final test. The along-field signal should be retired.

## Outputs

The manual workflow writes only `outputs/joint_core_orientation_reconciliation/`:

- `walk_forward_predictions.csv`
- `walk_forward_diagnostics.csv`
- `real_conditional_component_test.csv`
- `conditional_placebo_summary.csv`
- `conditional_placebo_by_season.csv`
- `conditional_placebo_test.csv`
- `season_reconciliation.csv`
- `reconciliation_status.csv`
- `summary.md`

## Guardrails

- Production remains frozen regardless of outcome.
- The failed standalone field-geometry confirmation remains part of the evidence record.
- No season can be removed or downweighted after the result is known.
- No alternate permutation seed, placebo count, threshold, feature definition, or model variant can substitute for failure of this predeclared test.
