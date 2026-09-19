# NFL 2026 Prospective Evaluation

The frozen `nfl_totals_paper_v1` strategy is evaluated prospectively without changing its model, thresholds, side, decision time, or grading rules from 2026 outcomes.

## What is graded

### Official entries

Only immutable entries created by the first eligible scheduled snapshot inside the frozen 24-hour decision window count toward the paper record.

Each official OVER is settled against its captured representative sportsbook total and price:

- WIN: actual game total is above the captured entry total;
- LOSS: actual game total is below the captured entry total;
- PUSH: actual game total equals the captured entry total.

Paper ROI uses the captured American price and one unit risked per settled entry. A flat -110 result is retained separately for research comparability.

### Calibration

Calibration uses **all immutable official decisions**, including NO PLAYs, not just selected entries. The frozen P(OVER) is compared with whether the game finished OVER the captured decision-time total. Pushes are excluded from binary calibration.

The report includes:

- Brier score;
- empirical base-rate Brier score;
- fixed probability-bin calibration;
- mean predicted P(OVER);
- observed OVER frequency.

### Closing-line value

For official entries:

`CLV points = near-kickoff benchmark total - captured entry total`

For an OVER, positive CLV means the frozen paper entry obtained a lower total than the later market.

### Missed / changed qualifiers

After a game's official 24-hour decision is frozen, later first-attempt scheduled live-board states are written as separate content-hashed `monitor_snapshot` records. They can never change the official entry.

The evaluation tracks:

- **late qualifier after freeze** — official NO PLAY later becomes QUALIFIES or STRONG;
- **official entry faded** — official QUALIFIES/STRONG later becomes NO PLAY;
- **tier upgraded later** — official QUALIFIES later reaches STRONG;
- **tier downgraded later** — official STRONG later becomes QUALIFIES.

This measures how much the fixed 24-hour cutoff differs from later information.

## Pre-registered evidence gate

The criteria are stored in `config/nfl_prospective_evaluation.yml` and were frozen before prospective outcomes accumulate.

Before the system can become `REVIEW_ELIGIBLE`, all of the following sample conditions must be met:

- at least **25 graded official entries**;
- at least **100 graded official decisions**;
- at least **20 entries with CLV**;
- entries spanning at least **8 distinct NFL weeks**;
- post-freeze monitor coverage of at least **80%** of official decisions.

Then the paper evidence must also satisfy:

- positive captured-price ROI;
- mean CLV above 0;
- median CLV at least 0;
- at least 50% of entries with positive CLV;
- Brier score no greater than 0.25;
- Brier score no worse than the empirical base-rate benchmark.

## Status meanings

**INSUFFICIENT_SAMPLE** — the pre-registered sample gate has not been met. Metrics are still shown, but early ROI or win rate must not be treated as validation.

**REVIEW_ELIGIBLE** — the sample gate and all pre-registered performance checks are met. This means only that the system is ready for human review.

**DO_NOT_PROMOTE** — the sample gate is met, but one or more pre-registered performance checks fail.

No status automatically authorizes wagering. `auto_validate_for_wagering` is permanently false in this evaluation plan.

## Outputs

The scheduled pipeline maintains:

- `graded_entries.csv`
- `graded_decisions.csv`
- `calibration_bins.csv`
- `signal_stability.csv`
- `performance_summary.csv`
- `evaluation_status.json`
- `prospective_evaluation.md`

The NFL site displays the current prospective evidence status and headline metrics.
