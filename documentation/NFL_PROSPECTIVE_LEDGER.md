# NFL 2026 Prospective Ledger

This ledger is the immutable evidence layer for `nfl_totals_paper_v1`.

## Decision capture

The mutable live board refreshes every three hours. A scheduled first-attempt run may write an immutable decision snapshot only when one or more 2026 regular-season games are inside the frozen 24-hour decision window.

The snapshot contains the line, representative and best OVER price, sportsbook names, market breadth, model residual prediction, OVER probability, projected total, venue/roof treatment, and 24/48/72-hour fixed-lead forecast fields.

For each game, the **first eligible scheduled snapshot inside the window is permanent**. A later refresh cannot create, erase, or upgrade the official decision.

Derived files:

- `official_decisions.csv`: one immutable decision per game, including NO PLAY;
- `official_entries.csv`: QUALIFIES/STRONG subset of official decisions;
- `entries_with_clv.csv`: official entries joined to the latest valid near-kickoff benchmark.

Manual runs and reruns are never official-eligible.

## Immutability

Source records are written under:

- `outputs/nfl/prospective/2026/decision_snapshots/`
- `outputs/nfl/prospective/2026/close_captures/`

Each filename contains the first 12 characters of the SHA-256 digest of its CSV bytes. Writes use exclusive-create mode and refuse to overwrite an existing path. `manifest.csv` recomputes and verifies every immutable file hash.

The derived CSVs may be rebuilt; the hashed source snapshots may not be edited or replaced.

## Entry price

The official paper entry uses the representative OVER price at the selected consensus total:

- `entry_total = closing_total`
- `entry_over_price = consensus_over_price`
- `entry_over_sportsbook = consensus_over_sportsbook`

The best price available at the same line is retained separately for line-shopping research, but it does not replace the official representative price.

## Near-kickoff CLV benchmark

A separate scheduled workflow runs twice per hour from September through January. It performs a local zero-API preflight first. The Odds API is called only when an already-official entry is within 105 minutes of kickoff.

Every valid first-attempt scheduled capture is immutable. The benchmark for a game is the **latest valid capture before kickoff**.

For an OVER entry:

`CLV points = benchmark close total - entry total`

Positive CLV means the paper entry obtained a lower total than the near-kickoff market.

## Scope

This ledger remains paper-only. It captures evidence for later evaluation; it does not establish that the model has a proven wagering edge.
