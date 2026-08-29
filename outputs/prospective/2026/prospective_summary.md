# 2026 Prospective Validation Ledger

Protocol version: **2026.1**

Protocol SHA-256: `7d419007007f79c5a6b0bad66fdee94f98d74627e37f16e406e0ca54fa87dc2c`

The immutable snapshot files are the source of truth. Derived ledgers and summaries can be rebuilt from them at any time.

## Frozen rules

- General: HGB UNDER edge >= 3.5, total >= 56.
- FCS: FCS-only HGB UNDER edge >= 7.5, total >= 56.
- Official entry: latest eligible scheduled snapshot at least 120 minutes before kickoff.
- CLV benchmark: latest immutable pre-kickoff market capture within the 90-minute capture window; median across books when available.

## Prospective results

| scope        |   official_games |   qualifying_entries |   settled_qualifying_entries |   graded_ex_pushes |   wins |   losses |   pushes |   hit_rate_ex_pushes |   net_units_1u_at_-110 |   roi_per_graded_entry |   qualifiers_with_clv |   average_clv_points |   median_clv_points |   positive_clv_rate |
|:-------------|-----------------:|---------------------:|-----------------------------:|-------------------:|-------:|---------:|---------:|---------------------:|-----------------------:|-----------------------:|----------------------:|---------------------:|--------------------:|--------------------:|
| ALL          |              376 |                    1 |                            0 |                  0 |      0 |        0 |        0 |                  nan |                      0 |                    nan |                     0 |                  nan |                 nan |                 nan |
| FCS-only HGB |               24 |                    1 |                            0 |                  0 |      0 |        0 |        0 |                  nan |                      0 |                    nan |                     0 |                  nan |                 nan |                 nan |
| GENERAL HGB  |              352 |                    0 |                            0 |                  0 |      0 |        0 |        0 |                  nan |                      0 |                    nan |                     0 |                  nan |                 nan |                 nan |

## Data integrity

- Immutable board snapshots: 3
- Immutable close captures: 1
- Official game entries selected: 376

Every immutable CSV filename contains the first 12 characters of its SHA-256 content hash. The workflow verifies those hashes before rebuilding derived results.

## Interpretation

These are prospective paper results, not a retrospective re-optimization. Protocol changes require a new version and apply prospectively only. Missing close captures remain missing rather than being backfilled from an unverified postgame line.