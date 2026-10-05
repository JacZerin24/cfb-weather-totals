# 2026 Prospective Validation Ledger

Protocol version: **2026.6**

Protocol SHA-256: `31e56b010c97f2b28e355236952dac28e3fa11e1c2a9f9e9a9937d4e3f876c98`

Immutable board snapshots and close captures remain the source of truth. Derived ledgers may apply only the explicitly documented integrity corrections in the active protocol.

## Frozen rules

- General: HGB UNDER edge >= 4, total >= 56.
- FCS: FCS-only HGB UNDER edge >= 7.5, total >= 56.
- Official entry: latest eligible scheduled snapshot at least 120 minutes before kickoff.
- CLV benchmark: latest immutable pre-kickoff market capture within the 90-minute capture window.

## Prospective results

| scope        |   official_games |   qualifying_entries |   settled_qualifying_entries |   excluded_qualifying_entries |   pending_qualifying_entries |   graded_ex_pushes |   wins |   losses |   pushes |   hit_rate_ex_pushes |   net_units_1u_at_-110 |   roi_per_graded_entry |   qualifiers_with_clv |   average_clv_points |   median_clv_points |   positive_clv_rate |
|:-------------|-----------------:|---------------------:|-----------------------------:|------------------------------:|-----------------------------:|-------------------:|-------:|---------:|---------:|---------------------:|-----------------------:|-----------------------:|----------------------:|---------------------:|--------------------:|--------------------:|
| ALL          |              872 |                   11 |                           10 |                             1 |                            0 |                 10 |      6 |        4 |        0 |             0.6      |               1.45455  |              0.145455  |                     4 |              0.875   |                0.25 |            0.5      |
| FCS-only HGB |              233 |                    7 |                            6 |                             1 |                            0 |                  6 |      4 |        2 |        0 |             0.666667 |               1.63636  |              0.272727  |                     3 |              1.33333 |                1    |            0.666667 |
| GENERAL HGB  |              639 |                    4 |                            4 |                             0 |                            0 |                  4 |      2 |        2 |        0 |             0.5      |              -0.181818 |             -0.0454545 |                     1 |             -0.5     |               -0.5  |            0        |

## Data-integrity corrections

- Official-entry rows selected through a documented eligibility metadata override: 109
- Qualifying entries explicitly excluded from performance grading: 1
  - Western Carolina at Campbell (game 401866625): POSTPONED / STALE - Western Carolina at Campbell was postponed from its original Saturday kickoff. The original prospective qualifier remains preserved, but the event changed kickoff/weather context and must not be settled, counted in units/ROI, or assigned CLV from the stale market snapshot.

## Data integrity

- Immutable board snapshots: 71
- Immutable close captures: 40
- Official game entries selected: 872

Every immutable CSV filename contains the first 12 characters of its SHA-256 content hash. The rebuild verifies those hashes before selecting entries.

## Interpretation

These are prospective paper results, not a retrospective re-optimization. Eligibility corrections are limited to documented automation metadata errors, and postponed/stale entries remain visible but do not count as wins, losses, units, ROI, or CLV.