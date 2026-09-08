# Pregame Forecast Model Run Trigger

The full archived NBM acquisition run `34213947929` passed the frozen source/QC gate before model fitting:

- 4,829 / 4,829 station-mapped games have complete 24h, 12h, and 6h forecasts.
- Every season from 2019 through 2025 has 100% three-lead coverage.
- Core continuous forecast fields are 100% complete.
- Availability-time leakage violations: 0.
- Kickoff valid-time offset violations: 0.
- Station-distance violations: 0.
- Duplicate game/lead rows: 0.
- `ready_for_modeling=True`.

The independent model-integrity workflow also passed all six preregistered self-tests.

This file triggers the already-frozen model workflow. It does not change any research threshold, model rule, feature definition, evidence gate, or production behavior.
