# Confirmatory Field-Geometry Research

## Status

Retrospective research only. This work does not modify the live 2026 model, weekly-board logic, pick thresholds, the frozen orientation shadow, or the official prospective ledger.

## Why this study exists

The joint-core robustness study isolated `along_field_wind` as the only component whose removal produced a statistically supported degradation in model performance. That is interesting, but it is not enough by itself to promote anything. The purpose of this study is to confirm or falsify that geometry signal using a narrower predeclared test.

## Primary hypothesis

The actual magnitude of wind along the real stadium field axis contains incremental information beyond the existing GENERAL HGB baseline.

The primary `along_magnitude` model is the baseline plus `alongwind_mph`, where the field is treated as an undirected 0-180 degree axis and along-field wind is a magnitude.

To be labeled `GEOMETRY_CONFIRMED_RETROSPECTIVELY`, both conditions must be met:

1. `along_magnitude` must satisfy the same four-part evidence gate used in the earlier joint research:
   - mean paired MAE delta below zero;
   - 95% game-level paired bootstrap interval entirely below zero;
   - 95% season-cluster interval entirely below zero;
   - improvement in at least 70% of test seasons.
2. The real field axes must beat a fixed 39-permutation venue-axis placebo test at randomization p <= 0.05 and perform better than the 5th percentile of placebo results.

The 39 placebo mappings preserve the observed field-axis distribution and venue orientation coverage, but break the true venue-to-axis pairing. This is intended to test whether actual geometry matters rather than whether HGB simply benefits from another nonlinear transformation of baseline wind speed.

## Secondary mechanism models

These are diagnostic and cannot substitute for failure of the primary confirmation:

- `cross_magnitude`: baseline + cross-field wind magnitude.
- `vector_magnitude`: baseline + along-field and cross-field wind magnitudes.
- `along_alignment`: baseline + cos(field-relative wind angle). Since baseline already contains wind speed, this asks whether pure relative alignment adds information.
- `alignment_pair`: baseline + along- and cross-alignment fractions.
- `joint_core`: the previously studied orientation vector plus leak-safe climate context.

## 2022 investigation

The earlier robustness work showed that omitting 2022 made the full joint-core result pass the retrospective evidence gate. This study does not remove 2022 or reinterpret that omission as evidence. Instead it produces descriptive diagnostics to understand what was unusual about that year.

Outputs compare 2022 with the other test seasons on:

- wind speed;
- along-field wind;
- cross-field wind;
- field-relative wind angle;
- along-field alignment;
- temperature and temperature anomaly;
- local wind percentile;
- closing total;
- venue coverage and whether venues were already seen in prior training;
- baseline, along-wind, and joint-core MAE.

It also compares 2022 versus other test seasons in fixed wind-speed and field-angle bins. These are explanatory diagnostics only and are not additional promotion gates.

## Isolation

The manual workflow:

- restores the historical `modeling_dataset.csv` artifact;
- reads the committed stadium location and orientation references;
- runs the self-test first;
- writes only `outputs/field_geometry_confirmation/`;
- uploads that directory as a workflow artifact;
- commits only that directory back to `main`.

Even a successful retrospective confirmation would only justify considering a new separately versioned prospective shadow challenger. It would not alter the frozen live 2026 protocol.
