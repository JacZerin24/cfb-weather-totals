from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import brier_score_loss, mean_absolute_error

from ..utils import ensure_dir, read_df, write_df
from .decision_time_holdout_2025 import (
    _build_market_snapshots,
    _load_market_rows,
)
from .forecast_native_bakeoff import _weather_features
from .model_bakeoff import BREAKEVEN, _prep
from .roi_search import _deep_models
from .team_context_phase2_eval import (
    LEAD,
    RULES,
    SELECTION_PATH,
    _feature_sets,
    _mask,
    _wilson,
)


DATA_PATH = 'data/nfl/processed/team_context_features.csv'
TEST_SEASON = 2025


def _load_selection() -> dict:
    path = Path(SELECTION_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f'{SELECTION_PATH} missing; run Phase 2 discovery first.'
        )
    return json.loads(path.read_text(encoding='utf-8'))


def _american_profit(odds: float) -> float:
    if not np.isfinite(odds) or odds == 0:
        return np.nan
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def _price_roi(
    frame: pd.DataFrame,
    price_column: str,
) -> tuple[int, float, float]:
    residual = pd.to_numeric(frame['market_residual'], errors='coerce')
    prices = pd.to_numeric(frame[price_column], errors='coerce')
    valid = residual.ne(0) & residual.notna() & prices.notna()
    units = 0.0
    for idx in frame.index[valid]:
        if residual.loc[idx] > 0:
            units += _american_profit(float(prices.loc[idx]))
        else:
            units -= 1.0
    graded = int(valid.sum())
    return graded, units, units / graded if graded else np.nan


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    nums: list[str],
    cats: list[str],
) -> pd.DataFrame:
    train_work = _prep(train, nums, cats)
    test_work = _prep(test, nums, cats)
    models = _deep_models(nums, cats)
    reg = models['reg_rf_leaf25'][1]
    cls = models['cls_rf_leaf25'][1]

    reg.fit(train_work[nums + cats], train_work['market_residual'])
    cls_train = train_work[train_work['market_residual'].ne(0)].copy()
    cls.fit(
        cls_train[nums + cats],
        cls_train['market_residual'].gt(0).astype(int),
    )

    keep = [
        c for c in [
            'game_id', 'season', 'week', 'gameday',
            'away_team', 'home_team',
            'decision_capture_at', 'decision_target_at',
            'decision_staleness_hours', 'decision_total',
            'consensus_over_price', 'best_over_price_same_line',
            'sportsbooks_at_selected_line', 'sportsbooks_in_snapshot',
            'market_total_min', 'market_total_max',
            'original_closing_total', 'actual_total_points',
            'market_residual',
        ]
        if c in test.columns
    ]
    out = test[keep].copy()
    out['reg_signal'] = reg.predict(test_work[nums + cats])
    out['over_probability'] = cls.predict_proba(
        test_work[nums + cats]
    )[:, 1]
    return out


def _prepare_test(
    historical: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = historical[historical['season'].lt(TEST_SEASON)].copy()
    test = historical[historical['season'].eq(TEST_SEASON)].copy()
    snap = snapshots[snapshots['lead_hours'].eq(LEAD)].copy()
    test = test.merge(
        snap.drop(
            columns=['season', 'week', 'home_team', 'away_team'],
            errors='ignore',
        ),
        on='game_id',
        how='inner',
    )
    if test.empty:
        return train, test

    test['original_closing_total'] = test['closing_total']
    test['closing_total'] = pd.to_numeric(
        test['decision_total'], errors='coerce'
    )
    test['market_residual'] = (
        pd.to_numeric(test['actual_total_points'], errors='coerce')
        - test['closing_total']
    )
    test['total_bin'] = pd.cut(
        test['closing_total'],
        bins=[0, 38, 42, 46, 50, 54, 100],
        labels=['<=38', '38-42', '42-46', '46-50', '50-54', '54+'],
    )
    return train, test


def _prediction_quality(frame: pd.DataFrame) -> dict:
    actual = pd.to_numeric(frame['market_residual'], errors='coerce')
    reg = pd.to_numeric(frame['reg_signal'], errors='coerce')
    prob = pd.to_numeric(frame['over_probability'], errors='coerce')
    valid_reg = actual.notna() & reg.notna()
    valid_prob = actual.ne(0) & prob.notna()
    binary = actual.gt(0).astype(int)
    return {
        'games': int(len(frame)),
        'mae_residual': (
            mean_absolute_error(actual[valid_reg], reg[valid_reg])
            if valid_reg.any() else np.nan
        ),
        'brier_over': (
            brier_score_loss(binary[valid_prob], prob[valid_prob])
            if valid_prob.any() else np.nan
        ),
    }


def _grade(frame: pd.DataFrame) -> dict:
    residual = pd.to_numeric(frame['market_residual'], errors='coerce')
    wins = int(residual.gt(0).sum())
    losses = int(residual.lt(0).sum())
    pushes = int(residual.eq(0).sum())
    graded = wins + losses
    flat_units = wins * (100 / 110) - losses
    low, high = _wilson(wins, graded)

    consensus_n, consensus_units, consensus_roi = _price_roi(
        frame, 'consensus_over_price'
    )
    best_n, best_units, best_roi = _price_roi(
        frame, 'best_over_price_same_line'
    )

    return {
        'plays': int(len(frame)),
        'graded': graded,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'hit_rate': wins / graded if graded else np.nan,
        'wilson95_low': low,
        'wilson95_high': high,
        'flat_units_minus110': flat_units,
        'flat_roi_minus110': flat_units / graded if graded else np.nan,
        'consensus_price_graded': consensus_n,
        'consensus_price_units': consensus_units,
        'consensus_price_roi': consensus_roi,
        'best_price_graded': best_n,
        'best_price_units': best_units,
        'best_price_roi': best_roi,
        'p_value_vs_minus110': (
            binomtest(
                wins,
                graded,
                p=BREAKEVEN,
                alternative='greater',
            ).pvalue
            if graded else np.nan
        ),
    }


def _paired_bootstrap(
    predictions: pd.DataFrame,
    samples: int = 5000,
    seed: int = 20261004,
) -> pd.DataFrame:
    baseline = predictions[
        predictions['system'].eq('v1_baseline')
    ].set_index('game_id')
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for challenger_name in [
        'reduced_football',
        'reduced_combined',
        'reduced_equal_blend',
    ]:
        challenger = predictions[
            predictions['system'].eq(challenger_name)
        ].set_index('game_id')
        common = baseline.index.intersection(challenger.index)
        if len(common) < 20:
            continue

        base = baseline.loc[common]
        ch = challenger.loc[common]
        actual = pd.to_numeric(base['market_residual'], errors='coerce').to_numpy()
        base_reg = pd.to_numeric(base['reg_signal'], errors='coerce').to_numpy()
        ch_reg = pd.to_numeric(ch['reg_signal'], errors='coerce').to_numpy()
        base_prob = pd.to_numeric(
            base['over_probability'], errors='coerce'
        ).to_numpy()
        ch_prob = pd.to_numeric(
            ch['over_probability'], errors='coerce'
        ).to_numpy()

        valid = (
            np.isfinite(actual)
            & np.isfinite(base_reg)
            & np.isfinite(ch_reg)
            & np.isfinite(base_prob)
            & np.isfinite(ch_prob)
        )
        actual = actual[valid]
        base_reg = base_reg[valid]
        ch_reg = ch_reg[valid]
        base_prob = base_prob[valid]
        ch_prob = ch_prob[valid]
        n = len(actual)
        if n < 20:
            continue

        binary = (actual > 0).astype(float)
        nonpush = actual != 0
        mae_diffs = np.empty(samples)
        brier_diffs = np.empty(samples)

        for i in range(samples):
            idx = rng.integers(0, n, size=n)
            mae_diffs[i] = (
                np.mean(np.abs(actual[idx] - ch_reg[idx]))
                - np.mean(np.abs(actual[idx] - base_reg[idx]))
            )
            keep = nonpush[idx]
            if not keep.any():
                brier_diffs[i] = np.nan
            else:
                sampled = idx[keep]
                brier_diffs[i] = (
                    np.mean((binary[sampled] - ch_prob[sampled]) ** 2)
                    - np.mean((binary[sampled] - base_prob[sampled]) ** 2)
                )

        brier_valid = brier_diffs[np.isfinite(brier_diffs)]
        rows.append({
            'challenger': challenger_name,
            'paired_games': n,
            'mae_diff_challenger_minus_v1': (
                np.mean(np.abs(actual - ch_reg))
                - np.mean(np.abs(actual - base_reg))
            ),
            'mae_diff_ci95_low': float(np.quantile(mae_diffs, 0.025)),
            'mae_diff_ci95_high': float(np.quantile(mae_diffs, 0.975)),
            'bootstrap_probability_mae_better': float(
                np.mean(mae_diffs < 0)
            ),
            'brier_diff_challenger_minus_v1': (
                np.mean((binary[nonpush] - ch_prob[nonpush]) ** 2)
                - np.mean((binary[nonpush] - base_prob[nonpush]) ** 2)
            ),
            'brier_diff_ci95_low': (
                float(np.quantile(brier_valid, 0.025))
                if len(brier_valid) else np.nan
            ),
            'brier_diff_ci95_high': (
                float(np.quantile(brier_valid, 0.975))
                if len(brier_valid) else np.nan
            ),
            'bootstrap_probability_brier_better': (
                float(np.mean(brier_valid < 0))
                if len(brier_valid) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def _write_no_selection(selection: dict) -> None:
    out = ensure_dir('outputs/nfl')
    pd.DataFrame([selection]).to_csv(
        out / 'team_context_phase2_decision_time_status.csv',
        index=False,
    )
    (
        out / 'team_context_phase2_decision_time_2025.md'
    ).write_text(
        '# NFL Phase 2 2025 Decision-Time Holdout\n\n'
        'No reduced football challenger passed the preregistered '
        'pre-2018 selection gates. No post-selection decision-time '
        'model was run.\n',
        encoding='utf-8',
    )


def main() -> None:
    selection = _load_selection()
    selected = selection.get('selected_candidate')
    selected_features = selection.get('selected_features') or []
    if not selected or not selected_features:
        _write_no_selection(selection)
        print(selection['selection_status'])
        return

    historical = read_df(DATA_PATH).copy()
    historical['season'] = pd.to_numeric(
        historical['season'], errors='coerce'
    )
    historical = historical[
        historical['team_context_eligible'].fillna(False).astype(bool)
    ].copy()
    historical = historical.dropna(
        subset=['closing_total', 'actual_total_points', 'market_residual']
    )
    historical = _weather_features(historical, LEAD)
    historical = historical[
        historical[f'forecast_complete_{LEAD}h'].fillna(False)
    ].copy()

    market = _load_market_rows()
    games_2025 = historical[
        historical['season'].eq(TEST_SEASON)
    ].copy()
    snapshots = _build_market_snapshots(games_2025, market)
    train, test = _prepare_test(historical, snapshots)
    if test.empty:
        raise RuntimeError(
            'No 2025 decision-time games matched Phase 2 research rows.'
        )

    feature_sets = _feature_sets(
        pd.concat([train, test], ignore_index=True),
        selected_features,
    )

    parts: list[pd.DataFrame] = []
    by_name: dict[str, pd.DataFrame] = {}
    for name, (nums, cats) in feature_sets.items():
        pred = _fit_predict(train, test, nums, cats)
        pred['system'] = name
        parts.append(pred)
        by_name[name] = pred

    baseline = by_name['v1_baseline'].set_index('game_id')
    football = by_name['reduced_football'].set_index('game_id')
    common = baseline.index.intersection(football.index)
    blend = baseline.loc[common].copy()
    blend['reg_signal'] = (
        baseline.loc[common, 'reg_signal']
        + football.loc[common, 'reg_signal']
    ) / 2.0
    blend['over_probability'] = (
        baseline.loc[common, 'over_probability']
        + football.loc[common, 'over_probability']
    ) / 2.0
    blend['system'] = 'reduced_equal_blend'
    parts.append(blend.reset_index())

    predictions = pd.concat(parts, ignore_index=True)

    quality_rows: list[dict] = []
    qualifier_rows: list[dict] = []
    agreement_rows: list[dict] = []

    for system, group in predictions.groupby('system'):
        quality_rows.append({
            'system': system,
            **_prediction_quality(group),
        })
        for rule, (reg_threshold, prob_threshold) in RULES.items():
            plays = group[_mask(group, reg_threshold, prob_threshold)]
            qualifier_rows.append({
                'system': system,
                'rule': rule,
                **_grade(plays),
            })

    baseline = predictions[
        predictions['system'].eq('v1_baseline')
    ].set_index('game_id')
    for challenger_name in [
        'reduced_football',
        'reduced_combined',
        'reduced_equal_blend',
    ]:
        challenger = predictions[
            predictions['system'].eq(challenger_name)
        ].set_index('game_id')
        common = baseline.index.intersection(challenger.index)
        for rule, (reg_threshold, prob_threshold) in RULES.items():
            base_mask = _mask(
                baseline.loc[common],
                reg_threshold,
                prob_threshold,
            )
            ch_mask = _mask(
                challenger.loc[common],
                reg_threshold,
                prob_threshold,
            )
            ids = common[
                base_mask.to_numpy() & ch_mask.to_numpy()
            ]
            agreement_rows.append({
                'challenger': challenger_name,
                'rule': rule,
                **_grade(baseline.loc[ids].reset_index()),
            })

    quality = pd.DataFrame(quality_rows)
    qualifiers = pd.DataFrame(qualifier_rows)
    agreement = pd.DataFrame(agreement_rows)
    bootstrap = _paired_bootstrap(predictions)

    write_df(
        snapshots,
        'outputs/nfl/team_context_phase2_decision_time_market_2025.csv',
    )
    write_df(
        predictions,
        'outputs/nfl/team_context_phase2_decision_time_predictions_2025.csv',
    )
    write_df(
        quality,
        'outputs/nfl/team_context_phase2_decision_time_quality_2025.csv',
    )
    write_df(
        qualifiers,
        'outputs/nfl/team_context_phase2_decision_time_qualifiers_2025.csv',
    )
    write_df(
        agreement,
        'outputs/nfl/team_context_phase2_decision_time_agreement_2025.csv',
    )
    write_df(
        bootstrap,
        'outputs/nfl/team_context_phase2_decision_time_bootstrap_2025.csv',
    )

    lines = [
        '# NFL Phase 2 Reduced Challenger — 2025 Decision-Time Holdout',
        '',
        f'- Pre-2018 selected candidate: **{selected}**',
        f'- Football feature count: **{len(selected_features)}**',
        '- Market line and price are from the archived 24-hour snapshot.',
        '',
        '## Prediction quality',
        '',
        quality.to_markdown(index=False),
        '',
        '## Fixed qualifier performance',
        '',
        qualifiers.to_markdown(index=False),
        '',
        '## Independent agreement with v1',
        '',
        agreement.to_markdown(index=False),
        '',
        '## Paired bootstrap versus v1',
        '',
        bootstrap.to_markdown(index=False),
    ]
    out = ensure_dir('outputs/nfl') / (
        'team_context_phase2_decision_time_2025.md'
    )
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(
        f'Completed Phase 2 decision-time holdout for {selected}.'
    )


if __name__ == '__main__':
    main()
