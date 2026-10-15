from __future__ import annotations

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
from .team_context_bakeoff import (
    RULES,
    _feature_sets,
    _paired_bootstrap,
    _qualifier_mask,
    _wilson,
)


DATA_PATH = 'data/nfl/processed/team_context_features.csv'
LEAD = 24
TEST_SEASON = 2025


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


def _grade(frame: pd.DataFrame) -> dict:
    residual = pd.to_numeric(frame['market_residual'], errors='coerce')
    wins = int(residual.gt(0).sum())
    losses = int(residual.lt(0).sum())
    pushes = int(residual.eq(0).sum())
    graded = wins + losses
    flat_units = wins * (100 / 110) - losses
    low, high = _wilson(wins, graded)
    consensus_n, consensus_units, consensus_roi = _price_roi(
        frame,
        'consensus_over_price',
    )
    best_n, best_units, best_roi = _price_roi(
        frame,
        'best_over_price_same_line',
    )
    return {
        'games': int(len(frame)),
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


def main() -> None:
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
    games_2025 = historical[historical['season'].eq(TEST_SEASON)].copy()
    snapshots = _build_market_snapshots(games_2025, market)
    train, test = _prepare_test(historical, snapshots)
    if test.empty:
        raise RuntimeError(
            'No 2025 decision-time games matched team-context research rows.'
        )

    feature_sets = _feature_sets(pd.concat([train, test], ignore_index=True))
    prediction_parts: list[pd.DataFrame] = []
    by_name: dict[str, pd.DataFrame] = {}

    for name, (nums, cats) in feature_sets.items():
        pred = _fit_predict(train, test, nums, cats)
        pred['system'] = name
        prediction_parts.append(pred)
        by_name[name] = pred

    baseline = by_name['v1_baseline'].set_index('game_id')
    football = by_name['football_context'].set_index('game_id')
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
    blend['system'] = 'equal_blend'
    prediction_parts.append(blend.reset_index())

    predictions = pd.concat(prediction_parts, ignore_index=True)

    quality_rows = []
    qualifier_rows = []
    agreement_rows = []

    for system, group in predictions.groupby('system'):
        quality_rows.append({
            'system': system,
            **_prediction_quality(group),
        })
        for rule, (reg_threshold, prob_threshold) in RULES.items():
            plays = group[
                _qualifier_mask(group, reg_threshold, prob_threshold)
            ].copy()
            qualifier_rows.append({
                'system': system,
                'rule': rule,
                'reg_threshold': reg_threshold,
                'probability_threshold': prob_threshold,
                **_grade(plays),
            })

    baseline = predictions[
        predictions['system'].eq('v1_baseline')
    ].set_index('game_id')
    for challenger_name in [
        'football_context', 'combined', 'equal_blend'
    ]:
        challenger = predictions[
            predictions['system'].eq(challenger_name)
        ].set_index('game_id')
        common = baseline.index.intersection(challenger.index)
        for rule, (reg_threshold, prob_threshold) in RULES.items():
            base_mask = _qualifier_mask(
                baseline.loc[common],
                reg_threshold,
                prob_threshold,
            )
            ch_mask = _qualifier_mask(
                challenger.loc[common],
                reg_threshold,
                prob_threshold,
            )
            ids = common[base_mask.to_numpy() & ch_mask.to_numpy()]
            plays = baseline.loc[ids].reset_index()
            agreement_rows.append({
                'challenger': challenger_name,
                'rule': rule,
                **_grade(plays),
            })

    quality = pd.DataFrame(quality_rows)
    qualifier = pd.DataFrame(qualifier_rows)
    agreement = pd.DataFrame(agreement_rows)
    bootstrap = _paired_bootstrap(
        predictions,
        'decision_time_2025',
        (TEST_SEASON,),
    )

    write_df(
        snapshots,
        'outputs/nfl/team_context_decision_time_market_snapshots_2025.csv',
    )
    write_df(
        predictions,
        'outputs/nfl/team_context_decision_time_predictions_2025.csv',
    )
    write_df(
        quality,
        'outputs/nfl/team_context_decision_time_quality_2025.csv',
    )
    write_df(
        qualifier,
        'outputs/nfl/team_context_decision_time_qualifiers_2025.csv',
    )
    write_df(
        agreement,
        'outputs/nfl/team_context_decision_time_agreement_2025.csv',
    )
    write_df(
        bootstrap,
        'outputs/nfl/team_context_decision_time_bootstrap_2025.csv',
    )

    lines = [
        '# NFL Team-Context Challenger — 2025 Decision-Time Holdout',
        '',
        'This test uses the archived 24-hour sportsbook total and price '
        'available before kickoff, plus the archived 24-hour JMA forecast.',
        '',
        'The frozen v1 system remains unchanged. Historical training targets '
        'still use final closing-market residuals because multi-season '
        'decision-time market archives are not available in the repository.',
        '',
        '## Prediction quality',
        '',
        quality.to_markdown(index=False),
        '',
        '## Fixed qualifier performance',
        '',
        qualifier.to_markdown(index=False),
        '',
        '## Independent agreement with v1',
        '',
        agreement.to_markdown(index=False),
        '',
        '## Paired bootstrap versus v1',
        '',
        (
            bootstrap.to_markdown(index=False)
            if not bootstrap.empty
            else '_No bootstrap comparisons._'
        ),
    ]
    out = ensure_dir('outputs/nfl') / (
        'team_context_decision_time_holdout_2025.md'
    )
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote team-context decision-time holdout under {out.parent}')


if __name__ == '__main__':
    main()
