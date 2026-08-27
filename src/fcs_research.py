from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .fcs_model import (
    FCS_CATEGORICAL_FEATURES,
    FCS_NUMERIC_FEATURES,
    FCS_QUALIFY_EDGE,
    FCS_QUALIFY_TOTAL,
    build_fcs_model,
    historical_fcs_training,
)
from .utils import ensure_dir, write_df

BREAKEVEN = 110 / 210
EDGE_GRID = [1.5, 2.5, 3.5, 5.0, 6.0, 7.5]
TOTAL_GRID = [0.0, 49.0, 52.0, 54.0, 56.0, 58.0, 60.0]


def _grade_under(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {'games': 0, 'graded': 0, 'wins': 0, 'losses': 0, 'pushes': 0, 'hit_rate': np.nan, 'roi_per_1u': np.nan}
    diff = pd.to_numeric(frame['actual_total_points'], errors='coerce') - pd.to_numeric(frame['closing_total'], errors='coerce')
    pushes = int(diff.eq(0).sum())
    wins = int(diff.lt(0).sum())
    losses = int(diff.gt(0).sum())
    graded = wins + losses
    net = wins * (100 / 110) - losses
    return {
        'games': int(len(frame)),
        'graded': graded,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'hit_rate': wins / graded if graded else np.nan,
        'roi_per_1u': net / graded if graded else np.nan,
    }


def _wilson(wins: int, n: int) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    z = 1.96
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return c - m, c + m


def walk_forward_predictions(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    features = FCS_NUMERIC_FEATURES + FCS_CATEGORICAL_FEATURES

    for season in sorted(data['season'].dropna().astype(int).unique()):
        train = data[data['season'] < season].copy()
        test = data[data['season'] == season].copy()
        # FCS market totals are only present from 2022 onward in the current dataset.
        # Requiring 500 prior games allows 2023-2025 to be tested walk-forward.
        if len(train) < 500 or len(test) < 100:
            continue
        model = build_fcs_model()
        model.fit(train[features], train['market_residual'])
        pred = model.predict(test[features])
        test = test.assign(pred_market_residual=pred)
        predictions.append(test)
        diagnostics.append({
            'test_season': season,
            'train_games': int(len(train)),
            'test_games': int(len(test)),
            'model_mae': float(mean_absolute_error(test['market_residual'], pred)),
            'zero_residual_baseline_mae': float(mean_absolute_error(test['market_residual'], np.zeros(len(test)))),
        })

    return (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.DataFrame(diagnostics),
    )


def threshold_grid(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for edge in EDGE_GRID:
        for minimum_total in TOTAL_GRID:
            mask = predictions['pred_market_residual'].le(-edge)
            if minimum_total > 0:
                mask &= predictions['closing_total'].ge(minimum_total)
            graded = _grade_under(predictions[mask].copy())
            rows.append({
                'under_edge_threshold': edge,
                'minimum_total': minimum_total,
                **graded,
            })
    return pd.DataFrame(rows).sort_values(['roi_per_1u', 'graded'], ascending=[False, False])


def selected_screen_by_season(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, group in predictions.groupby('season'):
        sub = group[
            group['pred_market_residual'].le(-FCS_QUALIFY_EDGE)
            & group['closing_total'].ge(FCS_QUALIFY_TOTAL)
        ].copy()
        rows.append({'season': int(season), **_grade_under(sub)})
    return pd.DataFrame(rows)


def data_summary(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby('season', as_index=False).agg(
        games_with_totals=('game_id', 'count'),
        avg_market_total=('closing_total', 'mean'),
        avg_market_residual=('market_residual', 'mean'),
        weather_games=('wind_mph', lambda s: int(s.notna().sum())),
    )


def write_markdown(
    data: pd.DataFrame,
    diagnostics: pd.DataFrame,
    grid: pd.DataFrame,
    by_season: pd.DataFrame,
) -> None:
    selected = grid[
        grid['under_edge_threshold'].eq(FCS_QUALIFY_EDGE)
        & grid['minimum_total'].eq(FCS_QUALIFY_TOTAL)
    ].iloc[0]
    wins = int(selected['wins'])
    graded = int(selected['graded'])
    low, high = _wilson(wins, graded)

    lines = [
        '# FCS Weather Totals Research',
        '',
        'This track uses **FCS-vs-FCS games only** and predicts `actual_total_points - closing_total`.',
        '',
        f'Historical FCS games with usable totals/results: **{len(data):,}**.',
        f'Walk-forward test seasons: **{int(by_season.season.min())}-{int(by_season.season.max())}**.' if not by_season.empty else 'No walk-forward seasons available.',
        '',
        '## Conservative FCS candidate screen',
        '',
        f'- Dedicated FCS HistGradientBoosting model',
        f'- Predicted UNDER edge: **{FCS_QUALIFY_EDGE:.1f}+ points**',
        f'- Market total: **{FCS_QUALIFY_TOTAL:.0f}+**',
        f'- Walk-forward record: **{wins}-{int(selected["losses"])}** ({selected["hit_rate"]:.1%})',
        f'- Paper ROI at -110: **{selected["roi_per_1u"]:.1%} per graded play**',
        f'- 95% Wilson interval for hit rate: **{low:.1%} to {high:.1%}**',
        '',
        '### By season',
        '',
        by_season.to_markdown(index=False) if not by_season.empty else '_No rows._',
        '',
        '## Model diagnostics',
        '',
        diagnostics.to_markdown(index=False) if not diagnostics.empty else '_No rows._',
        '',
        '## Threshold sensitivity',
        '',
        grid.head(20).to_markdown(index=False),
        '',
        '## Guardrails',
        '',
        '- The threshold grid is a research screen, so the selected rule is not treated as a guaranteed edge.',
        '- FCS market data in the current historical dataset begins in 2022, which limits the number of independent test seasons.',
        '- The FCS model MAE can be worse than a zero-residual baseline even when selective under subsets perform well; this is why the live site uses a strict screen rather than every model prediction.',
        '- Keep the FCS track in paper-tracking mode while 2026 live forecasts, closing totals, and results accumulate.',
        '- No FCS over strategy is promoted from this analysis.',
    ]
    out = ensure_dir('outputs') / 'fcs_research_summary.md'
    out.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    data = historical_fcs_training()
    if data.empty:
        raise RuntimeError('No historical FCS-vs-FCS training data is available.')
    predictions, diagnostics = walk_forward_predictions(data)
    if predictions.empty:
        raise RuntimeError('FCS walk-forward validation produced no test seasons.')

    grid = threshold_grid(predictions)
    by_season = selected_screen_by_season(predictions)
    summary = data_summary(data)

    write_df(summary, 'outputs/fcs_data_summary.csv')
    write_df(diagnostics, 'outputs/fcs_model_diagnostics.csv')
    write_df(grid, 'outputs/fcs_threshold_grid.csv')
    write_df(by_season, 'outputs/fcs_selected_screen_by_season.csv')
    write_markdown(data, diagnostics, grid, by_season)
    print(
        f'FCS research: {len(data)} historical games with totals; '
        f'{len(predictions)} walk-forward predictions.'
    )


if __name__ == '__main__':
    main()
