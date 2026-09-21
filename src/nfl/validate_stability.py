from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from ..utils import ensure_dir, load_yaml, read_df, write_df
from .model_bakeoff import (
    BREAKEVEN,
    CONTEXT_CATS,
    CONTEXT_NUMS,
    WEATHER_CATS,
    WEATHER_NUMS,
    _available_categorical,
    _available_numeric,
    _models,
    _prep,
    _settle,
)


def _bh_qvalues(p_values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    m = len(valid)
    if not m:
        return out
    running = 1.0
    adjusted: dict[int, float] = {}
    for rank in range(m, 0, -1):
        idx = valid.index[rank - 1]
        value = min(running, float(valid.loc[idx]) * m / rank)
        adjusted[idx] = value
        running = value
    for idx, value in adjusted.items():
        out.loc[idx] = value
    return out


def _wilson(wins: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    z = 1.96
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def _add_inference(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out['p_value_vs_minus110'] = out.apply(
        lambda row: (
            binomtest(
                int(row['wins']),
                int(row['graded']),
                p=BREAKEVEN,
                alternative='greater',
            ).pvalue
            if pd.notna(row.get('graded')) and int(row['graded']) > 0
            else np.nan
        ),
        axis=1,
    )
    out['q_value_bh'] = _bh_qvalues(out['p_value_vs_minus110'])
    intervals = out.apply(
        lambda row: _wilson(int(row['wins']), int(row['graded']))
        if pd.notna(row.get('graded')) and int(row['graded']) > 0
        else (np.nan, np.nan),
        axis=1,
    )
    out['wilson_low'] = [value[0] for value in intervals]
    out['wilson_high'] = [value[1] for value in intervals]
    return out


def _candidate_season_rows(
    df: pd.DataFrame,
    feature_set: str,
    model_name: str,
    threshold: float,
    min_train: int,
    min_test: int,
) -> pd.DataFrame:
    if feature_set == 'context_plus_weather':
        nums = _available_numeric(df, CONTEXT_NUMS + WEATHER_NUMS)
        cats = _available_categorical(df, CONTEXT_CATS + WEATHER_CATS)
    else:
        nums = _available_numeric(df, CONTEXT_NUMS)
        cats = _available_categorical(df, CONTEXT_CATS)

    work = _prep(df, nums, cats)
    rows: list[dict] = []
    for season in sorted(work['season'].dropna().astype(int).unique()):
        train = work[work['season'] < season].copy()
        test = work[work['season'] == season].copy()
        if len(train) < min_train or len(test) < min_test:
            continue
        model = _models(nums, cats)[model_name]
        model.fit(train[nums + cats], train['market_residual'])
        test = test.assign(pred_market_residual=model.predict(test[nums + cats]))
        plays = test[test['pred_market_residual'].abs() >= threshold].copy()
        if plays.empty:
            rows.append({
                'season': int(season), 'games': 0, 'graded': 0, 'wins': 0,
                'losses': 0, 'pushes': 0, 'hit_rate': np.nan,
                'roi_per_1u': np.nan, 'net_units_1u_each': 0.0,
            })
            continue
        result = _settle(plays)
        wins = int(result.eq('win').sum())
        losses = int(result.eq('loss').sum())
        graded = wins + losses
        units = float(result.map({'win': 100 / 110, 'loss': -1.0, 'push': 0.0}).sum())
        rows.append({
            'season': int(season),
            'games': int(len(plays)),
            'graded': graded,
            'wins': wins,
            'losses': losses,
            'pushes': int(result.eq('push').sum()),
            'hit_rate': wins / graded if graded else np.nan,
            'roi_per_1u': units / graded if graded else np.nan,
            'net_units_1u_each': units,
        })
    return pd.DataFrame(rows)


def _rule_by_season(df: pd.DataFrame) -> pd.DataFrame:
    outdoor = df[df['outdoor'].fillna(False)].copy()
    rule_defs = {
        'wind_10plus_under': outdoor['wind_mph'] >= 10,
        'wind_15plus_under': outdoor['wind_mph'] >= 15,
        'freezing_under': outdoor['temperature_f'] <= 32,
        'cold_windy_under': (outdoor['temperature_f'] <= 40) & (outdoor['wind_mph'] >= 12),
    }
    rows: list[dict] = []
    for name, mask in rule_defs.items():
        sample = outdoor.loc[mask.fillna(False)].copy()
        for season, group in sample.groupby('season'):
            residual = group['market_residual']
            wins = int((residual < 0).sum())
            losses = int((residual > 0).sum())
            pushes = int((residual == 0).sum())
            graded = wins + losses
            units = wins * (100 / 110) - losses
            rows.append({
                'rule': name,
                'season': int(season),
                'games': int(len(group)),
                'graded': graded,
                'wins': wins,
                'losses': losses,
                'pushes': pushes,
                'hit_rate': wins / graded if graded else np.nan,
                'roi_per_1u': units / graded if graded else np.nan,
                'avg_market_residual': float(residual.mean()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    settings = load_yaml('config/nfl_settings.yml')
    model_cfg = settings['modeling']
    df = read_df(settings['data']['processed_path']).copy()
    df = df.dropna(subset=['season', 'closing_total', 'actual_total_points', 'market_residual'])

    summary = _add_inference(read_df('outputs/nfl/model_bakeoff_summary.csv'))
    write_df(summary, 'outputs/nfl/model_multiple_testing_validation.csv')

    rules = _add_inference(read_df('outputs/nfl/simple_rule_screen.csv'))
    write_df(rules, 'outputs/nfl/rule_multiple_testing_validation.csv')

    diagnostics = read_df('outputs/nfl/model_bakeoff_diagnostics.csv')
    paired = diagnostics.pivot_table(
        index=['model', 'test_season'],
        columns='feature_set',
        values='model_mae',
    ).reset_index()
    paired['weather_minus_context_mae'] = paired['context_plus_weather'] - paired['context_only']
    incremental = (
        paired.groupby('model', as_index=False)
        .agg(
            seasons=('test_season', 'count'),
            mean_weather_minus_context_mae=('weather_minus_context_mae', 'mean'),
            median_weather_minus_context_mae=('weather_minus_context_mae', 'median'),
            seasons_weather_improved=(
                'weather_minus_context_mae',
                lambda values: int((values < 0).sum()),
            ),
        )
        .sort_values('mean_weather_minus_context_mae')
    )
    write_df(incremental, 'outputs/nfl/weather_increment_by_model.csv')

    eligible = summary[
        (pd.to_numeric(summary['graded'], errors='coerce') >= 100)
        & pd.to_numeric(summary['roi_per_1u'], errors='coerce').notna()
    ].copy()
    candidate = eligible.sort_values(
        ['roi_per_1u', 'graded'],
        ascending=[False, False],
    ).iloc[0]

    candidate_seasons = _candidate_season_rows(
        df,
        str(candidate['feature_set']),
        str(candidate['model']),
        float(candidate['threshold']),
        int(model_cfg.get('min_train_games', 1000)),
        int(model_cfg.get('min_test_games', 150)),
    )
    write_df(candidate_seasons, 'outputs/nfl/top_candidate_by_season.csv')

    rule_seasons = _rule_by_season(df)
    write_df(rule_seasons, 'outputs/nfl/weather_rules_by_season.csv')

    played_seasons = candidate_seasons[candidate_seasons['graded'] > 0].copy()
    positive_roi_seasons = int((played_seasons['roi_per_1u'] > 0).sum())
    above_break_even_seasons = int((played_seasons['hit_rate'] > BREAKEVEN).sum())

    lines = [
        '# NFL Statistical / Stability Validation',
        '',
        'This layer is deliberately skeptical: aggregate strategy rows are tested against the -110 break-even win probability, false-discovery-rate adjusted across the full model/threshold screen, and the top exploratory candidate is re-expressed season by season.',
        '',
        '## Top exploratory candidate',
        '',
        f"- Feature set: {candidate['feature_set']}",
        f"- Model: {candidate['model']}",
        f"- Threshold: {float(candidate['threshold']):.1f} predicted residual points",
        f"- Aggregate record: {int(candidate['wins'])}-{int(candidate['losses'])}-{int(candidate['pushes'])}",
        f"- Hit rate: {float(candidate['hit_rate']):.3%}",
        f"- Flat-stake ROI: {float(candidate['roi_per_1u']):.3%}",
        f"- One-sided exact p-value vs -110 break-even: {float(candidate['p_value_vs_minus110']):.4f}",
        f"- Benjamini-Hochberg q-value across screened model/threshold rows: {float(candidate['q_value_bh']):.4f}",
        '',
        '## Season stability',
        '',
        f"- Seasons with qualifying plays: {len(played_seasons)}",
        f"- Positive-ROI seasons: {positive_roi_seasons}",
        f"- Seasons above -110 break-even hit rate: {above_break_even_seasons}",
        '',
        candidate_seasons.to_markdown(index=False),
        '',
        '## Weather increment by model family',
        '',
        incremental.to_markdown(index=False),
        '',
        '## Interpretation guardrail',
        '',
        '- A visually attractive ROI is not enough. A candidate should not be promoted on this development sample when its exact test/FDR-adjusted evidence remains weak or season stability is inconsistent.',
        '- The same dataset was used to discover the candidate, so this is still development evidence rather than a pristine holdout.',
        '- Richer historical weather and forecast-at-lead-time reconstruction remain required before any live NFL qualifier is frozen.',
        '- Prospective 2026 tracking should be the final protection against research overfit.',
    ]
    out = ensure_dir('outputs/nfl') / 'statistical_validation.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote NFL statistical validation outputs under {out.parent}')


if __name__ == '__main__':
    main()
