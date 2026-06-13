from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import ensure_dir, read_df, write_df

BREAKEVEN = 110 / 210


def as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().map({'true': True, '1': True, 'yes': True, 'false': False, '0': False, 'no': False}).fillna(False).astype(bool)


def units(result: str) -> float:
    if result == 'win':
        return 100 / 110
    if result == 'loss':
        return -1.0
    return 0.0


def settle(actual: pd.Series, total: pd.Series, side) -> pd.Series:
    diff = actual - total
    if isinstance(side, str):
        win = diff < 0 if side == 'under' else diff > 0
    else:
        side = side.reindex(actual.index)
        win = ((side == 'under') & (diff < 0)) | ((side == 'over') & (diff > 0))
    return pd.Series(np.where(diff == 0, 'push', np.where(win, 'win', 'loss')), index=actual.index)


def wilson(wins: int, n: int) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    z = 1.96
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return c - m, c + m


def summarize(name: str, side: str, df: pd.DataFrame, result: pd.Series) -> dict:
    graded = result[result != 'push']
    wins = int((graded == 'win').sum())
    losses = int((graded == 'loss').sum())
    pushes = int((result == 'push').sum())
    n = wins + losses
    low, high = wilson(wins, n)
    net = float(result.map(units).sum()) if len(result) else 0.0
    hit = wins / n if n else np.nan
    return {
        'name': name,
        'side': side,
        'games': int(len(df)),
        'graded': n,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'hit_rate': hit,
        'hit_rate_minus_breakeven': hit - BREAKEVEN if n else np.nan,
        'hit_rate_wilson_low': low,
        'hit_rate_wilson_high': high,
        'net_units_1u_each': net,
        'roi_per_1u': net / n if n else np.nan,
        'avg_market_residual': float(df['market_residual'].mean()) if len(df) else np.nan,
        'median_market_residual': float(df['market_residual'].median()) if len(df) else np.nan,
    }


def prep(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.dropna(subset=['closing_total', 'actual_total_points', 'season']).copy()
    for c in ['closing_total', 'actual_total_points', 'market_residual', 'wind_mph', 'temperature_f', 'humidity', 'precipitation', 'snowfall', 'dewpoint_f', 'pressure']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan
    df['game_indoors_bool'] = as_bool(df['game_indoors']) if 'game_indoors' in df.columns else False
    df['outdoor'] = ~df['game_indoors_bool']
    df['precip_flag'] = df['precipitation'].fillna(0) > 0
    df['snow_flag'] = df['snowfall'].fillna(0) > 0
    if {'home_classification', 'away_classification'} <= set(df.columns):
        df['fbs_vs_fbs'] = (df['home_classification'].astype(str).str.lower() == 'fbs') & (df['away_classification'].astype(str).str.lower() == 'fbs')
    else:
        df['fbs_vs_fbs'] = False
    df['total_bin'] = pd.cut(df['closing_total'], [0, 42, 49, 56, 63, 100], labels=['<=42', '42-49', '49-56', '56-63', '63+'])
    df['weather_sample'] = df['temperature_f'].notna() | df['wind_mph'].notna()
    return df


def quality(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby('season').agg(
        games=('game_id', 'count'),
        with_weather=('weather_sample', 'sum'),
        avg_total=('closing_total', 'mean'),
        avg_actual_total=('actual_total_points', 'mean'),
        avg_residual=('market_residual', 'mean'),
    ).reset_index()


def groups(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ['wind_bin', 'temp_bin', 'total_bin', 'game_indoors_bool', 'precip_flag', 'snow_flag']:
        if col not in df.columns:
            continue
        for val, g in df.dropna(subset=[col]).groupby(col, observed=True):
            if len(g) < 25:
                continue
            rows.append({
                'grouping': col,
                'value': str(val),
                'games': len(g),
                'avg_total': g['closing_total'].mean(),
                'avg_actual': g['actual_total_points'].mean(),
                'avg_residual': g['market_residual'].mean(),
                'under_rate': g['went_under'].mean(),
                'over_rate': g['went_over'].mean(),
            })
    return pd.DataFrame(rows).sort_values(['grouping', 'avg_residual'])


def rule_tests(df: pd.DataFrame) -> pd.DataFrame:
    rulebook = [
        ('outdoor_wind_15_under', 'under', df['outdoor'] & (df['wind_mph'] >= 15)),
        ('outdoor_wind_20_under', 'under', df['outdoor'] & (df['wind_mph'] >= 20)),
        ('outdoor_wind_15_total_49plus_under', 'under', df['outdoor'] & (df['wind_mph'] >= 15) & (df['closing_total'] >= 49)),
        ('cold_35_under', 'under', df['outdoor'] & (df['temperature_f'] <= 35)),
        ('cold_40_wind_12_under', 'under', df['outdoor'] & (df['temperature_f'] <= 40) & (df['wind_mph'] >= 12)),
        ('precip_any_under', 'under', df['outdoor'] & df['precip_flag']),
        ('snow_any_under', 'under', df['outdoor'] & df['snow_flag']),
        ('calm_mild_over', 'over', df['outdoor'] & (df['wind_mph'] <= 5) & df['temperature_f'].between(50, 80) & (~df['precip_flag'])),
        ('indoor_over', 'over', df['game_indoors_bool']),
    ]
    rows = []
    for name, side, mask in rulebook:
        sub = df[mask.fillna(False)].copy()
        rows.append(summarize(name, side, sub, settle(sub['actual_total_points'], sub['closing_total'], side)))
    return pd.DataFrame(rows).sort_values('roi_per_1u', ascending=False)


def walk_forward(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    nums = ['closing_total', 'wind_mph', 'temperature_f', 'humidity', 'precipitation', 'snowfall', 'dewpoint_f', 'pressure']
    cats = ['game_indoors_bool', 'neutral_site', 'conference_game', 'line_provider', 'wind_bin', 'temp_bin', 'total_bin', 'fbs_vs_fbs']
    for c in cats:
        if c not in df.columns:
            df[c] = 'missing'
    model = Pipeline([
        ('prep', ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), nums),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cats),
        ])),
        ('ridge', Ridge(alpha=10.0)),
    ])
    pred_parts, diag = [], []
    for season in sorted(df['season'].dropna().astype(int).unique()):
        train = df[df['season'] < season].copy()
        test = df[df['season'] == season].copy()
        if len(train) < 1000 or len(test) < 100:
            continue
        model.fit(train[nums + cats], train['market_residual'])
        pred = model.predict(test[nums + cats])
        test = test.assign(pred_market_residual=pred)
        pred_parts.append(test)
        diag.append({
            'test_season': season,
            'train_games': len(train),
            'test_games': len(test),
            'model_mae': mean_absolute_error(test['market_residual'], pred),
            'zero_residual_baseline_mae': mean_absolute_error(test['market_residual'], np.zeros(len(test))),
            'avg_pred_residual': float(np.mean(pred)),
            'avg_actual_residual': float(test['market_residual'].mean()),
        })
    if not pred_parts:
        return pd.DataFrame(), pd.DataFrame(diag)
    pred_df = pd.concat(pred_parts, ignore_index=True)
    rows = []
    for th in [1.5, 2.5, 3.5, 5.0]:
        plays = pred_df[pred_df['pred_market_residual'].abs() >= th].copy()
        if plays.empty:
            rows.append({'threshold_points': th, 'games': 0})
            continue
        plays['side'] = np.where(plays['pred_market_residual'] > 0, 'over', 'under')
        row = summarize(f'walk_forward_ridge_abs_edge_{th}', 'model', plays, settle(plays['actual_total_points'], plays['closing_total'], plays['side']))
        row['threshold_points'] = th
        row['test_seasons'] = f"{int(pred_df['season'].min())}-{int(pred_df['season'].max())}"
        row['avg_abs_pred_edge'] = float(plays['pred_market_residual'].abs().mean())
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(diag)


def write_summary(df: pd.DataFrame, q: pd.DataFrame, g: pd.DataFrame, r: pd.DataFrame, wf: pd.DataFrame, diag: pd.DataFrame) -> None:
    lines = [
        '# Deep CFB Weather Totals Research Summary',
        '',
        'Target: `market_residual = actual_total_points - closing_total`.',
        '',
        f"Games with totals used for analysis: {len(df):,}",
        f"Seasons covered: {int(df['season'].min())}-{int(df['season'].max())}",
        '',
        '## Data quality by season', '', q.to_markdown(index=False), '',
        '## Weather group summary', '', g.head(30).to_markdown(index=False), '',
        '## Simple rule tests', '', r.to_markdown(index=False), '',
        '## Walk-forward model test', '', wf.to_markdown(index=False) if not wf.empty else '_No walk-forward rows._', '',
        '## Model diagnostics', '', diag.to_markdown(index=False) if not diag.empty else '_No diagnostics._', '',
        '## Defensibility notes', '',
        '- Rule tables are in-sample screening tools, not final evidence.',
        '- Walk-forward results matter more because each season is tested using only prior seasons.',
        '- Any positive rule still needs sensitivity checks by provider, season, conference, total range, and weather data quality.',
        '- Keep weekly outputs paper-tracking only until an out-of-sample edge is durable.',
    ]
    out = ensure_dir('outputs') / 'deep_research_summary.md'
    out.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    df = prep(read_df('data/processed/modeling_dataset.csv'))
    q = quality(df)
    g = groups(df)
    r = rule_tests(df)
    wf, diag = walk_forward(df)
    write_df(q, 'outputs/data_quality_summary.csv')
    write_df(g, 'outputs/weather_group_summary_deep.csv')
    write_df(r, 'outputs/rule_backtest_detailed.csv')
    write_df(wf, 'outputs/walk_forward_strategy_summary.csv')
    write_df(diag, 'outputs/model_diagnostics.csv')
    write_summary(df, q, g, r, wf, diag)
    print('Wrote deep research outputs')


if __name__ == '__main__':
    main()
