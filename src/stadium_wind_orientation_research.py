from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .deep_research import prep, settle, summarize
from .model_bakeoff import feature_lists, prep_features, reg_models
from .utils import ROOT, ensure_dir, read_df, write_df

ORIENTATION_FILE = ROOT / 'data/reference/stadium_orientations.csv'
OUT_DIR = ROOT / 'outputs/orientation_research'
QUALIFY_EDGE = 3.5
QUALIFY_TOTAL = 56.0
VARIANTS = {
    'baseline': [],
    'crosswind': ['crosswind_mph'],
    'components': ['crosswind_mph', 'alongwind_mph'],
}


def division_track(df: pd.DataFrame) -> pd.Series:
    h = df.get('home_classification', pd.Series('', index=df.index)).astype(str).str.lower()
    a = df.get('away_classification', pd.Series('', index=df.index)).astype(str).str.lower()
    return pd.Series(np.select(
        [h.eq('fbs') & a.eq('fbs'), h.eq('fcs') & a.eq('fcs')],
        ['FBS', 'FCS'], default='OTHER'), index=df.index)


def load_data() -> pd.DataFrame:
    df = read_df('data/processed/modeling_dataset.csv').copy()
    ori = pd.read_csv(ORIENTATION_FILE)
    for c in ['venue_id', 'field_axis_deg', 'axis_uncertainty_deg']:
        ori[c] = pd.to_numeric(ori.get(c), errors='coerce')
    measured = ori['field_axis_deg'].notna()
    if ori.loc[measured, 'venue_id'].duplicated().any():
        raise RuntimeError('Measured orientation venue_id values must be unique.')
    if (~ori.loc[measured, 'field_axis_deg'].between(0, 180, inclusive='left')).any():
        raise RuntimeError('field_axis_deg must be in [0, 180).')
    ori = ori[~ori['roof_behavior'].astype(str).str.lower().eq('indoor')]

    df['venue_id'] = pd.to_numeric(df.get('venue_id'), errors='coerce')
    df['wind_direction_degrees'] = pd.to_numeric(df.get('wind_direction_degrees'), errors='coerce')
    df = df.merge(ori, on='venue_id', how='left')
    df = prep(df)
    df['division_track'] = division_track(df)
    df['wind_direction_degrees'] = pd.to_numeric(df.get('wind_direction_degrees'), errors='coerce')

    wind_axis = df['wind_direction_degrees'] % 180.0
    field_axis = df['field_axis_deg'] % 180.0
    delta = (wind_axis - field_axis).abs()
    df['wind_field_angle_deg'] = np.minimum(delta, 180.0 - delta)
    rad = np.deg2rad(df['wind_field_angle_deg'])
    df['crosswind_mph'] = df['wind_mph'] * np.sin(rad)
    df['alongwind_mph'] = df['wind_mph'] * np.cos(rad)
    df['alignment_bin'] = pd.cut(
        df['wind_field_angle_deg'], [-0.001, 30, 60, 90.001],
        labels=['parallel_0_30', 'oblique_30_60', 'cross_60_90'])
    return df


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work['usable'] = (
        work['field_axis_deg'].notna()
        & work['wind_mph'].notna()
        & work['wind_direction_degrees'].notna())
    rows = []
    for track, g in work.groupby('division_track'):
        rows.append({'season': 'ALL', 'division_track': track,
                     'games': len(g), 'usable': int(g['usable'].sum()),
                     'usable_pct': float(g['usable'].mean())})
        for season, s in g.groupby('season'):
            rows.append({'season': int(season), 'division_track': track,
                         'games': len(s), 'usable': int(s['usable'].sum()),
                         'usable_pct': float(s['usable'].mean())})
    return pd.DataFrame(rows)


def descriptive(fbs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (wb, ab), g in fbs.dropna(subset=['wind_bin', 'alignment_bin']).groupby(
            ['wind_bin', 'alignment_bin'], observed=True):
        rows.append({
            'wind_bin': str(wb), 'alignment_bin': str(ab), 'games': len(g),
            'avg_wind_mph': float(g['wind_mph'].mean()),
            'avg_crosswind_mph': float(g['crosswind_mph'].mean()),
            'avg_market_residual': float(g['market_residual'].mean()),
            'under_rate': float(g['went_under'].mean()),
        })
    return pd.DataFrame(rows)


def qualifier(pred: pd.DataFrame, variant: str) -> dict:
    q = pred[(pred['pred_market_residual'] <= -QUALIFY_EDGE)
             & (pred['closing_total'] >= QUALIFY_TOTAL)].copy()
    row = summarize(variant, 'under', q, settle(
        q['actual_total_points'], q['closing_total'], 'under'))
    row['variant'] = variant
    return row


def walk_forward(fbs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_nums, cats = feature_lists(fbs)
    fbs = prep_features(fbs.copy(), cats)
    pred_parts = {name: [] for name in VARIANTS}
    diag = []

    for season in sorted(fbs['season'].dropna().astype(int).unique()):
        train = fbs[fbs['season'] < season].copy()
        test = fbs[fbs['season'] == season].copy()
        if len(train) < 1000 or len(test) < 100:
            continue
        d = {'test_season': season, 'train_games': len(train), 'test_games': len(test)}
        for name, extras in VARIANTS.items():
            nums = base_nums + [c for c in extras if c not in base_nums]
            model = reg_models(nums, cats)['hist_gradient_boosting']
            model.fit(train[nums + cats], train['market_residual'])
            scored = test.copy()
            scored['pred_market_residual'] = model.predict(test[nums + cats])
            pred_parts[name].append(scored)
            d[f'{name}_mae'] = mean_absolute_error(
                scored['market_residual'], scored['pred_market_residual'])
        diag.append(d)

    overall, by_season = [], []
    for name, parts in pred_parts.items():
        if not parts:
            continue
        pred = pd.concat(parts, ignore_index=True)
        r = qualifier(pred, name)
        r['test_season'] = 'ALL'
        r['mae'] = mean_absolute_error(pred['market_residual'], pred['pred_market_residual'])
        overall.append(r)
        for season, g in pred.groupby('season'):
            r = qualifier(g, name)
            r['test_season'] = int(season)
            r['mae'] = mean_absolute_error(g['market_residual'], g['pred_market_residual'])
            by_season.append(r)
    return pd.DataFrame(diag), pd.DataFrame(overall), pd.DataFrame(by_season)


def direction_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, g in df.dropna(subset=['wind_direction_degrees']).groupby('season'):
        x = pd.to_numeric(g['wind_direction_degrees'], errors='coerce').dropna()
        rows.append({
            'season': int(season), 'rows': len(x), 'unique_values': int(x.nunique()),
            'fraction_multiple_10deg': float(np.isclose(x % 10, 0).mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dir(OUT_DIR)
    df = load_data()
    cov = coverage(df)
    common = df[
        df['field_axis_deg'].notna()
        & df['wind_mph'].notna()
        & df['wind_direction_degrees'].notna()].copy()
    fbs = common[common['division_track'].eq('FBS')].copy()

    desc = descriptive(fbs)
    diag, overall, by_season = walk_forward(fbs)
    quality = direction_quality(df)

    write_df(cov, OUT_DIR / 'coverage_by_track_and_season.csv')
    write_df(quality, OUT_DIR / 'wind_direction_quality_by_season.csv')
    write_df(desc, OUT_DIR / 'fbs_descriptive_by_wind_and_alignment.csv')
    write_df(diag, OUT_DIR / 'fbs_hgb_walk_forward_diagnostics.csv')
    write_df(overall, OUT_DIR / 'fbs_hgb_qualifier_summary.csv')
    write_df(by_season, OUT_DIR / 'fbs_hgb_qualifier_by_season.csv')

    print('RESEARCH ONLY: no operational files were modified.')
    print(overall[['variant', 'graded', 'hit_rate', 'roi_per_1u', 'mae']].to_string(index=False))


if __name__ == '__main__':
    main()
