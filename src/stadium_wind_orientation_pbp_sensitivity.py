from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from .stadium_wind_orientation_mechanisms_v2 import cluster_se, metric_test
from .stadium_wind_orientation_pbp_mechanism import field_goal_distance, pass_flags
from .utils import ROOT, ensure_dir, read_df, write_df

OUT = ROOT / 'outputs/orientation_research'


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def game_explosive_sensitivity(plays: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    work = plays.copy()
    flags = work.apply(pass_flags, axis=1, result_type='expand')
    flags.columns = ['pass_attempt', 'pass_complete']
    work = pd.concat([work, flags], axis=1)
    work['yards_gained'] = pd.to_numeric(work.get('yards_gained'), errors='coerce')

    # Parser-integrity rule: generic Penalty rows can describe nullified plays.
    # Excluding them is a data-quality sensitivity, not a football threshold change.
    work['qa_pass_attempt'] = work['pass_attempt'] & ~work['play_type'].astype(str).str.lower().eq('penalty')
    work['qa_explosive_completion'] = (
        work['qa_pass_attempt'] & work['pass_complete'] & work['yards_gained'].ge(20)
    )
    attempts = work[work['qa_pass_attempt']].copy()
    game = attempts.groupby('game_id', as_index=False).agg(
        qa_pass_attempts=('play_id', 'size'),
        qa_explosive_completions=('qa_explosive_completion', 'sum'),
    )
    game['qa_explosive_pass_rate'] = game['qa_explosive_completions'] / game['qa_pass_attempts']
    frame = targets.merge(game, on='game_id', how='left').dropna(subset=['qa_explosive_pass_rate']).copy()

    rows = []
    for label, subset in [
        ('2014-2025', frame),
        ('2021-2025', frame[pd.to_numeric(frame['season'], errors='coerce') >= 2021]),
    ]:
        result = metric_test(subset, 'qa_explosive_pass_rate', '20+ completed pass rate; Penalty play rows excluded')
        result['period'] = label
        result['sensitivity'] = 'exclude_generic_penalty_play_rows'
        rows.append(result)
    return pd.DataFrame(rows)


def fg_design(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    work = frame.copy()
    numeric = [
        'fg_distance', 'wind_mph', 'temperature_f', 'humidity', 'precipitation',
        'dewpoint_f', 'pressure', 'closing_total',
    ]
    for col in numeric:
        work[col] = pd.to_numeric(work[col], errors='coerce')
    work['fg_distance_sq'] = work['fg_distance'] ** 2
    work['neutral_num'] = work.get('neutral_site', pd.Series(False, index=work.index)).map(as_bool).astype(float)
    work['is_cross'] = work['alignment'].eq('cross').astype(float)
    x = work[[
        'is_cross', 'fg_distance', 'fg_distance_sq', 'wind_mph', 'temperature_f', 'humidity',
        'precipitation', 'dewpoint_f', 'pressure', 'closing_total', 'neutral_num',
    ]].astype(float).reset_index(drop=True)
    dummies = pd.get_dummies(pd.DataFrame({
        'venue': pd.to_numeric(work['venue_id'], errors='coerce').astype('Int64').astype(str),
        'season': pd.to_numeric(work['season'], errors='coerce').astype('Int64').astype(str),
        'provider': work.get('line_provider', pd.Series('missing', index=work.index)).astype(str).fillna('missing'),
    }), drop_first=True, dtype=float).reset_index(drop=True)
    x = pd.concat([x, dummies], axis=1)
    x.insert(0, 'intercept', 1.0)
    return x.to_numpy(float), list(x.columns)


def fg_test(frame: pd.DataFrame, period: str) -> dict:
    required = [
        'fg_made', 'fg_distance', 'wind_mph', 'temperature_f', 'humidity', 'precipitation',
        'dewpoint_f', 'pressure', 'closing_total', 'game_id', 'venue_id', 'alignment',
    ]
    work = frame[frame['alignment'].isin(['cross', 'parallel'])].dropna(subset=required).copy()
    x, names = fg_design(work)
    y = work['fg_made'].astype(float).to_numpy()
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    se = cluster_se(x, residual, work['game_id'].to_numpy())
    idx = names.index('is_cross')
    z = beta[idx] / se[idx] if se[idx] > 0 else np.nan
    p = math.erfc(abs(float(z)) / math.sqrt(2)) if np.isfinite(z) else np.nan
    cross = work[work['alignment'].eq('cross')]
    parallel = work[work['alignment'].eq('parallel')]
    return {
        'period': period,
        'sensitivity': 'exclude_penalty_rows_and_require_Field_Goal_Good_for_make',
        'field_goal_attempts': len(work),
        'games_with_field_goal_attempt': int(work['game_id'].nunique()),
        'cross_attempts': len(cross),
        'parallel_attempts': len(parallel),
        'cross_make_rate': float(cross['fg_made'].mean()),
        'parallel_make_rate': float(parallel['fg_made'].mean()),
        'cross_avg_distance': float(cross['fg_distance'].mean()),
        'parallel_avg_distance': float(parallel['fg_distance'].mean()),
        'adjusted_cross_minus_parallel_make_probability': float(beta[idx]),
        'cluster_se_game': float(se[idx]),
        'p_two_sided': float(p),
    }


def field_goal_sensitivity(plays: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    play_type = plays['play_type'].astype(str)
    play_text = plays.get('play_text', pd.Series('', index=plays.index)).astype(str)
    fg_mask = play_type.str.contains('Field Goal', case=False, na=False) | play_text.str.contains('field goal', case=False, na=False)
    fg = plays[fg_mask].copy()

    # Generic Penalty rows can represent nullified attempts and are excluded.
    fg = fg[~fg['play_type'].astype(str).str.lower().eq('penalty')].copy()
    # Explicit play type is used so a blocked-FG return touchdown cannot be mistaken for a made kick.
    fg['fg_made'] = fg['play_type'].astype(str).str.lower().eq('field goal good').astype(float)
    distances = fg.apply(field_goal_distance, axis=1, result_type='expand')
    distances.columns = ['fg_distance', 'fg_distance_source']
    fg = pd.concat([fg, distances], axis=1)
    fg['fg_distance'] = pd.to_numeric(fg['fg_distance'], errors='coerce')
    fg = fg.merge(targets, on='game_id', how='left', suffixes=('', '_game'))

    full = fg_test(fg, '2014-2025')
    recent = fg_test(fg[pd.to_numeric(fg['season'], errors='coerce') >= 2021].copy(), '2021-2025')

    fg['distance_bin'] = pd.cut(fg['fg_distance'], [0, 39.999, 49.999, 100], labels=['<40', '40-49', '50+'])
    bins = fg.groupby(['distance_bin', 'alignment'], observed=True).agg(
        attempts=('fg_made', 'size'),
        makes=('fg_made', 'sum'),
        make_rate=('fg_made', 'mean'),
        avg_distance=('fg_distance', 'mean'),
    ).reset_index()
    return pd.DataFrame([full, recent]), bins


def main() -> None:
    plays = read_df('data/raw/orientation_pbp_10_15.csv').copy()
    targets = read_df('outputs/orientation_research/pbp_target_games.csv').copy()
    plays['game_id'] = pd.to_numeric(plays['game_id'], errors='coerce').astype('Int64')
    targets['game_id'] = pd.to_numeric(targets['game_id'], errors='coerce').astype('Int64')

    pass_results = game_explosive_sensitivity(plays, targets)
    fg_results, fg_bins = field_goal_sensitivity(plays, targets)
    write_df(pass_results, 'outputs/orientation_research/pbp_parser_sensitivity_explosive_pass.csv')
    write_df(fg_results, 'outputs/orientation_research/pbp_parser_sensitivity_field_goal.csv')
    write_df(fg_bins, 'outputs/orientation_research/pbp_parser_sensitivity_field_goal_bins.csv')

    lines = [
        '# Play-by-Play Parser-Integrity Sensitivity',
        '',
        '**This is a data-integrity sensitivity, not a new football hypothesis or threshold search.**',
        '',
        'The predeclared analysis found that generic `Penalty` play rows sometimes contain pass/field-goal text and that blocked field-goal return touchdowns can carry `scoring=true`. The sensitivity therefore excludes generic Penalty rows, and for the kicking sensitivity counts only explicit `Field Goal Good` play types as made kicks.',
        '',
        '## Explosive passing sensitivity', '', pass_results.to_markdown(index=False), '',
        '## Field-goal sensitivity', '', fg_results.to_markdown(index=False), '',
        '## Field-goal distance-bin sensitivity', '', fg_bins.to_markdown(index=False), '',
        'Interpretation should require the substantive conclusion to survive this parser-integrity check. The original predeclared results remain preserved and are not overwritten.',
    ]
    ensure_dir(OUT)
    (OUT / 'pbp_parser_sensitivity.md').write_text('\n'.join(lines), encoding='utf-8')
    print(pass_results.to_string(index=False))
    print(fg_results.to_string(index=False))


if __name__ == '__main__':
    main()
