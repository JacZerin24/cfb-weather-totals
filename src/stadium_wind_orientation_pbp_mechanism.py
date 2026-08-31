from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from .stadium_wind_orientation_mechanisms_v2 import cluster_se, metric_test
from .utils import ROOT, ensure_dir, load_yaml, read_df, write_df

PROTOCOL_PATH = ROOT / 'config/orientation_pbp_mechanism_2026.yml'
OUT = ROOT / 'outputs/orientation_research'


def cfg() -> dict:
    return load_yaml('config/orientation_pbp_mechanism_2026.yml')


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def text(value: Any) -> str:
    return '' if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).lower()


def pass_flags(row: pd.Series) -> tuple[bool, bool]:
    play_type = text(row.get('play_type'))
    play_text = text(row.get('play_text'))

    completion = (
        'pass reception' in play_type
        or 'pass completion' in play_type
        or 'passing touchdown' in play_type
        or bool(re.search(r'\bpass(?:ed)?\s+(?:complete|completed)\b', play_text))
        or bool(re.search(r'\bpass\s+to\b.*\bfor\s+-?\d+', play_text))
    ) and 'incomplete' not in play_text and 'intercept' not in play_text

    incompletion = (
        'pass incompletion' in play_type
        or 'incomplete pass' in play_type
        or 'pass incomplete' in play_type
        or 'pass incomplete' in play_text
        or 'incomplete pass' in play_text
    )
    interception = 'interception' in play_type or 'intercepted' in play_text
    attempt = completion or incompletion or interception
    return attempt, completion


def is_field_goal(row: pd.Series) -> bool:
    play_type = text(row.get('play_type'))
    play_text = text(row.get('play_text'))
    return 'field goal' in play_type or 'field goal' in play_text or bool(re.search(r'\bfg\b', play_type))


def field_goal_distance(row: pd.Series) -> tuple[float, str]:
    play_text = text(row.get('play_text'))
    patterns = [
        r'\b(\d{1,2})\s*(?:yd|yard)s?\s+(?:field\s+goal|fg)\b',
        r'\b(?:field\s+goal|fg)\b[^0-9]{0,20}(\d{1,2})\s*(?:yd|yard)s?\b',
        r'\b(\d{1,2})\s*(?:yd|yard)s?\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, play_text)
        if match:
            value = float(match.group(1))
            if 10 <= value <= 70:
                return value, 'text'
    ytg = pd.to_numeric(pd.Series([row.get('yards_to_goal')]), errors='coerce').iloc[0]
    if pd.notna(ytg):
        value = float(ytg) + 17.0
        if 10 <= value <= 75:
            return value, 'yards_to_goal_plus_17'
    return np.nan, 'missing'


def aggregate_explosive_passes(plays: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold = int(cfg()['outcomes']['explosive_passing']['explosive_yards_threshold'])
    work = plays.copy()
    flags = work.apply(pass_flags, axis=1, result_type='expand')
    flags.columns = ['pass_attempt', 'pass_complete']
    work = pd.concat([work, flags], axis=1)
    work['yards_gained'] = pd.to_numeric(work.get('yards_gained'), errors='coerce')
    work['explosive_completion'] = work['pass_complete'] & work['yards_gained'].ge(threshold)

    attempts = work[work['pass_attempt']].copy()
    game = attempts.groupby('game_id', as_index=False).agg(
        pbp_pass_attempts=('play_id', 'size'),
        pbp_completions=('pass_complete', 'sum'),
        explosive_pass_completions=('explosive_completion', 'sum'),
    )
    game['explosive_pass_rate'] = game['explosive_pass_completions'] / game['pbp_pass_attempts']
    game['pbp_completion_rate'] = game['pbp_completions'] / game['pbp_pass_attempts']

    qa = (
        work.groupby('play_type', dropna=False)
        .agg(rows=('play_id', 'size'), pass_attempts=('pass_attempt', 'sum'), completions=('pass_complete', 'sum'), explosive_completions=('explosive_completion', 'sum'))
        .reset_index()
        .sort_values(['pass_attempts', 'rows'], ascending=False)
    )
    return game, qa


def field_goal_plays(plays: pd.DataFrame) -> pd.DataFrame:
    work = plays[plays.apply(is_field_goal, axis=1)].copy()
    if work.empty:
        return work
    work['fg_made'] = work.get('scoring', pd.Series(False, index=work.index)).map(bool_value).astype(float)
    distances = work.apply(field_goal_distance, axis=1, result_type='expand')
    distances.columns = ['fg_distance', 'fg_distance_source']
    work = pd.concat([work, distances], axis=1)
    work['fg_distance'] = pd.to_numeric(work['fg_distance'], errors='coerce')
    return work


def field_goal_design(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    work = frame.copy()
    numeric = [
        'fg_distance', 'wind_mph', 'temperature_f', 'humidity', 'precipitation',
        'dewpoint_f', 'pressure', 'closing_total',
    ]
    for col in numeric:
        work[col] = pd.to_numeric(work[col], errors='coerce')
    work['fg_distance_sq'] = work['fg_distance'] ** 2
    work['neutral_num'] = work.get('neutral_site', pd.Series(False, index=work.index)).map(bool_value).astype(float)
    work['is_cross'] = work['alignment'].eq('cross').astype(float)
    x = work[['is_cross', 'fg_distance', 'fg_distance_sq', 'wind_mph', 'temperature_f', 'humidity', 'precipitation', 'dewpoint_f', 'pressure', 'closing_total', 'neutral_num']].astype(float).reset_index(drop=True)
    dummies = pd.get_dummies(pd.DataFrame({
        'venue': pd.to_numeric(work['venue_id'], errors='coerce').astype('Int64').astype(str),
        'season': pd.to_numeric(work['season'], errors='coerce').astype('Int64').astype(str),
        'provider': work.get('line_provider', pd.Series('missing', index=work.index)).astype(str).fillna('missing'),
    }), drop_first=True, dtype=float).reset_index(drop=True)
    x = pd.concat([x, dummies], axis=1)
    x.insert(0, 'intercept', 1.0)
    return x.to_numpy(float), list(x.columns)


def field_goal_test(frame: pd.DataFrame, period: str) -> dict[str, Any]:
    required = [
        'fg_made', 'fg_distance', 'wind_mph', 'temperature_f', 'humidity', 'precipitation',
        'dewpoint_f', 'pressure', 'closing_total', 'game_id', 'venue_id', 'alignment',
    ]
    work = frame[frame['alignment'].isin(['cross', 'parallel'])].dropna(subset=required).copy()
    if len(work) < 30:
        return {'period': period, 'field_goal_attempts': len(work)}

    x, names = field_goal_design(work)
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


def distance_bin_summary(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.dropna(subset=['fg_distance']).copy()
    work['distance_bin'] = pd.cut(
        work['fg_distance'], bins=[0, 39.999, 49.999, 100], labels=['<40', '40-49', '50+'])
    return (
        work.groupby(['distance_bin', 'alignment'], observed=True)
        .agg(attempts=('fg_made', 'size'), makes=('fg_made', 'sum'), make_rate=('fg_made', 'mean'), avg_distance=('fg_distance', 'mean'))
        .reset_index()
    )


def main() -> None:
    protocol = cfg()
    targets = read_df('outputs/orientation_research/pbp_target_games.csv').copy()
    plays = read_df('data/raw/orientation_pbp_10_15.csv').copy()
    for frame in [targets, plays]:
        frame['game_id'] = pd.to_numeric(frame['game_id'], errors='coerce').astype('Int64')

    explosive, pass_qa = aggregate_explosive_passes(plays)
    game_frame = targets.merge(explosive, on='game_id', how='left')
    game_frame = game_frame.dropna(subset=['explosive_pass_rate']).copy()

    full_explosive = metric_test(game_frame, 'explosive_pass_rate', '20+ yard completed passes per pass attempt')
    full_explosive['period'] = '2014-2025'
    recent_seasons = {int(v) for v in protocol['sample']['recent_era_secondary_check']}
    recent_game = game_frame[pd.to_numeric(game_frame['season'], errors='coerce').isin(recent_seasons)].copy()
    recent_explosive = metric_test(recent_game, 'explosive_pass_rate', '20+ yard completed passes per pass attempt')
    recent_explosive['period'] = '2021-2025'
    explosive_results = pd.DataFrame([full_explosive, recent_explosive])
    write_df(explosive_results, 'outputs/orientation_research/pbp_explosive_pass_tests.csv')
    write_df(pass_qa, 'outputs/orientation_research/pbp_pass_parsing_qa.csv')

    fg = field_goal_plays(plays)
    if not fg.empty:
        fg = fg.merge(targets, on='game_id', how='left', suffixes=('', '_game'))
    full_fg = field_goal_test(fg, '2014-2025') if not fg.empty else {'period': '2014-2025', 'field_goal_attempts': 0}
    recent_fg_frame = fg[pd.to_numeric(fg.get('season'), errors='coerce').isin(recent_seasons)].copy() if not fg.empty else fg
    recent_fg = field_goal_test(recent_fg_frame, '2021-2025') if not fg.empty else {'period': '2021-2025', 'field_goal_attempts': 0}
    fg_results = pd.DataFrame([full_fg, recent_fg])
    write_df(fg_results, 'outputs/orientation_research/pbp_field_goal_tests.csv')
    write_df(distance_bin_summary(fg), 'outputs/orientation_research/pbp_field_goal_distance_bins.csv') if not fg.empty else write_df(pd.DataFrame(), 'outputs/orientation_research/pbp_field_goal_distance_bins.csv')

    if not fg.empty:
        fg_type_qa = (
            fg.groupby(['play_type', 'fg_distance_source'], dropna=False)
            .agg(attempts=('fg_made', 'size'), makes=('fg_made', 'sum'), make_rate=('fg_made', 'mean'))
            .reset_index()
            .sort_values('attempts', ascending=False)
        )
    else:
        fg_type_qa = pd.DataFrame()
    write_df(fg_type_qa, 'outputs/orientation_research/pbp_field_goal_parsing_qa.csv')

    pull_manifest = read_df('outputs/orientation_research/pbp_pull_manifest.csv')
    distance_text_pct = float(fg['fg_distance_source'].eq('text').mean()) if not fg.empty else np.nan
    distance_available_pct = float(fg['fg_distance'].notna().mean()) if not fg.empty else np.nan
    qa_summary = pd.DataFrame([{
        'study_version': protocol['study_version'],
        'protocol_sha256': hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        'target_games': int(pull_manifest['target_games'].iloc[0]),
        'games_with_pbp': int(pull_manifest['games_with_pbp'].iloc[0]),
        'play_rows': int(pull_manifest['play_rows'].iloc[0]),
        'games_with_classified_pass_attempts': int(game_frame['game_id'].nunique()),
        'classified_pass_attempts': int(game_frame['pbp_pass_attempts'].sum()),
        'explosive_completions_20plus': int(game_frame['explosive_pass_completions'].sum()),
        'field_goal_attempts_identified': len(fg),
        'field_goal_distance_available_pct': distance_available_pct,
        'field_goal_distance_from_text_pct': distance_text_pct,
    }])
    write_df(qa_summary, 'outputs/orientation_research/pbp_mechanism_qa_summary.csv')

    lines = [
        '# Bounded Play-by-Play Stadium-Wind Mechanism Study',
        '',
        '**Research only. This is the predeclared final historical mechanism extension; it does not alter the operational model.**',
        '',
        'Locked sample: FBS outdoor games, raw wind >10 and <=15 mph, cross alignment 60-90 degrees, parallel alignment 0-30 degrees.',
        '',
        'No betting thresholds or new wind cutoffs are searched.',
        '',
        '## Explosive passing: completed passes gaining 20+ yards per pass attempt',
        '',
        explosive_results.to_markdown(index=False),
        '',
        '## Field-goal make probability, controlling for attempt distance',
        '',
        fg_results.to_markdown(index=False),
        '',
        '## Pull / parsing QA',
        '',
        qa_summary.to_markdown(index=False),
        '',
        '## Interpretation guardrail',
        '',
        'A lower explosive-pass rate under cross alignment would support a passing ball-flight mechanism. A lower distance-adjusted field-goal make probability would support a kicking mechanism. Null or contradictory results mean the historical orientation signal still lacks a clean play-level football mechanism. Regardless of outcome, the predeclared stopping rule ends historical mechanism mining after this study; future evidence should come primarily from the frozen 2026 prospective shadow evaluation.',
    ]
    ensure_dir(OUT)
    (OUT / 'pbp_mechanism_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(explosive_results.to_string(index=False))
    print(fg_results.to_string(index=False))
    print(qa_summary.to_string(index=False))


if __name__ == '__main__':
    main()
