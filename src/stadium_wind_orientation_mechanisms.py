from __future__ import annotations

import ast
import math
import re

import numpy as np
import pandas as pd

from .stadium_wind_orientation_research import build_research_frame
from .utils import ROOT, ensure_dir, read_df, write_df

OUT = ROOT / 'outputs/orientation_research'
LOW, HIGH = 10.0, 15.0


def slug(value: object) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(value).lower())


def parse_pair(value: object) -> tuple[float, float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan, np.nan
    match = re.match(r'^\s*([0-9.]+)\s*[-/]\s*([0-9.]+)\s*$', str(value).strip())
    if not match:
        return np.nan, np.nan
    return float(match.group(1)), float(match.group(2))


def num(value: object) -> float:
    if value is None:
        return np.nan
    try:
        return float(str(value).replace(',', '').strip())
    except Exception:
        return np.nan


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({'true', '1', 'yes', 'y'})


def game_stat_frame() -> tuple[pd.DataFrame, list[str]]:
    long = read_df('data/raw/cfbd_game_team_stats.csv').copy()
    if long.empty:
        return pd.DataFrame(), []
    long['game_id'] = pd.to_numeric(long['game_id'], errors='coerce')
    long['cat'] = long['category'].map(slug)
    cats = sorted(long['cat'].dropna().unique().tolist())
    wide = long.pivot_table(index=['game_id', 'team', 'home_away'], columns='cat', values='stat', aggfunc='first').reset_index()

    rows: list[dict] = []
    for gid, group in wide.groupby('game_id'):
        rec: dict[str, float] = {'game_id': gid}
        completions = attempts = pass_yards = 0.0
        rush_yards = rush_attempts = 0.0
        fg_made = fg_attempts = punts = punt_yards = 0.0
        have_pass_pair = have_pass_yards = have_rush = have_fg = have_punt = False
        for _, team in group.iterrows():
            ca = next((team.get(c) for c in ['completionattempts', 'completionsattempts'] if c in team.index), None)
            made, att = parse_pair(ca)
            if np.isfinite(made) and np.isfinite(att):
                completions += made
                attempts += att
                have_pass_pair = True
            passing = next((team.get(c) for c in ['netpassingyards', 'passingyards'] if c in team.index), None)
            passing = num(passing)
            if np.isfinite(passing):
                pass_yards += passing
                have_pass_yards = True
            rush_att = num(team.get('rushingattempts')) if 'rushingattempts' in team.index else np.nan
            rushing = num(team.get('rushingyards')) if 'rushingyards' in team.index else np.nan
            if np.isfinite(rush_att) and np.isfinite(rushing):
                rush_attempts += rush_att
                rush_yards += rushing
                have_rush = True
            fg_value = next((team.get(c) for c in ['fieldgoalsmadeattempts', 'fieldgoals'] if c in team.index), None)
            fg_m, fg_a = parse_pair(fg_value)
            if np.isfinite(fg_m) and np.isfinite(fg_a):
                fg_made += fg_m
                fg_attempts += fg_a
                have_fg = True
            p = num(team.get('punts')) if 'punts' in team.index else np.nan
            py = num(team.get('puntyards')) if 'puntyards' in team.index else np.nan
            if np.isfinite(p):
                punts += p
                have_punt = True
            if np.isfinite(py):
                punt_yards += py
        rec['combined_pass_attempts'] = attempts if have_pass_pair else np.nan
        rec['combined_completion_pct'] = completions / attempts if have_pass_pair and attempts > 0 else np.nan
        rec['combined_pass_yards_per_attempt'] = pass_yards / attempts if have_pass_pair and have_pass_yards and attempts > 0 else np.nan
        rec['combined_rush_yards_per_attempt'] = rush_yards / rush_attempts if have_rush and rush_attempts > 0 else np.nan
        rec['combined_fg_attempts'] = fg_attempts if have_fg else np.nan
        rec['combined_fg_pct'] = fg_made / fg_attempts if have_fg and fg_attempts > 0 else np.nan
        rec['combined_punts'] = punts if have_punt else np.nan
        rec['combined_punt_avg'] = punt_yards / punts if have_punt and punts > 0 and punt_yards > 0 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows), cats


def add_quarter_scoring(frame: pd.DataFrame) -> pd.DataFrame:
    games = read_df('data/raw/cfbd_games.csv').rename(columns={'id': 'game_id'}).copy()

    def parse_scores(value: object) -> list[float]:
        try:
            parsed = ast.literal_eval(str(value))
            return [float(item) for item in parsed] if isinstance(parsed, list) else []
        except Exception:
            return []

    home = games.get('homeLineScores', pd.Series('', index=games.index)).map(parse_scores)
    away = games.get('awayLineScores', pd.Series('', index=games.index)).map(parse_scores)
    quarter = pd.DataFrame({'game_id': pd.to_numeric(games['game_id'], errors='coerce')})
    for index in range(4):
        quarter[f'q{index + 1}_points'] = [
            (h[index] if len(h) > index else np.nan) + (a[index] if len(a) > index else np.nan)
            for h, a in zip(home, away)
        ]
    quarter['first_half_points'] = quarter['q1_points'] + quarter['q2_points']
    quarter['second_half_points'] = quarter['q3_points'] + quarter['q4_points']
    return frame.merge(quarter, on='game_id', how='left')


def design(frame: pd.DataFrame, include_cross: bool) -> tuple[np.ndarray, list[str]]:
    work = frame.copy()
    for col in ['wind_mph', 'temperature_f', 'humidity', 'precipitation', 'dewpoint_f', 'pressure', 'closing_total']:
        work[col] = pd.to_numeric(work.get(col), errors='coerce')
    work['neutral_num'] = as_bool(work.get('neutral_site', pd.Series(False, index=work.index))).astype(float)
    work['venue_fe'] = pd.to_numeric(work['venue_id'], errors='coerce').astype('Int64').astype(str)
    work['season_fe'] = pd.to_numeric(work['season'], errors='coerce').astype('Int64').astype(str)
    work['provider_fe'] = work.get('line_provider', pd.Series('missing', index=work.index)).astype(str).fillna('missing')
    nums = ['wind_mph', 'temperature_f', 'humidity', 'precipitation', 'dewpoint_f', 'pressure', 'closing_total', 'neutral_num']
    x = work[nums].astype(float).reset_index(drop=True)
    if include_cross:
        x.insert(0, 'is_cross', work['alignment'].eq('cross').astype(float).to_numpy())
    dummies = pd.get_dummies(work[['venue_fe', 'season_fe', 'provider_fe']], drop_first=True, dtype=float).reset_index(drop=True)
    x = pd.concat([x, dummies], axis=1)
    x.insert(0, 'intercept', 1.0)
    return x.to_numpy(float), list(x.columns)


def cluster_se(x: np.ndarray, residual: np.ndarray, groups: np.ndarray) -> np.ndarray:
    inv = np.linalg.pinv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]))
    unique = pd.unique(groups)
    for group in unique:
        idx = groups == group
        score = x[idx].T @ residual[idx]
        meat += np.outer(score, score)
    n, k = x.shape
    group_count = len(unique)
    correction = (group_count / (group_count - 1)) * ((n - 1) / max(1, n - k)) if group_count > 1 else 1
    return np.sqrt(np.maximum(np.diag(correction * (inv @ meat @ inv)), 0))


def metric_test(frame: pd.DataFrame, metric: str, expected: str) -> dict:
    work = frame[frame['alignment'].isin(['cross', 'parallel'])].dropna(subset=[metric, 'pressure']).copy()
    if len(work) < 100:
        return {'metric': metric, 'expected_direction': expected, 'games': len(work)}
    x, names = design(work, True)
    y = work[metric].astype(float).to_numpy()
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    se = cluster_se(x, residual, work['venue_id'].to_numpy())
    idx = names.index('is_cross')
    z = beta[idx] / se[idx] if se[idx] > 0 else np.nan
    p = math.erfc(abs(float(z)) / math.sqrt(2)) if np.isfinite(z) else np.nan
    cross = work.loc[work['alignment'].eq('cross'), metric]
    parallel = work.loc[work['alignment'].eq('parallel'), metric]
    return {
        'metric': metric,
        'expected_direction': expected,
        'games': len(work),
        'cross_games': len(cross),
        'parallel_games': len(parallel),
        'cross_mean': cross.mean(),
        'parallel_mean': parallel.mean(),
        'raw_cross_minus_parallel': cross.mean() - parallel.mean(),
        'adjusted_cross_minus_parallel': beta[idx],
        'cluster_se': se[idx],
        'p_two_sided': p,
    }


def main() -> None:
    frame, _ = build_research_frame()
    frame = frame[frame['division_track'].eq('FBS') & frame['outdoor']].copy()
    frame = frame[(frame['wind_mph'] > LOW) & (frame['wind_mph'] <= HIGH)].copy()
    frame['alignment'] = np.select(
        [frame['wind_field_angle_deg'] <= 30, frame['wind_field_angle_deg'] >= 60],
        ['parallel', 'cross'],
        default='oblique',
    )
    stats, cats = game_stat_frame()
    frame = frame.merge(stats, on='game_id', how='left')
    frame = add_quarter_scoring(frame)

    specs = [
        ('combined_pass_yards_per_attempt', 'lower if crosswind disrupts passing'),
        ('combined_completion_pct', 'lower if crosswind disrupts passing'),
        ('combined_pass_attempts', 'descriptive / adaptation'),
        ('combined_fg_pct', 'lower if crosswind disrupts kicking'),
        ('combined_fg_attempts', 'descriptive / scoring opportunities'),
        ('combined_punt_avg', 'descriptive special-teams mechanism'),
        ('combined_rush_yards_per_attempt', 'negative control; should be much weaker'),
        ('q1_points', 'lower'),
        ('q2_points', 'lower'),
        ('q3_points', 'lower'),
        ('q4_points', 'lower'),
        ('first_half_points', 'lower'),
        ('second_half_points', 'lower'),
        ('actual_total_points', 'lower'),
    ]
    results = pd.DataFrame([metric_test(frame, metric, expected) for metric, expected in specs if metric in frame.columns])
    write_df(results, 'outputs/orientation_research/mechanism_tests.csv')
    write_df(pd.DataFrame({'category_slug': cats}), 'outputs/orientation_research/mechanism_available_stat_categories.csv')

    recent = frame[pd.to_numeric(frame['season'], errors='coerce') >= 2021].copy()
    recent_results = pd.DataFrame([metric_test(recent, metric, expected) for metric, expected in specs if metric in recent.columns])
    write_df(recent_results, 'outputs/orientation_research/mechanism_tests_2021_2025.csv')

    lines = [
        '# Stadium Wind Orientation Football-Mechanism Study',
        '',
        '**Research only. This analysis does not alter the working model or Qualifier/Lean classifications.**',
        '',
        f'Primary regime was locked before this study: FBS outdoor games with {LOW:g}–<{HIGH:g} mph wind; cross = 60–90°, parallel = 0–30°.',
        '',
        '## Full 2014–2025 mechanism tests',
        '',
        results.to_markdown(index=False),
        '',
        '## Higher-resolution wind-direction era (2021–2025)',
        '',
        recent_results.to_markdown(index=False),
        '',
        '## Interpretation rule',
        '',
        'Passing/kicking effects that move in the hypothesized direction while the rushing negative control remains materially weaker would strengthen the physical-mechanism case. Results are robustness evidence, not untouched confirmatory evidence, because the primary wind regime came from the earlier discovery phase.',
    ]
    ensure_dir('outputs/orientation_research')
    (OUT / 'mechanism_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(results.to_string(index=False))


if __name__ == '__main__':
    main()
