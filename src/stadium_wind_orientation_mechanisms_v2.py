from __future__ import annotations

import ast
import math
import re

import numpy as np
import pandas as pd

from .stadium_wind_orientation_research import load_data
from .utils import ROOT, ensure_dir, read_df, write_df

OUT = ROOT / 'outputs/orientation_research'
LOW, HIGH = 10.0, 15.0


def num(value: object) -> float:
    try:
        return float(str(value).replace(',', '').strip())
    except Exception:
        return np.nan


def pair(value: object) -> tuple[float, float]:
    match = re.match(r'^\s*([0-9.]+)\s*[-/]\s*([0-9.]+)\s*$', str(value))
    return (float(match.group(1)), float(match.group(2))) if match else (np.nan, np.nan)


def team_stats_to_games() -> tuple[pd.DataFrame, list[str]]:
    raw = read_df('data/raw/cfbd_game_team_stats.csv').copy()
    raw['game_id'] = pd.to_numeric(raw['game_id'], errors='coerce')
    categories = sorted(raw['category'].dropna().astype(str).unique().tolist())
    wide = raw.pivot_table(
        index=['game_id', 'team', 'home_away'], columns='category', values='stat', aggfunc='first'
    ).reset_index()
    wide.columns.name = None

    rows: list[dict] = []
    for game_id, group in wide.groupby('game_id'):
        completions = attempts = passing_yards = 0.0
        rush_attempts = rush_yards = 0.0
        passing_tds = rushing_tds = kicking_points = turnovers = 0.0
        valid_pass = valid_rush = False
        for _, team in group.iterrows():
            completed, attempted = pair(team.get('completionAttempts'))
            if np.isfinite(completed) and np.isfinite(attempted):
                completions += completed
                attempts += attempted
                valid_pass = True
            value = num(team.get('netPassingYards'))
            if np.isfinite(value):
                passing_yards += value
            ra, ry = num(team.get('rushingAttempts')), num(team.get('rushingYards'))
            if np.isfinite(ra) and np.isfinite(ry):
                rush_attempts += ra
                rush_yards += ry
                valid_rush = True
            for column, target in [
                ('passingTDs', 'passing_tds'), ('rushingTDs', 'rushing_tds'),
                ('kickingPoints', 'kicking_points'), ('turnovers', 'turnovers'),
            ]:
                value = num(team.get(column))
                if np.isfinite(value):
                    if target == 'passing_tds': passing_tds += value
                    elif target == 'rushing_tds': rushing_tds += value
                    elif target == 'kicking_points': kicking_points += value
                    elif target == 'turnovers': turnovers += value
        rows.append({
            'game_id': game_id,
            'combined_pass_yards_per_attempt': passing_yards / attempts if valid_pass and attempts else np.nan,
            'combined_completion_pct': completions / attempts if valid_pass and attempts else np.nan,
            'combined_pass_attempts': attempts if valid_pass else np.nan,
            'combined_rush_yards_per_attempt': rush_yards / rush_attempts if valid_rush and rush_attempts else np.nan,
            'pass_play_share': attempts / (attempts + rush_attempts) if valid_pass and valid_rush and (attempts + rush_attempts) else np.nan,
            'combined_passing_tds': passing_tds,
            'combined_rushing_tds': rushing_tds,
            'combined_kicking_points': kicking_points,
            'combined_turnovers': turnovers,
        })
    return pd.DataFrame(rows), categories


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
    for i in range(4):
        quarter[f'q{i + 1}_points'] = [
            (h[i] if len(h) > i else np.nan) + (a[i] if len(a) > i else np.nan)
            for h, a in zip(home, away)
        ]
    quarter['first_half_points'] = quarter['q1_points'] + quarter['q2_points']
    quarter['second_half_points'] = quarter['q3_points'] + quarter['q4_points']
    return frame.merge(quarter, on='game_id', how='left')


def design(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    work = frame.copy()
    numeric = ['wind_mph', 'temperature_f', 'humidity', 'precipitation', 'dewpoint_f', 'pressure', 'closing_total']
    for col in numeric:
        work[col] = pd.to_numeric(work[col], errors='coerce')
    work['neutral_num'] = work.get('neutral_site', pd.Series(False, index=work.index)).astype(str).str.lower().isin(['true', '1', 'yes', 'y']).astype(float)
    x = work[numeric + ['neutral_num']].astype(float).reset_index(drop=True)
    x.insert(0, 'is_cross', work['alignment'].eq('cross').astype(float).to_numpy())
    dummies = pd.get_dummies(pd.DataFrame({
        'venue': pd.to_numeric(work['venue_id'], errors='coerce').astype('Int64').astype(str),
        'season': pd.to_numeric(work['season'], errors='coerce').astype('Int64').astype(str),
        'provider': work.get('line_provider', pd.Series('missing', index=work.index)).astype(str).fillna('missing'),
    }), drop_first=True, dtype=float).reset_index(drop=True)
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
    n, k, g = x.shape[0], x.shape[1], len(unique)
    correction = (g / (g - 1)) * ((n - 1) / max(1, n - k)) if g > 1 else 1.0
    return np.sqrt(np.maximum(np.diag(correction * inv @ meat @ inv), 0))


def metric_test(frame: pd.DataFrame, metric: str, interpretation: str) -> dict:
    required = [metric, 'pressure', 'wind_mph', 'temperature_f', 'humidity', 'precipitation', 'dewpoint_f', 'closing_total']
    work = frame[frame['alignment'].isin(['cross', 'parallel'])].dropna(subset=required).copy()
    x, names = design(work)
    y = pd.to_numeric(work[metric], errors='coerce').to_numpy(float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    se = cluster_se(x, residual, work['venue_id'].to_numpy())
    idx = names.index('is_cross')
    z = beta[idx] / se[idx] if se[idx] else np.nan
    p = math.erfc(abs(float(z)) / math.sqrt(2)) if np.isfinite(z) else np.nan
    cross = work.loc[work['alignment'].eq('cross'), metric].astype(float)
    parallel = work.loc[work['alignment'].eq('parallel'), metric].astype(float)
    return {
        'metric': metric,
        'interpretation': interpretation,
        'games': len(work),
        'cross_games': len(cross),
        'parallel_games': len(parallel),
        'cross_mean': float(cross.mean()),
        'parallel_mean': float(parallel.mean()),
        'raw_cross_minus_parallel': float(cross.mean() - parallel.mean()),
        'adjusted_cross_minus_parallel': float(beta[idx]),
        'cluster_se': float(se[idx]),
        'p_two_sided': float(p),
    }


def run_period(frame: pd.DataFrame, specs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame([metric_test(frame, metric, note) for metric, note in specs])


def main() -> None:
    frame = load_data()
    frame = frame[frame['division_track'].eq('FBS') & frame['outdoor']].copy()
    frame = frame[(frame['wind_mph'] > LOW) & (frame['wind_mph'] <= HIGH)].copy()
    frame['alignment'] = np.select(
        [frame['wind_field_angle_deg'] <= 30, frame['wind_field_angle_deg'] >= 60],
        ['parallel', 'cross'], default='oblique')
    stats, categories = team_stats_to_games()
    frame = add_quarter_scoring(frame.merge(stats, on='game_id', how='left'))

    specs = [
        ('combined_pass_yards_per_attempt', 'primary passing-efficiency mechanism'),
        ('combined_completion_pct', 'primary passing-accuracy mechanism'),
        ('combined_pass_attempts', 'offensive adaptation / volume'),
        ('pass_play_share', 'offensive adaptation / play selection'),
        ('combined_passing_tds', 'passing scoring mechanism'),
        ('combined_rush_yards_per_attempt', 'negative control: not expected to be uniquely crosswind-sensitive'),
        ('combined_rushing_tds', 'negative-control / scoring-channel check'),
        ('combined_kicking_points', 'FG+PAT aggregate only; not a clean field-goal test'),
        ('combined_turnovers', 'ball-security / disruption check'),
        ('q1_points', 'timing of scoring effect'), ('q2_points', 'timing of scoring effect'),
        ('q3_points', 'timing of scoring effect'), ('q4_points', 'timing of scoring effect'),
        ('first_half_points', 'timing of scoring effect'), ('second_half_points', 'timing of scoring effect'),
        ('actual_total_points', 'overall scoring outcome'), ('market_residual', 'market-adjusted scoring outcome'),
    ]
    full = run_period(frame, specs)
    recent = run_period(frame[pd.to_numeric(frame['season'], errors='coerce') >= 2021].copy(), specs)
    write_df(full, 'outputs/orientation_research/mechanism_tests.csv')
    write_df(recent, 'outputs/orientation_research/mechanism_tests_2021_2025.csv')
    write_df(pd.DataFrame({'category': categories}), 'outputs/orientation_research/mechanism_available_stat_categories.csv')

    lines = [
        '# Stadium Wind Orientation Football-Mechanism Study', '',
        '**Research only. No operational model or Qualifier/Lean status is changed by this analysis.**', '',
        'Locked regime: FBS outdoor games with raw wind >10 and <=15 mph; cross = 60-90 degrees, parallel = 0-30 degrees.', '',
        'Models control for venue, season, raw wind speed, temperature, humidity, precipitation, dew point, pressure, closing total, neutral-site status, and line provider; uncertainty is clustered by venue.', '',
        '## Full 2014-2025 results', '', full.to_markdown(index=False), '',
        '## Higher-resolution wind-direction era (2021-2025)', '', recent.to_markdown(index=False), '',
        '## Data limitation', '',
        'The CFBD game-team endpoint used here does not expose field-goal attempts/makes or punt distance in the returned category set. `combined_kicking_points` is therefore only an FG+PAT aggregate and cannot establish a field-goal-specific mechanism. A play-by-play study would be required for clean field-goal and explosive-pass tests.', '',
        '## Interpretation guardrail', '',
        'A convincing wind mechanism would ideally affect ball-flight-sensitive outcomes more clearly than negative controls. These results are robustness/mechanism evidence, not untouched confirmation, because the 10-15 mph regime was identified in the earlier historical discovery phase.',
    ]
    ensure_dir(OUT)
    (OUT / 'mechanism_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(full.to_string(index=False))


if __name__ == '__main__':
    main()
