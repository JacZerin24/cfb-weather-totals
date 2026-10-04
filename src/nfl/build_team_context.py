from __future__ import annotations

from io import BytesIO
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import requests

from ..utils import ensure_dir, read_df, write_df


PBP_URL = (
    'https://github.com/nflverse/nflverse-data/releases/download/'
    'pbp/play_by_play_{season}.parquet'
)
HISTORY_START = 2017
MODELED_START = 2018
END_SEASON = 2025
ROLLING_GAMES = 8
MIN_PRIOR_GAMES = 4
FORECAST_DATA = 'data/nfl/processed/forecast_native_dataset.csv'
OUT_TEAM_GAMES = 'data/nfl/processed/team_game_context.csv'
OUT_MATCHUPS = 'data/nfl/processed/team_context_features.csv'

PBP_COLUMNS = [
    'game_id', 'season', 'season_type', 'week', 'game_date',
    'home_team', 'away_team', 'posteam', 'defteam',
    'play_type', 'epa', 'success', 'yards_gained', 'down',
    'qb_dropback', 'pass_attempt', 'rush_attempt', 'sack',
    'interception', 'fumble_lost', 'qb_kneel', 'qb_spike',
    'two_point_attempt',
]

OFF_METRICS = [
    'epa_per_play',
    'success_rate',
    'pass_epa',
    'rush_epa',
    'explosive_rate',
    'pass_rate',
    'sack_rate',
    'turnover_rate',
    'early_down_epa',
]
DEF_METRICS = [f'def_{name}_allowed' for name in OFF_METRICS]
GAME_METRICS = OFF_METRICS + DEF_METRICS


def _download_pbp(season: int) -> pd.DataFrame:
    url = PBP_URL.format(season=season)
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(
                url,
                timeout=(20, 120),
                headers={'User-Agent': 'nfl-weather-totals-team-context/1.0'},
            )
            response.raise_for_status()
            payload = BytesIO(response.content)
            try:
                frame = pd.read_parquet(payload, columns=PBP_COLUMNS)
            except Exception:
                payload.seek(0)
                frame = pd.read_parquet(payload)
                keep = [c for c in PBP_COLUMNS if c in frame.columns]
                frame = frame[keep].copy()
            required = {
                'game_id', 'season', 'week', 'posteam', 'defteam',
                'epa', 'yards_gained',
            }
            missing = sorted(required - set(frame.columns))
            if missing:
                raise RuntimeError(
                    f'nflverse PBP {season} missing required columns: {missing}'
                )
            return frame
        except Exception as exc:
            last_error = exc
            sleep(min(20, 2 ** attempt))
    raise RuntimeError(f'Failed to download nflverse PBP {season}: {last_error}')


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors='coerce')


def _prepare_plays(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if 'season_type' in out.columns:
        out = out[out['season_type'].astype(str).str.upper().eq('REG')].copy()

    out = out[out['posteam'].notna() & out['defteam'].notna()].copy()
    out['epa'] = _numeric(out, 'epa', np.nan)
    out['success'] = _numeric(out, 'success', np.nan)
    out['yards_gained'] = _numeric(out, 'yards_gained', np.nan)
    out['down'] = _numeric(out, 'down', np.nan)

    dropback = _numeric(out, 'qb_dropback').eq(1)
    if 'qb_dropback' not in out.columns:
        dropback = (
            _numeric(out, 'pass_attempt').eq(1)
            | _numeric(out, 'sack').eq(1)
        )
    designed_rush = _numeric(out, 'rush_attempt').eq(1) & ~dropback
    scrimmage = dropback | designed_rush

    if 'play_type' in out.columns:
        play_type = out['play_type'].astype(str).str.lower()
        scrimmage &= play_type.isin(['pass', 'run'])

    scrimmage &= ~_numeric(out, 'qb_kneel').eq(1)
    scrimmage &= ~_numeric(out, 'qb_spike').eq(1)
    scrimmage &= ~_numeric(out, 'two_point_attempt').eq(1)

    out = out[scrimmage & out['epa'].notna()].copy()
    out['is_dropback'] = dropback.loc[out.index].astype(int)
    out['is_rush'] = designed_rush.loc[out.index].astype(int)
    out['is_explosive'] = out['yards_gained'].ge(20).astype(int)
    out['is_sack'] = _numeric(out, 'sack').eq(1).astype(int)
    out['is_turnover'] = (
        _numeric(out, 'interception').eq(1)
        | _numeric(out, 'fumble_lost').eq(1)
    ).astype(int)
    out['is_early_down'] = out['down'].isin([1, 2]).astype(int)
    return out


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors='coerce')
    return float(numeric.mean()) if numeric.notna().any() else np.nan


def _game_group_metrics(group: pd.DataFrame) -> pd.Series:
    dropbacks = group[group['is_dropback'].eq(1)]
    rushes = group[group['is_rush'].eq(1)]
    early = group[group['is_early_down'].eq(1)]
    return pd.Series({
        'epa_per_play': _safe_mean(group['epa']),
        'success_rate': _safe_mean(group['success']),
        'pass_epa': _safe_mean(dropbacks['epa']),
        'rush_epa': _safe_mean(rushes['epa']),
        'explosive_rate': _safe_mean(group['is_explosive']),
        'pass_rate': _safe_mean(group['is_dropback']),
        'sack_rate': _safe_mean(dropbacks['is_sack']),
        'turnover_rate': _safe_mean(group['is_turnover']),
        'early_down_epa': _safe_mean(early['epa']),
        'plays': int(len(group)),
    })


def _game_team_rows(plays: pd.DataFrame) -> pd.DataFrame:
    offense = (
        plays.groupby(
            ['game_id', 'season', 'week', 'posteam'],
            sort=False,
            observed=True,
        )
        .apply(_game_group_metrics, include_groups=False)
        .reset_index()
        .rename(columns={'posteam': 'team', 'plays': 'off_plays'})
    )

    defense = (
        plays.groupby(
            ['game_id', 'season', 'week', 'defteam'],
            sort=False,
            observed=True,
        )
        .apply(_game_group_metrics, include_groups=False)
        .reset_index()
        .rename(columns={'defteam': 'team', 'plays': 'def_plays'})
    )
    defense = defense.rename(
        columns={name: f'def_{name}_allowed' for name in OFF_METRICS}
    )

    dates = (
        plays[['game_id'] + [c for c in ['game_date', 'home_team', 'away_team'] if c in plays.columns]]
        .drop_duplicates('game_id')
    )
    out = offense.merge(
        defense[['game_id', 'team', 'def_plays'] + DEF_METRICS],
        on=['game_id', 'team'],
        how='outer',
    )
    out = out.merge(dates, on='game_id', how='left')
    out['game_date'] = pd.to_datetime(out.get('game_date'), errors='coerce')
    out['season'] = pd.to_numeric(out['season'], errors='coerce')
    out['week'] = pd.to_numeric(out['week'], errors='coerce')
    return out.sort_values(['team', 'game_date', 'season', 'week', 'game_id'])


def _pregame_rolls(team_games: pd.DataFrame) -> pd.DataFrame:
    out = team_games.copy()
    out['team_games_prior'] = out.groupby('team').cumcount()

    for metric in GAME_METRICS:
        values = pd.to_numeric(out[metric], errors='coerce')
        out[metric] = values
        out[f'pregame_{metric}'] = (
            out.assign(_value=values)
            .groupby('team', sort=False)['_value']
            .transform(
                lambda s: s.shift(1).rolling(
                    ROLLING_GAMES,
                    min_periods=MIN_PRIOR_GAMES,
                ).mean()
            )
        )
    return out


def _attach_side(
    games: pd.DataFrame,
    team_context: pd.DataFrame,
    side: str,
) -> pd.DataFrame:
    team_col = f'{side}_team'
    cols = ['game_id', 'team', 'team_games_prior'] + [
        f'pregame_{metric}' for metric in GAME_METRICS
    ]
    part = team_context[cols].copy()
    part = part.rename(
        columns={
            'team': team_col,
            'team_games_prior': f'{side}_team_games_prior',
            **{
                f'pregame_{metric}': f'{side}_{metric}'
                for metric in GAME_METRICS
            },
        }
    )
    return games.merge(part, on=['game_id', team_col], how='left')


def _matchup_features(games: pd.DataFrame) -> pd.DataFrame:
    out = games.copy()

    for metric in GAME_METRICS:
        home = pd.to_numeric(out[f'home_{metric}'], errors='coerce')
        away = pd.to_numeric(out[f'away_{metric}'], errors='coerce')
        out[f'team_{metric}_mean'] = (home + away) / 2.0
        out[f'team_{metric}_absdiff'] = (home - away).abs()

    matchup_pairs = {
        'epa': ('epa_per_play', 'def_epa_per_play_allowed'),
        'success': ('success_rate', 'def_success_rate_allowed'),
        'pass_epa': ('pass_epa', 'def_pass_epa_allowed'),
        'rush_epa': ('rush_epa', 'def_rush_epa_allowed'),
    }
    for label, (off_name, def_name) in matchup_pairs.items():
        home_attack = (
            pd.to_numeric(out[f'home_{off_name}'], errors='coerce')
            + pd.to_numeric(out[f'away_{def_name}'], errors='coerce')
        ) / 2.0
        away_attack = (
            pd.to_numeric(out[f'away_{off_name}'], errors='coerce')
            + pd.to_numeric(out[f'home_{def_name}'], errors='coerce')
        ) / 2.0
        out[f'matchup_{label}_mean'] = (home_attack + away_attack) / 2.0
        out[f'matchup_{label}_absdiff'] = (home_attack - away_attack).abs()

    return out


def main() -> None:
    if not Path(FORECAST_DATA).exists():
        raise FileNotFoundError(
            f'{FORECAST_DATA} is required. Restore/build the forecast-native '
            'dataset before team-context research.'
        )

    pieces = []
    for season in range(HISTORY_START, END_SEASON + 1):
        raw = _download_pbp(season)
        prepared = _prepare_plays(raw)
        print(
            f'NFL PBP {season}: {len(raw):,} raw rows -> '
            f'{len(prepared):,} eligible scrimmage plays'
        )
        pieces.append(prepared)

    plays = pd.concat(pieces, ignore_index=True)
    team_games = _game_team_rows(plays)
    team_context = _pregame_rolls(team_games)
    write_df(team_context, OUT_TEAM_GAMES)

    games = read_df(FORECAST_DATA).copy()
    games['season'] = pd.to_numeric(games['season'], errors='coerce')
    games = games[games['season'].between(MODELED_START, END_SEASON)].copy()
    games = _attach_side(games, team_context, 'home')
    games = _attach_side(games, team_context, 'away')
    games = _matchup_features(games)

    min_prior = pd.concat(
        [
            pd.to_numeric(games['home_team_games_prior'], errors='coerce'),
            pd.to_numeric(games['away_team_games_prior'], errors='coerce'),
        ],
        axis=1,
    ).min(axis=1)
    games['team_context_eligible'] = min_prior.ge(MIN_PRIOR_GAMES)

    write_df(games, OUT_MATCHUPS)

    modeled = games[games['season'].between(MODELED_START, END_SEASON)]
    coverage = (
        modeled.groupby('season', as_index=False)
        .agg(
            games=('game_id', 'count'),
            team_context_eligible=('team_context_eligible', 'sum'),
        )
    )
    coverage['coverage_rate'] = (
        coverage['team_context_eligible'] / coverage['games']
    )
    out = ensure_dir('outputs/nfl') / 'team_context_data_quality.csv'
    coverage.to_csv(out, index=False)

    print(f'Wrote {len(team_context):,} team-game rows to {OUT_TEAM_GAMES}')
    print(f'Wrote {len(games):,} matchup rows to {OUT_MATCHUPS}')
    print(coverage.to_string(index=False))


if __name__ == '__main__':
    main()
