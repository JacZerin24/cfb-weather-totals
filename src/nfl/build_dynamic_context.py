from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import ensure_dir, read_df, write_df


BASE_DATA = 'data/nfl/processed/modeling_dataset.csv'
FORECAST_TEAM_DATA = 'data/nfl/processed/team_context_features.csv'
OUT_DYNAMIC = 'data/nfl/processed/dynamic_team_context.csv'
OUT_TOURNAMENT = 'data/nfl/processed/model_tournament_dataset.csv'

SPANS = (4, 8, 16)
STATE_RATES = (0.05, 0.12)
MIN_TEAM_GAMES = 4


def _alpha(span: int) -> float:
    return 2.0 / (span + 1.0)


def _get(store: dict, key: tuple, default: float = np.nan) -> float:
    value = store.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _update_ewma(
    store: dict,
    key: tuple,
    value: float,
    span: int,
) -> None:
    if not np.isfinite(value):
        return
    old = _get(store, key)
    if not np.isfinite(old):
        store[key] = float(value)
        return
    a = _alpha(span)
    store[key] = float(a * value + (1.0 - a) * old)


def _safe_num(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _team_feature(
    ewma: dict,
    team: str,
    metric: str,
    span: int,
) -> float:
    return _get(ewma, (team, metric, span))


def _latent_projection(
    league_points: float,
    offense: dict[str, float],
    defense_allowed: dict[str, float],
    home: str,
    away: str,
) -> tuple[float, float, float]:
    base = league_points if np.isfinite(league_points) else 21.5
    home_points = (
        base
        + float(offense.get(home, 0.0))
        + float(defense_allowed.get(away, 0.0))
    )
    away_points = (
        base
        + float(offense.get(away, 0.0))
        + float(defense_allowed.get(home, 0.0))
    )
    return home_points, away_points, home_points + away_points


def main() -> None:
    if not Path(BASE_DATA).exists():
        raise FileNotFoundError(
            f'{BASE_DATA} missing; run the long-history NFL dataset build first.'
        )
    if not Path(FORECAST_TEAM_DATA).exists():
        raise FileNotFoundError(
            f'{FORECAST_TEAM_DATA} missing; build team context first.'
        )

    games = read_df(BASE_DATA).copy()
    games['season'] = pd.to_numeric(games['season'], errors='coerce')
    games['week'] = pd.to_numeric(games['week'], errors='coerce')
    games['gameday_dt'] = pd.to_datetime(games['gameday'], errors='coerce')
    games = games[
        games['season'].between(2006, 2025)
        & games['home_team'].notna()
        & games['away_team'].notna()
        & games['home_score'].notna()
        & games['away_score'].notna()
    ].copy()
    games = games.sort_values(
        ['gameday_dt', 'season', 'week', 'game_id'],
        kind='stable',
    )

    ewma: dict[tuple, float] = {}
    counts: defaultdict[str, int] = defaultdict(int)
    latent_offense = {
        rate: defaultdict(float)
        for rate in STATE_RATES
    }
    latent_defense_allowed = {
        rate: defaultdict(float)
        for rate in STATE_RATES
    }
    league_points = 21.5
    league_alpha = 0.02

    rows: list[dict] = []

    for _, game in games.iterrows():
        home = str(game['home_team'])
        away = str(game['away_team'])
        home_score = _safe_num(game['home_score'])
        away_score = _safe_num(game['away_score'])
        closing_total = _safe_num(game.get('closing_total'))
        residual = _safe_num(game.get('market_residual'))

        row: dict[str, object] = {
            'game_id': game['game_id'],
            'season': int(game['season']),
            'week': int(game['week']) if pd.notna(game['week']) else np.nan,
            'home_team': home,
            'away_team': away,
            'dyn_home_prior_games': counts[home],
            'dyn_away_prior_games': counts[away],
            'dyn_league_points_per_team': league_points,
        }

        for span in SPANS:
            home_pf = _team_feature(ewma, home, 'points_for', span)
            away_pf = _team_feature(ewma, away, 'points_for', span)
            home_pa = _team_feature(ewma, home, 'points_allowed', span)
            away_pa = _team_feature(ewma, away, 'points_allowed', span)
            home_resid = _team_feature(ewma, home, 'market_residual', span)
            away_resid = _team_feature(ewma, away, 'market_residual', span)
            home_abs_resid = _team_feature(
                ewma, home, 'abs_market_residual', span
            )
            away_abs_resid = _team_feature(
                ewma, away, 'abs_market_residual', span
            )

            row[f'dyn_home_points_for_ewma_{span}'] = home_pf
            row[f'dyn_away_points_for_ewma_{span}'] = away_pf
            row[f'dyn_home_points_allowed_ewma_{span}'] = home_pa
            row[f'dyn_away_points_allowed_ewma_{span}'] = away_pa
            row[f'dyn_home_market_resid_ewma_{span}'] = home_resid
            row[f'dyn_away_market_resid_ewma_{span}'] = away_resid
            row[f'dyn_market_resid_mean_{span}'] = np.nanmean(
                [home_resid, away_resid]
            ) if np.isfinite(home_resid) or np.isfinite(away_resid) else np.nan
            row[f'dyn_market_resid_absdiff_{span}'] = (
                abs(home_resid - away_resid)
                if np.isfinite(home_resid) and np.isfinite(away_resid)
                else np.nan
            )
            row[f'dyn_volatility_mean_{span}'] = np.nanmean(
                [home_abs_resid, away_abs_resid]
            ) if (
                np.isfinite(home_abs_resid)
                or np.isfinite(away_abs_resid)
            ) else np.nan

            if all(
                np.isfinite(x)
                for x in [home_pf, away_pf, home_pa, away_pa]
            ):
                projected = (
                    (home_pf + away_pa) / 2.0
                    + (away_pf + home_pa) / 2.0
                )
            else:
                projected = np.nan
            row[f'dyn_boxscore_projected_total_{span}'] = projected
            row[f'dyn_boxscore_edge_{span}'] = (
                projected - closing_total
                if np.isfinite(projected) and np.isfinite(closing_total)
                else np.nan
            )

        short_total = row.get('dyn_boxscore_projected_total_4')
        long_total = row.get('dyn_boxscore_projected_total_16')
        row['dyn_short_long_total_disagreement'] = (
            abs(float(short_total) - float(long_total))
            if (
                short_total is not None
                and long_total is not None
                and np.isfinite(_safe_num(short_total))
                and np.isfinite(_safe_num(long_total))
            )
            else np.nan
        )

        for rate in STATE_RATES:
            hp, ap, total = _latent_projection(
                league_points,
                latent_offense[rate],
                latent_defense_allowed[rate],
                home,
                away,
            )
            label = str(rate).replace('.', '')
            row[f'dyn_latent_home_points_k{label}'] = hp
            row[f'dyn_latent_away_points_k{label}'] = ap
            row[f'dyn_latent_total_k{label}'] = total
            row[f'dyn_latent_edge_k{label}'] = (
                total - closing_total
                if np.isfinite(closing_total)
                else np.nan
            )
            row[f'dyn_home_offense_state_k{label}'] = float(
                latent_offense[rate].get(home, 0.0)
            )
            row[f'dyn_away_offense_state_k{label}'] = float(
                latent_offense[rate].get(away, 0.0)
            )
            row[f'dyn_home_defense_allowed_state_k{label}'] = float(
                latent_defense_allowed[rate].get(home, 0.0)
            )
            row[f'dyn_away_defense_allowed_state_k{label}'] = float(
                latent_defense_allowed[rate].get(away, 0.0)
            )

        row['dynamic_context_eligible'] = (
            counts[home] >= MIN_TEAM_GAMES
            and counts[away] >= MIN_TEAM_GAMES
        )
        rows.append(row)

        for span in SPANS:
            _update_ewma(
                ewma, (home, 'points_for', span), home_score, span
            )
            _update_ewma(
                ewma, (away, 'points_for', span), away_score, span
            )
            _update_ewma(
                ewma, (home, 'points_allowed', span), away_score, span
            )
            _update_ewma(
                ewma, (away, 'points_allowed', span), home_score, span
            )
            if np.isfinite(residual):
                _update_ewma(
                    ewma, (home, 'market_residual', span), residual, span
                )
                _update_ewma(
                    ewma, (away, 'market_residual', span), residual, span
                )
                _update_ewma(
                    ewma,
                    (home, 'abs_market_residual', span),
                    abs(residual),
                    span,
                )
                _update_ewma(
                    ewma,
                    (away, 'abs_market_residual', span),
                    abs(residual),
                    span,
                )

        for rate in STATE_RATES:
            hp, ap, _ = _latent_projection(
                league_points,
                latent_offense[rate],
                latent_defense_allowed[rate],
                home,
                away,
            )
            home_error = home_score - hp
            away_error = away_score - ap
            latent_offense[rate][home] += rate * home_error / 2.0
            latent_defense_allowed[rate][away] += (
                rate * home_error / 2.0
            )
            latent_offense[rate][away] += rate * away_error / 2.0
            latent_defense_allowed[rate][home] += (
                rate * away_error / 2.0
            )

        game_pts_per_team = (home_score + away_score) / 2.0
        league_points = (
            league_alpha * game_pts_per_team
            + (1.0 - league_alpha) * league_points
        )
        counts[home] += 1
        counts[away] += 1

    dynamic = pd.DataFrame(rows)
    write_df(dynamic, OUT_DYNAMIC)

    forecast = read_df(FORECAST_TEAM_DATA).copy()
    merge_cols = [
        c for c in dynamic.columns
        if c == 'game_id' or c.startswith('dyn_')
        or c == 'dynamic_context_eligible'
    ]
    tournament = forecast.merge(
        dynamic[merge_cols],
        on='game_id',
        how='left',
        validate='one_to_one',
    )
    write_df(tournament, OUT_TOURNAMENT)

    tournament['season'] = pd.to_numeric(
        tournament['season'], errors='coerce'
    )
    quality = (
        tournament.groupby('season', as_index=False)
        .agg(
            games=('game_id', 'count'),
            dynamic_eligible=('dynamic_context_eligible', 'sum'),
        )
    )
    quality['dynamic_coverage'] = (
        quality['dynamic_eligible'] / quality['games']
    )
    out = ensure_dir('outputs/nfl') / 'model_tournament_dynamic_quality.csv'
    quality.to_csv(out, index=False)

    print(f'Wrote {len(dynamic):,} dynamic game rows to {OUT_DYNAMIC}')
    print(f'Wrote {len(tournament):,} tournament rows to {OUT_TOURNAMENT}')
    print(quality.to_string(index=False))


if __name__ == '__main__':
    main()
