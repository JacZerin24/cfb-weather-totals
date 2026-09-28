from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .cfbd_client import CFBDClient
from .deep_research import prep
from .inseason_feature_research import feature_set
from .inseason_features import merge_inseason_features
from .model_bakeoff import prep_features, reg_models
from .predict_week import add_live_categories, classify_row, normalize_games
from .utils import ensure_dir, read_df, write_df


def compact_live_features(
    records: list[dict],
    season: int,
    target_week: int,
    games: pd.DataFrame,
) -> pd.DataFrame:
    raw = pd.json_normalize(records, sep='.')
    if raw.empty or 'team' not in raw.columns:
        return pd.DataFrame(columns=['season', 'week', 'feature_team'])

    out = pd.DataFrame({
        'season': int(season),
        'week': int(target_week),
        'feature_team': raw['team'].astype(str),
    })
    for side in ('offense', 'defense'):
        plays = pd.to_numeric(raw.get(f'{side}.plays'), errors='coerce')
        drives = pd.to_numeric(raw.get(f'{side}.drives'), errors='coerce')
        out[f'adv_{side}_ppa'] = pd.to_numeric(raw.get(f'{side}.ppa'), errors='coerce')
        out[f'adv_{side}_success_rate'] = pd.to_numeric(raw.get(f'{side}.successRate'), errors='coerce')
        out[f'adv_{side}_plays_per_drive'] = plays / drives.replace(0, np.nan)
        out[f'adv_{side}_plays'] = plays
        out[f'adv_{side}_drives'] = drives

    prior = games[pd.to_numeric(games.get('week'), errors='coerce').lt(int(target_week))].copy()
    completed = prior.get('completed', pd.Series(True, index=prior.index)).fillna(False).astype(bool)
    prior = prior[completed]
    team_counts: dict[str, int] = {}
    for col in ('home_team', 'away_team'):
        if col in prior.columns:
            for team, count in prior[col].dropna().astype(str).value_counts().items():
                team_counts[team] = team_counts.get(team, 0) + int(count)
    out['adv_games_played'] = out['feature_team'].map(team_counts).fillna(0).astype(float)
    return out


def fetch_live_features(client: CFBDClient, games: pd.DataFrame, board: pd.DataFrame, season: int) -> pd.DataFrame:
    frames = []
    weeks = pd.to_numeric(board.get('week'), errors='coerce').dropna().astype(int).unique()
    for week in sorted(weeks):
        if week <= 1:
            continue
        records = client.get('/stats/season/advanced', {
            'year': int(season),
            'startWeek': 1,
            'endWeek': int(week) - 1,
            'classification': 'fbs',
            'excludeGarbageTime': True,
        })
        snap = compact_live_features(records, int(season), int(week), games)
        if not snap.empty:
            frames.append(snap)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['season', 'week', 'feature_team'])


def challenger_status(row: pd.Series) -> str:
    copy = row.copy()
    copy['pred_market_residual'] = row.get('inseason_pred_market_residual')
    status, _ = classify_row(copy)
    return status


def main() -> None:
    historical = prep(read_df('data/processed/modeling_dataset.csv'))
    feature_path = 'data/processed/team_inseason_game_advanced_features.csv'
    features = read_df(feature_path)
    if features.empty:
        raise RuntimeError(f'Missing required historical in-season features: {feature_path}')
    historical = merge_inseason_features(historical, features)

    board = read_df('outputs/weekly_board.csv')
    if board.empty:
        raise RuntimeError('weekly_board.csv is empty')
    season_values = pd.to_numeric(board.get('season'), errors='coerce').dropna().astype(int)
    if season_values.empty:
        raise RuntimeError('weekly board has no season')
    season = int(season_values.mode().iloc[0])

    client = CFBDClient()
    game_records = client.get('/games', {'year': season, 'seasonType': 'regular'})
    games = normalize_games(game_records, season)

    enrich_cols = [c for c in [
        'game_id', 'neutral_site', 'conference_game', 'home_pregame_elo', 'away_pregame_elo'
    ] if c in games.columns]
    if enrich_cols:
        board = board.merge(games[enrich_cols].drop_duplicates('game_id'), on='game_id', how='left')

    live_features = fetch_live_features(client, games, board, season)
    board = merge_inseason_features(board, live_features)
    board = add_live_categories(board)

    nums, cats = feature_set(historical, 'inseason_only')
    historical = prep_features(historical, cats)
    model = reg_models(nums, cats)['hist_gradient_boosting']
    model.fit(historical[nums + cats], historical['market_residual'])

    scored = board.copy()
    for col in nums:
        if col not in scored.columns:
            scored[col] = np.nan
        scored[col] = pd.to_numeric(scored[col], errors='coerce')
    for col in cats:
        if col not in scored.columns:
            scored[col] = 'missing'
        scored[col] = scored[col].astype(str).fillna('missing')

    eligible = (
        scored.get('division_track', pd.Series('', index=scored.index)).astype(str).eq('FBS')
        & pd.to_numeric(scored.get('closing_total'), errors='coerce').notna()
    )
    scored['inseason_pred_market_residual'] = np.nan
    if eligible.any():
        scored.loc[eligible, 'inseason_pred_market_residual'] = model.predict(scored.loc[eligible, nums + cats])
    scored['inseason_projected_total'] = pd.to_numeric(scored.get('closing_total'), errors='coerce') + scored['inseason_pred_market_residual']
    scored['inseason_abs_edge'] = scored['inseason_pred_market_residual'].abs()
    scored['inseason_side'] = np.where(scored['inseason_pred_market_residual'].lt(0), 'under', 'over')
    scored['challenger_status'] = scored.apply(
        lambda row: challenger_status(row) if pd.notna(row.get('inseason_pred_market_residual')) else '',
        axis=1,
    )
    scored['status_changed'] = scored['challenger_status'].astype(str).ne(scored.get('status', '').astype(str))
    scored['challenger_minus_baseline_edge'] = (
        scored['inseason_pred_market_residual'] - pd.to_numeric(scored.get('pred_market_residual'), errors='coerce')
    )
    scored['captured_at_utc'] = datetime.now(timezone.utc).isoformat()
    scored['research_only'] = True

    rename = {
        'status': 'baseline_status',
        'pred_market_residual': 'baseline_pred_market_residual',
        'model_projected_total': 'baseline_projected_total',
    }
    scored = scored.rename(columns=rename)
    keep = [c for c in [
        'captured_at_utc', 'research_only', 'season', 'week', 'game_id', 'start_date',
        'away_team', 'home_team', 'closing_total', 'line_provider',
        'baseline_status', 'baseline_pred_market_residual', 'baseline_projected_total',
        'challenger_status', 'inseason_pred_market_residual', 'inseason_projected_total',
        'inseason_abs_edge', 'inseason_side', 'challenger_minus_baseline_edge', 'status_changed',
        'home_inseason_adv_games_played', 'away_inseason_adv_games_played',
        'home_inseason_adv_offense_ppa', 'away_inseason_adv_offense_ppa',
        'home_inseason_adv_offense_success_rate', 'away_inseason_adv_offense_success_rate',
        'home_inseason_adv_defense_ppa', 'away_inseason_adv_defense_ppa',
        'home_inseason_adv_defense_success_rate', 'away_inseason_adv_defense_success_rate',
    ] if c in scored.columns]

    out_dir = ensure_dir(f'outputs/inseason_shadow/{season}')
    snapshot = scored[keep].copy()
    write_df(snapshot, f'outputs/inseason_shadow/{season}/latest.csv')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    write_df(snapshot, f'outputs/inseason_shadow/{season}/shadow_{stamp}.csv')
    print(
        f'Wrote research-only in-season form shadow for {len(snapshot)} games; '
        f'{int(snapshot.get("status_changed", pd.Series(dtype=bool)).fillna(False).sum())} status disagreement(s).'
    )


if __name__ == '__main__':
    main()
