"""Read-only, fail-closed point-in-time coverage and UNDER price audit.

Run from repo root. No network calls, betting actions, or production writes.
The source historical CFBD closing_total/weather fields are deliberately NOT
accepted as timestamped evidence of a price or forecast available before kickoff.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (48, 24, 6, 2)


def load_csv(path: str | None, required: tuple[str, ...], name: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=list(required))
    frame = pd.read_csv(path, low_memory=False)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'{name} missing columns: {sorted(missing)}')
    if frame.duplicated().all() and len(frame):
        raise ValueError(f'{name} appears fully duplicated')
    return frame


def parse_time(s: pd.Series) -> pd.Series:
    # UTC conversion is required. Explicitly reject timezone-naive strings.
    values = s.astype('string')
    naive = values.notna() & ~values.str.contains(r'(?:Z|[+-]\d\d:\d\d)$', case=False, regex=True, na=False)
    if naive.any():
        raise ValueError('All source timestamps must explicitly specify UTC or a timezone offset')
    return pd.to_datetime(s, utc=True, errors='coerce')


def clean_games(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    for c in ('game_id', 'season', 'actual_total_points'):
        g[c] = pd.to_numeric(g[c], errors='coerce')
    g['kickoff'] = parse_time(g['start_date'])
    g = g.dropna(subset=['game_id', 'season', 'kickoff', 'actual_total_points']).copy()
    g['game_id'] = g['game_id'].astype('int64')
    if g['game_id'].duplicated().any():
        raise ValueError('Games must be one row per unique game_id')
    return g


def group_vintages(source: pd.DataFrame, time_col: str) -> dict[int, pd.DataFrame]:
    if source.empty:
        return {}
    source = source[source[time_col].notna()].sort_values(time_col, kind='stable')
    return {int(game_id): frame for game_id, frame in source.groupby('game_id', dropna=True)}


def best_row(groups: dict[int, pd.DataFrame], game_id: int, decision: pd.Timestamp, time_col: str) -> pd.Series | None:
    source = groups.get(game_id)
    if source is None or source.empty:
        return None
    candidates = source.loc[source[time_col] <= decision]
    return candidates.iloc[-1] if len(candidates) else None


def audit(games: pd.DataFrame, odds: pd.DataFrame, weather: pd.DataFrame,
          predictions: pd.DataFrame, lead_hours: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    g = clean_games(games)
    o, w, p = odds.copy(), weather.copy(), predictions.copy()
    for x in (o, w, p):
        x['game_id'] = pd.to_numeric(x['game_id'], errors='coerce').astype('Int64')
    for x, columns in ((o, ('captured_at_utc', 'quoted_kickoff_utc')),
                       (w, ('issued_at_utc', 'available_at_utc', 'valid_at_utc')),
                       (p, ('generated_at_utc', 'training_cutoff_utc'))):
        for column in columns:
            x[column] = parse_time(x[column])
    # Never silently select missing source timestamps or a non-forecast weather record.
    if len(w):
        w = w[w['weather_type'].astype('string').str.lower() == 'forecast'].copy()
    odds_groups = group_vintages(o, 'captured_at_utc')
    weather_groups = group_vintages(w, 'available_at_utc')
    prediction_groups = group_vintages(p, 'generated_at_utc')
    rows = []
    for item in g.itertuples(index=False):
        kickoff, gid = item.kickoff, int(item.game_id)
        for hours in lead_hours:
            decision = kickoff - pd.Timedelta(hours=hours)
            record = {'game_id': gid, 'season': int(item.season), 'kickoff_utc': kickoff.isoformat(),
                      'decision_utc': decision.isoformat(), 'lead_hours': int(hours),
                      'quote_ok': False, 'forecast_ok': False, 'prediction_ok': False,
                      'eligible': False, 'exclusion_reason': '', 'bookmaker': '',
                      'bet_total': np.nan, 'under_odds_american': np.nan,
                      'pred_market_residual': np.nan, 'result': '', 'profit_units': np.nan}
            quote = best_row(odds_groups, gid, decision, 'captured_at_utc')
            issues = []
            if quote is None:
                issues.append('NO_PREGAME_ODDS_SNAPSHOT')
            else:
                quote_age = (decision - quote['captured_at_utc']).total_seconds() / 3600
                kickoff_diff = abs((quote['quoted_kickoff_utc'] - kickoff).total_seconds() / 60) if pd.notna(quote['quoted_kickoff_utc']) else float('inf')
                total = pd.to_numeric(quote['total'], errors='coerce')
                price = pd.to_numeric(quote['under_odds_american'], errors='coerce')
                if not (0 <= quote_age <= 12 and kickoff_diff <= 45 and pd.notna(total) and 1 <= total <= 110
                        and pd.notna(price) and (price <= -100 or price >= 100)
                        and pd.notna(quote['bookmaker']) and str(quote['bookmaker']).strip()
                        and pd.notna(quote['quote_id']) and str(quote['quote_id']).strip()):
                    issues.append('ODDS_TIME_PRICE_OR_KICKOFF_INVALID')
                else:
                    record.update(quote_ok=True, bookmaker=str(quote['bookmaker']),
                                  bet_total=float(total), under_odds_american=float(price))
            candidate = best_row(weather_groups, gid, decision, 'available_at_utc')
            if candidate is None:
                issues.append('NO_PREGAME_FORECAST_VINTAGE')
            else:
                issued, available, valid = (candidate[k] for k in ('issued_at_utc', 'available_at_utc', 'valid_at_utc'))
                age = (decision - available).total_seconds() / 3600 if pd.notna(available) else float('inf')
                valid_err = abs((valid - kickoff).total_seconds() / 60) if pd.notna(valid) else float('inf')
                winds = pd.to_numeric(candidate['wind_mph'], errors='coerce')
                temperature = pd.to_numeric(candidate['temperature_f'], errors='coerce')
                if not (pd.notna(issued) and pd.notna(available) and issued <= available <= decision
                        and age <= 18 and valid_err <= 90 and pd.notna(winds) and pd.notna(temperature)
                        and pd.notna(candidate['vintage_id']) and str(candidate['vintage_id']).strip()):
                    issues.append('FORECAST_TIME_VALIDITY_OR_FIELDS_INVALID')
                else:
                    record['forecast_ok'] = True
            prediction = best_row(prediction_groups, gid, decision, 'generated_at_utc')
            if prediction is None:
                issues.append('NO_ARCHIVED_PREGAME_PREDICTION')
            else:
                pred = pd.to_numeric(prediction['pred_market_residual'], errors='coerce')
                training = prediction['training_cutoff_utc']
                if not (pd.notna(pred) and pd.notna(training) and training < decision
                        and record['quote_ok'] and record['forecast_ok']
                        and str(prediction['quote_id']) == str(quote['quote_id'])
                        and str(prediction['weather_vintage_id']) == str(candidate['vintage_id'])
                        and prediction['generated_at_utc'] <= decision):
                    issues.append('PREDICTION_PROVENANCE_INVALID')
                else:
                    record['prediction_ok'] = True
                    record['pred_market_residual'] = float(pred)
            record['eligible'] = not issues
            record['exclusion_reason'] = ';'.join(issues)
            if record['eligible'] and record['pred_market_residual'] <= -4 and record['bet_total'] >= 56:
                total_score = float(item.actual_total_points)
                if total_score == record['bet_total']:
                    record.update(result='push', profit_units=0.0)
                elif total_score < record['bet_total']:
                    price = record['under_odds_american']
                    record.update(result='win', profit_units=100 / abs(price) if price < 0 else price / 100)
                else:
                    record.update(result='loss', profit_units=-1.0)
            rows.append(record)
    return pd.DataFrame(rows)


def summarize(audited: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    grouped = audited.groupby(['season', 'lead_hours'], as_index=False).agg(
        games=('game_id', 'count'), quotes=('quote_ok', 'sum'), forecasts=('forecast_ok', 'sum'),
        archived_predictions=('prediction_ok', 'sum'), complete_replays=('eligible', 'sum'),
        settled_qualifiers=('result', lambda s: int(s.isin(['win', 'loss', 'push']).sum())),
    )
    scored = audited[audited['result'].isin(['win', 'loss', 'push'])]
    stake = len(scored)
    return grouped, {
        'status': 'VALIDATED_PREGAME_REPLAY' if stake else 'BLOCKED_NO_VERIFIABLE_BETS',
        'evaluated_game_horizons': int(len(audited)),
        'distinct_games': int(audited['game_id'].nunique()),
        'complete_game_horizons': int(audited['eligible'].sum()),
        'graded_qualifiers': stake,
        'wins': int((scored['result'] == 'win').sum()),
        'losses': int((scored['result'] == 'loss').sum()),
        'pushes': int((scored['result'] == 'push').sum()),
        'roi_per_stake': float(scored['profit_units'].sum() / stake) if stake else None,
        'important': 'No verifiable bets means ROI is unavailable, not 0%. Multiple lead horizons must never be summed as independent wagers; market access and transaction costs are not demonstrated.',
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--games', required=True, help='Original modeling_dataset.csv (scores and kickoff times)')
    ap.add_argument('--odds', help='Timestamped bookmaker quotes CSV')
    ap.add_argument('--weather', help='Timestamped archived forecast vintages CSV')
    ap.add_argument('--predictions', help='Archived pregame prediction CSV with exact quote and vintage ids')
    ap.add_argument('--out', default='research/phase1/local_results')
    args = ap.parse_args()
    games = load_csv(args.games, ('game_id','season','start_date','actual_total_points'), 'games')
    odds = load_csv(args.odds, ('game_id','captured_at_utc','quoted_kickoff_utc','quote_id','bookmaker','total','under_odds_american'), 'odds')
    weather = load_csv(args.weather, ('game_id','weather_type','issued_at_utc','available_at_utc','valid_at_utc','vintage_id','wind_mph','temperature_f'), 'weather')
    predictions = load_csv(args.predictions, ('game_id','generated_at_utc','training_cutoff_utc','quote_id','weather_vintage_id','pred_market_residual'), 'predictions')
    audits = audit(games, odds, weather, predictions)
    grouped, report = summarize(audits)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audits.to_csv(out/'audit_by_game_horizon.csv', index=False)
    grouped.to_csv(out/'coverage_by_season_horizon.csv', index=False)
    (out/'gate_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
