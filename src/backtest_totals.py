from __future__ import annotations

import pandas as pd

from .utils import read_df, write_df, load_yaml


def payout_units(result: str, odds: int = -110) -> float:
    if result == 'push':
        return 0.0
    if result == 'loss':
        return -1.0
    if odds < 0:
        return 100 / abs(odds)
    return odds / 100


def apply_rule(df: pd.DataFrame, rule: dict, odds: int) -> dict:
    sample = df.copy()
    if 'wind_min_mph' in rule and 'wind_mph' in sample.columns:
        sample = sample[sample['wind_mph'] >= float(rule['wind_min_mph'])]
    if 'gust_min_mph' in rule and 'wind_gust_mph' in sample.columns:
        sample = sample[sample['wind_gust_mph'] >= float(rule['gust_min_mph'])]
    if 'temp_max_f' in rule and 'temperature_f' in sample.columns:
        sample = sample[sample['temperature_f'] <= float(rule['temp_max_f'])]

    sample = sample.dropna(subset=['actual_total_points', 'closing_total'])
    side = rule.get('side', 'under')
    if side == 'under':
        outcomes = sample.apply(lambda r: 'win' if r.actual_total_points < r.closing_total else ('push' if r.actual_total_points == r.closing_total else 'loss'), axis=1)
    else:
        outcomes = sample.apply(lambda r: 'win' if r.actual_total_points > r.closing_total else ('push' if r.actual_total_points == r.closing_total else 'loss'), axis=1)

    units = outcomes.map(lambda x: payout_units(x, odds)).sum() if len(outcomes) else 0.0
    bets = int((outcomes != 'push').sum()) if len(outcomes) else 0
    wins = int((outcomes == 'win').sum()) if len(outcomes) else 0
    losses = int((outcomes == 'loss').sum()) if len(outcomes) else 0
    return {
        'rule': rule['name'],
        'side': side,
        'games_matched': int(len(sample)),
        'bets_no_pushes': bets,
        'wins': wins,
        'losses': losses,
        'hit_rate_no_pushes': wins / bets if bets else None,
        'net_units_risk_1u_each': units,
        'roi_per_1u_risked': units / bets if bets else None,
    }


def main() -> None:
    cfg = load_yaml('config/model_thresholds.yml')
    odds = int(cfg['backtest']['default_odds'])
    df = read_df('data/processed/modeling_dataset.csv')
    rows = [apply_rule(df, rule, odds) for rule in cfg['backtest']['test_rules']]
    out = write_df(pd.DataFrame(rows), 'outputs/backtest_summary.csv')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
