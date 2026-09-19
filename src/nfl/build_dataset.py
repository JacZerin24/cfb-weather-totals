from __future__ import annotations

import pandas as pd

from ..utils import load_yaml, read_df, write_df


OUTDOOR_ROOFS = {'outdoors', 'open'}
INDOOR_ROOFS = {'closed', 'dome'}


def _as_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')


def main() -> None:
    settings = load_yaml('config/nfl_settings.yml')
    data_cfg = settings['data']
    df = read_df(data_cfg['raw_path']).copy()

    _as_numeric(
        df,
        [
            'season', 'week', 'home_score', 'away_score', 'total', 'total_line',
            'spread_line', 'temp', 'wind', 'home_rest', 'away_rest',
        ],
    )

    score_sum = df['home_score'] + df['away_score']
    if 'total' in df.columns:
        df['actual_total_points'] = df['total'].where(df['total'].notna(), score_sum)
    else:
        df['actual_total_points'] = score_sum

    df['closing_total'] = df['total_line']
    df['market_residual'] = df['actual_total_points'] - df['closing_total']
    df['went_over'] = df['market_residual'] > 0
    df['went_under'] = df['market_residual'] < 0
    df['push'] = df['market_residual'] == 0

    roof = df.get('roof', pd.Series('', index=df.index)).astype(str).str.lower().str.strip()
    df['outdoor'] = roof.isin(OUTDOOR_ROOFS)
    df['game_indoors_bool'] = roof.isin(INDOOR_ROOFS)
    df['weather_exposed'] = df['outdoor']

    df['temperature_f'] = pd.to_numeric(df.get('temp'), errors='coerce')
    df['wind_mph'] = pd.to_numeric(df.get('wind'), errors='coerce')
    df['wind_bin'] = pd.cut(
        df['wind_mph'],
        bins=[-1, 5, 10, 15, 20, 200],
        labels=['0-5', '5-10', '10-15', '15-20', '20+'],
    )
    df['temp_bin'] = pd.cut(
        df['temperature_f'],
        bins=[-100, 20, 32, 45, 60, 75, 90, 200],
        labels=['<=20', '20-32', '32-45', '45-60', '60-75', '75-90', '90+'],
    )
    df['total_bin'] = pd.cut(
        df['closing_total'],
        bins=[0, 38, 42, 46, 50, 54, 100],
        labels=['<=38', '38-42', '42-46', '46-50', '50-54', '54+'],
    )

    # Piecewise features let linear models test the commonly observed idea that
    # wind may matter little until a threshold is crossed, while tree models can
    # still learn their own non-linear relationships.
    df['wind_excess_10'] = (df['wind_mph'] - 10).clip(lower=0)
    df['wind_excess_15'] = (df['wind_mph'] - 15).clip(lower=0)
    df['cold_excess_below_32'] = (32 - df['temperature_f']).clip(lower=0)
    df['extreme_cold_below_20'] = (20 - df['temperature_f']).clip(lower=0)
    df['wind_10plus'] = (df['wind_mph'] >= 10).astype(int)
    df['wind_15plus'] = (df['wind_mph'] >= 15).astype(int)
    df['cold_32_or_less'] = (df['temperature_f'] <= 32).astype(int)
    df['cold_windy'] = ((df['temperature_f'] <= 40) & (df['wind_mph'] >= 12)).astype(int)

    gameday = pd.to_datetime(df.get('gameday'), errors='coerce')
    df['month'] = gameday.dt.month.astype('Int64')
    df['line_provider'] = 'nflverse/PFR closing total'

    out = write_df(df, data_cfg['processed_path'])
    usable = df['closing_total'].notna() & df['actual_total_points'].notna()
    outdoor_weather = usable & df['outdoor'] & df['wind_mph'].notna()
    print(f'Wrote {len(df):,} rows to {out}')
    print(f'Usable closing-total outcomes: {int(usable.sum()):,}')
    print(f'Usable outdoor games with wind: {int(outdoor_weather.sum()):,}')


if __name__ == '__main__':
    main()
