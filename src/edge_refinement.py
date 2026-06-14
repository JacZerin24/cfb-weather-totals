from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd

from .deep_research import BREAKEVEN, prep, settle, units, wilson
from .model_bakeoff import feature_lists, prep_features, reg_models
from .utils import ROOT, ensure_dir, read_df, write_df

HGB = 'hist_gradient_boosting'
CONSENSUS_MODELS = ['ridge', 'elastic_net', 'extra_trees']
ALL_MODELS = [HGB] + CONSENSUS_MODELS
AMERICAN_PRICE = -110
MIN_GRADED_FOR_SHORTLIST = 100
TARGET_ROI = 0.10


@dataclass
class CandidateFilter:
    name: str
    category: str
    description: str
    fn: Callable[[pd.DataFrame], pd.Series]


def max_drawdown(unit_series: pd.Series) -> float:
    if unit_series.empty:
        return 0.0
    equity = unit_series.cumsum()
    running_max = equity.cummax().clip(lower=0)
    drawdown = equity - running_max
    return float(drawdown.min())


def safe_bool(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(default)
    return s.astype(str).str.lower().isin(['true', '1', 'yes', 'y'])


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype='float64')
    return pd.to_numeric(df[col], errors='coerce')


def slug(value: str) -> str:
    return (
        str(value)
        .lower()
        .replace('+', 'plus')
        .replace('>=', 'ge')
        .replace('<=', 'le')
        .replace('>', 'gt')
        .replace('<', 'lt')
        .replace('%', 'pct')
        .replace('°', '')
        .replace(' ', '_')
        .replace('/', '_')
        .replace('-', '_')
        .replace('(', '')
        .replace(')', '')
        .replace('.', 'p')
        .replace('__', '_')
        .strip('_')
    )


def add_line_market_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lines_path = ROOT / 'data/raw/cfbd_lines.csv'
    if not lines_path.exists():
        return out
    try:
        lines = read_df('data/raw/cfbd_lines.csv').copy()
    except Exception:
        return out
    if lines.empty:
        return out
    rename = {}
    if 'id' in lines.columns and 'game_id' not in lines.columns:
        rename['id'] = 'game_id'
    if 'overUnder' in lines.columns and 'over_under' not in lines.columns:
        rename['overUnder'] = 'over_under'
    if rename:
        lines = lines.rename(columns=rename)
    if 'game_id' not in lines.columns or 'over_under' not in lines.columns:
        return out
    lines['over_under'] = pd.to_numeric(lines['over_under'], errors='coerce')
    lines = lines.dropna(subset=['game_id', 'over_under'])
    if lines.empty:
        return out
    market = lines.groupby('game_id')['over_under'].agg(
        line_provider_count='count',
        line_total_min='min',
        line_total_max='max',
        line_total_median='median',
        line_total_mean='mean',
    ).reset_index()
    market['line_total_range'] = market['line_total_max'] - market['line_total_min']
    out = out.merge(market, on='game_id', how='left')
    out['selected_vs_market_median'] = numeric(out, 'closing_total') - numeric(out, 'line_total_median')
    return out


def add_weather_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create clean yes/no weather flags from numeric fields plus text descriptions.

    CFBD weather fields are not always complete. Some games have measurable precipitation,
    while others only have text like "Light Rain" or "Thunderstorms". These flags make the
    refinement grid more honest than relying on precipitation > 0 alone.
    """
    out = df.copy()
    text_cols = [
        c for c in [
            'weather_condition',
            'weather_condition_code',
            'weather',
            'conditions',
            'weather_description',
        ]
        if c in out.columns
    ]
    if text_cols:
        text = out[text_cols].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
    else:
        text = pd.Series('', index=out.index)

    precip_words = r'rain|shower|drizzle|mist|thunder|storm|snow|sleet|freez|wintry|flurr'
    rain_words = r'rain|shower|drizzle|thunder|storm'
    snow_words = r'snow|sleet|flurr|wintry|freez'
    thunder_words = r'thunder|t-storm|storm'
    wintry_words = r'snow|sleet|freez|wintry|flurr'

    precip_num = numeric(out, 'precipitation')
    snow_num = numeric(out, 'snowfall')

    out['precip_flag'] = (precip_num > 0) | text.str.contains(precip_words, regex=True, na=False)
    out['rain_flag'] = (precip_num > 0) | text.str.contains(rain_words, regex=True, na=False)
    out['snow_flag'] = (snow_num > 0) | text.str.contains(snow_words, regex=True, na=False)
    out['thunder_flag'] = text.str.contains(thunder_words, regex=True, na=False)
    out['wintry_flag'] = (snow_num > 0) | text.str.contains(wintry_words, regex=True, na=False)
    return out


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def add_team_style_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    home_pass = first_existing(out, ['home_prior_pass_attempts', 'home_prior_passing_attempts'])
    away_pass = first_existing(out, ['away_prior_pass_attempts', 'away_prior_passing_attempts'])
    home_rush = first_existing(out, ['home_prior_rush_attempts', 'home_prior_rushing_attempts'])
    away_rush = first_existing(out, ['away_prior_rush_attempts', 'away_prior_rushing_attempts'])
    if home_pass and away_pass and home_rush and away_rush:
        hp = numeric(out, home_pass)
        ap = numeric(out, away_pass)
        hr = numeric(out, home_rush)
        ar = numeric(out, away_rush)
        out['home_prior_pass_rate_proxy'] = hp / (hp + hr).replace(0, np.nan)
        out['away_prior_pass_rate_proxy'] = ap / (ap + ar).replace(0, np.nan)
        out['game_prior_pass_rate_proxy'] = (hp + ap) / (hp + ap + hr + ar).replace(0, np.nan)

    home_plays = first_existing(out, ['home_prior_total_plays', 'home_prior_plays'])
    away_plays = first_existing(out, ['away_prior_total_plays', 'away_prior_plays'])
    if home_plays and away_plays:
        out['game_prior_pace_proxy'] = numeric(out, home_plays) + numeric(out, away_plays)

    home_pass_yards = first_existing(out, ['home_prior_pass_yards', 'home_prior_passing_yards'])
    away_pass_yards = first_existing(out, ['away_prior_pass_yards', 'away_prior_passing_yards'])
    home_rush_yards = first_existing(out, ['home_prior_rush_yards', 'home_prior_rushing_yards'])
    away_rush_yards = first_existing(out, ['away_prior_rush_yards', 'away_prior_rushing_yards'])
    if home_pass_yards and away_pass_yards and home_rush_yards and away_rush_yards:
        pyd = numeric(out, home_pass_yards) + numeric(out, away_pass_yards)
        ryd = numeric(out, home_rush_yards) + numeric(out, away_rush_yards)
        out['game_prior_pass_yard_share_proxy'] = pyd / (pyd + ryd).replace(0, np.nan)

    ypp_candidates = [
        ('home_prior_yards_per_play', 'away_prior_yards_per_play', 'game_prior_yards_per_play_proxy'),
        ('home_prior_yards_per_attempt', 'away_prior_yards_per_attempt', 'game_prior_yards_per_attempt_proxy'),
        ('home_prior_passing_yards_per_attempt', 'away_prior_passing_yards_per_attempt', 'game_prior_pass_yards_per_attempt_proxy'),
        ('home_prior_net_passing_yards_per_attempt', 'away_prior_net_passing_yards_per_attempt', 'game_prior_net_pass_yards_per_attempt_proxy'),
    ]
    for h, a, new_col in ypp_candidates:
        if h in out.columns and a in out.columns:
            out[new_col] = (numeric(out, h) + numeric(out, a)) / 2

    return out


def percentile_mask(df: pd.DataFrame, col: str, q: float, direction: str = 'ge') -> pd.Series:
    s = numeric(df, col)
    if s.notna().sum() < 50:
        return pd.Series(False, index=df.index)
    cutoff = s.quantile(q)
    if direction == 'ge':
        return s >= cutoff
    return s <= cutoff


def train_walk_forward_predictions(df: pd.DataFrame) -> pd.DataFrame:
    nums, cats = feature_lists(df)
    df = prep_features(df, cats)
    models = reg_models(nums, cats)
    parts = []
    for season in sorted(df['season'].dropna().astype(int).unique()):
        train = df[df['season'] < season].copy()
        test = df[df['season'] == season].copy()
        if len(train) < 1000 or len(test) < 100:
            continue
        pred_cols = {}
        for name in ALL_MODELS:
            model = models[name]
            model.fit(train[nums + cats], train['market_residual'])
            pred_cols[f'pred_{name}'] = model.predict(test[nums + cats])
        test = test.assign(**pred_cols)
        parts.append(test)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out['hgb_pred_market_residual'] = out[f'pred_{HGB}']
    out['hgb_abs_edge'] = out['hgb_pred_market_residual'].abs()
    out['hgb_side'] = np.where(out['hgb_pred_market_residual'] > 0, 'over', 'under')
    for name in CONSENSUS_MODELS:
        out[f'{name}_agrees_under'] = out[f'pred_{name}'] < 0
        out[f'{name}_strongly_opposes_under'] = out[f'pred_{name}'] > 1.5
    out['other_model_under_count'] = sum(out[f'{name}_agrees_under'].astype(int) for name in CONSENSUS_MODELS)
    out['all_model_under_count'] = out['other_model_under_count'] + (out['hgb_pred_market_residual'] < 0).astype(int)
    out['any_model_strongly_opposes_under'] = np.logical_or.reduce([out[f'{name}_strongly_opposes_under'] for name in CONSENSUS_MODELS])
    return out


def base_hgb_under(df: pd.DataFrame, threshold: float = 3.5) -> pd.Series:
    return (numeric(df, 'hgb_pred_market_residual') <= -threshold) & numeric(df, 'closing_total').notna()


def recent_period(season: int) -> str:
    if season >= 2022:
        return '2022-2025'
    if season >= 2019:
        return '2019-2021'
    return '2016-2018'


def grade_subset(df: pd.DataFrame, filt: CandidateFilter) -> tuple[dict, pd.DataFrame]:
    mask = filt.fn(df).fillna(False)
    plays = df[mask].copy()
    if plays.empty:
        return {}, pd.DataFrame()
    plays['side'] = 'under'
    plays['result'] = settle(plays['actual_total_points'], plays['closing_total'], plays['side'])
    outcomes = plays['result']
    graded_mask = outcomes != 'push'
    graded = outcomes[graded_mask]
    wins = int((graded == 'win').sum())
    losses = int((graded == 'loss').sum())
    pushes = int((outcomes == 'push').sum())
    n = wins + losses
    net_units = float(outcomes.map(units).sum()) if len(outcomes) else 0.0
    hit = wins / n if n else np.nan
    roi = net_units / n if n else np.nan
    low, high = wilson(wins, n)

    plays_sorted = plays.sort_values([c for c in ['season', 'week', 'game_id'] if c in plays.columns]).copy()
    plays_sorted['units'] = plays_sorted['result'].map(units)
    dd = max_drawdown(plays_sorted['units'])

    by_season_rows = []
    positive_seasons = 0
    losing_seasons = 0
    for season, g in plays.groupby('season'):
        out = g['result']
        sg = out[out != 'push']
        sw = int((sg == 'win').sum())
        sl = int((sg == 'loss').sum())
        sn = sw + sl
        snet = float(out.map(units).sum()) if len(out) else 0.0
        sroi = snet / sn if sn else np.nan
        if pd.notna(sroi) and sroi > 0:
            positive_seasons += 1
        if pd.notna(sroi) and sroi < 0:
            losing_seasons += 1
        by_season_rows.append({
            'filter_name': filt.name,
            'category': filt.category,
            'season': int(season),
            'graded': int(sn),
            'wins': sw,
            'losses': sl,
            'hit_rate': sw / sn if sn else np.nan,
            'net_units_1u_each': snet,
            'roi_per_1u': sroi,
        })

    season_count = len(by_season_rows)
    recent = plays[plays['season'].astype(int) >= 2022]
    if recent.empty:
        recent_roi = np.nan
        recent_graded = 0
    else:
        rout = recent['result']
        rgraded = rout[rout != 'push']
        recent_graded = int(len(rgraded))
        recent_roi = float(rout.map(units).sum()) / recent_graded if recent_graded else np.nan

    available_notes = []
    if 'line_total_range' in plays.columns and plays['line_total_range'].notna().any():
        available_notes.append(f"avg provider range {plays['line_total_range'].mean():.2f}")
    if 'selected_vs_market_median' in plays.columns and plays['selected_vs_market_median'].notna().any():
        available_notes.append(f"avg selected-vs-median {plays['selected_vs_market_median'].mean():+.2f}")
    clv_proxy_note = '; '.join(available_notes) if available_notes else 'true current-to-close CLV not available in historical file'

    overfit_flags = []
    if n < MIN_GRADED_FOR_SHORTLIST:
        overfit_flags.append('small_sample')
    if recent_graded < 30:
        overfit_flags.append('thin_recent_sample')
    if pd.notna(recent_roi) and recent_roi < 0:
        overfit_flags.append('negative_recent_roi')
    if season_count and positive_seasons / season_count < 0.60:
        overfit_flags.append('weak_season_stability')
    if pd.notna(roi) and roi >= 0.20 and n < 150:
        overfit_flags.append('high_roi_thin_sample')

    pos_season_rate = positive_seasons / season_count if season_count else np.nan
    score = 0.0
    if pd.notna(roi):
        score += roi * 100
    if pd.notna(recent_roi):
        score += recent_roi * 80
    if pd.notna(pos_season_rate):
        score += pos_season_rate * 15
    score += min(n, 500) / 500 * 10
    score += max(dd, -50) / 50 * 10

    row = {
        'filter_name': filt.name,
        'category': filt.category,
        'description': filt.description,
        'games': int(len(plays)),
        'graded': int(n),
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'hit_rate': hit,
        'hit_rate_minus_breakeven': hit - BREAKEVEN if n else np.nan,
        'hit_rate_wilson_low': low,
        'hit_rate_wilson_high': high,
        'net_units_1u_each': net_units,
        'roi_per_1u': roi,
        'target_roi_10pct_gap': roi - TARGET_ROI if pd.notna(roi) else np.nan,
        'recent_2022_plus_graded': recent_graded,
        'recent_2022_plus_roi': recent_roi,
        'season_count': season_count,
        'positive_seasons': positive_seasons,
        'losing_seasons': losing_seasons,
        'positive_season_rate': pos_season_rate,
        'max_drawdown_units': dd,
        'avg_closing_total': float(plays['closing_total'].mean()) if 'closing_total' in plays.columns else np.nan,
        'avg_hgb_abs_edge': float(plays['hgb_abs_edge'].mean()) if 'hgb_abs_edge' in plays.columns else np.nan,
        'avg_market_residual': float(plays['market_residual'].mean()) if 'market_residual' in plays.columns else np.nan,
        'line_movement_clv_proxy_note': clv_proxy_note,
        'overfit_flags': ','.join(overfit_flags) if overfit_flags else 'none',
        'shortlist_candidate': bool(n >= MIN_GRADED_FOR_SHORTLIST and pd.notna(recent_roi) and recent_roi > 0 and pos_season_rate >= 0.60 and pd.notna(roi) and roi > 0),
        'score': score,
    }
    return row, pd.DataFrame(by_season_rows)


def add_filter(filters: list[CandidateFilter], name: str, category: str, description: str, fn: Callable[[pd.DataFrame], pd.Series]) -> None:
    filters.append(CandidateFilter(slug(name), category, description, fn))


def condition_catalog() -> list[tuple[str, str, Callable[[pd.DataFrame], pd.Series]]]:
    """Reusable condition catalog for interaction mining.

    Conditions are intentionally interpretable and not too granular. The guardrails later
    decide whether a combo is too small or unstable.
    """
    return [
        ('total_ge_56', 'total >= 56', lambda d: numeric(d, 'closing_total') >= 56),
        ('total_ge_58', 'total >= 58', lambda d: numeric(d, 'closing_total') >= 58),
        ('total_ge_60', 'total >= 60', lambda d: numeric(d, 'closing_total') >= 60),
        ('total_60_63', 'total 60-63', lambda d: (numeric(d, 'closing_total') >= 60) & (numeric(d, 'closing_total') < 63)),
        ('wind_ge_10', 'wind >= 10 mph', lambda d: numeric(d, 'wind_mph') >= 10),
        ('wind_ge_12', 'wind >= 12 mph', lambda d: numeric(d, 'wind_mph') >= 12),
        ('wind_ge_15', 'wind >= 15 mph', lambda d: numeric(d, 'wind_mph') >= 15),
        ('wind_ge_18', 'wind >= 18 mph', lambda d: numeric(d, 'wind_mph') >= 18),
        ('cold_le_55', 'temp <= 55F', lambda d: numeric(d, 'temperature_f') <= 55),
        ('cold_le_50', 'temp <= 50F', lambda d: numeric(d, 'temperature_f') <= 50),
        ('cold_le_45', 'temp <= 45F', lambda d: numeric(d, 'temperature_f') <= 45),
        ('cold_le_40', 'temp <= 40F', lambda d: numeric(d, 'temperature_f') <= 40),
        ('heat_ge_85', 'temp >= 85F', lambda d: numeric(d, 'temperature_f') >= 85),
        ('heat_ge_90', 'temp >= 90F', lambda d: numeric(d, 'temperature_f') >= 90),
        ('heat_ge_95', 'temp >= 95F', lambda d: numeric(d, 'temperature_f') >= 95),
        ('humid_ge_70', 'humidity >= 70%', lambda d: numeric(d, 'humidity') >= 70),
        ('humid_ge_80', 'humidity >= 80%', lambda d: numeric(d, 'humidity') >= 80),
        ('humid_ge_90', 'humidity >= 90%', lambda d: numeric(d, 'humidity') >= 90),
        ('dewpoint_ge_65', 'dewpoint >= 65F', lambda d: numeric(d, 'dewpoint_f') >= 65),
        ('dewpoint_ge_70', 'dewpoint >= 70F', lambda d: numeric(d, 'dewpoint_f') >= 70),
        ('dewpoint_ge_75', 'dewpoint >= 75F', lambda d: numeric(d, 'dewpoint_f') >= 75),
        ('precip_any', 'any precipitation flag', lambda d: safe_bool(d, 'precip_flag') | (numeric(d, 'precipitation') > 0)),
        ('rain_flag', 'rain/thunder text or precip amount', lambda d: safe_bool(d, 'rain_flag')),
        ('snow_flag', 'snow/wintry flag', lambda d: safe_bool(d, 'snow_flag')),
        ('thunder_flag', 'thunderstorm text flag', lambda d: safe_bool(d, 'thunder_flag')),
        ('wintry_flag', 'wintry/sleet/freezing flag', lambda d: safe_bool(d, 'wintry_flag')),
        ('high_pass_rate', 'top 30% pass-rate proxy', lambda d: percentile_mask(d, 'game_prior_pass_rate_proxy', 0.70, 'ge')),
        ('high_pace', 'top 30% pace proxy', lambda d: percentile_mask(d, 'game_prior_pace_proxy', 0.70, 'ge')),
        ('high_pass_yard_share', 'top 30% pass-yard-share proxy', lambda d: percentile_mask(d, 'game_prior_pass_yard_share_proxy', 0.70, 'ge')),
        ('provider_count_ge_3', '3+ line providers', lambda d: numeric(d, 'line_provider_count') >= 3),
        ('line_range_ge_2', 'provider total range >= 2', lambda d: numeric(d, 'line_total_range') >= 2),
        ('selected_above_median_1', 'selected total >= 1 above provider median', lambda d: numeric(d, 'selected_vs_market_median') >= 1),
    ]


def build_filters(df: pd.DataFrame) -> list[CandidateFilter]:
    filters: list[CandidateFilter] = []

    def add(name: str, category: str, description: str, fn: Callable[[pd.DataFrame], pd.Series]) -> None:
        add_filter(filters, name, category, description, fn)

    # Baselines and edge-strength tests.
    add('hgb_under_3p5_baseline', 'baseline', 'All HGB under candidates with at least a 3.5-point predicted edge.', lambda d: base_hgb_under(d, 3.5))
    add('hgb_under_5p0_baseline', 'baseline', 'All HGB under candidates with at least a 5.0-point predicted edge.', lambda d: base_hgb_under(d, 5.0))
    add('hgb_under_7p0_baseline', 'baseline', 'All HGB under candidates with at least a 7.0-point predicted edge.', lambda d: base_hgb_under(d, 7.0))

    # Total buckets and ranges.
    for total in [52, 54, 56, 58, 60, 63, 66, 70]:
        add(f'hgb_under_3p5_total_ge_{total}', 'high_total_bucket', f'HGB under 3.5+ where closing total is at least {total}.', lambda d, total=total: base_hgb_under(d, 3.5) & (numeric(d, 'closing_total') >= total))
    total_ranges = [(48, 52), (52, 56), (56, 60), (60, 63), (63, 66), (66, 70), (70, 1000)]
    for lo, hi in total_ranges:
        label = f'{lo}_{hi if hi < 999 else "plus"}'
        desc = f'HGB under 3.5+ with closing total {lo} to {hi}.' if hi < 999 else f'HGB under 3.5+ with closing total {lo}+.'
        add(f'hgb_under_3p5_total_{label}', 'high_total_bucket', desc, lambda d, lo=lo, hi=hi: base_hgb_under(d, 3.5) & (numeric(d, 'closing_total') >= lo) & (numeric(d, 'closing_total') < hi))

    # Single weather filters.
    add('hgb_under_3p5_outdoor_only', 'weather_single', 'HGB under 3.5+ excluding indoor games.', lambda d: base_hgb_under(d, 3.5) & ~safe_bool(d, 'game_indoors_bool'))
    for wind in [5, 10, 12, 15, 18, 20, 25]:
        add(f'hgb_under_3p5_wind_ge_{wind}', 'weather_single', f'HGB under 3.5+ with wind at least {wind} mph.', lambda d, wind=wind: base_hgb_under(d, 3.5) & (numeric(d, 'wind_mph') >= wind))
    for temp in [55, 50, 45, 40, 35, 32, 25]:
        add(f'hgb_under_3p5_temp_le_{temp}', 'weather_single', f'HGB under 3.5+ with temperature at or below {temp}F.', lambda d, temp=temp: base_hgb_under(d, 3.5) & (numeric(d, 'temperature_f') <= temp))
    for temp in [80, 85, 90, 95]:
        add(f'hgb_under_3p5_temp_ge_{temp}', 'weather_single', f'HGB under 3.5+ with temperature at or above {temp}F.', lambda d, temp=temp: base_hgb_under(d, 3.5) & (numeric(d, 'temperature_f') >= temp))
    for humid in [60, 70, 80, 90]:
        add(f'hgb_under_3p5_humidity_ge_{humid}', 'weather_single', f'HGB under 3.5+ with humidity at least {humid}%.', lambda d, humid=humid: base_hgb_under(d, 3.5) & (numeric(d, 'humidity') >= humid))
    for dew in [60, 65, 70, 75]:
        add(f'hgb_under_3p5_dewpoint_ge_{dew}', 'weather_single', f'HGB under 3.5+ with dewpoint at least {dew}F.', lambda d, dew=dew: base_hgb_under(d, 3.5) & (numeric(d, 'dewpoint_f') >= dew))

    flag_defs = [
        ('precip_any', 'precipitation flag or measurable precipitation', lambda d: safe_bool(d, 'precip_flag') | (numeric(d, 'precipitation') > 0)),
        ('rain_flag', 'rain/thunder text or measurable precipitation', lambda d: safe_bool(d, 'rain_flag')),
        ('snow_flag', 'snow flag or measurable snowfall', lambda d: safe_bool(d, 'snow_flag') | (numeric(d, 'snowfall') > 0)),
        ('thunder_flag', 'thunderstorm text flag', lambda d: safe_bool(d, 'thunder_flag')),
        ('wintry_flag', 'wintry/sleet/freezing flag', lambda d: safe_bool(d, 'wintry_flag')),
    ]
    for name, desc, fn in flag_defs:
        add(f'hgb_under_3p5_{name}', 'weather_single', f'HGB under 3.5+ with {desc}.', lambda d, fn=fn: base_hgb_under(d, 3.5) & fn(d))

    # Existing core combinations, retained for comparability.
    add('hgb_under_3p5_high_total_wind15', 'weather_total_combo', 'HGB under 3.5+, total 56+, wind 15+ mph.', lambda d: base_hgb_under(d, 3.5) & (numeric(d, 'closing_total') >= 56) & (numeric(d, 'wind_mph') >= 15))
    add('hgb_under_3p5_high_total_cold50', 'weather_total_combo', 'HGB under 3.5+, total 56+, temp <= 50F.', lambda d: base_hgb_under(d, 3.5) & (numeric(d, 'closing_total') >= 56) & (numeric(d, 'temperature_f') <= 50))
    add('hgb_under_3p5_high_total_precip', 'weather_total_combo', 'HGB under 3.5+, total 56+, precipitation present.', lambda d: base_hgb_under(d, 3.5) & (numeric(d, 'closing_total') >= 56) & safe_bool(d, 'precip_flag'))
    add('hgb_under_3p5_high_total_wind_or_precip', 'weather_total_combo', 'HGB under 3.5+, total 56+, wind 15+ or precipitation.', lambda d: base_hgb_under(d, 3.5) & (numeric(d, 'closing_total') >= 56) & ((numeric(d, 'wind_mph') >= 15) | safe_bool(d, 'precip_flag')))

    # Provider / market-position filters.
    if 'line_provider' in df.columns:
        providers = [p for p in df['line_provider'].dropna().astype(str).value_counts().head(12).index.tolist() if p.lower() != 'nan']
        for provider in providers:
            add(f'hgb_under_3p5_provider_{provider}', 'provider', f'HGB under 3.5+ using line provider {provider}.', lambda d, provider=provider: base_hgb_under(d, 3.5) & (d['line_provider'].astype(str) == provider))

    for count in [2, 3, 4, 5]:
        add(f'hgb_under_3p5_provider_count_ge_{count}', 'line_market_proxy', f'HGB under 3.5+ where at least {count} line providers are available.', lambda d, count=count: base_hgb_under(d, 3.5) & (numeric(d, 'line_provider_count') >= count))
    for rng in [0.5, 1, 1.5, 2, 3]:
        add(f'hgb_under_3p5_line_range_ge_{rng}', 'line_market_proxy', f'HGB under 3.5+ where provider total range is at least {rng} points.', lambda d, rng=rng: base_hgb_under(d, 3.5) & (numeric(d, 'line_total_range') >= rng))
    for margin in [0.5, 1, 1.5, 2]:
        add(f'hgb_under_3p5_selected_total_above_median_{margin}', 'line_market_proxy', f'HGB under 3.5+ where selected total is at least {margin} above provider median.', lambda d, margin=margin: base_hgb_under(d, 3.5) & (numeric(d, 'selected_vs_market_median') >= margin))

    # Model-consensus filters.
    add('hgb_under_3p5_two_plus_models_under', 'model_consensus', 'HGB under 3.5+ and at least 2 other regression models also lean under.', lambda d: base_hgb_under(d, 3.5) & (numeric(d, 'other_model_under_count') >= 2))
    add('hgb_under_3p5_all_models_under', 'model_consensus', 'HGB under 3.5+ and all tracked regression models lean under.', lambda d: base_hgb_under(d, 3.5) & (numeric(d, 'all_model_under_count') >= 4))
    add('hgb_under_3p5_no_model_strong_opposition', 'model_consensus', 'HGB under 3.5+ with no other model strongly leaning over.', lambda d: base_hgb_under(d, 3.5) & ~safe_bool(d, 'any_model_strongly_opposes_under'))
    add('hgb_under_5p0_two_plus_models_under', 'model_consensus', 'HGB under 5.0+ and at least 2 other regression models also lean under.', lambda d: base_hgb_under(d, 5.0) & (numeric(d, 'other_model_under_count') >= 2))
    add('hgb_under_7p0_no_model_strong_opposition', 'model_consensus', 'HGB under 7.0+ with no other model strongly leaning over.', lambda d: base_hgb_under(d, 7.0) & ~safe_bool(d, 'any_model_strongly_opposes_under'))

    # Dynamic team-style filters.
    add('hgb_under_3p5_high_pass_rate_top30', 'team_style', 'HGB under 3.5+ where combined prior pass-rate proxy is in the top 30%.', lambda d: base_hgb_under(d, 3.5) & percentile_mask(d, 'game_prior_pass_rate_proxy', 0.70, 'ge'))
    add('hgb_under_3p5_high_pace_top30', 'team_style', 'HGB under 3.5+ where combined prior pace/play-volume proxy is in the top 30%.', lambda d: base_hgb_under(d, 3.5) & percentile_mask(d, 'game_prior_pace_proxy', 0.70, 'ge'))
    add('hgb_under_3p5_high_pass_yard_share_top30', 'team_style', 'HGB under 3.5+ where prior pass-yard share proxy is in the top 30%.', lambda d: base_hgb_under(d, 3.5) & percentile_mask(d, 'game_prior_pass_yard_share_proxy', 0.70, 'ge'))

    # Exhaustive-but-controlled weather and team-style interaction grid.
    cat = {key: (desc, fn) for key, desc, fn in condition_catalog()}

    # Weather-only pairs/triples people intuitively care about.
    weather_pair_keys = [
        ('cold_le_55', 'wind_ge_10'), ('cold_le_55', 'wind_ge_12'), ('cold_le_50', 'wind_ge_12'), ('cold_le_45', 'wind_ge_12'),
        ('cold_le_40', 'wind_ge_12'), ('cold_le_50', 'wind_ge_15'), ('cold_le_45', 'wind_ge_15'),
        ('cold_le_55', 'precip_any'), ('cold_le_50', 'precip_any'), ('cold_le_45', 'precip_any'),
        ('cold_le_50', 'rain_flag'), ('cold_le_45', 'rain_flag'), ('cold_le_50', 'snow_flag'), ('cold_le_45', 'snow_flag'),
        ('wind_ge_10', 'precip_any'), ('wind_ge_12', 'precip_any'), ('wind_ge_15', 'precip_any'),
        ('wind_ge_10', 'rain_flag'), ('wind_ge_12', 'rain_flag'), ('wind_ge_15', 'rain_flag'),
        ('wind_ge_10', 'snow_flag'), ('wind_ge_12', 'snow_flag'), ('wind_ge_15', 'snow_flag'),
        ('heat_ge_85', 'humid_ge_70'), ('heat_ge_85', 'humid_ge_80'), ('heat_ge_90', 'humid_ge_70'),
        ('heat_ge_85', 'dewpoint_ge_65'), ('heat_ge_90', 'dewpoint_ge_70'),
        ('humid_ge_80', 'dewpoint_ge_70'), ('humid_ge_90', 'dewpoint_ge_70'),
    ]
    for a, b in weather_pair_keys:
        desc_a, fn_a = cat[a]
        desc_b, fn_b = cat[b]
        add(f'hgb_under_3p5_combo_{a}_{b}', 'weather_combo_grid', f'HGB under 3.5+ with {desc_a} and {desc_b}.', lambda d, fn_a=fn_a, fn_b=fn_b: base_hgb_under(d, 3.5) & fn_a(d) & fn_b(d))

    weather_triple_keys = [
        ('cold_le_50', 'wind_ge_12', 'precip_any'),
        ('cold_le_45', 'wind_ge_12', 'precip_any'),
        ('cold_le_50', 'wind_ge_12', 'snow_flag'),
        ('cold_le_45', 'wind_ge_12', 'snow_flag'),
        ('wind_ge_12', 'rain_flag', 'total_ge_56'),
        ('wind_ge_12', 'precip_any', 'total_ge_56'),
        ('cold_le_50', 'wind_ge_12', 'total_ge_56'),
        ('cold_le_50', 'precip_any', 'total_ge_56'),
        ('heat_ge_85', 'humid_ge_70', 'total_ge_56'),
        ('heat_ge_90', 'humid_ge_70', 'total_ge_56'),
        ('heat_ge_85', 'dewpoint_ge_70', 'high_pace'),
    ]
    for a, b, c in weather_triple_keys:
        descs = [cat[k][0] for k in (a, b, c)]
        fns = [cat[k][1] for k in (a, b, c)]
        add(f'hgb_under_3p5_combo_{a}_{b}_{c}', 'weather_combo_grid', 'HGB under 3.5+ with ' + ', '.join(descs[:-1]) + f', and {descs[-1]}.', lambda d, fns=fns: base_hgb_under(d, 3.5) & fns[0](d) & fns[1](d) & fns[2](d))

    # High-total + each weather condition and a few total-range + weather tests.
    weather_keys = ['wind_ge_10', 'wind_ge_12', 'wind_ge_15', 'wind_ge_18', 'cold_le_55', 'cold_le_50', 'cold_le_45', 'cold_le_40', 'heat_ge_85', 'heat_ge_90', 'heat_ge_95', 'humid_ge_70', 'humid_ge_80', 'dewpoint_ge_65', 'dewpoint_ge_70', 'precip_any', 'rain_flag', 'snow_flag', 'thunder_flag', 'wintry_flag']
    for total_key in ['total_ge_56', 'total_ge_58', 'total_ge_60', 'total_60_63']:
        desc_total, fn_total = cat[total_key]
        for key in weather_keys:
            desc_wx, fn_wx = cat[key]
            add(f'hgb_under_3p5_combo_{total_key}_{key}', 'weather_total_combo_grid', f'HGB under 3.5+ with {desc_total} and {desc_wx}.', lambda d, fn_total=fn_total, fn_wx=fn_wx: base_hgb_under(d, 3.5) & fn_total(d) & fn_wx(d))

    # Team-style + weather/total/market interaction grid.
    style_keys = ['high_pass_rate', 'high_pace', 'high_pass_yard_share']
    style_weather_keys = ['wind_ge_10', 'wind_ge_12', 'wind_ge_15', 'precip_any', 'rain_flag', 'snow_flag', 'cold_le_50', 'cold_le_45', 'heat_ge_85', 'heat_ge_90', 'humid_ge_70', 'humid_ge_80', 'dewpoint_ge_70']
    for style_key in style_keys:
        desc_style, fn_style = cat[style_key]
        for wx_key in style_weather_keys:
            desc_wx, fn_wx = cat[wx_key]
            add(f'hgb_under_3p5_combo_{style_key}_{wx_key}', 'weather_team_style_grid', f'HGB under 3.5+ with {desc_style} and {desc_wx}.', lambda d, fn_style=fn_style, fn_wx=fn_wx: base_hgb_under(d, 3.5) & fn_style(d) & fn_wx(d))
        for total_key in ['total_ge_56', 'total_ge_58', 'total_ge_60', 'total_60_63']:
            desc_total, fn_total = cat[total_key]
            add(f'hgb_under_3p5_combo_{style_key}_{total_key}', 'team_style_total_grid', f'HGB under 3.5+ with {desc_style} and {desc_total}.', lambda d, fn_style=fn_style, fn_total=fn_total: base_hgb_under(d, 3.5) & fn_style(d) & fn_total(d))
            for wx_key in ['wind_ge_12', 'wind_ge_15', 'precip_any', 'rain_flag', 'cold_le_50', 'heat_ge_85', 'humid_ge_70', 'dewpoint_ge_70']:
                desc_wx, fn_wx = cat[wx_key]
                add(f'hgb_under_3p5_combo_{style_key}_{total_key}_{wx_key}', 'weather_team_style_total_grid', f'HGB under 3.5+ with {desc_style}, {desc_total}, and {desc_wx}.', lambda d, fn_style=fn_style, fn_total=fn_total, fn_wx=fn_wx: base_hgb_under(d, 3.5) & fn_style(d) & fn_total(d) & fn_wx(d))

    # Market-position plus good historical signals.
    for market_key in ['provider_count_ge_3', 'line_range_ge_2', 'selected_above_median_1']:
        desc_market, fn_market = cat[market_key]
        for signal_key in ['total_ge_56', 'total_ge_60', 'wind_ge_12', 'high_pass_rate']:
            desc_signal, fn_signal = cat[signal_key]
            add(f'hgb_under_3p5_combo_{market_key}_{signal_key}', 'market_signal_combo_grid', f'HGB under 3.5+ with {desc_market} and {desc_signal}.', lambda d, fn_market=fn_market, fn_signal=fn_signal: base_hgb_under(d, 3.5) & fn_market(d) & fn_signal(d))

    return filters


def write_summary(summary: pd.DataFrame, by_season: pd.DataFrame) -> None:
    out = ensure_dir('outputs') / 'edge_refinement_methodology_summary.md'
    lines = [
        '# Edge Refinement Summary',
        '',
        'This module starts with the HGB under signal and tests whether additional filters improve the profile without relying only on raw ROI.',
        '',
        'This expanded version now includes a controlled interaction grid for cold, heat, humidity, dewpoint, wind, rain, snow, thunder, high totals, team style, model consensus, and line-market proxies.',
        '',
        '## Ranking logic',
        '',
        'Rows are ranked by a stability-first score that considers overall ROI, recent ROI, positive-season rate, sample size, and drawdown. A high ROI with a tiny sample is flagged rather than treated as a production rule.',
        '',
        '## Top shortlist candidates',
        '',
    ]
    if summary.empty:
        lines.append('_No edge-refinement rows generated._')
    else:
        shortlist = summary[summary['shortlist_candidate']].sort_values(['score', 'graded'], ascending=[False, False]).head(30)
        if shortlist.empty:
            lines.append('_No rows cleared the shortlist guardrails yet._')
        else:
            cols = [c for c in ['filter_name', 'category', 'graded', 'hit_rate', 'roi_per_1u', 'recent_2022_plus_roi', 'positive_season_rate', 'max_drawdown_units', 'overfit_flags', 'score'] if c in shortlist.columns]
            lines.append(shortlist[cols].to_markdown(index=False))
        lines.extend(['', '## Top rows by ROI before guardrails', ''])
        top_roi = summary.sort_values(['roi_per_1u', 'graded'], ascending=[False, False]).head(30)
        cols = [c for c in ['filter_name', 'category', 'graded', 'hit_rate', 'roi_per_1u', 'recent_2022_plus_roi', 'positive_season_rate', 'max_drawdown_units', 'overfit_flags'] if c in top_roi.columns]
        lines.append(top_roi[cols].to_markdown(index=False))
        lines.extend(['', '## Best rows by category', ''])
        cat_parts = []
        for cat_name, g in summary.sort_values(['score', 'graded'], ascending=[False, False]).groupby('category'):
            cat_parts.append(g.head(5))
        if cat_parts:
            by_cat = pd.concat(cat_parts, ignore_index=True)
            cols = [c for c in ['category', 'filter_name', 'graded', 'hit_rate', 'roi_per_1u', 'recent_2022_plus_roi', 'positive_season_rate', 'overfit_flags', 'score'] if c in by_cat.columns]
            lines.append(by_cat[cols].to_markdown(index=False))
    lines.extend([
        '',
        '## CLV / line movement note',
        '',
        'The historical dataset does not yet contain a true timestamped current line and final closing line for each betting decision. This module adds provider-dispersion proxies where available, but true CLV must be tracked live: decision-line total, closing total, and whether the strategy beat the close.',
        '',
        '## Production guardrails',
        '',
        '- Do not use a filter as a production rule only because it clears 10% historical ROI.',
        '- Prefer rows with at least 100 graded plays, positive recent ROI, and positive seasons across most years.',
        '- Treat tiny high-ROI filters as research leads, not plays.',
        '- Continue using live paper tracking before staking real money.',
    ])
    out.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    df = prep(read_df('data/processed/modeling_dataset.csv'))
    df = add_line_market_context(df)
    df = add_weather_flags(df)
    df = add_team_style_features(df)
    pred = train_walk_forward_predictions(df)
    if pred.empty:
        print('No walk-forward predictions generated for edge refinement.')
        write_df(pd.DataFrame(), 'outputs/edge_refinement_summary.csv')
        write_df(pd.DataFrame(), 'outputs/edge_refinement_by_season.csv')
        write_df(pd.DataFrame(), 'outputs/edge_combo_grid_summary.csv')
        write_df(pd.DataFrame(), 'outputs/edge_combo_grid_by_season.csv')
        return

    pred = add_weather_flags(pred)
    filters = build_filters(pred)
    rows = []
    season_parts = []
    inventory = []
    for filt in filters:
        inventory.append({'filter_name': filt.name, 'category': filt.category, 'description': filt.description})
        row, season_df = grade_subset(pred, filt)
        if row:
            rows.append(row)
        if not season_df.empty:
            season_parts.append(season_df)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(['score', 'graded'], ascending=[False, False])
    by_season = pd.concat(season_parts, ignore_index=True) if season_parts else pd.DataFrame()
    shortlist = summary[summary['shortlist_candidate']].copy() if not summary.empty else pd.DataFrame()
    if not shortlist.empty:
        shortlist = shortlist.sort_values(['score', 'graded'], ascending=[False, False])

    write_df(summary, 'outputs/edge_refinement_summary.csv')
    write_df(by_season, 'outputs/edge_refinement_by_season.csv')
    write_df(shortlist, 'outputs/edge_refinement_shortlist.csv')
    write_df(pd.DataFrame(inventory), 'outputs/edge_refinement_filter_inventory.csv')

    # Alias files for the expanded interaction-grid work, so future dashboard/code can reference either name.
    write_df(summary, 'outputs/edge_combo_grid_summary.csv')
    write_df(by_season, 'outputs/edge_combo_grid_by_season.csv')
    write_df(shortlist, 'outputs/edge_combo_grid_shortlist.csv')

    write_summary(summary, by_season)
    print(f'Wrote {len(summary):,} edge-refinement rows from {len(filters):,} filters and {len(shortlist):,} shortlist rows')


if __name__ == '__main__':
    main()
