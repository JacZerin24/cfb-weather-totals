from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import brier_score_loss, mean_absolute_error

from ..utils import ensure_dir, read_df, write_df
from .forecast_native_bakeoff import (
    CONTEXT_CATS,
    CONTEXT_NUMS,
    _available_categorical,
    _available_numeric,
)
from .model_bakeoff import BREAKEVEN, _prep
from .roi_search import _deep_models


DATA_PATH = 'data/nfl/processed/team_context_long_history.csv'
MIN_TRAIN = 700
TEST_SEASONS = tuple(range(2009, 2018))
DISCOVERY_SEASONS = tuple(range(2009, 2014))
CONFIRMATION_SEASONS = tuple(range(2014, 2018))
SELECTION_PATH = 'outputs/nfl/team_context_phase2_selection.json'

RULES = {
    'qualifies_3pt_60pct': (3.0, 0.60),
    'strong_4pt_60pct': (4.0, 0.60),
}

FEATURE_GROUPS = {
    'efficiency_core': [
        'team_epa_per_play_mean',
        'team_epa_per_play_absdiff',
        'team_def_epa_per_play_allowed_mean',
        'team_success_rate_mean',
        'team_success_rate_absdiff',
        'team_def_success_rate_allowed_mean',
        'matchup_epa_mean',
        'matchup_success_mean',
    ],
    'passing_core': [
        'team_pass_epa_mean',
        'team_pass_epa_absdiff',
        'team_def_pass_epa_allowed_mean',
        'matchup_pass_epa_mean',
        'team_sack_rate_mean',
        'team_def_sack_rate_allowed_mean',
        'team_pass_rate_mean',
        'team_pass_rate_absdiff',
    ],
    'rushing_core': [
        'team_rush_epa_mean',
        'team_rush_epa_absdiff',
        'team_def_rush_epa_allowed_mean',
        'matchup_rush_epa_mean',
    ],
    'explosive_turnover': [
        'team_explosive_rate_mean',
        'team_explosive_rate_absdiff',
        'team_def_explosive_rate_allowed_mean',
        'team_turnover_rate_mean',
        'team_turnover_rate_absdiff',
        'team_def_turnover_rate_allowed_mean',
    ],
    'early_down': [
        'team_early_down_epa_mean',
        'team_early_down_epa_absdiff',
        'team_def_early_down_epa_allowed_mean',
    ],
}
FEATURE_GROUPS['efficiency_plus_passing'] = list(
    dict.fromkeys(
        FEATURE_GROUPS['efficiency_core']
        + FEATURE_GROUPS['passing_core']
    )
)
FEATURE_GROUPS['compact_all'] = list(
    dict.fromkeys(
        FEATURE_GROUPS['efficiency_core']
        + FEATURE_GROUPS['passing_core']
        + FEATURE_GROUPS['rushing_core']
        + FEATURE_GROUPS['explosive_turnover']
        + FEATURE_GROUPS['early_down']
    )
)


def _decision_roof(value: object) -> str:
    roof = str(value or '').strip().lower()
    if roof == 'outdoors':
        return 'outdoors'
    if roof == 'dome':
        return 'dome'
    if roof in {'open', 'closed'}:
        return 'retractable'
    return 'unknown'


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out['season'] = pd.to_numeric(out['season'], errors='coerce')
    if 'decision_roof' not in out.columns:
        out['decision_roof'] = out.get(
            'roof', pd.Series('', index=out.index)
        ).map(_decision_roof)
    return out


def _feature_columns(
    frame: pd.DataFrame,
    football_columns: list[str],
) -> tuple[list[str], list[str]]:
    nums = _available_numeric(
        frame,
        list(CONTEXT_NUMS) + list(football_columns),
    )
    cats = _available_categorical(frame, list(CONTEXT_CATS))
    return nums, cats


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    football_columns: list[str],
) -> pd.DataFrame:
    nums, cats = _feature_columns(
        pd.concat([train, test], ignore_index=True),
        football_columns,
    )
    train_work = _prep(train, nums, cats)
    test_work = _prep(test, nums, cats)

    models = _deep_models(nums, cats)
    reg = models['reg_rf_leaf25'][1]
    cls = models['cls_rf_leaf25'][1]

    reg.fit(train_work[nums + cats], train_work['market_residual'])
    cls_train = train_work[train_work['market_residual'].ne(0)].copy()
    cls.fit(
        cls_train[nums + cats],
        cls_train['market_residual'].gt(0).astype(int),
    )

    out = test[
        [
            c for c in [
                'game_id', 'season', 'week', 'gameday',
                'away_team', 'home_team',
                'closing_total', 'actual_total_points',
                'market_residual',
            ]
            if c in test.columns
        ]
    ].copy()
    out['reg_signal'] = reg.predict(test_work[nums + cats])
    out['over_probability'] = cls.predict_proba(
        test_work[nums + cats]
    )[:, 1]
    return out


def _walk_forward(frame: pd.DataFrame) -> pd.DataFrame:
    systems = {'context_only': []}
    systems.update(FEATURE_GROUPS)
    parts: list[pd.DataFrame] = []

    for season in TEST_SEASONS:
        train = frame[frame['season'].lt(season)].copy()
        test = frame[frame['season'].eq(season)].copy()
        if len(train) < MIN_TRAIN or test.empty:
            continue

        for name, football_columns in systems.items():
            missing = [
                c for c in football_columns
                if c not in frame.columns
            ]
            if missing:
                raise RuntimeError(
                    f'{name} missing preregistered columns: {missing}'
                )
            pred = _fit_predict(train, test, football_columns)
            pred['system'] = name
            parts.append(pred)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _prediction_metrics(frame: pd.DataFrame) -> dict:
    residual = pd.to_numeric(frame['market_residual'], errors='coerce')
    reg = pd.to_numeric(frame['reg_signal'], errors='coerce')
    prob = pd.to_numeric(frame['over_probability'], errors='coerce')

    valid_reg = residual.notna() & reg.notna()
    valid_prob = residual.ne(0) & prob.notna()
    binary = residual.gt(0).astype(int)

    return {
        'games': int(len(frame)),
        'mae_residual': (
            mean_absolute_error(residual[valid_reg], reg[valid_reg])
            if valid_reg.any() else np.nan
        ),
        'brier_over': (
            brier_score_loss(binary[valid_prob], prob[valid_prob])
            if valid_prob.any() else np.nan
        ),
    }


def _grade(frame: pd.DataFrame) -> dict:
    residual = pd.to_numeric(frame['market_residual'], errors='coerce')
    wins = int(residual.gt(0).sum())
    losses = int(residual.lt(0).sum())
    pushes = int(residual.eq(0).sum())
    graded = wins + losses
    flat_units = wins * (100 / 110) - losses
    return {
        'plays': int(len(frame)),
        'graded': graded,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'hit_rate': wins / graded if graded else np.nan,
        'flat_roi_minus110': flat_units / graded if graded else np.nan,
        'p_value_vs_minus110': (
            binomtest(
                wins,
                graded,
                p=BREAKEVEN,
                alternative='greater',
            ).pvalue
            if graded else np.nan
        ),
    }


def _mask(
    frame: pd.DataFrame,
    reg_threshold: float,
    probability_threshold: float,
) -> pd.Series:
    return (
        pd.to_numeric(frame['reg_signal'], errors='coerce')
        .ge(reg_threshold)
        & pd.to_numeric(frame['over_probability'], errors='coerce')
        .ge(probability_threshold)
    )


def _block_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    blocks = {
        'discovery_2009_2013': DISCOVERY_SEASONS,
        'confirmation_2014_2017': CONFIRMATION_SEASONS,
        'all_pre2018_oos': TEST_SEASONS,
    }
    rows: list[dict] = []

    for block, seasons in blocks.items():
        subset = predictions[predictions['season'].isin(seasons)]
        for system, group in subset.groupby('system'):
            row = {
                'block': block,
                'system': system,
                **_prediction_metrics(group),
            }
            for rule, (reg_threshold, prob_threshold) in RULES.items():
                plays = group[
                    _mask(group, reg_threshold, prob_threshold)
                ]
                grade = _grade(plays)
                prefix = 'qualifies' if rule.startswith('qualifies') else 'strong'
                for key, value in grade.items():
                    row[f'{prefix}_{key}'] = value
            rows.append(row)
    return pd.DataFrame(rows)


def _season_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (season, system), group in predictions.groupby(
        ['season', 'system']
    ):
        rows.append({
            'season': int(season),
            'system': system,
            **_prediction_metrics(group),
        })
    return pd.DataFrame(rows)


def _gain(
    baseline: pd.Series,
    candidate: pd.Series,
) -> dict:
    base_mae = float(baseline['mae_residual'])
    cand_mae = float(candidate['mae_residual'])
    base_brier = float(baseline['brier_over'])
    cand_brier = float(candidate['brier_over'])

    mae_diff = cand_mae - base_mae
    brier_diff = cand_brier - base_brier
    relative_mae_gain = (
        (base_mae - cand_mae) / base_mae
        if np.isfinite(base_mae) and base_mae != 0 else np.nan
    )
    relative_brier_gain = (
        (base_brier - cand_brier) / base_brier
        if np.isfinite(base_brier) and base_brier != 0 else np.nan
    )
    composite_gain = (
        0.5 * relative_mae_gain + 0.5 * relative_brier_gain
        if np.isfinite(relative_mae_gain)
        and np.isfinite(relative_brier_gain)
        else np.nan
    )
    return {
        'mae_diff_candidate_minus_baseline': mae_diff,
        'brier_diff_candidate_minus_baseline': brier_diff,
        'relative_mae_gain': relative_mae_gain,
        'relative_brier_gain': relative_brier_gain,
        'composite_gain': composite_gain,
    }


def _selection_table(
    blocks: pd.DataFrame,
    seasons: pd.DataFrame,
) -> pd.DataFrame:
    baseline_blocks = (
        blocks[blocks['system'].eq('context_only')]
        .set_index('block')
    )
    baseline_seasons = (
        seasons[seasons['system'].eq('context_only')]
        .set_index('season')
    )

    rows: list[dict] = []
    for candidate, columns in FEATURE_GROUPS.items():
        candidate_blocks = (
            blocks[blocks['system'].eq(candidate)]
            .set_index('block')
        )
        discovery = _gain(
            baseline_blocks.loc['discovery_2009_2013'],
            candidate_blocks.loc['discovery_2009_2013'],
        )
        confirmation = _gain(
            baseline_blocks.loc['confirmation_2014_2017'],
            candidate_blocks.loc['confirmation_2014_2017'],
        )

        positive_confirmation_seasons = 0
        season_composites: list[float] = []
        candidate_seasons = (
            seasons[seasons['system'].eq(candidate)]
            .set_index('season')
        )
        for season in CONFIRMATION_SEASONS:
            if (
                season not in baseline_seasons.index
                or season not in candidate_seasons.index
            ):
                continue
            gain = _gain(
                baseline_seasons.loc[season],
                candidate_seasons.loc[season],
            )
            comp = gain['composite_gain']
            season_composites.append(comp)
            if np.isfinite(comp) and comp > 0:
                positive_confirmation_seasons += 1

        eligible = bool(
            np.isfinite(discovery['composite_gain'])
            and discovery['composite_gain'] > 0
            and np.isfinite(confirmation['composite_gain'])
            and confirmation['composite_gain'] > 0
            and confirmation['mae_diff_candidate_minus_baseline'] <= 0.05
            and confirmation['brier_diff_candidate_minus_baseline'] <= 0.002
            and positive_confirmation_seasons >= 2
        )

        rows.append({
            'candidate': candidate,
            'feature_count': len(columns),
            'eligible': eligible,
            'positive_confirmation_seasons': positive_confirmation_seasons,
            'mean_confirmation_season_composite': (
                float(np.nanmean(season_composites))
                if season_composites else np.nan
            ),
            **{
                f'discovery_{key}': value
                for key, value in discovery.items()
            },
            **{
                f'confirmation_{key}': value
                for key, value in confirmation.items()
            },
        })

    return pd.DataFrame(rows)


def _select(selection: pd.DataFrame) -> dict:
    eligible = selection[selection['eligible']].copy()
    if eligible.empty:
        return {
            'selected_candidate': None,
            'selected_features': [],
            'selection_status': 'NO_CANDIDATE_PASSED_PREREGISTERED_GATES',
        }

    eligible = eligible.sort_values(
        [
            'confirmation_composite_gain',
            'feature_count',
            'candidate',
        ],
        ascending=[False, True, True],
        kind='stable',
    )
    winner = eligible.iloc[0]
    candidate = str(winner['candidate'])
    return {
        'selected_candidate': candidate,
        'selected_features': FEATURE_GROUPS[candidate],
        'selection_status': 'SELECTED_BY_PREREGISTERED_GATES',
        'feature_count': int(winner['feature_count']),
        'confirmation_composite_gain': float(
            winner['confirmation_composite_gain']
        ),
        'discovery_composite_gain': float(
            winner['discovery_composite_gain']
        ),
    }


def main() -> None:
    df = _prepare_frame(read_df(DATA_PATH))
    df = df[df['season'].between(2006, 2017)].copy()
    df = df[
        df['team_context_eligible'].fillna(False).astype(bool)
    ].copy()
    df = df.dropna(
        subset=[
            'closing_total',
            'actual_total_points',
            'market_residual',
        ]
    )

    predictions = _walk_forward(df)
    if predictions.empty:
        raise RuntimeError(
            'Long-history team-context discovery produced no predictions.'
        )

    blocks = _block_rows(predictions)
    seasons = _season_rows(predictions)
    selection = _selection_table(blocks, seasons)
    result = _select(selection)

    write_df(
        predictions,
        'outputs/nfl/team_context_long_discovery_predictions.csv',
    )
    write_df(
        blocks,
        'outputs/nfl/team_context_long_discovery_blocks.csv',
    )
    write_df(
        seasons,
        'outputs/nfl/team_context_long_discovery_by_season.csv',
    )
    write_df(
        selection,
        'outputs/nfl/team_context_long_discovery_selection_table.csv',
    )

    out_dir = ensure_dir('outputs/nfl')
    selection_path = out_dir / 'team_context_phase2_selection.json'
    selection_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding='utf-8',
    )

    lines = [
        '# NFL Long-History Team-Context Discovery',
        '',
        'Feature selection in this report uses only pre-2018 games.',
        '',
        '## Preregistered selection result',
        '',
        f"- Status: **{result['selection_status']}**",
        f"- Selected candidate: **{result.get('selected_candidate')}**",
        f"- Selected feature count: **{len(result.get('selected_features', []))}**",
        '',
        '## Candidate selection table',
        '',
        selection.to_markdown(index=False),
        '',
        '## Block performance',
        '',
        blocks.to_markdown(index=False),
        '',
        'No 2018-2025 result is used by the selection function above.',
    ]
    (out_dir / 'team_context_long_discovery.md').write_text(
        '\n'.join(lines),
        encoding='utf-8',
    )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
