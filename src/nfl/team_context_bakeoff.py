from __future__ import annotations

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
    _feature_columns,
    _weather_features,
)
from .model_bakeoff import BREAKEVEN, _prep
from .roi_search import _deep_models


DATA_PATH = 'data/nfl/processed/team_context_features.csv'
LEAD = 24
MIN_TRAIN = 700
TEST_SEASONS = (2021, 2022, 2023, 2024, 2025)
DEVELOPMENT_SEASONS = (2021, 2022, 2023, 2024)
FINAL_HOLDOUT_SEASON = 2025
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20261004

RULES = {
    'qualifies_3pt_60pct': (3.0, 0.60),
    'strong_4pt_60pct': (4.0, 0.60),
}


def _football_features(frame: pd.DataFrame) -> list[str]:
    return sorted(
        c for c in frame.columns
        if (
            (c.startswith('team_') and c.endswith(('_mean', '_absdiff')))
            or c.startswith('matchup_')
        )
        and pd.to_numeric(frame[c], errors='coerce').notna().any()
    )


def _feature_sets(
    frame: pd.DataFrame,
) -> dict[str, tuple[list[str], list[str]]]:
    baseline_nums, baseline_cats = _feature_columns(frame, LEAD, True)
    football = _football_features(frame)

    context_nums = _available_numeric(frame, CONTEXT_NUMS)
    context_cats = _available_categorical(frame, CONTEXT_CATS)

    football_nums = list(dict.fromkeys(context_nums + football))
    combined_nums = list(dict.fromkeys(baseline_nums + football))

    return {
        'v1_baseline': (baseline_nums, baseline_cats),
        'football_context': (football_nums, context_cats),
        'combined': (combined_nums, baseline_cats),
    }


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    nums: list[str],
    cats: list[str],
) -> pd.DataFrame:
    train_work = _prep(train, nums, cats)
    test_work = _prep(test, nums, cats)

    models = _deep_models(nums, cats)
    reg = models['reg_rf_leaf25'][1]
    cls = models['cls_rf_leaf25'][1]

    reg.fit(train_work[nums + cats], train_work['market_residual'])

    cls_train = train_work[train_work['market_residual'].ne(0)].copy()
    cls_target = cls_train['market_residual'].gt(0).astype(int)
    cls.fit(cls_train[nums + cats], cls_target)

    out = test[
        [
            c for c in [
                'game_id', 'season', 'week', 'gameday',
                'away_team', 'home_team', 'closing_total',
                'actual_total_points', 'market_residual',
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
    feature_sets = _feature_sets(frame)
    parts: list[pd.DataFrame] = []

    for season in TEST_SEASONS:
        train = frame[frame['season'].lt(season)].copy()
        test = frame[frame['season'].eq(season)].copy()
        if len(train) < MIN_TRAIN or test.empty:
            continue

        season_predictions: dict[str, pd.DataFrame] = {}
        for name, (nums, cats) in feature_sets.items():
            pred = _fit_predict(train, test, nums, cats)
            pred['system'] = name
            season_predictions[name] = pred
            parts.append(pred)

        baseline = season_predictions['v1_baseline'].set_index('game_id')
        football = season_predictions['football_context'].set_index('game_id')
        common = baseline.index.intersection(football.index)
        blend = baseline.loc[common].copy()
        blend['reg_signal'] = (
            baseline.loc[common, 'reg_signal']
            + football.loc[common, 'reg_signal']
        ) / 2.0
        blend['over_probability'] = (
            baseline.loc[common, 'over_probability']
            + football.loc[common, 'over_probability']
        ) / 2.0
        blend['system'] = 'equal_blend'
        parts.append(blend.reset_index())

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _wilson(wins: int, graded: int) -> tuple[float, float]:
    if graded <= 0:
        return np.nan, np.nan
    z = 1.96
    p = wins / graded
    denom = 1 + z * z / graded
    center = (p + z * z / (2 * graded)) / denom
    margin = (
        z
        * np.sqrt(
            (p * (1 - p) + z * z / (4 * graded))
            / graded
        )
        / denom
    )
    return center - margin, center + margin


def _prediction_metrics(frame: pd.DataFrame) -> dict:
    residual = pd.to_numeric(frame['market_residual'], errors='coerce')
    reg = pd.to_numeric(frame['reg_signal'], errors='coerce')
    valid_reg = residual.notna() & reg.notna()

    outcome = residual.gt(0).astype(int)
    prob = pd.to_numeric(frame['over_probability'], errors='coerce')
    valid_prob = residual.ne(0) & prob.notna()

    return {
        'games': int(len(frame)),
        'mae_residual': (
            mean_absolute_error(residual[valid_reg], reg[valid_reg])
            if valid_reg.any() else np.nan
        ),
        'brier_over': (
            brier_score_loss(outcome[valid_prob], prob[valid_prob])
            if valid_prob.any() else np.nan
        ),
        'mean_pred_residual': float(reg[valid_reg].mean())
        if valid_reg.any() else np.nan,
        'mean_over_probability': float(prob[valid_prob].mean())
        if valid_prob.any() else np.nan,
    }


def _grade_qualifier(frame: pd.DataFrame) -> dict:
    residual = pd.to_numeric(frame['market_residual'], errors='coerce')
    wins = int(residual.gt(0).sum())
    losses = int(residual.lt(0).sum())
    pushes = int(residual.eq(0).sum())
    graded = wins + losses
    flat_units = wins * (100 / 110) - losses
    low, high = _wilson(wins, graded)
    return {
        'games': int(len(frame)),
        'graded': graded,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'hit_rate': wins / graded if graded else np.nan,
        'wilson95_low': low,
        'wilson95_high': high,
        'flat_units_minus110': flat_units,
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


def _qualifier_mask(
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


def _summary_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict] = []
    qualifier_rows: list[dict] = []
    agreement_rows: list[dict] = []

    subsets = {
        'development_2021_2024': DEVELOPMENT_SEASONS,
        'holdout_2025': (FINAL_HOLDOUT_SEASON,),
        'all_oos_2021_2025': TEST_SEASONS,
    }

    for subset_name, seasons in subsets.items():
        subset = predictions[predictions['season'].isin(seasons)].copy()
        for system, group in subset.groupby('system'):
            prediction_rows.append({
                'subset': subset_name,
                'system': system,
                **_prediction_metrics(group),
            })
            for rule, (reg_threshold, prob_threshold) in RULES.items():
                plays = group[
                    _qualifier_mask(group, reg_threshold, prob_threshold)
                ].copy()
                qualifier_rows.append({
                    'subset': subset_name,
                    'system': system,
                    'rule': rule,
                    'reg_threshold': reg_threshold,
                    'probability_threshold': prob_threshold,
                    **_grade_qualifier(plays),
                })

        baseline = subset[
            subset['system'].eq('v1_baseline')
        ].set_index('game_id')
        for challenger_name in [
            'football_context', 'combined', 'equal_blend'
        ]:
            challenger = subset[
                subset['system'].eq(challenger_name)
            ].set_index('game_id')
            common = baseline.index.intersection(challenger.index)
            if common.empty:
                continue
            for rule, (reg_threshold, prob_threshold) in RULES.items():
                base_mask = _qualifier_mask(
                    baseline.loc[common],
                    reg_threshold,
                    prob_threshold,
                )
                challenger_mask = _qualifier_mask(
                    challenger.loc[common],
                    reg_threshold,
                    prob_threshold,
                )
                ids = common[base_mask.to_numpy() & challenger_mask.to_numpy()]
                plays = baseline.loc[ids].reset_index()
                agreement_rows.append({
                    'subset': subset_name,
                    'challenger': challenger_name,
                    'rule': rule,
                    'reg_threshold': reg_threshold,
                    'probability_threshold': prob_threshold,
                    **_grade_qualifier(plays),
                })

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(qualifier_rows),
        pd.DataFrame(agreement_rows),
    )


def _season_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (season, system), group in predictions.groupby(
        ['season', 'system']
    ):
        base = {
            'season': int(season),
            'system': system,
            **_prediction_metrics(group),
        }
        for rule, (reg_threshold, prob_threshold) in RULES.items():
            plays = group[
                _qualifier_mask(group, reg_threshold, prob_threshold)
            ]
            graded = _grade_qualifier(plays)
            prefix = rule.replace('_3pt_60pct', '').replace(
                '_4pt_60pct', ''
            )
            base[f'{prefix}_plays'] = graded['games']
            base[f'{prefix}_hit_rate'] = graded['hit_rate']
            base[f'{prefix}_roi'] = graded['flat_roi_minus110']
        rows.append(base)
    return pd.DataFrame(rows)


def _paired_bootstrap(
    predictions: pd.DataFrame,
    subset_name: str,
    seasons: tuple[int, ...],
) -> pd.DataFrame:
    subset = predictions[predictions['season'].isin(seasons)]
    baseline = subset[
        subset['system'].eq('v1_baseline')
    ].set_index('game_id')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict] = []

    for challenger_name in [
        'football_context', 'combined', 'equal_blend'
    ]:
        challenger = subset[
            subset['system'].eq(challenger_name)
        ].set_index('game_id')
        common = baseline.index.intersection(challenger.index)
        if len(common) < 20:
            continue

        base = baseline.loc[common]
        ch = challenger.loc[common]
        actual = pd.to_numeric(base['market_residual'], errors='coerce').to_numpy()
        base_reg = pd.to_numeric(base['reg_signal'], errors='coerce').to_numpy()
        ch_reg = pd.to_numeric(ch['reg_signal'], errors='coerce').to_numpy()
        base_prob = pd.to_numeric(
            base['over_probability'], errors='coerce'
        ).to_numpy()
        ch_prob = pd.to_numeric(
            ch['over_probability'], errors='coerce'
        ).to_numpy()

        valid = (
            np.isfinite(actual)
            & np.isfinite(base_reg)
            & np.isfinite(ch_reg)
            & np.isfinite(base_prob)
            & np.isfinite(ch_prob)
        )
        actual = actual[valid]
        base_reg = base_reg[valid]
        ch_reg = ch_reg[valid]
        base_prob = base_prob[valid]
        ch_prob = ch_prob[valid]
        n = len(actual)
        if n < 20:
            continue

        mae_diffs = np.empty(BOOTSTRAP_SAMPLES)
        brier_diffs = np.empty(BOOTSTRAP_SAMPLES)
        binary = (actual > 0).astype(float)
        nonpush = actual != 0

        for i in range(BOOTSTRAP_SAMPLES):
            idx = rng.integers(0, n, size=n)
            mae_diffs[i] = (
                np.mean(np.abs(actual[idx] - ch_reg[idx]))
                - np.mean(np.abs(actual[idx] - base_reg[idx]))
            )
            bmask = nonpush[idx]
            if not bmask.any():
                brier_diffs[i] = np.nan
            else:
                sampled = idx[bmask]
                brier_diffs[i] = (
                    np.mean((binary[sampled] - ch_prob[sampled]) ** 2)
                    - np.mean((binary[sampled] - base_prob[sampled]) ** 2)
                )

        brier_valid = brier_diffs[np.isfinite(brier_diffs)]
        rows.append({
            'subset': subset_name,
            'challenger': challenger_name,
            'paired_games': n,
            'mae_diff_challenger_minus_v1': (
                np.mean(np.abs(actual - ch_reg))
                - np.mean(np.abs(actual - base_reg))
            ),
            'mae_diff_ci95_low': float(np.quantile(mae_diffs, 0.025)),
            'mae_diff_ci95_high': float(np.quantile(mae_diffs, 0.975)),
            'bootstrap_probability_mae_better': float(
                np.mean(mae_diffs < 0)
            ),
            'brier_diff_challenger_minus_v1': (
                np.mean((binary[nonpush] - ch_prob[nonpush]) ** 2)
                - np.mean((binary[nonpush] - base_prob[nonpush]) ** 2)
            ),
            'brier_diff_ci95_low': (
                float(np.quantile(brier_valid, 0.025))
                if len(brier_valid) else np.nan
            ),
            'brier_diff_ci95_high': (
                float(np.quantile(brier_valid, 0.975))
                if len(brier_valid) else np.nan
            ),
            'bootstrap_probability_brier_better': (
                float(np.mean(brier_valid < 0))
                if len(brier_valid) else np.nan
            ),
        })

    return pd.DataFrame(rows)


def main() -> None:
    df = read_df(DATA_PATH).copy()
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df = df[
        df['season'].between(min(TEST_SEASONS) - 3, max(TEST_SEASONS))
    ].copy()
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
    df = _weather_features(df, LEAD)
    df = df[df[f'forecast_complete_{LEAD}h'].fillna(False)].copy()

    predictions = _walk_forward(df)
    if predictions.empty:
        raise RuntimeError('NFL team-context challenger produced no predictions.')

    prediction_summary, qualifier_summary, agreement_summary = (
        _summary_tables(predictions)
    )
    seasons = _season_table(predictions)
    bootstrap = pd.concat(
        [
            _paired_bootstrap(
                predictions,
                'development_2021_2024',
                DEVELOPMENT_SEASONS,
            ),
            _paired_bootstrap(
                predictions,
                'holdout_2025',
                (FINAL_HOLDOUT_SEASON,),
            ),
        ],
        ignore_index=True,
    )

    write_df(
        predictions,
        'outputs/nfl/team_context_oos_predictions.csv',
    )
    write_df(
        prediction_summary,
        'outputs/nfl/team_context_prediction_summary.csv',
    )
    write_df(
        qualifier_summary,
        'outputs/nfl/team_context_qualifier_summary.csv',
    )
    write_df(
        agreement_summary,
        'outputs/nfl/team_context_agreement_summary.csv',
    )
    write_df(
        seasons,
        'outputs/nfl/team_context_by_season.csv',
    )
    write_df(
        bootstrap,
        'outputs/nfl/team_context_bootstrap.csv',
    )

    football_features = _football_features(df)
    lines = [
        '# NFL Team-Context Challenger — Walk-Forward Results',
        '',
        'This report implements the preregistered NFL team-context challenger. '
        'The frozen v1 protocol is not modified.',
        '',
        '## Fixed design',
        '',
        f'- Primary forecast horizon: {LEAD} hours.',
        f'- Football rolling window: 8 completed games; minimum prior games: 4.',
        '- Models: RF leaf-25 regression + RF leaf-25 classifier.',
        '- Systems: v1 baseline, football context, combined, fixed 50/50 blend.',
        '- Development OOS: 2021-2024.',
        '- Final historical holdout: 2025.',
        '- Qualifier thresholds are inherited unchanged from v1.',
        f'- Football feature count: {len(football_features)}.',
        '',
        '## Prediction quality',
        '',
        prediction_summary.to_markdown(index=False),
        '',
        '## Fixed qualifier performance',
        '',
        qualifier_summary.to_markdown(index=False),
        '',
        '## Independent agreement with v1',
        '',
        (
            agreement_summary.to_markdown(index=False)
            if not agreement_summary.empty
            else '_No agreement plays._'
        ),
        '',
        '## Season-by-season',
        '',
        seasons.to_markdown(index=False),
        '',
        '## Paired bootstrap versus v1',
        '',
        (
            bootstrap.to_markdown(index=False)
            if not bootstrap.empty
            else '_No bootstrap comparisons._'
        ),
        '',
        'Negative MAE/Brier differences favor the challenger. Bootstrap '
        'intervals are descriptive uncertainty checks, not a license to '
        'retune the frozen v1 rule.',
    ]
    out = ensure_dir('outputs/nfl') / 'team_context_bakeoff.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote team-context challenger outputs under {out.parent}')


if __name__ == '__main__':
    main()
