from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .utils import ROOT, ensure_dir


def csv_preview(path: Path, max_rows: int = 10) -> str:
    if not path.exists():
        return '_Not available yet._'
    df = pd.read_csv(path)
    if df.empty:
        return '_File exists but has no rows._'
    return df.head(max_rows).to_markdown(index=False)


def append_markdown(lines: list[str], heading: str, path: Path, fallback: str) -> None:
    lines.extend(['', heading, ''])
    if path.exists():
        lines.append(path.read_text(encoding='utf-8'))
    else:
        lines.append(fallback)


def run_edge_refinement_if_needed() -> None:
    # The manual workflow already runs src.report before the dashboard is built.
    # Running edge refinement here avoids needing to edit the workflow YAML directly.
    try:
        from . import edge_refinement
        edge_refinement.main()
    except Exception as exc:  # Keep the rest of the report/dashboard generation alive.
        print(f'Edge refinement step failed; continuing report generation: {exc}')


def main() -> None:
    run_edge_refinement_if_needed()
    outputs = ensure_dir('outputs')
    generated = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines = [
        '# Weekly CFB Weather Totals Report',
        '',
        f'Generated: {generated}',
        '',
        '> Paper tracking only. This report is not a guarantee of profit or betting advice.',
        '',
        '## Weekly picks',
        '',
        csv_preview(ROOT / 'outputs/weekly_picks.csv'),
        '',
        '## Starter backtest summary',
        '',
        csv_preview(ROOT / 'outputs/backtest_summary.csv'),
        '',
        '## Walk-forward strategy summary',
        '',
        csv_preview(ROOT / 'outputs/walk_forward_strategy_summary.csv'),
        '',
        '## Model bake-off summary',
        '',
        csv_preview(ROOT / 'outputs/model_bakeoff_summary.csv'),
        '',
        '## Model edge recent-period validation',
        '',
        csv_preview(ROOT / 'outputs/model_edge_by_recent_period.csv'),
        '',
        '## Edge refinement shortlist',
        '',
        csv_preview(ROOT / 'outputs/edge_refinement_shortlist.csv'),
        '',
        '## Detailed rule summary',
        '',
        csv_preview(ROOT / 'outputs/rule_backtest_detailed.csv'),
    ]
    append_markdown(lines, '## Edge refinement detail', ROOT / 'outputs/edge_refinement_methodology_summary.md', '_Edge refinement summary not generated yet._')
    append_markdown(lines, '## Model edge validation detail', ROOT / 'outputs/model_edge_validation_summary.md', '_Model edge validation summary not generated yet._')
    append_markdown(lines, '## Model bake-off detail', ROOT / 'outputs/model_bakeoff_summary.md', '_Model bake-off summary not generated yet._')
    append_markdown(lines, '## Deep research summary', ROOT / 'outputs/deep_research_summary.md', '_Deep research summary not generated yet._')
    append_markdown(lines, '## Starter research summary', ROOT / 'outputs/research_summary.md', '_Starter research summary not generated yet._')
    out = outputs / 'weekly_report.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
