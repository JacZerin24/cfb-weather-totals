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


def main() -> None:
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
        '## Detailed rule summary',
        '',
        csv_preview(ROOT / 'outputs/rule_backtest_detailed.csv'),
    ]
    append_markdown(lines, '## Model edge validation detail', ROOT / 'outputs/model_edge_validation_summary.md', '_Model edge validation summary not generated yet._')
    append_markdown(lines, '## Model bake-off detail', ROOT / 'outputs/model_bakeoff_summary.md', '_Model bake-off summary not generated yet._')
    append_markdown(lines, '## Deep research summary', ROOT / 'outputs/deep_research_summary.md', '_Deep research summary not generated yet._')
    append_markdown(lines, '## Starter research summary', ROOT / 'outputs/research_summary.md', '_Starter research summary not generated yet._')
    out = outputs / 'weekly_report.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
