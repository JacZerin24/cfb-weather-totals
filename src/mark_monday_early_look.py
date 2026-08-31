from __future__ import annotations

import os
from pathlib import Path

from .utils import ROOT

MONDAY_EARLY_LOOK_CRON = '0 14 * * 1'
BANNER_ID = 'monday-early-look-banner'
REPORT_NOTICE = '> **Monday Early Look — preliminary / non-official.** Use this as a baseline for the week. Official prospective snapshots begin Thursday; Monday does not count toward the frozen 2026 prospective or orientation evaluation records.'


def is_monday_early_look() -> bool:
    event_name = os.getenv('WEEKLY_EVENT_NAME', os.getenv('GITHUB_EVENT_NAME', '')).strip()
    event_schedule = os.getenv('WEEKLY_EVENT_SCHEDULE', '').strip()
    return event_name == 'schedule' and event_schedule == MONDAY_EARLY_LOOK_CRON


def mark_html(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f'Monday early-look marker expected dashboard file: {path}')

    text = path.read_text(encoding='utf-8')
    if BANNER_ID in text:
        return
    if '<body>' not in text:
        raise RuntimeError(f'Could not find <body> in dashboard file: {path}')

    banner = f'''
  <div id="{BANNER_ID}" style="background:#7c2d12;color:#ffedd5;border-bottom:1px solid rgba(251,146,60,.45);padding:10px 14px;text-align:center;font-weight:800;">
    Monday Early Look — preliminary / non-official. Use this as a baseline for the week; official prospective snapshots begin Thursday.
  </div>'''
    text = text.replace('<body>', '<body>' + banner, 1)
    path.write_text(text, encoding='utf-8')


def mark_report(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f'Monday early-look marker expected report file: {path}')

    text = path.read_text(encoding='utf-8')
    if REPORT_NOTICE in text:
        return

    title = '# Weekly CFB Weather Totals Report'
    if title not in text:
        raise RuntimeError(f'Could not find report title in: {path}')

    text = text.replace(title, f'{title}\n\n{REPORT_NOTICE}', 1)
    path.write_text(text, encoding='utf-8')


def main() -> None:
    if not is_monday_early_look():
        print('Standard weekly run; Monday early-look label not applied.')
        return

    mark_html(ROOT / 'docs/index.html')
    mark_html(ROOT / 'outputs/live_dashboard.html')
    mark_report(ROOT / 'outputs/weekly_report.md')
    print('Applied Monday Early Look preliminary/non-official label to dashboard and weekly report.')


if __name__ == '__main__':
    main()
