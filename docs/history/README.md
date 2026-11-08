# Read-only historical CFB week viewer

Archived weekly boards are **generated into the GitHub Pages deployment artifact**, not
committed to this directory. The deployment runs `python -m src.build_week_history`
against existing immutable pregame snapshot CSVs and the derived prospective ledger.

Only the public website's HTML is affected. Forecast scoring, thresholds, grades,
immutable archives, and operational workflows remain unchanged.

The published `history/index.html` redirects to the most recent completed week.
The live site's season/week selector also links to individual archived week pages.

The Pages-only `src.preview_week_results` step shows verified completed scores
and in-memory frozen-protocol paper results on the current weekly board even on Sundays.
This does not rewrite the official graded ledger or alter the model pipeline.
