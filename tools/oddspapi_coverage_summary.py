from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

PATH = Path('outputs/oddspapi_bookmaker_coverage.csv')


def main() -> None:
    df = pd.read_csv(PATH)
    df['valid_full_game_total'] = df['valid_full_game_total'].astype(str).str.lower().eq('true')

    books = list(dict.fromkeys(df['bookmaker'].astype(str)))
    valid = df.loc[df['valid_full_game_total']].copy()
    target = set(valid['fixture_id'].astype(str))

    by_book = {
        book: set(valid.loc[valid['bookmaker'].eq(book), 'fixture_id'].astype(str))
        for book in books
    }

    print(f'Matched FCS fixtures represented: {df["fixture_id"].nunique()}')
    print(f'FCS fixtures with at least one valid full-game total: {len(target)}')
    print('Per-book valid-total coverage:')
    for book in sorted(books, key=lambda b: (-len(by_book[b]), books.index(b))):
        print(f'  {book}: {len(by_book[book])}/{len(target)}')

    print('Exact minimum subsets preserving the full observed union:')
    found = []
    for size in range(1, len(books) + 1):
        for combo in combinations(books, size):
            union = set().union(*(by_book[b] for b in combo))
            if union == target:
                found.append(combo)
                if len(found) >= 10:
                    break
        if found:
            break
    for combo in found:
        print(f'  {len(combo)} books: {", ".join(combo)}')

    # A second view excludes Kalshi from sportsbook-source selection because its
    # ladder is structurally different from conventional sportsbook main totals.
    sportsbook_books = [b for b in books if b != 'kalshi']
    sportsbook_target = set().union(*(by_book[b] for b in sportsbook_books))
    print(f'Conventional-sportsbook union (excluding Kalshi): {len(sportsbook_target)} fixtures')
    found = []
    for size in range(1, len(sportsbook_books) + 1):
        for combo in combinations(sportsbook_books, size):
            union = set().union(*(by_book[b] for b in combo))
            if union == sportsbook_target:
                found.append(combo)
                if len(found) >= 10:
                    break
        if found:
            break
    for combo in found:
        print(f'  {len(combo)} sportsbook books: {", ".join(combo)}')


if __name__ == '__main__':
    main()
