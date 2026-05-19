# -*- coding: utf-8 -*-
"""
odds_utils.py
-------------
Shared odds matching utilities used by all backtest and prediction scripts.
Uses full-name matching with last-name fallback only when no collision exists.
"""


def american_to_implied(o):
    o = float(o)
    return 100 / (o + 100) if o > 0 else abs(o) / (abs(o) + 100)


def remove_vig(o1, o2):
    i1 = american_to_implied(o1)
    i2 = american_to_implied(o2)
    return i1 / (i1 + i2), i2 / (i1 + i2)


def normalize(n):
    return str(n).lower().strip()


def last_name(n):
    return normalize(n).split()[-1]


def build_odds_map(odds_rows):
    """
    Build odds lookup dict from raw DB rows.
    Keys: (name1, name2, ym) where ym = 'YYYY-MM'
    Strategy:
      - Always store full normalized name keys
      - Store last-name keys ONLY when no collision exists for that month
    """
    # Count last-name occurrences per month to detect collisions
    last_counts = {}
    for f1n, f2n, date, o1, o2 in odds_rows:
        if not o1 or not o2:
            continue
        ym = str(date)[:7]
        for n in [f1n, f2n]:
            k = (last_name(n), ym)
            last_counts[k] = last_counts.get(k, 0) + 1

    odds_map = {}
    for f1n, f2n, date, o1, o2 in odds_rows:
        if not o1 or not o2:
            continue
        ym = str(date)[:7]
        o1f, o2f = float(o1), float(o2)
        n1 = normalize(f1n)
        n2 = normalize(f2n)
        l1 = last_name(f1n)
        l2 = last_name(f2n)

        # Full name keys — always
        odds_map[(n1, n2, ym)] = (o1f, o2f)
        odds_map[(n2, n1, ym)] = (o2f, o1f)

        # Last name keys — only when no collision
        if last_counts.get((l1, ym), 0) <= 1 and last_counts.get((l2, ym), 0) <= 1:
            odds_map[(l1, l2, ym)] = (o1f, o2f)
            odds_map[(l2, l1, ym)] = (o2f, o1f)

    return odds_map


def lookup_odds(odds_map, f1n, f2n, ym):
    """
    Look up odds with fallback chain:
      1. Full normalized name
      2. Last name only (if no collision — already filtered in build_odds_map)
    """
    n1 = normalize(f1n)
    n2 = normalize(f2n)

    entry = odds_map.get((n1, n2, ym))
    if entry:
        return entry

    # Last name fallback
    entry = odds_map.get((last_name(f1n), last_name(f2n), ym))
    return entry


def fmt_odds(o):
    if o is None:
        return "N/A"
    return ("+" + str(int(o))) if o > 0 else str(int(o))


def payout(odds, stake=100):
    """Return profit on winning bet."""
    return stake * (odds / 100) if odds > 0 else stake * (100 / abs(odds))
