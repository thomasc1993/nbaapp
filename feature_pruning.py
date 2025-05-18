#!/usr/bin/env python3
"""
feature_pruning.py
==================

Playoffs-only interaction-feature builder + MI/ρ pruner.

OUTPUTS
-------
1. kept.json          ← JSON list of kept interaction feature names
2. kept_tuples.txt    ← One-line-per-entry mapping in the form

       "feat_name": ("col_a", "col_b"),

   (Path set with -T/--tuples-out.)

Usage (unchanged + new flag)
---------------------------

    python3 feature_pruning.py nba.db \
        --rho 0.92 --top 75 \
        --out kept.json \
        --tuples-out kept_tuples.txt
"""

from __future__ import annotations

# ───────────── stdlib
import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Set

# ───────────── 3rd-party
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# ───────────── project local
from features_player_props import (           # noqa: E402
    BASE_STATS, TEAM_STATS, OPP_STATS, PLAYER_ADVANCED_TRACKING,
    EXTRA_FEATURES, INTERACTIONS, INTERACTION_NAMES, CAT_FLAGS, PLAYOFF_COLS, ODD_COLS_NUMERIC, ODD_COLS
)
from sql_player_props_playoffs import (       # noqa: E402
    VIEW_SETUP_SQL, PLAYER_GAMES_SQL,
)

# ───────────── logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
log = logging.getLogger("feature_pruning")

MUST_HAVE_COLS: set[str] = set(
    BASE_STATS + TEAM_STATS + OPP_STATS +
    PLAYER_ADVANCED_TRACKING + EXTRA_FEATURES + CAT_FLAGS + PLAYOFF_COLS + ODD_COLS_NUMERIC + ODD_COLS
)

# ==================================================================
# helpers
# ==================================================================
def fetch_raw(db: Path) -> pd.DataFrame:
    """Return playoff-only rows sorted by game_date ASC."""
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(db))
        conn.executescript(VIEW_SETUP_SQL)
        sql = f"""
        WITH base AS ({PLAYER_GAMES_SQL.rstrip(';')})
        SELECT * FROM base
        WHERE is_playoffs = 1
        ORDER BY game_date ASC;
        """
        df = pd.read_sql_query(sql, conn, parse_dates=["game_date"])
    finally:
        if conn:
            conn.close()

    df.columns = [str(c) for c in df.columns]
    return df


def build_interactions(raw: pd.DataFrame,
                       pairs: Dict[str, Tuple[str, str]]
                       ) -> pd.DataFrame:
    """Multiply valid column pairs → interaction matrix."""
    feats, skipped = {}, []
    for name, (a, b) in pairs.items():
        if a not in raw.columns or b not in raw.columns:
            skipped.append(name)
            continue
        feats[name] = raw[a] * raw[b]
    if skipped:
        log.info("Skipped %d interaction(s) (missing inputs).", len(skipped))
    return pd.DataFrame(feats)


def prune(X: pd.DataFrame, y: pd.Series,
          rho: float, top_k: int) -> List[str]:
    """MI rank → Pearson ρ redundancy prune → top-k list."""
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    mi = mutual_info_regression(X, y, random_state=0)
    order = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    corr = X.corr().abs()
    keep: Set[str] = set()
    for feat in order.index:
        if any(corr.loc[feat, list(keep)] > rho):
            continue
        keep.add(feat)
        if len(keep) == top_k:
            break
    return sorted(keep)


def write_tuples(path: Path, kept: List[str],
                 tuples: Dict[str, Tuple[str, str]]) -> None:
    """Write one-liner tuple file exactly as demanded."""
    with path.open("w", encoding="utf-8") as fh:
        for feat in kept:
            a, b = tuples[feat]
            fh.write(f"\"{feat}\": (\"{a}\", \"{b}\"),\n")
    log.info("✓ kept-tuple map (%d) → %s", len(kept), path)


# ==================================================================
# main
# ==================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("db_file", type=Path, help="Path to nba.db")
    ap.add_argument("--rho", type=float, default=0.92,
                    help="Correlation threshold for redundancy pruning")
    ap.add_argument("--top", type=int, default=75,
                    help="Max number of kept interactions")
    ap.add_argument("-t", "--target", default="pts", help="Target column")
    ap.add_argument("-o", "--out", type=Path, default=Path("kept.json"),
                    help="Where to dump JSON list of kept names")
    ap.add_argument("-T", "--tuples-out", type=Path,
                    default=Path("kept_tuples.txt"),
                    help="Where to dump name→(a,b) lines")
    args = ap.parse_args()

    # 1 ─ fetch
    log.info("Loading playoff games from %s …", args.db_file)
    raw = fetch_raw(args.db_file)
    if args.target not in raw.columns:
        raise SystemExit(f"Target «{args.target}» not in SQL frame.")

    # 2 ─ interactions
    X = build_interactions(raw, INTERACTIONS)

    # 3 ─ prune
    kept = prune(X, raw[args.target], args.rho, args.top)

    # 4 ─ save JSON list
    args.out.write_text(json.dumps(kept, indent=2))
    log.info("✓ kept %d cols → %s", len(kept), args.out)

    # 5 ─ build & write tuple map
    tuples = {
        f: (INTERACTIONS.get(f) or INTERACTION_NAMES[f])
        for f in kept
    }
    write_tuples(args.tuples_out, kept, tuples)


if __name__ == "__main__":
    main()
