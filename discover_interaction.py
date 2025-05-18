#!/usr/bin/env python3
"""
build_mi_interactions.py
─────────────────────────────────────────────────────────────────────────────
Standalone command-line tool that mines the **mutual-information strongest**  
pair-wise products of raw numeric box-score stats and the target (`pts` by
default).  Results are saved to two CSVs and echoed in copy-paste-friendly
formats.

╭───────────────────────────── WORKFLOW ─────────────────────────────╮
│ 1.  View-setup + SELECT from sql_player_props_playoffs.py          │
│ 2.  Keep only numeric, non-“*_rank” columns + target               │
│ 3.  Compute MI(target, each_single_stat)                           │
│ 4.  Pick the TOP-N single stats                                    │
│ 5.  For every pair among the TOP-N, compute MI(target, Xi*Xj)      │
│ 6.  Sort by MI, then apply a diversity quota per base stat         │
│ 7.  Print & write:                                                 │
│       • mi_singular.csv      (all single-stat MI scores)           │
│       • mi_interactions.csv  (kept interactions & scores)          │
╰────────────────────────────────────────────────────────────────────╯

CLI flags
---------
--db               SQLite file (default: nba.db)
--target           target column (default: pts)
--top-n            how many single leaders to consider (default: 400)
--k-inter          how many interactions to output (default: 200)
--max-per-feat     quota appearances per base stat (default: 3)
--sample-rows      rows to sample for MI calc (default: 10000; 0 = all)
--out-dir          directory to drop CSVs (default: cwd)

Example
-------
$ python build_mi_interactions.py --db nba.db --top-n 500 --k-inter 250 \
      --max-per-feat 4 --sample-rows 15000 --out-dir output/
"""

from __future__ import annotations

import argparse
import collections
import itertools
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# ──────────────────────────────────────────────────────────────────────
#  SQL TEXT (imported from project module)
# ──────────────────────────────────────────────────────────────────────
try:
    from sql_player_props_playoffs import VIEW_SETUP_SQL, PLAYER_GAMES_SQL
except ImportError:
    sys.stderr.write(
        "❌  Could not import VIEW_SETUP_SQL / PLAYER_GAMES_SQL "
        "from sql_player_props_playoffs.py.  Adjust PYTHONPATH.\n"
    )
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────
#  CONSTANT DEFAULTS  (overridable by CLI)
# ──────────────────────────────────────────────────────────────────────
DEF_DB             = Path("nba.db")
DEF_TARGET         = "pts"
DEF_TOP_N          = 400
DEF_K_INTER        = 200
DEF_MAX_PER_FEAT   = 3
DEF_SAMPLE_ROWS    = 10_000
DEF_OUTDIR         = Path(".")

EXCLUDE_STATS      : set[str]      = {"team_id", "player_id", "opponent_team_id"}
EXCLUDE_PATTERNS   : Tuple[str, ...] = ()      # e.g. ("_rank_pct",)

# ──────────────────────────────────────────────────────────────────────
#  UTILITIES
# ──────────────────────────────────────────────────────────────────────
def run_query(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Execute the view-setup DDL and then pull **playoff-only** rows.
    
    We wrap the original `PLAYER_GAMES_SQL` in a sub-query called `tgf`
    and filter on `tgf.is_playoffs = 1`, so nothing upstream has to be
    modified.
    """
    with conn:
        # 1) create / refresh the SQL view(s)
        conn.executescript(VIEW_SETUP_SQL)

        # 2) playoff-only SELECT
        playoff_sql = f"""
        SELECT *
        FROM (
            {PLAYER_GAMES_SQL.rstrip(';')}
        ) AS tgf
        WHERE tgf.is_playoffs = 1;
        """

        df = pd.read_sql_query(
            playoff_sql,
            conn,
            parse_dates=["game_date"],
            coerce_float=True,
        )

    # 3) sanity-clean duplicated column names (can happen in complex joins)
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].unique()
        sys.stderr.write(f"⚠  Dropping duplicate columns: {', '.join(dupes)}\n")
        df = df.loc[:, ~df.columns.duplicated()]

    return df


def numeric_without_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric frame w/ bool→uint8 and no '*_rank' cols."""
    df = df.copy()
    # Bools to numeric
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype("uint8")
    num = df.select_dtypes(include=[np.number])
    drop = num.columns[num.columns.str.contains(r"_rank", case=False)]
    return num.drop(columns=drop)

def greedy_diverse(df_pairs: pd.DataFrame,
                   k: int,
                   max_per_feat: int) -> pd.DataFrame:
    """
    Enforce per-feature quota on a pairs DataFrame sorted by 'score' DESC.
    """
    keep, used = [], collections.Counter()
    for _, row in df_pairs.iterrows():
        if len(keep) == k:
            break
        if used[row.f1] >= max_per_feat or used[row.f2] >= max_per_feat:
            continue
        keep.append(row)
        used[row.f1] += 1
        used[row.f2] += 1
    return pd.DataFrame(keep)

def implied_prob(usd_odds: pd.Series) -> pd.Series:
    """Quick helper (not used here but handy)."""
    return np.where(
        usd_odds < 0,
        -usd_odds / (-usd_odds + 100),
        100 / (usd_odds + 100),
    )

# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────
def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="build_mi_interactions.py",
                                 description="Mutual-information interaction miner.")
    ap.add_argument("--db", type=Path, default=DEF_DB, help="SQLite DB file")
    ap.add_argument("--target", default=DEF_TARGET, help="Target column name")
    ap.add_argument("--top-n", type=int, default=DEF_TOP_N,
                    help="How many single-stat MI leaders to consider")
    ap.add_argument("--k-inter", type=int, default=DEF_K_INTER,
                    help="How many interactions to output in the end")
    ap.add_argument("--max-per-feat", type=int, default=DEF_MAX_PER_FEAT,
                    help="Quota per base stat in final interactions")
    ap.add_argument("--sample-rows", type=int, default=DEF_SAMPLE_ROWS,
                    help="Row sample for MI calc (0 = all rows)")
    ap.add_argument("--out-dir", type=Path, default=DEF_OUTDIR,
                    help="Directory to write the CSVs")
    args = ap.parse_args(argv)

    # ─── Validate + prepare I/O paths
    db_path: Path = args.db.expanduser().resolve()
    if not db_path.is_file():
        sys.stderr.write(f"❌  DB not found: {db_path}\n"); sys.exit(1)

    out_dir: Path = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_single  = out_dir / "mi_singular.csv"
    csv_inter   = out_dir / "mi_interactions.csv"

    # ─── Load data
    print(f"🔌  Connecting to {db_path} …")
    with sqlite3.connect(str(db_path)) as conn:
        df_raw = run_query(conn)
    print(f"✅  Retrieved {len(df_raw):,} rows × {len(df_raw.columns):,} cols.")

    # ─── Prep numeric frame
    if args.target not in df_raw.columns:
        sys.stderr.write(f"❌  Target '{args.target}' not in SQL output.\n")
        sys.exit(1)

    df_num = numeric_without_rank(df_raw).dropna(subset=[args.target])
    y = df_num[args.target]
    df_num = df_num.fillna(df_num.median(numeric_only=True))

    # ─── Single-stat MI
    X_sing = df_num.drop(columns=[args.target])
    sample_rows = args.sample_rows if args.sample_rows > 0 else len(X_sing)
    if len(X_sing) > sample_rows:
        sel_idx = np.random.choice(X_sing.index, sample_rows, replace=False)
        X_sample = X_sing.loc[sel_idx]
        y_sample = y.loc[sel_idx]
    else:
        X_sample, y_sample = X_sing, y

    mi_vals = mutual_info_regression(X_sample, y_sample, random_state=42)
    mi_series = pd.Series(mi_vals, index=X_sample.columns, name="mi").sort_values(ascending=False)

    # ─── Select TOP-N stats
    top_stats = mi_series.head(args.top_n).index.tolist()
    top_stats = [s for s in top_stats if s not in EXCLUDE_STATS]
    if EXCLUDE_PATTERNS:
        top_stats = [s for s in top_stats
                     if not any(p.lower() in s.lower() for p in EXCLUDE_PATTERNS)]

    if len(top_stats) < 2:
        sys.stderr.write("❌  Need at least two stats after filtering.\n")
        sys.exit(1)

    print(f"📈  Computing pair-wise MI among {len(top_stats)} stats …")
    pairs, scores = [], []
    for i, s1 in enumerate(top_stats):
        v1 = df_num[s1].to_numpy()
        for s2 in top_stats[i+1:]:
            prod = (v1 * df_num[s2].to_numpy()).reshape(-1, 1)
            mi = mutual_info_regression(prod, y, discrete_features=False, random_state=42)[0]
            pairs.append((s1, s2)); scores.append(mi)

    df_pairs = (pd.DataFrame(pairs, columns=["f1", "f2"])
                  .assign(score=scores)
                  .sort_values("score", ascending=False))

    df_keep = greedy_diverse(df_pairs, args.k_inter, args.max_per_feat)

    # ─── OUTPUT
    print(f"\n🏁  Final interaction list ({len(df_keep)} rows):")
    for i, r in enumerate(df_keep.itertuples(index=False), 1):
        print(f"{i:3d}. {r.f1} * {r.f2}   MI={r.score:.6f}")

    mi_series.to_csv(csv_single, float_format="%.6f")
    df_keep.to_csv(csv_inter, index=False, float_format="%.6f")

    print(f"\n📝  Wrote:")
    print(f"    • Single-stat MI scores  → {csv_single}")
    print(f"    • Interaction MI scores  → {csv_inter}")

    # ─── copy-paste helpers
    print("\n──────────── dict format ────────────")
    for r in df_keep.itertuples(index=False):
        key = f"{r.f1}_{r.f2}"
        print(f'"{key}": ("{r.f1}", "{r.f2}"),')
    print("\n──────────── quoted list ────────────")
    quoted = ", ".join(f'"{r.f1}_{r.f2}"' for r in df_keep.itertuples(index=False))
    print(quoted)

# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
