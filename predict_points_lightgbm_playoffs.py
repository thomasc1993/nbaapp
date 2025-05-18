#!/usr/bin/env python3
"""
predict_points_lightgbm_playoffs.py
===================================

Standalone prediction script for the *LightGBMPlayoffModel*.

Predicts player stats for upcoming PLAYOFF games only. Defaults to 'pts' target.

Key principles
--------------
1. **Loads Playoff Artifacts:** Uses target-specific playoff model artifacts.
2. **Checks Playoff Game:** Only proceeds if the player's next game is a playoff game.
3. **Calculates Correct Context:** Recalculates playoff series context and team
   sequence features (streaks, rest) accurately for the upcoming game.
4. **Delegates FE:** Uses the `prepare_new_data` method from the loaded
   `LightGBMPlayoffModel` instance to ensure feature alignment.
5. **Schema Parity:** SQL queries aim to match playoff training inputs where relevant.

**REWRITTEN:** Incorporates fixes for streak logic, missing general features,
and playoff context recalculation to ensure correct logic mimicking
teamsgamefeatures.py where appropriate for prediction time. Includes player
regular season tracking stats via TEMP VIEW matching training.
"""

from __future__ import annotations

# ───────────────────────────── stdlib ──────────────────────────────
import argparse
import json
import logging
import math
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ───────────────────────────── 3rd‑party ───────────────────────────
import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

from sql_player_props_playoffs import VIEW_SETUP_SQL, PLAYER_GAMES_SQL
from features_player_props import BASE_STATS, TEAM_STATS, OPP_STATS, PLAYER_ADVANCED_TRACKING, INTERACTIONS, INTERACTION_NAMES, EXTRA_FEATURES, CAT_FLAGS, PLAYOFF_COLS   # <- new line


# --- Import the STANDALONE PLAYOFF model class and its helpers ---
from player_props_playoffs_lightgbm_alt import (
    LightGBMPlayoffModel,
    _NoOpScaler,
    # Import playoff artifact path functions
    get_playoff_model_path,
    get_playoff_features_path,
    get_playoff_cat_features_path
)

# ───────────────────────────── logging ─────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s [PlayoffPredict]", # Add identifier
    datefmt="%Y‑%m‑%d %H:%M:%S",
)
# Use a distinct logger name for playoff predictions
logger = logging.getLogger("playoff_predict_standalone")

# ──────────────────────────── constants ────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
ARTIFACT_DIR = SCRIPT_DIR # Assume artifacts are in the same directory

# --- Define Database Path (same logic as original script) ---
try:
    # Assumes structure like /project_root/stats/models/script.py
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    DB_FILE = PROJECT_ROOT / "nba.db"
    if not DB_FILE.exists():
        logger.warning(f"Default database file not found at: {DB_FILE}. "
                       "Using relative 'nba.db'. Use --db-file argument.")
        DB_FILE = Path("nba.db") # Fallback to relative path
except Exception:
     logger.warning("Could not determine project root reliably. "
                    "Defaulting DB path to 'nba.db' (relative to CWD). "
                    "Use --db-file argument for specific path.")
     DB_FILE = Path("nba.db")

# Playoff context fields required for accurate prediction state
# Based on teamsgamefeatures.py output schema for playoff context
# We will calculate these for the upcoming game
PLAYOFF_COLS = PLAYOFF_COLS
# ─────────────────────── helper utilities (Copied) ─────────────────
def _american_to_decimal(odds: int) -> float:
    if odds == 0: return 1.0
    return 1 + (odds / 100) if odds > 0 else 1 + (100 / abs(odds))

def _implied_prob(decimal_odds: float) -> float:
    return 0.0 if decimal_odds <= 0 else 1 / decimal_odds

def _kelly_fraction(p: float, decimal_odds: float) -> float:
    if decimal_odds <= 1.0: return 0.0
    b = decimal_odds - 1
    if b == 0: return 0.0
    kelly = (p * b - (1 - p)) / b
    return max(0, kelly)

# ─────────────────── PLAYOFF artefact / model cache ────────────────
_PLAYOFF_MODEL_CACHE: Dict[str, LightGBMPlayoffModel] = {}

def _load_playoff_model(target: str, artifact_dir: Path = ARTIFACT_DIR) -> LightGBMPlayoffModel:
    """
    Singleton-load the fitted PLAYOFF LightGBM model for a specific target.
    (Unchanged from original)
    """
    global _PLAYOFF_MODEL_CACHE
    if target in _PLAYOFF_MODEL_CACHE:
        logger.debug(f"Returning cached playoff model for target '{target}'.")
        return _PLAYOFF_MODEL_CACHE[target]

    logger.info(f"Loading PLAYOFF artefacts for target '{target}' from: {artifact_dir} …")
    model_filename = get_playoff_model_path(target).name
    feats_filename = get_playoff_features_path(target).name
    cats_filename  = get_playoff_cat_features_path(target).name
    model_file = artifact_dir / model_filename
    feats_file = artifact_dir / feats_filename
    cats_file = artifact_dir / cats_filename

    logger.debug(f"Attempting to load model: {model_file}")
    logger.debug(f"Attempting to load features: {feats_file}")
    logger.debug(f"Attempting to load categoricals: {cats_file}")

    for fp in (model_file, feats_file, cats_file):
        if not fp.exists():
            raise FileNotFoundError(f"Required PLAYOFF artifact file missing for target '{target}': {fp}")

    mdl = LightGBMPlayoffModel(target=target)
    try:
        mdl.model = joblib.load(model_file)
        mdl.feature_names = joblib.load(feats_file)
        mdl.final_categorical_features = joblib.load(cats_file)
        mdl.scaler = _NoOpScaler()
        logger.info(
            f"PLAYOFF artefacts loaded for target '{target}' — {len(mdl.feature_names)} features ({len(mdl.final_categorical_features)} categoricals)"
        )
        _PLAYOFF_MODEL_CACHE[target] = mdl
        return mdl
    except Exception as e:
        logger.error(f"Failed to load playoff artifacts for target '{target}': {e}", exc_info=True)
        raise

# ───────────────────────── SQL helpers (Adapted/Copied) ────────────
def _connect_db(db_path: Path) -> sqlite3.Connection:
    """Helper to connect to the database (copied)."""
    try:
        logger.debug(f"Attempting to connect to database: {db_path}")
        if not db_path.exists():
             raise FileNotFoundError(f"Database file does not exist at: {db_path}")
        conn = sqlite3.connect(db_path)
        logger.debug(f"Database connection successful: {db_path}")
        return conn
    except (sqlite3.Error, FileNotFoundError) as e:
        logger.error(f"Error connecting to database at {db_path}: {e}")
        raise ConnectionError(f"Failed to connect to database: {db_path}") from e

def _future_games(conn: sqlite3.Connection) -> Dict[str, List[dict]]:
    """
    Return *one record per team* for every game that is today or later.

    Fix (2025-05-04)
    ----------------
    • Force `team_id` and `opponent_team_id` to the same Int64 dtype
      **before** any further processing.
    • Duplication is only done when the source table stores one row
      per fixture.  If the table already holds both team rows
      (`is_home` ∈ {0, 1}), we do not duplicate again.
    """

    logger.debug("Fetching future games schedule …")

    query = """
        SELECT
            gs.game_id,
            gs.game_date,
            gs.game_time,
            gs.team_id,             -- home side in DB
            gs.opponent_team_id,    -- away side in DB
            gs.is_home,
            COALESCE(gs.is_playoffs, 0) AS is_playoffs,
            gs.season
        FROM gameschedules gs
        ORDER BY gs.game_date, gs.game_time
    """
    df = pd.read_sql_query(query, conn, parse_dates=["game_date"])
    logger.debug("Fetched %d raw schedule rows.", len(df))

    # ─── 1. Normalise the ID columns *immediately* ───────────────────────────
    id_cols = ["team_id", "opponent_team_id"]
    df[id_cols] = (df[id_cols]
                   .apply(pd.to_numeric, errors="coerce")
                   .astype("Int64"))

    # ─── 2. Build a robust UTC timestamp  ───────────────────────────────────
    def _combine_date_time(row_date, row_time):
        d_parsed = pd.to_datetime(str(row_date), errors="coerce", utc=True)
        t_parsed = pd.to_datetime(str(row_time), errors="coerce", utc=True)

        if (not pd.isna(d_parsed)) and (d_parsed.time() != pd.Timestamp.min.time()):
            return d_parsed
        if (not pd.isna(t_parsed)) and (t_parsed.date() != pd.Timestamp.min.date()):
            return t_parsed

        d_only = pd.to_datetime(row_date, errors="coerce").strftime("%Y-%m-%d")
        t_only = (pd.to_datetime(str(row_time), errors="coerce")
                  .strftime("%H:%M:%S") if pd.notna(row_time) else "00:00:00")
        return pd.to_datetime(f"{d_only} {t_only}", utc=True, errors="coerce")

    df["game_datetime_utc"] = df.apply(
        lambda r: _combine_date_time(r["game_date"], r["game_time"]), axis=1
    )
    df = df.dropna(subset=["game_datetime_utc"])

    # ─── 3. Keep only today-and-forward  ─────────────────────────────────────
    today_utc = pd.Timestamp.utcnow().normalize()          # 00:00 UTC today
    df = df.loc[df["game_datetime_utc"].dt.normalize() >= today_utc].copy()
    logger.debug("Remaining future rows after date filter: %d", len(df))

    # ─── 4. Duplicate rows *iff* we still have one row per fixture ───────────
    if df.groupby("game_id").size().eq(1).all():          # exactly one row ⇢ home only
        away_df = df.copy()
        away_df[["team_id", "opponent_team_id"]] = away_df[["opponent_team_id", "team_id"]]
        away_df["is_home"] = 1 - away_df["is_home"]
        df = pd.concat([df, away_df], ignore_index=True)
        logger.debug("Table had only home rows – duplicated to create away rows.")
    else:
        logger.debug("Table already contains both home and away rows – no duplication.")

    # ─── 5. Final housekeeping  ─────────────────────────────────────────────
    df["game_date_dt"] = df["game_datetime_utc"].dt.tz_convert(None).dt.normalize()
    df["game_date"] = df["game_date_dt"].dt.strftime("%Y-%m-%d")
    df["game_time"] = df["game_datetime_utc"].dt.tz_convert(None)
    df["is_playoffs"] = df["is_playoffs"].astype(int)
    df["season"] = df["season"].astype(str)

    # ─── 6. Build the `{game_id: [list-of-team-dicts]}` mapping ─────────────
    out: Dict[str, List[dict]] = {
        gid: grp.to_dict("records") for gid, grp in df.groupby("game_id")
    }

    logger.debug(
        "Built future-games map with %d game_ids (%d total team rows).",
        len(out), len(df)
    )
    return out

def _player_history(conn: sqlite3.Connection, player_id: int) -> pd.DataFrame:
    """
    Fetch every historical game row for `player_id`, using the *same* VIEW and
    query-construction pattern employed by the XGBoost helper:

        • `VIEW_SETUP_SQL`  – all CREATE TEMP VIEW … statements
        • `PLAYER_GAMES_SQL` – the giant SELECT used at train-time

    Only the data-clean-up / default-fill logic differs, matching the original
    LightGBM pipeline.
    """
    try:
        # ────────────────────────────── 1. VIEWs ──────────────────────────────
        logger.debug("Fetching historical data for player_id: %s", player_id)
        conn.executescript(VIEW_SETUP_SQL)                     # build helper views
        logger.debug("Executed VIEW_SETUP_SQL – TEMP VIEWs created.")

        # ────────────────────────────── 2. query ──────────────────────────────
        base_sql = PLAYER_GAMES_SQL.rstrip().rstrip(";")       # trim final “;”
        player_sql = f"""{base_sql}
WHERE pgf.player_id = ?            -- positional placeholder
ORDER BY pgf.game_date;
"""

        df = pd.read_sql_query(
            player_sql,
            conn,
            params=(player_id,),                               # safe binding
            parse_dates=["game_date"],
        )
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.tz_localize(None)
        logger.debug("Fetched %d historical rows for player %s.", len(df), player_id)

        # drop duplicate column names that can arise from VIEW joins
        df = df.loc[:, ~df.columns.duplicated()]

        # ──────────────────────── 3. Vegas-derived cols ───────────────────────
        raw_odds_cols = ["moneyline_odds", "spread_line", "total_line",
                         "over_odds", "under_odds"]
        for c in raw_odds_cols:
            df[c] = pd.to_numeric(df.get(c, 0), errors="coerce")

        def _implied_prob(american):
            american = np.asarray(american, dtype="float64")
            prob = np.empty_like(american)
            neg = american < 0
            prob[neg]  = -american[neg] / (-american[neg] + 100.0)     # favourites
            prob[~neg] = 100.0 / (american[~neg] + 100.0)              # underdogs
            prob[np.isnan(american)] = np.nan
            return prob

        df["p_ml"]     = _implied_prob(df["moneyline_odds"])
        df["fav_flag"] = (df["moneyline_odds"] < 0).astype("int8")
        df["abs_spread"] = df["spread_line"].abs()
        df["teim"]       = (df["total_line"] / 2) - (df["spread_line"] / 2)
        df["opim"]       = (df["total_line"] / 2) + (df["spread_line"] / 2)
        df[["p_ml", "abs_spread", "teim", "opim"]] = (
            df[["p_ml", "abs_spread", "teim", "opim"]].fillna(0)
        )

        # ──────────────────────── 4. season / win flags ───────────────────────
        if "season" in df.columns:
            df["season"] = df["season"].astype(str)
        else:                                                  # derive if missing
            df["season"] = df["game_date"].apply(
                lambda d: f"{d.year-1}-{str(d.year)[-2:]}" if d.month < 9
                else f"{d.year}-{str(d.year+1)[-2:]}"
            ).astype(str)

        if "win" not in df.columns:
            if "team_wl" in df.columns:
                df["win"] = df["team_wl"].apply(
                    lambda x: 1 if isinstance(x, str) and x.upper() == "W" else 0
                ).astype(int)
            else:
                df["win"] = 0
        df["win"] = pd.to_numeric(df["win"], errors="coerce").fillna(0).astype(int)

        # ─────────────────────── 5. playoff context cols ──────────────────────
        for col in PLAYOFF_COLS:
            if col in df.columns:
                if col == "series_prev_game_margin":
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                elif col == "series_record":
                    df[col] = (pd.to_numeric(df[col], errors="coerce")
                               .fillna(0).astype(int))
                else:
                    df[col] = (pd.to_numeric(df[col], errors="coerce")
                               .fillna(0).astype(int))
            else:
                df[col] = 0 if col != "series_prev_game_margin" else 0.0

        # ───────────────── 6. congestion / rest-days safeguards ───────────────
        for col in ["rest_days", "is_first_night_back_to_back"]:
            for prefix in ["team_", "opponent_"]:
                full = f"{prefix}{col}"
                if full not in df.columns:
                    df[full] = 0
                df[full] = (pd.to_numeric(df[full], errors="coerce")
                            .fillna(99 if col == "rest_days" else 0)
                            .astype(int))

        # ───────────────── 7. RS tracking metric default-fills ────────────────
        rs_tracking_cols = [   # must match VIEW select list
            "rs_cs_fg3a", "rs_cs_fg3_pct", "rs_cs_efg_pct", "rs_pu_fga",
            "rs_pu_efg_pct", "rs_drives", "rs_drive_pts_pct", "rs_drive_fg_pct",
            "rs_drive_ast_pct", "rs_paint_touches", "rs_paint_touch_fg_pct",
            "rs_paint_touch_pts_pct", "rs_paint_touch_passes_pct",
            "rs_post_touches", "rs_post_touch_fg_pct", "rs_post_touch_pts_pct",
            "rs_post_touch_ast_pct", "rs_ab_fg_pct", "rs_ab_fga", "rs_ab_fgm",
            "rs_bc_fg_pct", "rs_bc_fga", "rs_bc_fgm", "rs_c3_fg_pct",
            "rs_c3_fga", "rs_c3_fgm", "rs_paint_fg_pct", "rs_paint_fga",
            "rs_paint_fgm", "rs_lc3_fg_pct", "rs_lc3_fga", "rs_lc3_fgm",
            "rs_mr_fg_pct", "rs_mr_fga", "rs_mr_fgm", "rs_ra_fg_pct",
            "rs_ra_fga", "rs_ra_fgm", "rs_rc3_fg_pct", "rs_rc3_fga",
            "rs_rc3_fgm", "opp_rs_d_fg_pct", "opp_rs_d_fga", "opp_rs_d_fgm",
            "opp_rs_ov_pct_plusminus", "opp_rs_2p_freq", "opp_rs_2p_plusminus",
            "opp_rs_3p_freq", "opp_rs_3p_plusminus", "opp_rs_lt6f_freq",
            "opp_rs_lt6f_plusminus", "opp_rs_lt10f_freq",
            "opp_rs_lt10f_plusminus", "opp_rs_gt15f_freq",
            "opp_rs_gt15f_plusminus", "opp_rs_lt_06_pct", "opp_rs_lt_10_pct",
            "opp_rs_gt_15_pct", "opp_rs_fg2_pct", "opp_rs_fg3_pct",
        ]
        for col in rs_tracking_cols:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        return df

    # ───────────────────────────── error handling ─────────────────────────────
    except (pd.errors.DatabaseError, sqlite3.Error) as e:
        logger.error("Database error fetching player history for %s: %s",
                     player_id, e)
        raise


# --- Sigma Calculation Helpers (Unchanged - Uses MAIN model residuals) ---
def _latest_two_runs(conn: sqlite3.Connection,
                     *,
                     ignore_substring: str | None = None) -> tuple[str, str]:
    """
    Return the two most-recent distinct *model_run* identifiers that are present
    in `training_predictions`.

    Parameters
    ----------
    ignore_substring
        If given, rows whose *model_run* contains this substring are discarded
        **before** the “need ≥ 2” test.  Pass `None` (default) to accept *all*
        runs – regular-season and/or playoff – exactly what the standalone
        playoff script needs.
    """
    runs = pd.read_sql_query(
        "SELECT DISTINCT model_run "
        "FROM   training_predictions "
        "ORDER  BY model_run DESC",
        conn,
    )["model_run"].tolist()

    if ignore_substring:
        runs = [r for r in runs if ignore_substring not in r]

    if len(runs) < 2:
        raise ValueError("Need ≥ 2 model_run batches in training_predictions for σ.")

    return runs[0], runs[1]

def _player_sigma(conn: sqlite3.Connection, player_id: int) -> Tuple[float, str, int]:
    """Calculates sigma based on MAIN model residuals (copied)."""
    logger.debug(f"Calculating sigma for player_id: {player_id} using MAIN model residuals...")
    try:
        run_new, run_prev = _latest_two_runs(conn)
    except ValueError as e:
        logger.error(f"Cannot calculate sigma, failed to get latest main model runs: {e}")
        raise

    method = "pooled"; n_points = 0
    try:
        player_res_df = pd.read_sql_query(
            "SELECT residual FROM training_predictions WHERE player_id = ? AND model_run IN (?, ?)",
            conn, params=(player_id, run_new, run_prev),
        )
        player_res = player_res_df["residual"].dropna()
        n_player_points = len(player_res)
        PLAYER_SIGMA_MIN_N = 75 # Threshold for player-specific sigma
        if n_player_points >= PLAYER_SIGMA_MIN_N:
            variance = np.mean(player_res**2); sigma = float(math.sqrt(variance))
            method = "player_specific"; n_points = n_player_points
            logger.info("Using player-specific σ (N=%d, main runs): %.4f", n_points, sigma)
            return sigma, method, n_points
        else:
            logger.info("Insufficient player data (N=%d, main runs) for specific σ, using pooled.", n_player_points)
            pooled_res_df = pd.read_sql_query(
                "SELECT residual FROM training_predictions WHERE model_run IN (?, ?)",
                conn, params=(run_new, run_prev),
            )
            pooled_res = pooled_res_df["residual"].dropna()
            if pooled_res.empty: raise ValueError("No residuals found in main model runs for pooled sigma.")
            n_pooled_points = len(pooled_res)
            variance = np.mean(pooled_res**2); sigma = float(math.sqrt(variance))
            method = "pooled"; n_points = n_pooled_points
            logger.info("Using pooled σ (N=%d, main runs): %.4f", n_points, sigma)
            return sigma, method, n_points
    except (pd.errors.DatabaseError, sqlite3.Error) as e:
        logger.error(f"Database error calculating sigma for player {player_id} (main runs): {e}")
        raise
    except ValueError as e:
        logger.error(f"Value error calculating sigma for player {player_id} (main runs): {e}")
        raise

# --- Team Congestion Features (REVISED) ---
def _team_congestion_features(glogs: pd.DataFrame) -> pd.DataFrame:
    """
    Build congestion / streak features for *playoff* team logs.

    ── What’s different from the old version? ─────────────────────────────
    ▸ `win_streak`, `losing_streak`, `road_trip_length`
      are saved **after** they’ve been updated with the result/location of
      the current row’s game.  
      Therefore the most-recent row now shows the team’s *current* streak
      (e.g. DEN win-streak = 1 after their latest victory), fixing issues
      where the script thought a team was still on a skid it had just ended.
    ----------------------------------------------------------------------
    """
    if glogs.empty:
        # return skeleton frame with expected columns
        exp_cols = list(glogs.columns) + [
            "road_trip_length", "is_back_to_back", "is_three_in_four",
            "is_five_in_seven", "win_streak", "losing_streak", "rest_days",
            "is_first_night_back_to_back",
        ]
        return pd.DataFrame(columns=exp_cols)

    logger.debug("Calculating team congestion features (post-game state)…")

    # ensure chronological order for each team
    glogs = (glogs
             .sort_values(["team_id", "game_date"])
             .reset_index(drop=True))

    rows: list[dict] = []

    for tid, grp in glogs.groupby("team_id", sort=False):
        grp = grp.reset_index(drop=True)       # tidy index

        current_win_streak     = 0
        current_losing_streak  = 0
        current_road_trip_len  = 0
        prev_date              = None          # date of previous game

        for idx, row in grp.iterrows():
            game_dt      = row["game_date"]
            is_home_game = int(row["is_home"]) == 1

            # ── rest-day maths (state before this game) ───────────────────
            if prev_date is None:
                rest_days = 99     # first game of log
            else:
                gap       = (game_dt - prev_date).days
                rest_days = max(0, gap - 1)

            is_b2b        = int(rest_days == 0)          # played yesterday
            three_in_four = int(idx >= 2 and
                                (game_dt - grp.loc[idx - 2, "game_date"]).days <= 3)
            five_in_seven = int(idx >= 4 and
                                (game_dt - grp.loc[idx - 4, "game_date"]).days <= 6)

            # ── update streak / road-trip counters with RESULT of this game ─
            wl = str(row.get("wl", "")).upper()
            if wl == "W":
                current_win_streak    += 1
                current_losing_streak  = 0
            elif wl == "L":
                current_losing_streak += 1
                current_win_streak     = 0
            else:                                   # draw / missing -> reset
                current_win_streak = current_losing_streak = 0

            # road-trip length *after* this game
            current_road_trip_len = (current_road_trip_len + 1) if not is_home_game else 0

            # ── build row dict reflecting the **post-game** state ──────────
            r = row.to_dict()
            r.update({
                "rest_days":                rest_days,
                "is_back_to_back":          is_b2b,
                "is_three_in_four":         three_in_four,
                "is_five_in_seven":         five_in_seven,
                "road_trip_length":         current_road_trip_len,
                "win_streak":               current_win_streak,
                "losing_streak":            current_losing_streak,
            })
            rows.append(r)

            prev_date = game_dt     # move window

    # -------- create DataFrame & first-night-B2B flag ----------------------
    processed = (pd.DataFrame(rows)
                 .sort_values(["team_id", "game_date"])
                 .reset_index(drop=True))

    processed["next_game_date"] = processed.groupby("team_id")["game_date"].shift(-1)
    processed["is_first_night_back_to_back"] = (
        (processed["next_game_date"] - processed["game_date"]).dt.days == 1
    ).fillna(False).astype(int)
    processed = processed.drop(columns=["next_game_date"])

    # guarantee column order / presence
    final_cols = (glogs.columns.tolist() +
                  ["road_trip_length", "is_back_to_back", "is_three_in_four",
                   "is_five_in_seven", "win_streak", "losing_streak",
                   "rest_days", "is_first_night_back_to_back"])
    processed = processed.reindex(columns=list(dict.fromkeys(final_cols)))

    logger.debug("Finished congestion features (post-game state).")
    return processed



# ───────────────── PLAYOFF prediction core (REVISED CONTEXT CALC) ────────────────
def predict_player_playoffs(
    player_id: int,
    target: str, # Target stat required to load correct model
    *,
    line: Optional[float] = None,
    over_odds: Optional[int] = None,
    under_odds: Optional[int] = None,
    db_path: Path = DB_FILE,
    artifact_dir: Path = ARTIFACT_DIR,
) -> Dict[str, Any]:
    """
    Predict the **next scheduled PLAYOFF** game stat for *player_id*.
    Includes corrected playoff context calculation.
    """
    conn: Optional[sqlite3.Connection] = None
    logger.info(f"Attempting PLAYOFF prediction for player {player_id}, target '{target}'...")

    try:
        conn = _connect_db(db_path)

        # Find player's current team and name (copied)
        logger.debug(f"Finding current team and names for player_id {player_id}...")
        team_id_df = pd.read_sql_query("SELECT team_id FROM player_game_features WHERE player_id = ? ORDER BY game_date DESC LIMIT 1", conn, params=(player_id,))
        if team_id_df.empty: raise ValueError(f"No historical games found for player {player_id} in DB.")
        team_id = int(team_id_df.iloc[0]["team_id"])
        player_name = conn.execute("SELECT name FROM players WHERE player_id = ?", (player_id,)).fetchone()
        player_name = player_name[0] if player_name else f"ID {player_id}"
        team_name = conn.execute("SELECT team_city || ' ' || team_name FROM teams WHERE team_id = ?", (team_id,)).fetchone()
        team_name = team_name[0] if team_name else f"ID {team_id}"
        logger.info(f"Player '{player_name}' on team '{team_name}'.")

        # Find Player's Next Game (using modified _future_games)
        logger.debug("Identifying next scheduled game...")
        upc_games_dict = _future_games(conn)
        if not upc_games_dict: raise ValueError("No upcoming games found in schedule.")
        upcoming_game_info: Optional[dict] = None
        for game_id, game_records in upc_games_dict.items():
            for record in game_records:
                if record["team_id"] == team_id:
                    upcoming_game_info = record; upcoming_game_info['game_id'] = game_id; break
            if upcoming_game_info: break
        if upcoming_game_info is None: raise ValueError(f"No scheduled game found for player {player_id}'s team {team_id}.")

        # <<< !!! PLAYOFF CHECK !!! >>>
        is_next_game_playoffs = bool(upcoming_game_info.get("is_playoffs", 0))
        game_date_str = upcoming_game_info.get('game_date', 'N/A') # Already string format
        game_id_str = upcoming_game_info.get('game_id', 'N/A')
        upcoming_season = upcoming_game_info.get('season') # Get season for context
        if not is_next_game_playoffs:
            logger.warning(f"Player {player_id}'s next game (ID: {game_id_str} on {game_date_str}) is NOT a playoff game. Use the main prediction script.")
            raise ValueError(f"Player {player_id}'s next game is not flagged as playoffs.")
        elif not upcoming_season:
             logger.error(f"Cannot determine season for upcoming playoff game {game_id_str}. Cannot calculate context.")
             raise ValueError(f"Missing season for upcoming playoff game {game_id_str}.")
        else:
            logger.info(f"Next game (ID: {game_id_str} on {game_date_str}, Season: {upcoming_season}) IS a playoff game. Proceeding...")

        # --- Calculate CORRECT Playoff Context for Upcoming Game ---
        logger.info("Calculating playoff context for upcoming game...")
        opponent_team_id = upcoming_game_info["opponent_team_id"]
        calculated_playoff_context: Dict[str, Any] = {"is_playoffs": 1}

        # Fetch relevant historical playoff games for this specific series
        # Need team_id, opponent_team_id, season, game_date, wl, series_number, pts, opp_pts, playoff_round, has_home_court
        series_hist_df = pd.read_sql_query(
            f"""
            SELECT game_date, team_id, opponent_team_id, wl, series_number, pts, opp_pts, playoff_round, has_home_court
            FROM teams_game_features
            WHERE season = ?
              AND is_playoffs = 1
              AND ((team_id = ? AND opponent_team_id = ?) OR (team_id = ? AND opponent_team_id = ?))
            ORDER BY game_date
            """,
            conn,
            params=(upcoming_season, team_id, opponent_team_id, opponent_team_id, team_id),
            parse_dates=["game_date"]
        )
        series_hist_df['game_date'] = pd.to_datetime(series_hist_df['game_date']).dt.tz_localize(None)

        # Find last game played *in this series*
        last_game_in_series = None
        if not series_hist_df.empty:
             # Get unique games (avoid team/opponent duplication for same game)
             unique_games_in_series = series_hist_df.drop_duplicates(subset=['game_date', 'series_number']).sort_values('game_date')
             if not unique_games_in_series.empty:
                  last_game_in_series = series_hist_df[series_hist_df['game_date'] == unique_games_in_series['game_date'].max()]
                  # Ensure we get both team rows for the last game to find margin easily
                  if len(last_game_in_series) > 2: # Should be 0 or 2 rows if data is clean
                       logger.warning(f"Found {len(last_game_in_series)} rows for last game in series, expected 2. Taking latest two.")
                       last_game_in_series = last_game_in_series.iloc[-2:]


        # Determine upcoming game state
        if last_game_in_series is None or last_game_in_series.empty:
            # This is Game 1 of the series
            logger.info("Identified upcoming game as Game 1 of the series.")
            upcoming_series_number = 1
            wins_before_upcoming = 0
            opp_wins_before_upcoming = 0
            last_game_margin = np.nan # No previous game margin
            # Need to determine round and HCA potentially from schedule or assume defaults
            calculated_playoff_context["playoff_round"] = 1 # Assume round 1 if no history? Needs better logic maybe.
            calculated_playoff_context["has_home_court"] = upcoming_game_info["is_home"] # Assume HCA if home G1
        else:
            # Calculate from last game in series
            last_game_this_team = last_game_in_series[last_game_in_series['team_id'] == team_id].iloc[0]
            last_game_opponent = last_game_in_series[last_game_in_series['team_id'] == opponent_team_id].iloc[0]

            # --- Start Replacement ---
            # 1. Get the raw value
            raw_value = last_game_this_team['series_number']

            # 2. Attempt numeric conversion, coercing errors to NaN
            numeric_value = pd.to_numeric(raw_value, errors='coerce')

            # 3. Check if the conversion resulted in NaN. If so, use 0. Otherwise, use the numeric value.
            #    Make sure to convert to int *after* handling NaN.
            if pd.isna(numeric_value):
                last_series_number = 0
            else:
                # It's a valid number, convert it to a standard Python int
                last_series_number = int(numeric_value)
            # --- End Replacement ---
            upcoming_series_number = last_series_number + 1

            # Count wins from full series history DataFrame
            team_wins_series = len(series_hist_df[(series_hist_df['team_id'] == team_id) & (series_hist_df['wl'] == 'W')])
            opp_wins_series = len(series_hist_df[(series_hist_df['team_id'] == opponent_team_id) & (series_hist_df['wl'] == 'W')])
            wins_before_upcoming = team_wins_series
            opp_wins_before_upcoming = opp_wins_series

            # Calculate margin from last game
            last_pts = pd.to_numeric(last_game_this_team['pts'], errors='coerce')
            last_opp_pts = pd.to_numeric(last_game_this_team['opp_pts'], errors='coerce') # opp_pts for player's team perspective
            if pd.notna(last_pts) and pd.notna(last_opp_pts):
                 last_game_margin = float(last_pts - last_opp_pts)
            else:
                 last_game_margin = np.nan
                 logger.warning(f"Could not calculate last game margin for series (Game ID related to date {last_game_this_team['game_date']}). Missing pts/opp_pts.")

            # Carry over round and HCA from last game (should be constant in series)
            # --- Start Replacement for playoff_round ---
            raw_value_round = last_game_this_team['playoff_round']
            numeric_value_round = pd.to_numeric(raw_value_round, errors='coerce')
            # Default to 1 if NaN (as per original fillna(1))
            if pd.isna(numeric_value_round):
                calculated_playoff_context["playoff_round"] = 1
            else:
                calculated_playoff_context["playoff_round"] = int(numeric_value_round)
            # --- End Replacement for playoff_round ---
            # --- Start Replacement for has_home_court ---
            raw_value_hca = last_game_this_team['has_home_court']
            numeric_value_hca = pd.to_numeric(raw_value_hca, errors='coerce')
            # Default to 0 if NaN (as per original fillna(0))
            if pd.isna(numeric_value_hca):
                calculated_playoff_context["has_home_court"] = 0
            else:
                calculated_playoff_context["has_home_court"] = int(numeric_value_hca)
            # --- End Replacement for has_home_court ---

            logger.info(f"Upcoming Game {upcoming_series_number}. Wins Before: Team={wins_before_upcoming}, Opp={opp_wins_before_upcoming}. Last Margin={last_game_margin:.1f}")


        # Calculate remaining context based on upcoming state
        calculated_playoff_context["series_number"] = upcoming_series_number
        calculated_playoff_context["series_record"] = wins_before_upcoming # Wins BEFORE this game
        calculated_playoff_context["series_prev_game_margin"] = last_game_margin if pd.notna(last_game_margin) else None # Use None for DB

        is_game_6 = int(upcoming_series_number == 6)
        is_game_7 = int(upcoming_series_number == 7)
        calculated_playoff_context["is_game_6"] = is_game_6
        calculated_playoff_context["is_game_7"] = is_game_7

        # Elimination/Win logic (using correct upcoming state)
        # Re-evaluate 'can_win_series' and 'is_elimination_game' from the *team's* perspective:
        # Team faces elimination if *opponent* has 3 wins (or G7)
        calculated_playoff_context["is_elimination_game"] = int((opp_wins_before_upcoming == 3) or is_game_7)
        # Team can win the series if *they* have 3 wins (or G7)
        calculated_playoff_context["can_win_series"] = int((wins_before_upcoming == 3) or is_game_7)


        calculated_playoff_context["series_score_diff"] = wins_before_upcoming - opp_wins_before_upcoming

        # Update the upcoming_game_info dict with ACCURATE context
        upcoming_game_info.update(calculated_playoff_context)


        # ── NEW: Vegas odds & derived features ──────────────────────────────
        vegas_row = conn.execute(
            """
            SELECT moneyline_odds,
                spread_line,
                total_line,
                over_odds,
                under_odds
            FROM   historical_nba_odds
            WHERE  game_id = ?
            AND  team_id = ?
            """,
            (upcoming_game_info["game_id"], team_id)
        ).fetchone()

        if vegas_row:
            moneyline, spread, total, over_odds, under_odds = vegas_row
            # Raw values ------------------------------------------------------
            upcoming_game_info["moneyline_odds"] = moneyline
            upcoming_game_info["spread_line"]    = spread
            upcoming_game_info["total_line"]     = total
            upcoming_game_info["over_odds"]      = over_odds
            upcoming_game_info["under_odds"]     = under_odds

            # Derived values --------------------------------------------------
            def _implied_prob(usd):
                return (-usd / (-usd + 100)) if usd < 0 else (100 / (usd + 100))

            if moneyline is not None:
                upcoming_game_info["p_ml"]  = _implied_prob(moneyline)
                upcoming_game_info["fav_flag"] = int(moneyline < 0)
            else:
                upcoming_game_info["p_ml"] = 0.0
                upcoming_game_info["fav_flag"] = 0

            if spread is not None and total is not None:
                upcoming_game_info["abs_spread"] = abs(spread)
                upcoming_game_info["teim"] = (total / 2) - (spread / 2)   # team-implied
                upcoming_game_info["opim"] = (total / 2) + (spread / 2)   # opp-implied
            else:
                upcoming_game_info["abs_spread"] = 0.0
                upcoming_game_info["teim"] = 0.0
                upcoming_game_info["opim"] = 0.0
        else:
            logger.warning("No Vegas odds found for upcoming game; using zero-filled defaults.")
            for col in ["moneyline_odds","spread_line","total_line","over_odds","under_odds",
                        "p_ml","fav_flag","abs_spread","teim","opim"]:
                upcoming_game_info[col] = 0


        logger.info("Successfully calculated upcoming playoff context.")

        # --- Get Team Congestion Features for state BEFORE upcoming game ---
        # Fetch all team logs needed for congestion calculation up to day before upcoming game
        logger.debug("Fetching team logs for congestion calculation (up to yesterday)...")
        # Need team_meta_logs up to the day *before* the upcoming game
        upcoming_game_date_dt = pd.to_datetime(upcoming_game_info['game_date_dt'])
        team_meta_logs = pd.read_sql_query(
             f"""SELECT game_id, game_date, team_id, is_home, wl
                 FROM teams_game_features
                 WHERE game_date < ? AND (team_id = ? OR team_id = ?)
                 ORDER BY game_date""",
              conn,
              params=(upcoming_game_date_dt.strftime('%Y-%m-%d'), team_id, opponent_team_id),
              parse_dates=['game_date']
        )
        team_meta_logs['game_date'] = pd.to_datetime(team_meta_logs['game_date']).dt.tz_localize(None)


        # Calculate congestion features using REVISED logic
        processed_team_logs = _team_congestion_features(team_meta_logs)

        # Get the LATEST calculated congestion state for team and opponent
        latest_congestion: Dict[str, Any] = {}
        congestion_cols = [ # These are calculated by _team_congestion_features based on state *before* game
             "win_streak", "losing_streak", "road_trip_length", "is_back_to_back",
             "is_three_in_four", "is_five_in_seven", "rest_days",
             "is_first_night_back_to_back"
        ]
        for prefix, tid in (("team_", team_id), ("opponent_", opponent_team_id)):
             latest_log = processed_team_logs[processed_team_logs["team_id"] == tid]
             if not latest_log.empty:
                  latest_log_row = latest_log.sort_values("game_date").iloc[-1]
                  for col in congestion_cols:
                       if col in latest_log_row: latest_congestion[f"{prefix}{col}"] = latest_log_row[col]
             else: logger.warning(f"No processed log found for {prefix}id {tid} for congestion.")

        # Add latest congestion features to upcoming game info
        # Also add non-calculated context like travel/altitude from schedule/last game if needed by model
        # (Assuming travel/altitude are static and can be taken from schedule or player history last row later)
        upcoming_game_info.update(latest_congestion)
        upcoming_game_info["team_is_home"] = upcoming_game_info["is_home"] # Redundant but matches old player_history join convention
        upcoming_game_info["opponent_is_home"] = 1 - upcoming_game_info["is_home"]


        # --- Load PLAYOFF Model Artifacts for the SPECIFIC TARGET ---
        model_instance = _load_playoff_model(target=target, artifact_dir=artifact_dir)
        if model_instance.target != target: logger.warning(f"Loaded model target '{model_instance.target}' differs from requested '{target}'.")

        # Pull Full Player History (already includes many team/opp features from DB and RS tracking via TEMP VIEW)
        player_hist_df = _player_history(conn, player_id)
        if player_hist_df.empty: raise ValueError(f"Historical data retrieval returned empty for player {player_id}.")

        # Calculate Prediction Sigma (Reusing main model sigma logic - unchanged)
        try:
            sigma, sigma_method, sigma_n = _player_sigma(conn, player_id)
            logger.info(f"Sigma calculation for player {player_id} (using main runs): Value={sigma:.4f}, Method='{sigma_method}', N={sigma_n}")
        except ValueError as sigma_err:
             logger.error(f"Could not calculate sigma for player {player_id}: {sigma_err}"); sigma, sigma_method, sigma_n = None, "error", 0; logger.warning("Proceeding without valid sigma.")
        except Exception as e:
             logger.error(f"Unexpected error during sigma calculation: {e}", exc_info=True); sigma, sigma_method, sigma_n = None, "error", 0; logger.warning("Proceeding without valid sigma.")

        # --- Prepare Data Row using PLAYOFF Model's FE ---
        logger.debug("Preparing feature row for PLAYOFF prediction...")
        # Create a template based on the structure of historical data rows
        # Use the last historical row as a base, then overwrite with upcoming game context
        if player_hist_df.empty:
             raise ValueError(f"Cannot create prediction row template, no history for player {player_id}")
        new_row_template = player_hist_df.iloc[-1].to_dict()

        # Overwrite template with known upcoming game info and CALCULATED context
        # Map calculated context keys to potential feature names (e.g., team_win_streak)
        context_mapping = {
            "playoff_round": "playoff_round", "series_number": "series_number",
            "series_record": "series_record", "is_elimination_game": "is_elimination_game",
            "can_win_series": "can_win_series", "has_home_court": "has_home_court",
            "is_game_6": "is_game_6", "is_game_7": "is_game_7",
            "series_score_diff": "series_score_diff", "series_prev_game_margin": "series_prev_game_margin",
            "is_playoffs": "is_playoffs", "team_win_streak": "win_streak",
            "team_losing_streak": "losing_streak", "team_road_trip_length": "road_trip_length",
            "team_is_back_to_back": "is_back_to_back", "team_is_three_in_four": "is_three_in_four",
            "team_is_five_in_seven": "is_five_in_seven", "team_rest_days": "rest_days",
            "team_is_first_night_b2b": "is_first_night_back_to_back", "opponent_win_streak": "opponent_win_streak",
            "opponent_losing_streak": "opponent_losing_streak", "opponent_road_trip_length": "opponent_road_trip_length",
            "opponent_is_back_to_back": "opponent_is_back_to_back", "opponent_is_three_in_four": "opponent_is_three_in_four",
            "opponent_is_five_in_seven": "opponent_is_five_in_seven", "opponent_rest_days": "opponent_rest_days",
            "opponent_is_first_night_b2b": "opponent_is_first_night_back_to_back", "game_id": "game_id",
            "team_id": "team_id", "player_id": "player_id", "opponent_team_id": "opponent_team_id", "is_home": "is_home",
            "team_is_home": "team_is_home", "opponent_is_home": "opponent_is_home",
            "season": "season", "travel_distance": "travel_distance",
            "team_travel_distance": "travel_distance", "is_high_altitude": "is_high_altitude",
            "team_is_high_altitude": "is_high_altitude"
        }

        context_mapping.update({
            "moneyline_odds": "moneyline_odds",
            "spread_line":    "spread_line",
            "total_line":     "total_line",
            "over_odds":      "over_odds",
            "under_odds":     "under_odds",
            "p_ml":           "p_ml",
            "fav_flag":       "fav_flag",
            "abs_spread":     "abs_spread",
            "teim":           "teim",
            "opim":           "opim",
        })

        # Overwrite the template
        for template_key, context_key in context_mapping.items():
            if template_key in new_row_template:
                if context_key in upcoming_game_info:
                    # Handle potential type mismatches explicitly
                    target_dtype = player_hist_df[template_key].dtype if template_key in player_hist_df else None
                    value_to_set = upcoming_game_info[context_key]
                    # --- START CORRECTION ---
                    if pd.api.types.is_numeric_dtype(target_dtype) and not isinstance(value_to_set, (int, float, np.number)):
                        # Convert to numeric, then check for NaN separately
                        numeric_value = pd.to_numeric(value_to_set, errors='coerce')
                        if pd.isna(numeric_value):
                            new_row_template[template_key] = 0 # Assign 0 if conversion resulted in NaN
                        else:
                            # Optional: Try to preserve integer type if original was integer
                            if pd.api.types.is_integer_dtype(target_dtype):
                                try:
                                    new_row_template[template_key] = int(numeric_value)
                                except (ValueError, TypeError): # Handle cases where float can't be cleanly int
                                     new_row_template[template_key] = numeric_value # Assign the float
                            else:
                                new_row_template[template_key] = numeric_value # Assign the successfully converted number
                    elif pd.api.types.is_datetime64_any_dtype(target_dtype):
                         new_row_template[template_key] = pd.to_datetime(value_to_set, errors='coerce')
                    elif pd.api.types.is_string_dtype(target_dtype) or isinstance(target_dtype, object):
                         new_row_template[template_key] = str(value_to_set) if value_to_set is not None else ''
                    else: # Default assignment
                         new_row_template[template_key] = value_to_set
            # else: # Don't warn for every single column in the large mapping
                # logger.debug(f"Key '{template_key}' (for context '{context_key}') not in historical template.")

        # Add missing columns from player_hist_df that weren't in context_mapping
        for col in player_hist_df.columns:
            if col not in new_row_template:
                 new_row_template[col] = player_hist_df.iloc[-1][col] # Copy from last row

        # -------------------------------------------------------------------------
        # Add missing columns from player_hist_df that weren't in context_mapping
        # -------------------------------------------------------------------------
        for col in player_hist_df.columns:
            if col not in new_row_template:
                new_row_template[col] = player_hist_df.iloc[-1][col]  # Copy from last row

        # >>> PLACE THE FILL-FROM-LAST-GAME FLAG PATCH **RIGHT HERE** <<<

        place_holder_flags = [
            "player_game_is_injury", "player_game_is_available",
            "player_game_is_starter", "player_game_is_bench",
        ]

        for col in place_holder_flags:
            # create the key if it wasn’t in the template
            if col not in new_row_template:
                new_row_template[col] = np.nan

            # back-fill from the latest non-null historical value
            if pd.isna(new_row_template[col]) or new_row_template[col] == "":
                last_valid = player_hist_df[col].dropna()
                new_row_template[col] = last_valid.iloc[-1] if not last_valid.empty else 0  # hard-default


        # ✱✱ INSERT the safeguard here ─────────────────────────────────────
        required_odds_cols = [
            "moneyline_odds", "spread_line", "total_line",
            "over_odds", "under_odds",
            "p_ml", "fav_flag", "abs_spread", "teim", "opim"
        ]
        for col in required_odds_cols:
            # make sure the key exists
            if col not in new_row_template:
                new_row_template[col] = 0.0
            # overwrite with the upcoming-game value if we fetched it
            if col in upcoming_game_info:
                new_row_template[col] = upcoming_game_info[col]
        # ─────────────────────────────────────────────────────────────────-

        # Set target to NaN, game_date correctly
        new_row_template[model_instance.target] = np.nan
        new_row_template['game_date'] = upcoming_game_info['game_date_dt'] # Use datetime object


        # Create DataFrame for prediction
        new_game_df = pd.DataFrame([new_row_template])

        # Ensure types are consistent with historical df before passing to prepare_new_data
        for col in player_hist_df.columns:
             if col in new_game_df.columns:
                  try:
                       # Handle specific types first
                       if pd.api.types.is_datetime64_any_dtype(player_hist_df[col]):
                            new_game_df[col] = pd.to_datetime(new_game_df[col], errors='coerce')
                       elif pd.api.types.is_numeric_dtype(player_hist_df[col]):
                            # Convert to numeric, fill NaNs appropriately (e.g., 0)
                            new_game_df[col] = pd.to_numeric(new_game_df[col], errors='coerce').fillna(0) # Use 0 for missing numerics
                            # Try to preserve original integer type if possible
                            if pd.api.types.is_integer_dtype(player_hist_df[col].dtype):
                                 try:
                                      new_game_df[col] = new_game_df[col].astype(player_hist_df[col].dtype)
                                 except (ValueError, TypeError): # Handle potential Int64 conversion issues if float remains
                                      new_game_df[col] = new_game_df[col].astype(int) # Fallback to standard int

                       elif pd.api.types.is_bool_dtype(player_hist_df[col]):
                            new_game_df[col] = new_game_df[col].astype(bool) # Ensure boolean
                       elif pd.api.types.is_string_dtype(player_hist_df[col]) or isinstance(player_hist_df[col].dtype, object): # Catch object type too
                            new_game_df[col] = new_game_df[col].astype(str).fillna('') # Ensure string, fill NaNs with empty string
                       # Add more specific type handling if needed (e.g., categorical)

                  except Exception as e:
                       logger.warning(f"Type casting issue for column '{col}' during prediction row finalization: {e}")


        # Delegate feature prep to PLAYOFF model instance
        features_for_prediction = model_instance.prepare_new_data(player_hist_df, new_game_df)
        logger.debug(f"Playoff feature row prepared with {features_for_prediction.shape[1]} features.")
        logger.debug(f"Feature columns: {features_for_prediction.columns.tolist()}") # Log columns being used

        # --- Make PLAYOFF Prediction ---
        logger.debug("Making PLAYOFF prediction...")
        prediction_value = model_instance.predict(features_for_prediction)
        predicted_target_value = float(prediction_value[0])
        logger.info(f"Predicted PLAYOFF {model_instance.target}: {predicted_target_value:.3f}")

        # Format Results
        result = {
            "prediction_type": "playoffs", "player_id": player_id, "team_id": team_id,
            "opponent_team_id": opponent_team_id, "game_id": game_id_str,
            "game_date": game_date_str, "is_playoffs": True,
            "target_stat": model_instance.target, "predicted_value": predicted_target_value,
            "model_sigma": sigma, "sigma_method": sigma_method, "sigma_n": sigma_n,
            # Add calculated context to output for verification
            "calculated_context": calculated_playoff_context,
            "calculated_congestion": latest_congestion,
        }

        # Optional Betting Edge Calculation (Unchanged)
        if line is not None and over_odds is not None and under_odds is not None:
            if sigma is not None and sigma > 0:
                logger.debug("Calculating betting analysis for playoff prediction...")
                z = (line - predicted_target_value) / sigma; p_over = 1.0 - norm.cdf(z); p_under = 1.0 - p_over
                d_over = _american_to_decimal(over_odds); d_under = _american_to_decimal(under_odds)
                result["betting_analysis"] = {
                    "line": line, "z_score": z,
                    "over": {"am_odds": over_odds, "dec_odds": d_over, "prob": round(p_over, 4), "imp_prob": round(_implied_prob(d_over), 4), "kelly": round(_kelly_fraction(p_over, d_over), 4)},
                    "under": {"am_odds": under_odds, "dec_odds": d_under, "prob": round(p_under, 4), "imp_prob": round(_implied_prob(d_under), 4), "kelly": round(_kelly_fraction(p_under, d_under), 4)},
                }
                logger.debug("Playoff betting analysis complete.")
            else: logger.warning("Cannot calculate betting analysis due to missing/invalid sigma."); result["betting_analysis"] = None

        return result

    except (ValueError, FileNotFoundError, ConnectionError) as e:
         logger.error(f"PLAYOFF prediction failed for player {player_id}, target '{target}': {e}"); raise
    except Exception as e:
        logger.exception(f"Unexpected error during PLAYOFF prediction for player {player_id}, target '{target}': {e}"); raise
    finally:
        if conn: conn.close(); logger.debug("Database connection closed.")


# ───────────────────── PLAYOFF CLI glue (Unchanged) ──────────────────
def _cli_playoffs() -> None:
    """Command Line Interface setup and execution for PLAYOFF predictions."""
    parser = argparse.ArgumentParser(
        description=f"Predict NBA player stats for upcoming PLAYOFF games using the standalone playoff model. "
                    f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--player-id", type=int, required=True, help="NBA player_id (e.g., 201939 for Stephen Curry)")
    parser.add_argument(
        "--target", type=str, default="pts", required=False,
        help="Target stat to predict (default: pts) - MUST match trained playoff model artifacts."
    )
    parser.add_argument("--line", type=float, help="Prop bet line for the target stat (e.g., 28.5)")
    parser.add_argument("--over-odds", type=int, help="American odds for the OVER (e.g., -110)")
    parser.add_argument("--under-odds", type=int, help="American odds for the UNDER (e.g., -110)")
    parser.add_argument("--db-file", type=str, default=str(DB_FILE), help="Path to the SQLite database file.")
    parser.add_argument("--artifact-dir", type=str, default=str(ARTIFACT_DIR), help="Directory containing playoff model artifacts.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("playoff_predict_standalone").setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")

    if args.line is not None and (args.over_odds is None or args.under_odds is None):
        parser.error("--line requires --over-odds and --under-odds.")
    if (args.over_odds is not None or args.under_odds is not None) and args.line is None:
         parser.error("--over-odds/--under-odds require --line.")

    try:
        db_path_to_use = Path(args.db_file)
        artifact_path_to_use = Path(args.artifact_dir)
        logger.info(f"Using database: {db_path_to_use}")
        logger.info(f"Using artifact directory: {artifact_path_to_use}")

        # Call the main playoff prediction function
        prediction_result = predict_player_playoffs(
            player_id=args.player_id,
            target=args.target,
            line=args.line,
            over_odds=args.over_odds,
            under_odds=args.under_odds,
            db_path=db_path_to_use,
            artifact_dir=artifact_path_to_use
        )
        # Use default=str to handle potential non-serializable types like np.nan or pd.NA
        print(json.dumps(prediction_result, indent=2, default=str))
        sys.exit(0) # Success

    except (ValueError, FileNotFoundError, ConnectionError) as e:
         logger.error(f"Playoff prediction failed: {e}")
         print(f"ERROR: {e}", file=sys.stderr)
         sys.exit(1) # Failure exit code
    except Exception:
         logger.exception("An unexpected critical error occurred during playoff prediction.")
         print(f"ERROR: An unexpected critical error occurred. Check logs.", file=sys.stderr)
         sys.exit(1) # Failure exit code

if __name__ == "__main__":
    _cli_playoffs()