#!/usr/bin/env python3
"""
predict_points_xgboost_playoffs.py
====================================

Standalone prediction script for the *XGBoostPlayoffModel*.

Predicts player stats for upcoming PLAYOFF games only using a pre-trained
XGBoost model.

Key principles
--------------
1. **Loads XGBoost Playoff Artifacts:** Uses target-specific XGBoost playoff model artifacts.
2. **Checks Playoff Game:** Only proceeds if the player's next game is a playoff game.
3. **Calculates Correct Context:** Recalculates playoff series context and team
   sequence features (streaks, rest) accurately for the upcoming game.
4. **Replicates Feature Generation:** Creates a temporary instance of the
   `LightGBMPlayoffModel` class from the *original LightGBM script* to run its
   full `feature_engineering` pipeline on data with corrected context.
5. **Applies XGBoost Preprocessing:** Selects final features based on loaded XGBoost
   artifact, converts categoricals, and ensures numeric types.
6. **Schema Parity:** SQL queries aim to match training inputs where relevant.

**REWRITTEN:** Incorporates fixes mirroring the corrected LightGBM script:
streak logic, missing general features, and playoff context recalculation.
Includes fix for congestion calculation KeyError using iloc.
**ADDED:** Creation of TEMP VIEW player_tracking_rs_for_join in _player_history.
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
import xgboost as xgb # Import XGBoost
from scipy.stats import norm
# Removed SimpleImputer import, using fillna(0) instead before predict
from sql_player_props_playoffs import VIEW_SETUP_SQL, PLAYER_GAMES_SQL
from features_player_props import BASE_STATS, TEAM_STATS, OPP_STATS, PLAYER_ADVANCED_TRACKING, INTERACTIONS, INTERACTION_NAMES, EXTRA_FEATURES, CAT_FLAGS, PLAYOFF_COLS   # <- new line


# --- Import the XGBOOST model class and its helpers ---
try:
    from player_props_playoffs_xgboost import (
        XGBoostPlayoffModel,
        _NoOpScaler, # Keep if used by the XGBoostPlayoffModel class structure
        get_xgboost_model_path,
        get_xgboost_features_path,
        get_xgboost_cat_features_path,
        DEFAULT_ARTIFACTS_DIR_XGBOOST,
    )
except ImportError as ie_xgb:
     print(f"ERROR: Failed to import XGBoost training components from "
           f"'player_props_playoffs_xgboost.py'. Check script existence and content. Details: {ie_xgb}", file=sys.stderr)
     sys.exit(1)

# --- Import the LIGHTGBM model class FOR FEATURE ENGINEERING ---
try:
    from player_props_playoffs_lightgbm_alt import (
        LightGBMPlayoffModel as LightGBMFeatureGeneratorModel, # Rename for clarity
    )
except ImportError as ie_lgbm:
    print(f"ERROR: Failed to import LightGBM components needed for feature generation "
          f"from 'player_props_playoffs_lightgbm_alt.py'. Check script existence. Details: {ie_lgbm}", file=sys.stderr)
    sys.exit(1)

# ───────────────────────────── logging ─────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s [PlayoffPredictXGB]", # Add identifier
    datefmt="%Y‑%m‑%d %H:%M:%S",
)
logger = logging.getLogger("playoff_predict_xgboost") # Distinct logger

# ──────────────────────────── constants ────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
ARTIFACT_DIR = DEFAULT_ARTIFACTS_DIR_XGBOOST # Default to XGBoost artifacts

# --- Define Database Path ---
try:
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    DB_FILE = PROJECT_ROOT / "nba.db"
    if not DB_FILE.exists():
        logger.warning(f"Default database file not found at: {DB_FILE}. Using relative 'nba.db'. Use --db-file argument.")
        DB_FILE = Path("nba.db")
except Exception:
     logger.warning("Could not determine project root reliably. Defaulting DB path to 'nba.db' (relative to CWD). Use --db-file argument for specific path.")
     DB_FILE = Path("nba.db")

# Playoff context fields required for accurate prediction state
# Mirrors the corrected LightGBM script's definition
PLAYOFF_COLS = [
    "playoff_round", "series_number", "series_record", "is_elimination_game",
    "can_win_series", "has_home_court", "is_game_6", "is_game_7",
    "series_score_diff", "series_prev_game_margin", "is_playoffs",
]

# General sequence/congestion columns to calculate
CONGESTION_COLS = [
    "win_streak", "losing_streak", "road_trip_length", "is_back_to_back",
    "is_three_in_four", "is_five_in_seven", "rest_days",
    "is_first_night_back_to_back"
]

# --- Define LGBM FE Constants Locally (if needed by FE instance) ---
LGBM_DEFAULT_ROLLING_WINDOWS = [1, 2, 3, 5, 7, 10]
logger.debug(f"Using locally defined LGBM rolling windows for FE: {LGBM_DEFAULT_ROLLING_WINDOWS}")

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

# ─────────────────── XGBOOST artefact / model cache ────────────────
_XGBOOST_MODEL_CACHE: Dict[str, XGBoostPlayoffModel] = {}

def _load_xgboost_model(target: str, artifact_dir: Path = ARTIFACT_DIR) -> XGBoostPlayoffModel:
    """
    Singleton-load the fitted XGBOOST PLAYOFF model for a specific target.
    (Unchanged from original XGB script)
    """
    global _XGBOOST_MODEL_CACHE
    if target in _XGBOOST_MODEL_CACHE:
        logger.debug(f"Returning cached XGBoost playoff model for target '{target}'.")
        return _XGBOOST_MODEL_CACHE[target]

    logger.info(f"Loading XGBOOST PLAYOFF artefacts for target '{target}' from: {artifact_dir} …")
    model_filename = get_xgboost_model_path(target).name
    feats_filename = get_xgboost_features_path(target).name
    cats_filename = get_xgboost_cat_features_path(target).name
    model_file = artifact_dir / model_filename
    feats_file = artifact_dir / feats_filename
    cats_file = artifact_dir / cats_filename

    logger.debug(f"Attempting to load XGBoost model: {model_file}")
    logger.debug(f"Attempting to load XGBoost features: {feats_file}")
    logger.debug(f"Attempting to load XGBoost categoricals (names): {cats_file}")

    for fp in (model_file, feats_file, cats_file):
        if not fp.exists():
            raise FileNotFoundError(f"Required XGBOOST PLAYOFF artifact file missing for target '{target}': {fp}")

    mdl = XGBoostPlayoffModel(target=target, xgboost_artifacts_dir=artifact_dir)
    try:
        mdl.model = joblib.load(model_file)
        mdl.feature_names = joblib.load(feats_file)
        mdl.final_categorical_features = joblib.load(cats_file)
        mdl.scaler = _NoOpScaler()
        logger.info(
            f"XGBOOST PLAYOFF artefacts loaded for target '{target}' — {len(mdl.feature_names)} features ({len(mdl.final_categorical_features)} categoricals by name)"
        )
        _XGBOOST_MODEL_CACHE[target] = mdl
        return mdl
    except Exception as e:
        logger.error(f"Failed to load XGBoost playoff artifacts for target '{target}': {e}", exc_info=True)
        raise

# ───────────────────────── SQL helpers (CORRECTED) ────────────
# Using the corrected versions from the rewritten LightGBM script
def _connect_db(db_path: Path) -> sqlite3.Connection:
    """Helper to connect to the database."""
    try:
        logger.debug(f"Attempting to connect to database: {db_path}")
        if not db_path.exists():
             raise FileNotFoundError(f"Database file does not exist at: {db_path}")
        conn = sqlite3.connect(db_path)
        # Use WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;") # 5 seconds
        conn.row_factory = sqlite3.Row # Access columns by name
        logger.debug(f"Database connection successful: {db_path}")
        return conn
    except (sqlite3.Error, FileNotFoundError) as e:
        logger.error(f"Error connecting to database at {db_path}: {e}")
        raise ConnectionError(f"Failed to connect to database: {db_path}") from e

def _future_games(conn: sqlite3.Connection) -> Dict[int, List[dict]]:
    """
    Return *one record per team* for every game that is today or later.

    • Forces `team_id` and `opponent_team_id` to Int64 right away.
    • If the schedule table stores only the home row, we duplicate it
      to create the away row.
    • Keys of the returned mapping are plain `int` game-ids (the same
      type the LightGBM script now emits), so both predictors align.
    """

    logger.debug("Fetching future games schedule …")

    query = """
        SELECT
            gs.game_id,
            gs.game_date,
            gs.game_time,
            gs.team_id,             -- home side in DB
            gs.opponent_team_id,    -- away side
            gs.is_home,
            COALESCE(gs.is_playoffs, 0) AS is_playoffs,
            gs.season
        FROM gameschedules gs
        ORDER BY gs.game_date, gs.game_time
    """
    df = pd.read_sql_query(query, conn, parse_dates=["game_date"])
    logger.debug("Fetched %d raw schedule rows.", len(df))

    # 1 ── normalise ID columns ------------------------------------------------
    id_cols = ["team_id", "opponent_team_id"]
    df[id_cols] = df[id_cols].apply(pd.to_numeric, errors="coerce").astype("Int64")

    # 2 ── build robust UTC datetime ------------------------------------------
    def _combine_date_time(row_date, row_time):
        d_parsed = pd.to_datetime(str(row_date), errors="coerce", utc=True)
        t_parsed = pd.to_datetime(str(row_time), errors="coerce", utc=True)

        if (not pd.isna(d_parsed)) and (d_parsed.time() != pd.Timestamp.min.time()):
            return d_parsed
        if (not pd.isna(t_parsed)) and (t_parsed.date() != pd.Timestamp.min.date()):
            return t_parsed

        d_only = pd.to_datetime(row_date, errors="coerce").strftime("%Y-%m-%d")
        t_only = (
            pd.to_datetime(str(row_time), errors="coerce").strftime("%H:%M:%S")
            if pd.notna(row_time)
            else "00:00:00"
        )
        return pd.to_datetime(f"{d_only} {t_only}", utc=True, errors="coerce")

    df["game_datetime_utc"] = df.apply(
        lambda r: _combine_date_time(r["game_date"], r["game_time"]), axis=1
    )
    df = df.dropna(subset=["game_datetime_utc"])

    # 3 ── keep only today-and-forward ----------------------------------------
    today_utc = pd.Timestamp.utcnow().normalize()  # 00:00 UTC today
    df = df.loc[df["game_datetime_utc"].dt.normalize() >= today_utc].copy()
    logger.debug("Remaining future rows after date filter: %d", len(df))

    # 4 ── duplicate away row *iff* one row per fixture ------------------------
    if df.groupby("game_id").size().eq(1).all():        # exactly one row → home only
        away_df = df.copy()
        away_df[["team_id", "opponent_team_id"]] = away_df[
            ["opponent_team_id", "team_id"]
        ]
        away_df["is_home"] = 1 - away_df["is_home"]
        df = pd.concat([df, away_df], ignore_index=True)
        logger.debug("Schedule held only home rows – duplicated to create away rows.")
    else:
        logger.debug("Schedule already contains both home and away rows – no duplication.")

    # 5 ── housekeeping --------------------------------------------------------
    df["game_date_dt"] = df["game_datetime_utc"].dt.tz_convert(None).dt.normalize()
    df["game_date"] = df["game_date_dt"].dt.strftime("%Y-%m-%d")
    df["game_time"] = df["game_datetime_utc"].dt.tz_convert(None)
    df["is_playoffs"] = df["is_playoffs"].astype(int)
    df["season"] = df["season"].astype(str)

    # 6 ── build {game_id: [row-dicts]} ---------------------------------------
    future_map: Dict[int, List[dict]] = {
        gid: grp.to_dict("records") for gid, grp in df.groupby("game_id")
    }
    logger.debug(
        "Built future-games map with %d game_ids (%d total team rows).",
        len(future_map),
        len(df),
    )
    return future_map


def _player_history(conn: sqlite3.Connection, player_id: int, target: str) -> pd.DataFrame:
    """
    Retrieve ALL historical player rows, ensuring necessary columns and
    creating the TEMP VIEWs for regular season tracking stats.
    Uses SQL query structure identical to the training script.
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

        df = df.loc[:, ~df.columns.duplicated()]

         # ----------------------------------------------------------------------
        # 3.  Vegas-derived numeric columns  (exactly the same math as training)
        # ----------------------------------------------------------------------
        raw_odds_cols = [
            "moneyline_odds", "spread_line", "total_line",
            "over_odds", "under_odds"
        ]
        for c in raw_odds_cols:
            df[c] = pd.to_numeric(df.get(c, 0), errors="coerce")

        def _implied_prob(american):
            """
            Convert American odds (scalar, array, or Series) → implied probability
            without triggering divide-by-zero warnings.
            """
            american = np.asarray(american, dtype="float64")

            prob = np.empty_like(american)
            neg_mask = american < 0          # favourites
            pos_mask = ~neg_mask             # underdogs & even/zero

            # favourites: -odds / (-odds + 100)
            prob[neg_mask] = (
                -american[neg_mask] /
                (-american[neg_mask] + 100.0)
            )

            # underdogs / even odds: 100 / (odds + 100)
            prob[pos_mask] = (
                100.0 /
                (american[pos_mask] + 100.0)
            )

            # keep NaN where original value was NaN
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

        # Ensure necessary columns exist and have correct types (rest of the function is unchanged)
        if 'season' in df.columns: df['season'] = df['season'].astype(str)
        else:
            df['season'] = df['game_date'].apply(lambda d: f"{d.year - 1}-{str(d.year)[-2:]}" if d.month < 9 else f"{d.year}-{str(d.year + 1)[-2:]}")
        df['season'] = df['season'].astype(str)

        if 'win' not in df.columns:
             if 'team_wl' in df.columns: df['win'] = df['team_wl'].apply(lambda x: 1 if isinstance(x, str) and x.upper() == 'W' else 0).astype(int)
             else: df['win'] = 0
        df['win'] = pd.to_numeric(df['win'], errors='coerce').fillna(0).astype(int)

        # Default fill playoff columns
        for col in PLAYOFF_COLS:
             if col in df.columns:
                 if col == 'series_prev_game_margin': df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                 elif col == 'series_record': df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int) # Use numeric wins before
                 else: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
             else: df[col] = 0 if col != 'series_prev_game_margin' else 0.0

        # Default fill new general features
        for col in ['rest_days', 'is_first_night_back_to_back']:
            for prefix in ['team_', 'opponent_']:
                full_col = f'{prefix}{col}'
                if full_col not in df.columns: df[full_col] = 0
                df[full_col] = pd.to_numeric(df[full_col], errors='coerce').fillna(99 if col=='rest_days' else 0).astype(int)

        # Ensure opponent_vs_player cols exist
        ovp_cols = ["opponent_vs_player_fgm_allowed", "opponent_vs_player_fga_allowed", "opponent_vs_player_pts_allowed"]
        for col in ovp_cols:
             if col not in df.columns: df[col] = 0.0
             else: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # Ensure target column exists and is numeric
        if target not in df.columns:
             df[target] = np.nan
        df[target] = pd.to_numeric(df[target], errors='coerce')

        # Ensure other FE inputs exist and are numeric
        numeric_fe_inputs = ['min', 'fga', 'fta', 'fg3a', 'fgm', 'ftm', 'fg3m', 'usage_rate', 'poss']
        for col in numeric_fe_inputs:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else: df[col] = 0.0

        return df
    except (pd.errors.DatabaseError, sqlite3.Error) as e:
        logger.error(f"Database error fetching player history for {player_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing player history for {player_id}: {e}", exc_info=True)
        raise

# --- Sigma Calculation Helpers (Unchanged - Uses MAIN model residuals) ---
def _latest_two_runs(conn: sqlite3.Connection) -> tuple[str, str]:
    """
    Return the two most-recent *model_run* identifiers that exist in the
    `training_predictions` table – no filtering on playoff / regular-season
    naming.  Raises if fewer than two distinct batches are found.
    """
    logger.debug("Fetching latest two model_run batches for σ …")
    try:
        runs: list[str] = (
            pd.read_sql_query(
                "SELECT DISTINCT model_run "
                "FROM   training_predictions "
                "ORDER  BY model_run DESC",
                conn,
            )["model_run"]
            .dropna()
            .tolist()
        )

        if len(runs) < 2:
            raise ValueError("Need at least two model_run batches to estimate σ.")

        logger.debug("Latest model runs selected for σ: %s, %s", runs[0], runs[1])
        return runs[0], runs[1]

    except (pd.errors.DatabaseError, sqlite3.Error) as e:
        logger.error("Database error while fetching model_run list: %s", e)
        raise

def _player_sigma(conn: sqlite3.Connection, player_id: int) -> Tuple[float | None, str, int]:
    """Calculates sigma based on MAIN model residuals."""
    logger.debug(f"Calculating sigma for player_id: {player_id} using MAIN model residuals...")
    try:
        run_new, run_prev = _latest_two_runs(conn)
    except ValueError as e:
        logger.error(f"Cannot calculate sigma, failed to get latest main model runs: {e}")
        return None, "error_no_runs", 0

    method = "pooled"; n_points = 0; sigma = None
    PLAYER_SIGMA_MIN_N = 75 # Min points for player-specific sigma
    try:
        player_res_df = pd.read_sql_query(
            "SELECT residual FROM training_predictions WHERE player_id = ? AND model_run IN (?, ?)",
            conn, params=(player_id, run_new, run_prev),
        )
        player_res = player_res_df["residual"].dropna()
        n_player_points = len(player_res)

        if n_player_points >= PLAYER_SIGMA_MIN_N:
            variance = np.mean(player_res**2)
            if variance >= 0 and np.isfinite(variance): sigma = float(math.sqrt(variance))
            method = "player_specific"; n_points = n_player_points
            if sigma is not None and np.isfinite(sigma): logger.info("Using player-specific σ (N=%d, main runs): %.4f", n_points, sigma)
            else: logger.warning("Player-specific sigma calc invalid. Falling back."); sigma = None

        if sigma is None: # Fallback to pooled
            pooled_res_df = pd.read_sql_query(
                "SELECT residual FROM training_predictions WHERE model_run IN (?, ?)", conn, params=(run_new, run_prev)
            )
            pooled_res = pooled_res_df["residual"].dropna()
            if pooled_res.empty: return None, "error_pooled_empty", 0
            n_pooled_points = len(pooled_res)
            variance = np.mean(pooled_res**2)
            if variance >= 0 and np.isfinite(variance): sigma = float(math.sqrt(variance))
            method = "pooled"; n_points = n_pooled_points
            if sigma is not None and np.isfinite(sigma): logger.info("Using pooled σ (N=%d, main runs): %.4f", n_points, sigma)
            else: return None, "error_pooled_calc", n_pooled_points

        if sigma is None or not np.isfinite(sigma) or sigma <= 0: return None, method, n_points
        return sigma, method, n_points

    except (pd.errors.DatabaseError, sqlite3.Error) as e:
        logger.error(f"DB error calculating sigma for player {player_id}: {e}"); return None, f"error_{method}_db", n_points
    except Exception as e:
        logger.error(f"Unexpected error calculating sigma for {player_id}: {e}", exc_info=True); return None, f"error_{method}_unexpected", n_points

# --- NO _team_congestion_features needed - calculated within main function ---

# ───────────────── XGBOOST PLAYOFF prediction core (CORRECTED CONTEXT) ─────
def predict_player_playoffs_xgboost(
    player_id: int,
    target: str, # Target stat required
    *,
    line: Optional[float] = None,
    over_odds: Optional[int] = None,
    under_odds: Optional[int] = None,
    db_path: Path = DB_FILE,
    artifact_dir: Path = ARTIFACT_DIR,
) -> Dict[str, Any]:
    """
    Predict the **next scheduled PLAYOFF** game stat for *player_id* using XGBoost.
    Includes corrected context calculation and replicates LightGBM feature engineering.
    Includes fix for congestion calculation KeyError using iloc.
    """
    conn: Optional[sqlite3.Connection] = None
    logger.info(f"Attempting XGBoost PLAYOFF prediction for player {player_id}, target '{target}'...")

    # --- Instantiate the TEMPORARY LightGBM model for FE ---
    try:
        lgbm_fe_generator = LightGBMFeatureGeneratorModel(
            target=target, rolling_windows=LGBM_DEFAULT_ROLLING_WINDOWS
        )
        logger.debug("Temporary LightGBM FE instance created.")
    except Exception as e:
        raise RuntimeError("Could not set up feature engineering helper.") from e

    try:
        conn = _connect_db(db_path)

        # Find player's current team and name
        logger.debug(f"Finding current team and names for player_id {player_id}...")
        team_id_df = pd.read_sql_query("SELECT team_id FROM player_game_features WHERE player_id = ? ORDER BY game_date DESC LIMIT 1", conn, params=(player_id,))
        if team_id_df.empty: raise ValueError(f"No historical games found for player {player_id} in DB.")
        team_id = int(team_id_df.iloc[0]["team_id"])
        player_name_res = conn.execute("SELECT name FROM players WHERE player_id = ?", (player_id,)).fetchone()
        player_name = player_name_res['name'] if player_name_res else f"ID {player_id}"
        team_name_res = conn.execute("SELECT team_city || ' ' || team_name AS team_full_name FROM teams WHERE team_id = ?", (team_id,)).fetchone()
        team_name = team_name_res['team_full_name'] if team_name_res else f"ID {team_id}"
        logger.info(f"Player '{player_name}' on team '{team_name}'.")

        # Find Player's Next Game
        logger.debug("Identifying next scheduled game...")
        upc_games_dict = _future_games(conn)
        if not upc_games_dict: raise ValueError("No upcoming games found in schedule.")
        upcoming_game_info: Optional[dict] = None

        sorted_game_ids = sorted(upc_games_dict.keys(), key=lambda gid: min(g.get('game_time', pd.Timestamp.max) for g in upc_games_dict[gid] if isinstance(g.get('game_time'), pd.Timestamp)))
        for game_id in sorted_game_ids:
            game_records = upc_games_dict[game_id]
            for record in game_records:
                 if record["team_id"] == team_id:
                     upcoming_game_info = record.copy()
                     upcoming_game_info['game_id'] = game_id
                     break
            if upcoming_game_info: break
        if upcoming_game_info is None: raise ValueError(f"No valid scheduled game found for player {player_id}'s team {team_id}.")

        # <<< !!! PLAYOFF CHECK !!! >>>
        is_next_game_playoffs = bool(upcoming_game_info.get("is_playoffs", 0))
        game_date_str = upcoming_game_info.get('game_date', 'N/A') # Date string
        game_date_dt = upcoming_game_info.get('game_date_dt') # Datetime object
        game_id_str = upcoming_game_info.get('game_id', 'N/A')
        upcoming_season = upcoming_game_info.get('season')
        if not is_next_game_playoffs:
            raise ValueError(f"Player {player_id}'s next game (ID: {game_id_str} on {game_date_str}) is NOT a playoff game.")
        elif not upcoming_season:
             raise ValueError(f"Missing season for upcoming playoff game {game_id_str}.")
        else:
            logger.info(f"Next game (ID: {game_id_str} on {game_date_str}, Season: {upcoming_season}) IS a playoff game. Proceeding...")

        # --- Calculate CORRECT Playoff Context for Upcoming Game ---
        logger.info("Calculating playoff context for upcoming game...")
        opponent_team_id = upcoming_game_info["opponent_team_id"]
        calculated_playoff_context: Dict[str, Any] = {"is_playoffs": 1}
        # Fetch relevant historical playoff games for this specific series
        series_hist_df = pd.read_sql_query(
            f"""
            SELECT game_date, team_id, opponent_team_id, wl, series_number, pts, opp_pts, playoff_round, has_home_court
            FROM teams_game_features
            WHERE season = ? AND is_playoffs = 1
              AND ((team_id = ? AND opponent_team_id = ?) OR (team_id = ? AND opponent_team_id = ?))
            ORDER BY game_date
            """,
            conn, params=(upcoming_season, team_id, opponent_team_id, opponent_team_id, team_id),
            parse_dates=["game_date"]
        )
        series_hist_df['game_date'] = pd.to_datetime(series_hist_df['game_date']).dt.tz_localize(None)
        series_hist_df[['pts', 'opp_pts', 'series_number', 'playoff_round', 'has_home_court']] = \
            series_hist_df[['pts', 'opp_pts', 'series_number', 'playoff_round', 'has_home_court']].apply(pd.to_numeric, errors='coerce')

        last_game_in_series_df = None
        if not series_hist_df.empty:
             unique_games_in_series = series_hist_df.drop_duplicates(subset=['game_date', 'series_number']).sort_values('game_date')
             if not unique_games_in_series.empty:
                  last_game_date = unique_games_in_series['game_date'].max()
                  # Ensure the last game date is before the upcoming game date
                  if game_date_dt and last_game_date < game_date_dt:
                      last_game_in_series_df = series_hist_df[series_hist_df['game_date'] == last_game_date]
                      if len(last_game_in_series_df) > 2: last_game_in_series_df = last_game_in_series_df.iloc[-2:]
                  else:
                      logger.info("Last game in DB is on or after upcoming game date, assuming Game 1 or context reset.")

        # Determine upcoming game state based on fetched history
        if last_game_in_series_df is None or last_game_in_series_df.empty:
            logger.info("Upcoming game is Game 1 of the series.")
            upcoming_series_number = 1
            wins_before_upcoming, opp_wins_before_upcoming = 0, 0
            last_game_margin = np.nan
            calculated_playoff_context["playoff_round"] = 1 # Assume Round 1 if no history
            calculated_playoff_context["has_home_court"] = upcoming_game_info["is_home"] # HCA if home G1
        else:
            last_game_this_team = last_game_in_series_df[last_game_in_series_df['team_id'] == team_id].iloc[0]
            last_series_number = int(last_game_this_team['series_number']) if pd.notna(last_game_this_team['series_number']) else 0
            upcoming_series_number = last_series_number + 1

            # Count wins from full series history DataFrame up to the last game
            wins_before_upcoming = len(series_hist_df[(series_hist_df['team_id'] == team_id) & (series_hist_df['wl'] == 'W')])
            opp_wins_before_upcoming = len(series_hist_df[(series_hist_df['team_id'] == opponent_team_id) & (series_hist_df['wl'] == 'W')])

            last_pts = last_game_this_team['pts']
            last_opp_pts = last_game_this_team['opp_pts'] # pts scored by opponent in that game
            if pd.notna(last_pts) and pd.notna(last_opp_pts): last_game_margin = float(last_pts - last_opp_pts)
            else: last_game_margin = np.nan

            calculated_playoff_context["playoff_round"] = int(last_game_this_team['playoff_round']) if pd.notna(last_game_this_team['playoff_round']) else 1
            calculated_playoff_context["has_home_court"] = int(last_game_this_team['has_home_court']) if pd.notna(last_game_this_team['has_home_court']) else 0
            logger.info(f"Upcoming Game {upcoming_series_number}. Wins Before: Team={wins_before_upcoming}, Opp={opp_wins_before_upcoming}. Last Margin={last_game_margin if pd.notna(last_game_margin) else 'N/A'}")

        # Calculate remaining context
        calculated_playoff_context["series_number"] = upcoming_series_number
        calculated_playoff_context["series_record"] = wins_before_upcoming
        calculated_playoff_context["series_prev_game_margin"] = last_game_margin if pd.notna(last_game_margin) else None
        is_game_6 = int(upcoming_series_number == 6); calculated_playoff_context["is_game_6"] = is_game_6
        is_game_7 = int(upcoming_series_number == 7); calculated_playoff_context["is_game_7"] = is_game_7
        calculated_playoff_context["is_elimination_game"] = int((opp_wins_before_upcoming == 3) or is_game_7)
        calculated_playoff_context["can_win_series"] = int((wins_before_upcoming == 3) or is_game_7)
        calculated_playoff_context["series_score_diff"] = wins_before_upcoming - opp_wins_before_upcoming
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

        # --- Calculate CORRECT Team Congestion Features for state BEFORE upcoming game ---
        logger.info("Calculating team congestion features for state before upcoming game...")
        calculated_congestion = {}
        # Fetch logs for team and opponent ending *before* the upcoming game date
        if game_date_dt: # Ensure we have the date
            team_logs_hist = pd.read_sql_query(
                 f"""SELECT game_date, team_id, is_home, wl
                     FROM teams_game_features
                     WHERE game_date < ? AND (team_id = ? OR team_id = ?)
                     ORDER BY game_date""", # Keep initial sort for filtering
                  conn, params=(game_date_dt.strftime('%Y-%m-%d'), team_id, opponent_team_id),
                  parse_dates=['game_date']
            )
            team_logs_hist['game_date'] = pd.to_datetime(team_logs_hist['game_date']).dt.tz_localize(None)

            # --- BEGIN FIXED CONGESTION CALCULATION ---
            processed_congestion_logs = pd.DataFrame() # Placeholder for combined results
            if not team_logs_hist.empty:
                all_rows = []
                # Ensure sorted by date within team BEFORE grouping
                team_logs_hist = team_logs_hist.sort_values(["team_id", "game_date"])

                for tid_cong, grp_cong in team_logs_hist.groupby("team_id", sort=False):
                    # Reset index *within* the group to get 0-based positional index easily
                    grp_cong = grp_cong.reset_index(drop=True)
                    current_win_streak, current_losing_streak, current_road_trip = 0, 0, 0
                    prev_wl, prev_date = None, None

                    # Iterate using the new 0-based index from the group's reset_index
                    for iloc_pos, row in grp_cong.iterrows(): # Now iloc_pos is the 0-based integer position
                         d0 = row["game_date"]
                         r_cong = row.to_dict()
                         rest = (d0 - prev_date).days if prev_date is not None else 999
                         rest_days = max(0, rest - 1) if rest != 999 else 99
                         r_cong["win_streak"] = current_win_streak # Streak BEFORE this game
                         r_cong["losing_streak"] = current_losing_streak
                         r_cong["rest_days"] = rest_days

                         is_b2b = int(rest == 1)

                         # Calculate three_in_four using iloc for position-based lookback
                         three_in_four = 0
                         if iloc_pos >= 2: # Check if there are at least 2 previous games in this group
                             prev_date_3in4 = grp_cong.iloc[iloc_pos - 2]["game_date"]
                             if pd.notna(prev_date_3in4):
                                 three_in_four = int((d0 - prev_date_3in4).days <= 3)

                         # Calculate five_in_seven using iloc for position-based lookback
                         five_in_seven = 0
                         if iloc_pos >= 4: # Check if there are at least 4 previous games in this group
                             prev_date_5in7 = grp_cong.iloc[iloc_pos - 4]["game_date"]
                             if pd.notna(prev_date_5in7):
                                 five_in_seven = int((d0 - prev_date_5in7).days <= 6)

                         r_cong["is_back_to_back"] = is_b2b
                         r_cong["is_three_in_four"] = three_in_four # Assign fixed value
                         r_cong["is_five_in_seven"] = five_in_seven # Assign fixed value
                         r_cong["road_trip_length"] = current_road_trip if row["is_home"] == 0 else 0

                         all_rows.append(r_cong)

                         # Update state for next iteration
                         wl_val = str(row.get("wl", "")).upper()
                         if wl_val == "W": current_win_streak += 1; current_losing_streak = 0
                         elif wl_val == "L": current_losing_streak += 1; current_win_streak = 0
                         else: current_win_streak, current_losing_streak = 0, 0
                         current_road_trip = (current_road_trip + 1) if row["is_home"] == 0 else 0
                         prev_wl, prev_date = wl_val, d0

                if all_rows:
                    # Reconstruct the DataFrame from the processed rows
                    processed_congestion_logs = pd.DataFrame(all_rows)
                    # Need to re-sort if the order matters downstream (e.g., for first_night_b2b)
                    processed_congestion_logs = processed_congestion_logs.sort_values(["team_id", "game_date"]).reset_index(drop=True)

                    # Calculate is_first_night_back_to_back AFTER processing all rows
                    processed_congestion_logs['next_game_date'] = processed_congestion_logs.groupby('team_id')['game_date'].shift(-1)
                    processed_congestion_logs['is_first_night_back_to_back'] = ((processed_congestion_logs['next_game_date'] - processed_congestion_logs['game_date']).dt.days == 1).fillna(False).astype(int)
                    processed_congestion_logs = processed_congestion_logs.drop(columns=['next_game_date'])
            # --- END FIXED CONGESTION CALCULATION ---

            # Get the LATEST calculated congestion state for team and opponent
            for prefix, tid_get in (("team_", team_id), ("opponent_", opponent_team_id)):
                latest_log = processed_congestion_logs[processed_congestion_logs["team_id"] == tid_get]
                if not latest_log.empty:
                    latest_log_row = latest_log.iloc[-1] # Already sorted by date
                    for col in CONGESTION_COLS:
                        if col in latest_log_row and pd.notna(latest_log_row[col]):
                            calculated_congestion[f"{prefix}{col}"] = latest_log_row[col]
                        else: # Default if missing
                            calculated_congestion[f"{prefix}{col}"] = 99 if col == 'rest_days' else 0
                else: # Default if no history for team
                    logger.warning(f"No history found for {prefix}id {tid_get} for congestion calc.")
                    for col in CONGESTION_COLS: calculated_congestion[f"{prefix}{col}"] = 99 if col == 'rest_days' else 0

            upcoming_game_info.update(calculated_congestion)
            logger.info("Successfully calculated upcoming congestion features.")
        else:
            logger.warning("Cannot calculate congestion features, missing upcoming game date.")
            # Add default congestion values
            for prefix in ["team_", "opponent_"]:
                 for col in CONGESTION_COLS: upcoming_game_info[f"{prefix}{col}"] = 99 if col == 'rest_days' else 0


        # --- Load XGBOOST Model Artifacts ---
        xgb_model_instance = _load_xgboost_model(target=target, artifact_dir=artifact_dir)
        if xgb_model_instance.target != target:
             logger.warning(f"Loaded XGBoost model target '{xgb_model_instance.target}' differs from requested '{target}'.")

        # --- Prepare Data for Prediction ---
        logger.debug("Preparing feature row for XGBoost PLAYOFF prediction...")

        # 1. Pull Full Player History (includes newly added context columns)
        # Note: _player_history now creates the TEMP VIEW but doesn't use it in the main query
        player_hist_df = _player_history(conn, player_id, target)
        # Historical context in player_hist_df will be overwritten by calculated context for the prediction row

        # 2. Create the new game row template
        if player_hist_df.empty:
             logger.warning(f"No historical data for player {player_id}. Prediction row based on defaults + calculated context.")
             new_row_template = {'player_id': player_id, 'team_id': team_id}
             # Add other essential keys if needed by FE, with defaults
             new_row_template['season'] = upcoming_season
             new_row_template['game_date'] = game_date_dt
        else:
             new_row_template = player_hist_df.iloc[-1].to_dict()

        # 3. Update template with CORRECT calculated context and base info
        # Define mapping from calculated context keys to feature names in template
        context_mapping = {
             "playoff_round": "playoff_round", "series_number": "series_number", "series_record": "series_record",
             "is_elimination_game": "is_elimination_game", "can_win_series": "can_win_series", "has_home_court": "has_home_court",
             "is_game_6": "is_game_6", "is_game_7": "is_game_7", "series_score_diff": "series_score_diff",
             "series_prev_game_margin": "series_prev_game_margin", "is_playoffs": "is_playoffs",
             "team_win_streak": "team_win_streak", "team_losing_streak": "team_losing_streak", "team_road_trip_length": "team_road_trip_length",
             "team_is_back_to_back": "team_is_back_to_back", "team_is_three_in_four": "team_is_three_in_four", "team_is_five_in_seven": "team_is_five_in_seven",
             "team_rest_days": "team_rest_days", "team_is_first_night_b2b": "team_is_first_night_b2b", # Use correct key name
             "opponent_win_streak": "opponent_win_streak", "opponent_losing_streak": "opponent_losing_streak", "opponent_road_trip_length": "opponent_road_trip_length",
             "opponent_is_back_to_back": "opponent_is_back_to_back", "opponent_is_three_in_four": "opponent_is_three_in_four", "opponent_is_five_in_seven": "opponent_is_five_in_seven",
             "opponent_rest_days": "opponent_rest_days", "opponent_is_first_night_b2b": "opponent_is_first_night_b2b", # Use correct key name
             "game_id": "game_id", "team_id": "team_id", "opponent_team_id": "opponent_team_id",
             "is_home": "is_home", "team_is_home": "is_home", # team_is_home is just 'is_home' from player perspective
             "season": "season",
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

        for template_key, context_key in context_mapping.items():
            if context_key in upcoming_game_info:
                # Ensure template_key exists before assigning (should if history query worked)
                # if template_key not in new_row_template: new_row_template[template_key] = None # Add key if missing? Risky.
                new_row_template[template_key] = upcoming_game_info[context_key]

        # -------------------------------------------------------------------------
        # Fallback: fill injury / availability / starter flags from last valid game
        # -------------------------------------------------------------------------
        place_holder_flags = [
            "player_is_injury", "player_is_available",
            "player_is_starter", "player_is_bench",
        ]

        for col in place_holder_flags:
            # guarantee the key exists in the template
            if col not in new_row_template:
                new_row_template[col] = np.nan

            # if still NaN / empty, back-fill from latest non-null historical value
            if pd.isna(new_row_template[col]) or new_row_template[col] == "":
                if col in player_hist_df.columns:
                    last_valid_series = player_hist_df[col].dropna()
                    if not last_valid_series.empty:
                        new_row_template[col] = last_valid_series.iloc[-1]
                    else:
                        new_row_template[col] = 0         # hard default when no history
                else:
                    new_row_template[col] = 0             # column absent in history
        # -------------------------------------------------------------------------

        new_row_template[target] = np.nan
        new_row_template['game_date'] = game_date_dt # Use datetime object
        # Ensure opponent_is_home is opposite of team's is_home
        if 'is_home' in new_row_template and pd.notna(new_row_template['is_home']):
            new_row_template['opponent_is_home'] = 1 - new_row_template['is_home']
        else: new_row_template['opponent_is_home'] = 0 # Default if is_home missing

        new_game_df = pd.DataFrame([new_row_template])
        # Ensure types consistent with historical df
        for col in player_hist_df.columns:
             if col in new_game_df.columns:
                  try:
                       if pd.api.types.is_datetime64_any_dtype(player_hist_df[col]):
                           new_game_df[col] = pd.to_datetime(new_game_df[col])
                       elif pd.api.types.is_numeric_dtype(player_hist_df[col]):
                           # Handle None from calculated context (like prev margin) -> map to NaN then fill
                           new_val = pd.to_numeric(new_game_df[col], errors='coerce')
                           new_game_df[col] = new_val.fillna(0.0 if pd.api.types.is_float_dtype(player_hist_df[col]) else 0)
                           # Preserve integer types if possible (allow float conversion if needed)
                           target_type = player_hist_df[col].dtype
                           if pd.api.types.is_integer_dtype(target_type) and new_game_df[col].apply(lambda x: x == int(x)).all():
                               new_game_df[col] = new_game_df[col].astype(target_type)
                           else: # Convert to float if necessary or if original was float
                               new_game_df[col] = new_game_df[col].astype(float)
                       elif pd.api.types.is_string_dtype(player_hist_df[col]):
                           new_game_df[col] = new_game_df[col].astype(str).fillna('')
                       elif pd.api.types.is_categorical_dtype(player_hist_df[col]):
                           # Handle categoricals carefully if present in history
                           new_game_df[col] = new_game_df[col].astype(player_hist_df[col].dtype)
                       elif pd.api.types.is_bool_dtype(player_hist_df[col]):
                           new_game_df[col] = new_game_df[col].fillna(False).astype(bool)

                  except Exception as e: logger.warning(f"Type casting issue for column {col} (Target type: {player_hist_df[col].dtype}): {e}")

        # 4. Combine historical and new game data with CORRECT context, sort
        cols_to_use = player_hist_df.columns.union(new_game_df.columns)
        player_hist_df = player_hist_df.reindex(columns=cols_to_use)
        new_game_df = new_game_df.reindex(columns=cols_to_use)
        combo = pd.concat([player_hist_df, new_game_df], ignore_index=True)
        combo.sort_values(["player_id", "game_date"], inplace=True, na_position='last')
        combo.reset_index(drop=True, inplace=True)
        new_game_index = combo.index[-1] # Index position of the new row

        combo = combo.copy()

        # 5. Calculate Career/Season Averages on Combined Data (using corrected target col)
        logger.debug(f"Calculating shifted averages for '{target}' on combined data...")
        combo[target] = pd.to_numeric(combo[target], errors='coerce') # Ensure numeric again
        player_group_combo = combo.groupby('player_id')
        combo[f"{target}_career_avg"] = player_group_combo[target].transform(lambda s: s.shift(1).expanding(min_periods=1).mean()).fillna(0)
        player_season_group_combo = combo.groupby(['player_id', 'season'], observed=True, sort=False)
        combo[f"{target}_season_avg"] = player_season_group_combo[target].transform(lambda s: s.shift(1).expanding(min_periods=1).mean()).fillna(0)

        # 6. Run Feature Engineering (using imported LGBM FE generator)
        logger.debug("Running imported LightGBM feature engineering pipeline...")
        try:
            # Ensure the FE function handles potential NaNs robustly
            combo_engineered = lgbm_fe_generator.feature_engineering(combo.copy())
            # --- FIX: Add .copy() here to de-fragment the engineered DataFrame ---
            combo_engineered = combo_engineered.copy()
            # --------------------------------------------------------------------
            logger.debug(f"Feature engineering complete. Engineered DataFrame shape: {combo_engineered.shape}") # Optional: log shape
        except Exception as fe_err:
            logger.error(f"Error during LightGBM feature engineering step: {fe_err}", exc_info=True)
            raise RuntimeError("Feature generation failed during prediction.") from fe_err

        # 7. Extract the engineered row for the new game
        try:
            # Use the known index position for reliability
            new_row_engineered = combo_engineered.iloc[[new_game_index]]
            if new_row_engineered.empty: raise RuntimeError("Engineered row for the new game is unexpectedly empty.")
        except IndexError:
            logger.error(f"Cannot locate new game row at index {new_game_index} after feature engineering. Engineered df shape: {combo_engineered.shape}, combo df shape: {combo.shape}", exc_info=True)
            raise RuntimeError("Cannot reliably locate new game row after feature engineering.")
        except Exception as e:
            logger.error(f"Unexpected error extracting engineered row: {e}", exc_info=True)
            raise RuntimeError("Unexpected error locating new game row after feature engineering.")

        # 8. Select FINAL features required by the loaded XGBoost model
        final_feature_names_xgb = xgb_model_instance.feature_names
        logger.debug(f"Selecting {len(final_feature_names_xgb)} final features required by XGBoost model.")
        missing_model_features = [f for f in final_feature_names_xgb if f not in new_row_engineered.columns]
        if missing_model_features:
            logger.warning(f"Adding missing XGBoost model features with NaN: {missing_model_features}")
            for f in missing_model_features: new_row_engineered[f] = np.nan # Adding to single row DF, less likely to fragment significantly

        try:
            # Ensure order matches training - necessary for XGBoost
            # This copy already helps for features_for_prediction
            features_for_prediction = new_row_engineered[final_feature_names_xgb].copy()
        except KeyError as e:
            missing_keys = list(set(final_feature_names_xgb) - set(new_row_engineered.columns))
            logger.error(f"Cannot select final XGBoost features. Missing keys: {missing_keys}. Available columns after FE: {new_row_engineered.columns.tolist()}")
            raise ValueError(f"Cannot select final XGBoost features. Missing: {missing_keys}") from e

        # 9. Convert Categorical Features to Codes for XGBoost
        categorical_names_xgb = xgb_model_instance.final_categorical_features
        logger.debug(f"Converting {len(categorical_names_xgb)} features to integer codes (or handling)...")
        for col in categorical_names_xgb:
            if col in features_for_prediction.columns:
                # XGBoost handles NaNs internally if enable_categorical=True, but if not, factorize needs handling
                # Using factorize assumes non-numeric means categorical string/object
                if not pd.api.types.is_numeric_dtype(features_for_prediction[col]):
                     # Factorize handles NaNs by assigning -1
                     codes, _ = pd.factorize(features_for_prediction[col], sort=True)
                     features_for_prediction[col] = codes.astype("int32") # Use int32 for codes
                else:
                     # If it's somehow numeric but listed as categorical, ensure integer type? Or leave as is?
                     # Let's assume it should be treated as categorical -> convert to string then factorize
                     logger.warning(f"Column '{col}' is numeric but listed in XGBoost categoricals. Converting to string then factorizing.")
                     codes, _ = pd.factorize(features_for_prediction[col].astype(str), sort=True)
                     features_for_prediction[col] = codes.astype("int32")
            else:
                 logger.warning(f"Categorical feature '{col}' required by model but not found in engineered data. Setting to default code -1.")
                 features_for_prediction[col] = -1 # Assign default code for missing categorical

        # 10. Final Numeric Check and Imputation
        # Check before imputation
        nan_cols_before_impute = features_for_prediction.columns[features_for_prediction.isna().any()].tolist()
        if nan_cols_before_impute:
             logger.warning(f"NaNs detected before prediction in columns: {nan_cols_before_impute}. Imputing with 0.")
             features_for_prediction.fillna(0, inplace=True) # Simple imputation

        # Ensure all columns are numeric types acceptable by XGBoost (int, float)
        for col in features_for_prediction.columns:
             if not pd.api.types.is_numeric_dtype(features_for_prediction[col]):
                  logger.error(f"Non-numeric column '{col}' (type: {features_for_prediction[col].dtype}) remains AFTER categorical conversion and imputation. Attempting final numeric conversion.")
                  features_for_prediction[col] = pd.to_numeric(features_for_prediction[col], errors='coerce').fillna(0)

        # The re-ordering step might also return a non-fragmented view/copy
        if list(features_for_prediction.columns) != final_feature_names_xgb: # Line 905 could be around here or just before predict
             logger.warning("Re-ordering columns to match model training order.")
             features_for_prediction = features_for_prediction[final_feature_names_xgb]

        logger.debug(f"XGBoost feature row prepared. Shape: {features_for_prediction.shape}, Columns: {features_for_prediction.columns.tolist()}")

        # --- Calculate Prediction Sigma ---
        sigma, sigma_method, sigma_n = _player_sigma(conn, player_id)
        if sigma is not None: logger.info(f"Sigma: Value={sigma:.4f}, Method='{sigma_method}', N={sigma_n}")
        else: logger.warning(f"Sigma calculation failed (Method: {sigma_method}).")

        # --- Make XGBOOST Prediction ---
        logger.debug("Making XGBoost PLAYOFF prediction...")
        try:
            # If model was trained with enable_categorical=True, need DMatrix or correct types
            # Assuming standard predict method here. Add enable_categorical if needed.
            prediction_value = xgb_model_instance.model.predict(features_for_prediction)
            predicted_target_value = float(prediction_value[0])
        except Exception as pred_err:
             logger.error(f"XGBoost model.predict() failed: {pred_err}", exc_info=True)
             # Optionally log feature types/values here for debugging
             logger.error(f"Features dtypes:\n{features_for_prediction.dtypes}")
             logger.error(f"Features values:\n{features_for_prediction.iloc[0].to_dict()}")
             raise RuntimeError("XGBoost prediction execution failed.") from pred_err

        logger.info(f"Predicted XGBoost PLAYOFF {target}: {predicted_target_value:.3f}")

        # Format Results
        result = {
            "prediction_type": "playoffs_xgboost", "player_id": player_id, "player_name": player_name, "team_name": team_name,
            "team_id": team_id, "opponent_team_id": opponent_team_id, "game_id": game_id_str,
            "game_date": game_date_str, "is_playoffs": True, "target_stat": target,
            "predicted_value": round(predicted_target_value, 3),
            "model_sigma": round(sigma, 3) if sigma is not None else None,
            "sigma_method": sigma_method, "sigma_n": sigma_n,
            # Optionally include calculated context for verification
            # "calculated_context": calculated_playoff_context,
            # "calculated_congestion": calculated_congestion
        }

        # Optional Betting Edge Calculation
        if line is not None and over_odds is not None and under_odds is not None:
            if sigma is not None and sigma > 1e-6:
                logger.debug("Calculating betting analysis...")
                z = (line - predicted_target_value) / sigma; p_over = 1.0 - norm.cdf(z); p_under = 1.0 - p_over
                d_over = _american_to_decimal(over_odds); d_under = _american_to_decimal(under_odds)
                result["betting_analysis"] = {
                    "line": line, "z_score": round(z, 3),
                    "over": {"am_odds": over_odds, "dec_odds": round(d_over, 3), "prob": round(p_over, 4), "imp_prob": round(_implied_prob(d_over), 4), "kelly": round(_kelly_fraction(p_over, d_over), 4)},
                    "under": {"am_odds": under_odds, "dec_odds": round(d_under, 3), "prob": round(p_under, 4), "imp_prob": round(_implied_prob(d_under), 4), "kelly": round(_kelly_fraction(p_under, d_under), 4)},
                }
            else: result["betting_analysis"] = None

        return result

    except (ValueError, FileNotFoundError, ConnectionError, RuntimeError) as e:
         logger.error(f"XGBoost PLAYOFF prediction failed for player {player_id}, target '{target}': {e}")
         raise # Re-raise specific handled errors
    except Exception as e:
        logger.exception(f"Unexpected error during XGBoost PLAYOFF prediction for {player_id}, target '{target}': {e}")
        raise # Re-raise unexpected errors
    finally:
        if conn: conn.close(); logger.debug("Database connection closed.")

# ───────────────────── XGBOOST PLAYOFF CLI glue (Unchanged) ──────────────────
def _cli_xgboost() -> None:
    """Command Line Interface setup and execution for XGBOOST PLAYOFF predictions."""
    parser = argparse.ArgumentParser(
        description=f"Predict NBA player stats for upcoming PLAYOFF games using the standalone XGBoost playoff model.\n"
                    f"Requires LightGBM feature engineering logic from player_props_playoffs_lightgbm_alt.py.\n"
                    f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--player-id", type=int, required=True, help="NBA player_id (e.g., 201939 for Stephen Curry)")
    parser.add_argument(
        "--target", type=str, default="pts", required=False,
        help="Target stat to predict (default: pts) - MUST match trained XGBoost playoff model artifacts."
    )
    parser.add_argument("--line", type=float, help="Prop bet line for the target stat (e.g., 28.5)")
    parser.add_argument("--over-odds", type=int, help="American odds for the OVER (e.g., -110)")
    parser.add_argument("--under-odds", type=int, help="American odds for the UNDER (e.g., -110)")
    parser.add_argument("--db-file", type=str, default=str(DB_FILE), help="Path to the SQLite database file.")
    parser.add_argument("--artifact-dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR_XGBOOST), help="Directory containing XGBoost playoff model artifacts.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("playoff_predict_xgboost").setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")

    if args.line is not None and (args.over_odds is None or args.under_odds is None):
        parser.error("--line requires --over-odds and --under-odds.")
    if (args.over_odds is not None or args.under_odds is not None) and args.line is None:
         parser.error("--over-odds/--under-odds require --line.")

    try:
        db_path_to_use = Path(args.db_file).resolve()
        artifact_path_to_use = Path(args.artifact_dir).resolve()
        logger.info(f"Using database: {db_path_to_use}")
        logger.info(f"Using XGBoost artifact directory: {artifact_path_to_use}")
        if not db_path_to_use.exists(): raise FileNotFoundError(f"Database file not found: {db_path_to_use}")
        if not artifact_path_to_use.exists(): raise FileNotFoundError(f"Artifact directory not found: {artifact_path_to_use}")
        if not artifact_path_to_use.is_dir(): raise NotADirectoryError(f"Artifact path is not a directory: {artifact_path_to_use}")

        prediction_result = predict_player_playoffs_xgboost(
            player_id=args.player_id, target=args.target, line=args.line,
            over_odds=args.over_odds, under_odds=args.under_odds,
            db_path=db_path_to_use, artifact_dir=artifact_path_to_use
        )
        print(json.dumps(prediction_result, indent=2, default=str))
        sys.exit(0)

    except (ValueError, FileNotFoundError, ConnectionError, RuntimeError, NotADirectoryError) as e:
         logger.error(f"XGBoost Playoff prediction failed: {e}")
         print(f"\nERROR: {e}", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
         logger.exception("An unexpected critical error occurred during XGBoost playoff prediction.")
         print(f"\nERROR: An unexpected critical error occurred. Check logs. ({type(e).__name__})", file=sys.stderr)
         sys.exit(1)

if __name__ == "__main__":
    _cli_xgboost()