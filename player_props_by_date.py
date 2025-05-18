#!/usr/bin/env python3
"""
player_props_by_date.py
=======================

Analyse NBA player points prop bets using model predictions and market odds,
incorporating odds movement analysis from open to current.

Handles both regular season and playoff games. For playoff games, it calls
the **Optuna-tuned LGBM Pipeline MetaPlayoffModel prediction script** (`predict_points_meta_model_lightgbm.py`).
For regular season games, it runs only the standard prediction model.

Workflow Change (2025-05-02 v2.7 - Odds Movement Analysis):
1. Fetch schedule from DB.
2. Fetch CURRENT odds/props from Odds API (**Live/Upcoming Only**).
3. Extract players listed in API props.
4. Match API players to DB player IDs.
5. Run predictions ONLY for players found in API props:
   - Regular Season: Standard LGBM model.
   - Playoffs: **Call `predict_points_meta_model_lightgbm.predict_player_playoffs_meta`.**
6. For each player/market with current odds:
   a. Call `player_odds_movement.py` to get OPEN odds from BettingPros.
   b. Calculate Line Movement (Current Line - Open Line).
   c. Calculate Price Movement (Current Price - Open Price) for Over/Under.
   d. Compare Model Prediction vs. OPEN Odds.
   e. Compare Model Prediction vs. CURRENT Odds (existing analysis).
   f. Interpret Movement Context (Does movement support/contradict initial model value?).
7. Perform final analysis using the prediction, CURRENT odds, OPEN odds comparison, and MOVEMENT context.
8. Log detailed analysis including movement data to DB.

Updated 2025-05-02 (Odds Movement Version)
-----------------------------------------
* **Odds Movement Integration:** Calls `player_odds_movement.py` via subprocess.
* **Movement Calculation:** Calculates line and price movement.
* **Open Odds Analysis:** Compares model prediction against scraped open odds.
* **Movement Context:** Interprets line movement relative to initial model edge.
* **Enhanced Analysis:** Final betting decision considers current edge + movement context.
* **DB Schema Update:** Added columns for open odds, movements, context.
* **Console Output:** Updated to display movement analysis.
* **Meta Playoff Prediction:** Updated to use Optuna LGBM Pipeline Meta Model (v2.8).
* **API-First Workflow:** Retained from v2.6.
* **Conditional Prediction:** Retained from v2.6.
* **RMSE Logic:** Retained from v2.6.
* **Prediction Caching:** Retained from v2.6.
* **Dependencies:** Imports functions from prediction modules and uses subprocess.
* **Monte Carlo:** Retained.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import subprocess # <<< NEW IMPORT
import sys
import time
import unicodedata
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import requests
# import requests_cache # <<< REMOVED
from scipy.stats import norm

# --- Import the Meta Playoff prediction function ---
# <<< CHANGE 1: Import from the LGBM meta model script instead of the Ridge meta model script >>>
try:
    from predict_points_meta_model_lightgbm import predict_player_playoffs_meta
    # <<< CHANGE 2: Get the logger corresponding to the imported script >>>
    logger_meta = logging.getLogger("playoff_predict_meta_lgbm_optuna") # Get logger if needed
except ImportError as ie_meta:
    print(f"ERROR: Failed to import Meta Playoff prediction function from "
          # <<< CHANGE 3: Update error message to reflect the correct script name >>>
          f"'predict_points_meta_model_lightgbm.py'. Check script existence. Details: {ie_meta}", file=sys.stderr)
    sys.exit(1)

# --- Import the function to get open odds ---
# Assuming player_odds_movement.py is callable and has a fetch_and_scrape_player_odds function
# If not, we'll rely on subprocess call
# try:
#     from player_odds_movement import fetch_and_scrape_player_odds as get_open_odds_func
# except ImportError:
#     get_open_odds_func = None
#     logger.warning("Could not import fetch_and_scrape_player_odds from player_odds_movement.py. Will rely on subprocess.")
# Using subprocess as primary method based on request wording "call the script"

# ───────────────────────── Constants ─────────────────────────

# --- File Paths & DB ---
DB_FILE = "nba.db"
ENV_FILE_PATH = ".env"
SCRIPT_DIR = Path(__file__).parent.resolve()
PLAYER_ODDS_MOVEMENT_SCRIPT_PATH = SCRIPT_DIR / "player_odds_movement.py" # <<< NEW CONSTANT

# --- Define prediction modules ---
PREDICTION_MODULE_NAME_STD = "predict_points_lightgbm"
# <<< CHANGE 4: Update the constant to reflect the new playoff meta script >>>
PREDICTION_MODULE_NAME_PLAYOFF_META = "predict_points_meta_model_lightgbm"

PREDICTION_CACHE_FILE = "prediction_cache.json"
DB_TRAINING_PREDS_TABLE = "training_predictions"
DB_ANALYSIS_TABLE = "player_betting_analysis"
DB_PLAYERS_TABLE = "players"
DB_GAMESCHEDULES_TABLE = "gameschedules"
DB_TEAMS_TABLE = "teams"

# --- Default Artifact Dirs for Meta Predictor ---
# <<< CHANGE 5: Update default artifact directory for the new meta model >>>
DEFAULT_META_ARTIFACTS_DIR = SCRIPT_DIR / "meta_model_lgbm_optuna_playoff_artifacts"
DEFAULT_LGBM_PLAYOFF_ARTIFACTS_DIR = SCRIPT_DIR # Assumes relative to script dir
DEFAULT_XGB_PLAYOFF_ARTIFACTS_DIR = SCRIPT_DIR / "xgboost_playoff_artifacts_v7" # Example path

# --- API Configuration ---
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_API_SPORT_KEY = "basketball_nba"
ODDS_API_REGIONS = "us"
ODDS_API_MARKETS = "player_points"
ODDS_API_ODDS_FORMAT = "american"
ODDS_API_DATE_FORMAT = "iso"
ODDS_API_TIMEOUT = 10 # seconds
DEFAULT_BOOKMAKER = "fanduel"

# --- Analysis Parameters ---
DEFAULT_RMSE_FALLBACK = 10.0
MONTE_CARLO_SAMPLE_SIZE = 250_000
KELLY_EDGE_THRESHOLD = 0.05
PLAYER_SIGMA_MIN_N = 75
DEFAULT_PLAYOFF_TARGET_STAT = "pts"
MOVEMENT_LINE_STABLE_THRESHOLD = 0.5 # Definition of stable line movement (+/- this value)

# --- Team Name Normalization ---
TEAM_NORMALISE_MAP = {
    "los angeles clippers": "la clippers",
}

# --- Logging ---
LOG_FORMAT = "%(asctime)s — %(levelname)s — %(message)s"
LOG_DATE_FORMAT = "%Y‑%m‑%d %H:%M:%S"

# ───────────────────────── Setup & Configuration ─────────────────────────

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__) # Main logger for this script

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but .* was fitted with feature names",
    category=UserWarning,
)

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "nba-bet-analyser/2.7-odds-movement" # Updated user agent

# ───────────────────────── Helper Functions ─────────────────────────
# (remove_diacritics, normalise_team, format_api_date,
#  implied_prob, american_to_decimal, kelly_fraction unchanged)
def remove_diacritics(text: str) -> str:
    if not isinstance(text, str): return ""
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def normalise_team(name: str) -> str:
    if not isinstance(name, str): return "unknown"
    base = remove_diacritics(name).lower().strip()
    return TEAM_NORMALISE_MAP.get(base, base)

def format_api_date(date_str: str) -> str:
    try:
        dt = pd.to_datetime(date_str).tz_localize(None)
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    except ValueError:
        logger.warning(f"Could not parse date string '{date_str}'. Using as is.")
        if isinstance(date_str, str) and "T" in date_str and date_str.endswith("Z"):
            return date_str
        elif isinstance(date_str, str):
            return f"{date_str}T00:00:00Z" # Basic fallback if it looks like just a date
        else:
            logger.error(f"Unexpected type for date_str: {type(date_str)}. Returning default.")
            return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def implied_prob(american_odds: int) -> float:
    if not isinstance(american_odds, (int, float)) or american_odds == 0: return 0.0
    try:
        if american_odds > 0: return 100 / (american_odds + 100)
        else: return abs(american_odds) / (abs(american_odds) + 100)
    except (TypeError, ZeroDivisionError): return 0.0

def american_to_decimal(american_odds: int) -> float:
    if not isinstance(american_odds, (int, float)) or american_odds == 0: return 1.0
    try:
        if american_odds > 0: return 1.0 + american_odds / 100.0
        else: return 1.0 + 100.0 / abs(american_odds)
    except (TypeError, ZeroDivisionError): return 1.0

def kelly_fraction(p: float, decimal_odds: float) -> float:
    if decimal_odds <= 1.0: return 0.0
    b = decimal_odds - 1.0
    if b == 0: return 0.0
    try:
        fraction = (p * b - (1.0 - p)) / b
        return max(0.0, min(fraction, 1.0))
    except (TypeError, ZeroDivisionError): return 0.0


# ────────────────── Prediction Cache Handling (MODIFIED) ─────────────────
# (load_prediction_cache and save_prediction_cache remain the same as v2.6, but validation updated)
def load_prediction_cache(cache_file_path: Path, current_key: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Loads prediction cache for the specific game_date/game_id key.
    Cache structure expected: {cache_key: {player_id_str: {"standard": {...}, "playoff_meta": {...}}}}
    """
    if not cache_file_path.exists():
        logger.info("Prediction cache file not found. Will generate predictions.")
        return None
    try:
        with open(cache_file_path, 'r') as f:
            full_cache = json.load(f)
        if not isinstance(full_cache, dict):
            logger.warning(f"Prediction cache file {cache_file_path} is not a valid JSON object. Discarding.")
            return None

        if current_key in full_cache:
            logger.info(f"Prediction cache HIT for key: {current_key}")
            cached_data = full_cache[current_key]
            # Validate the structure for the key
            if isinstance(cached_data, dict):
                valid_player_data = {}
                for pid, data in cached_data.items():
                    # Expect player data to be a dict containing 'standard' or 'playoff_meta'
                    if isinstance(data, dict):
                        is_valid = False
                        if 'standard' in data and isinstance(data['standard'], dict):
                            is_valid = True
                        # <<< CHANGE 6: Validate the prediction_type for the LGBM meta model >>>
                        if 'playoff_meta' in data and isinstance(data['playoff_meta'], dict):
                            if data['playoff_meta'].get("prediction_type") == "playoffs_meta_lgbm_optuna":
                                is_valid = True

                        if is_valid:
                            valid_player_data[str(pid)] = data
                        else:
                            logger.warning(f"Invalid data structure or prediction_type in cache for player {pid} under key {current_key}. Skipping player.")
                return valid_player_data
            else:
                logger.warning(f"Invalid data format in cache for key {current_key}. Discarding.")
                return None
        else:
            logger.info(f"Prediction cache MISS for key: {current_key}. Will generate predictions.")
            return None
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Error loading prediction cache file {cache_file_path}: {e}. Will generate predictions.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading prediction cache {cache_file_path}: {e}", exc_info=True)
        return None

def save_prediction_cache(cache_file_path: Path, current_key: str, new_predictions: Dict[str, Dict[str, Any]]) -> None:
    """
    Saves newly generated predictions to the cache file under the current key.
    Input structure: {player_id_str: {"standard": {...}}} OR {player_id_str: {"playoff_meta": {...}}}
    """
    full_cache = {}
    if cache_file_path.exists():
        try:
            with open(cache_file_path, 'r') as f:
                loaded_data = json.load(f)
                if isinstance(loaded_data, dict):
                    full_cache = loaded_data
                else:
                    logger.warning(f"Existing cache file {cache_file_path} was not a valid JSON object. Overwriting.")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not read existing cache file {cache_file_path} before saving: {e}. Starting fresh.")
            full_cache = {}
        except Exception as e:
                logger.error(f"Unexpected error reading existing cache {cache_file_path}: {e}. Starting fresh.", exc_info=True)
                full_cache = {}

    # Get existing data for the key or initialize
    current_key_data = full_cache.get(current_key, {})

    # Update the data for the current key by merging new predictions player by player
    for pid_str, sources in new_predictions.items():
        player_entry = current_key_data.setdefault(pid_str, {})
        if 'standard' in sources:
            player_entry['standard'] = sources['standard']
        if 'playoff_meta' in sources:
            player_entry['playoff_meta'] = sources['playoff_meta'] # Store the full meta result

    # Put updated/new data back into the full cache structure
    full_cache[current_key] = current_key_data

    try:
        with open(cache_file_path, 'w') as f:
            json.dump(full_cache, f, indent=2, default=str) # Use default=str for non-serializable types like Path
        logger.info(f"Saved/Updated prediction cache for key: {current_key}")
    except (IOError, TypeError) as e:
        logger.error(f"Error saving prediction cache to {cache_file_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving prediction cache {cache_file_path}: {e}", exc_info=True)


# ───────────────────────── Environment/Args Loading (MODIFIED)──────────────────────
# (Unchanged from previous version, except argument help text)
def _parse_env_file(p: Path) -> dict[str, str]:
    if not p.exists(): return {}
    env: dict[str, str] = {}
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("'\"")
    except Exception as e: logger.error(f"Error reading env file {p}: {e}")
    return env

def parse_args() -> argparse.Namespace:
    # <<< CHANGE 7: Update argument parser description >>>
    ap = argparse.ArgumentParser(description="Analyse NBA player points prop bets using model predictions (LGBM Optuna meta playoffs).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--game_date", help=f"Schedule ISO date (e.g., {datetime.now(timezone.utc).strftime('%Y-%m-%d')})")
    g.add_argument("--game_id", help="Exact gameschedules.game_id")
    ap.add_argument("--db-file", default=str(DB_FILE), help="Path to the SQLite database file.")
    ap.add_argument("--api-key", help="The Odds API key for live/upcoming odds.")
    # Removed --api-key-historical
    ap.add_argument("--env-file", default=ENV_FILE_PATH, help="Path to .env file for API keys.")
    ap.add_argument("--bookmakers", default=DEFAULT_BOOKMAKER, help="Comma-separated bookmakers or 'all'.")
    ap.add_argument("--dry-run", action="store_true", help="Print analysis instead of inserting into DB.")
    ap.add_argument("--sample-size", type=int, default=MONTE_CARLO_SAMPLE_SIZE, help="Monte Carlo sample size.")
    ap.add_argument("--rebuild-cached", action="store_true", help="Force regeneration of predictions, ignoring cache.")
    ap.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    # Add arguments for base model artifact paths needed by meta predictor
    # <<< CHANGE 8: Update help text for meta artifact directory argument >>>
    ap.add_argument("--meta-artifact-dir", type=str, default=str(DEFAULT_META_ARTIFACTS_DIR),
        help="Directory containing LGBM Optuna META-MODEL PIPELINE artifacts. Passed to meta predictor.")
    ap.add_argument("--lgbm-playoff-artifact-dir", type=str, default=str(DEFAULT_LGBM_PLAYOFF_ARTIFACTS_DIR),
        help="Directory containing LightGBM PLAYOFF model artifacts (used by meta predictor).")
    ap.add_argument("--xgb-playoff-artifact-dir", type=str, default=str(DEFAULT_XGB_PLAYOFF_ARTIFACTS_DIR),
        help="Directory containing XGBoost PLAYOFF model artifacts (used by meta predictor).")
    # <<< NEW ARGUMENT (Optional) >>>
    ap.add_argument("--odds-script-path", type=str, default=str(PLAYER_ODDS_MOVEMENT_SCRIPT_PATH),
        help="Path to the player_odds_movement.py script.")

    return ap.parse_args()

def resolve_api_key(args: argparse.Namespace) -> Optional[str]:
    """Resolves the single Odds API key."""
    key_live = args.api_key or os.getenv("ODDS_API_KEY")
    if not key_live:
        env = _parse_env_file(Path(args.env_file).expanduser())
        key_live = key_live or env.get("ODDS_API_KEY")
    if not key_live: logger.warning("Live Odds API key not found.")
    return key_live


# ─────────────────── Prediction Module Import (MODIFIED) ────────────────────
# (Unchanged from previous version, except log message)
def import_prediction_functions() -> Tuple[Optional[callable], Optional[callable]]:
    """Imports prediction functions from standard and meta playoff modules."""
    predict_player_std = None
    predict_player_meta = None # Only need the meta function

    # Import Standard Prediction Function
    try:
        module_std = __import__(PREDICTION_MODULE_NAME_STD)
        if hasattr(module_std, "predict_player"):
            logger.info(f"Using standard prediction function from module: {PREDICTION_MODULE_NAME_STD}")
            predict_player_std = getattr(module_std, "predict_player")
        else:
            raise AttributeError(f"'predict_player' not found in {PREDICTION_MODULE_NAME_STD}")
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load STANDARD predict_player function: {e}. Standard predictions unavailable.")

    # Import Meta Playoff Prediction Function (already imported globally, just assign)
    if 'predict_player_playoffs_meta' in globals():
        # <<< CHANGE 9: Update log message for the meta function >>>
        logger.info(f"Using META playoff prediction function from module: {PREDICTION_MODULE_NAME_PLAYOFF_META}")
        predict_player_meta = predict_player_playoffs_meta # Assign the imported function
    else:
        # This case should be caught by the initial import check, but good to be safe
        logger.error(f"Failed to assign META PLAYOFF prediction function. It was not imported correctly.")


    # --- Updated Checks ---
    if predict_player_std is None and predict_player_meta is None:
        logger.critical("No prediction functions could be loaded. Exiting.")
        sys.exit(1)
    if predict_player_std is None:
            logger.warning("Standard prediction function failed to load.")
    if predict_player_meta is None:
        logger.warning("Meta Playoff prediction function failed to load.")

    return predict_player_std, predict_player_meta


# ─────────────────── API Interaction (Unchanged) ──────────────────────
# (Unchanged from previous version)
def _odds_api_get(endpoint: str, params: dict, api_key: Optional[str]) -> Optional[dict | list]:
    if not api_key:
        logger.error(f"API key missing for request to endpoint: {endpoint}")
        return None
    url = f"{ODDS_API_BASE_URL}/{endpoint}"
    params['apiKey'] = api_key
    try:
        r = SESSION.get(url, params=params, timeout=ODDS_API_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout: logger.error(f"Timeout error requesting Odds API: {endpoint}")
    except requests.exceptions.HTTPError as e: logger.error(f"HTTP error for Odds API {endpoint}: {e} (Status: {r.status_code if 'r' in locals() else 'N/A'})")
    except requests.exceptions.RequestException as e: logger.error(f"Request exception for Odds API {endpoint}: {e}")
    except json.JSONDecodeError as e: logger.error(f"JSON decode error for Odds API {endpoint}: {e} - Response: {r.text[:200] if 'r' in locals() else 'N/A'}...")
    except Exception as e: logger.error(f"Unexpected error during API request to {endpoint}: {e}")
    return None

def fetch_odds_api_games(api_key: str, bookmakers: Optional[str] = None) -> list:
    """Fetches LIVE/UPCOMING games from the Odds API."""
    endpoint = f"sports/{ODDS_API_SPORT_KEY}/odds"
    params = {"regions": ODDS_API_REGIONS, "markets": "h2h", "oddsFormat": ODDS_API_ODDS_FORMAT, "dateFormat": ODDS_API_DATE_FORMAT}
    logger.info("Fetching LIVE/UPCOMING games from Odds API.") # Only fetches live now

    if bookmakers and bookmakers.lower() != 'all': params["bookmakers"] = bookmakers
    logger.debug(f"Fetching Odds API games: Endpoint={endpoint}, Params={params}")
    result = _odds_api_get(endpoint, params, api_key)
    return result if isinstance(result, list) else []

def fetch_odds_api_props(api_key: str, event_id: str, bookmakers: Optional[str] = None, market:str = ODDS_API_MARKETS) -> list:
    """Fetches LIVE/UPCOMING player props for a specific event."""
    endpoint = f"sports/{ODDS_API_SPORT_KEY}/events/{event_id}/odds"
    params = {"markets": market, "oddsFormat": ODDS_API_ODDS_FORMAT, "dateFormat": ODDS_API_DATE_FORMAT}
    if bookmakers and bookmakers.lower() != 'all': params["bookmakers"] = bookmakers
    logger.debug(f"Fetching LIVE Odds API props: Endpoint={endpoint}, Params={params}") # Only live now

    result = _odds_api_get(endpoint, params, api_key)
    # Result for event odds endpoint is a single dictionary (containing bookmakers list), not a list directly.
    if isinstance(result, dict): return [result] # Wrap dict in list for consistency downstream
    elif isinstance(result, list):
        # Should technically not happen for event endpoint, but handle just in case API changes
        logger.warning(f"Received unexpected list format from event odds endpoint for {event_id}. Processing anyway.")
        return result
    else: return []

# ────────────────────── SQLite Operations (MODIFIED) ────────────────────

def ensure_analysis_table(conn: sqlite3.Connection) -> None:
    """Creates or alters the analysis table, adding columns for odds movement."""
    cursor = conn.execute(f"PRAGMA table_info({DB_ANALYSIS_TABLE})")
    columns = {col[1]: col[2] for col in cursor.fetchall()} # name -> type

    # Base schema with added movement columns
    base_create_sql = f"""
        CREATE TABLE IF NOT EXISTS {DB_ANALYSIS_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            odds_api_game_id TEXT,
            game_date TEXT NOT NULL,
            player_id TEXT NOT NULL,
            player_name TEXT,
            team_id TEXT,
            team_name TEXT,
            opponent_team_id TEXT,
            opponent_name TEXT,
            bookmakers TEXT NOT NULL,
            market TEXT NOT NULL,
            line REAL NOT NULL,          -- CURRENT Line from API
            price INTEGER NOT NULL,       -- CURRENT Price from API
            bet_type TEXT NOT NULL,       -- 'over' or 'under'
            predicted_points REAL,         -- Stores STANDARD prediction OR META prediction for playoffs
            prediction_source TEXT,       -- e.g., 'standard', 'playoff_meta_lgbm_optuna'
            predicted_points_lgbm REAL,    -- Stores base LGBM prediction when source is meta, else NULL
            predicted_points_xgb REAL,     -- Stores base XGB prediction when source is meta, else NULL
            model_win_pct REAL,            -- Model win% vs CURRENT odds
            rmse REAL,                     -- Stores the RMSE used for analysis (from standard or meta result)
            z_score REAL,                  -- Z-score vs CURRENT line
            monte_carlo_win_pct REAL,      -- MC win% vs CURRENT line
            implied_book_win_pct REAL,     -- Implied win% from CURRENT odds
            kelly_fraction REAL,           -- Kelly fraction vs CURRENT odds
            edge REAL,                     -- Edge vs CURRENT odds (model_win_pct - implied_book_win_pct)
            alert_text TEXT,
            -- NEW COLUMNS FOR MOVEMENT ANALYSIS --
            open_line REAL,                -- OPEN Line (e.g., from BettingPros scrape)
            open_price_over INTEGER,       -- OPEN Over Price
            open_price_under INTEGER,      -- OPEN Under Price
            line_movement REAL,            -- current_line - open_line
            price_movement_over INTEGER,   -- current_over_price - open_price_over
            price_movement_under INTEGER,  -- current_under_price - open_price_under
            model_edge_vs_open REAL,       -- Edge model had vs OPEN odds for this bet_type
            movement_context TEXT,         -- e.g., "Supports Model", "Contradicts Model", "Stable"
            -- END NEW COLUMNS --
            created_at TEXT DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'now', 'utc'))
        );
        """
    # Index definition (includes prediction_source as part of the key)
    # <<< CHANGE 10: Ensure index definition includes the prediction_source >>>
    index_sql = f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_analysis_unique ON {DB_ANALYSIS_TABLE}
            (game_id, player_id, bookmakers, market, line, bet_type, game_date, prediction_source);
        """

    # --- Columns to Add/Check ---
    columns_to_add = {
        "prediction_source": "TEXT", # Needs to store the new source string
        "predicted_points_lgbm": "REAL",
        "predicted_points_xgb": "REAL",
        # New movement columns
        "open_line": "REAL",
        "open_price_over": "INTEGER",
        "open_price_under": "INTEGER",
        "line_movement": "REAL",
        "price_movement_over": "INTEGER",
        "price_movement_under": "INTEGER",
        "model_edge_vs_open": "REAL",
        "movement_context": "TEXT"
    }

    try:
        with conn:
            conn.execute(base_create_sql) # Create table if not exists

            # Add missing columns individually
            for col_name, col_type in columns_to_add.items():
                if col_name not in columns:
                    logger.info(f"Adding '{col_name}' column to {DB_ANALYSIS_TABLE}...")
                    try:
                        conn.execute(f"ALTER TABLE {DB_ANALYSIS_TABLE} ADD COLUMN {col_name} {col_type};")
                    except sqlite3.OperationalError as alter_err:
                        if "duplicate column name" not in str(alter_err): raise alter_err
                        else: logger.warning(f"Attempted to add '{col_name}' column, but it already exists: {alter_err}")

            # Drop old index if it exists and differs? Simpler to just try creating new one.
            # conn.execute(f"DROP INDEX IF EXISTS idx_analysis_unique;") # Drop old index just in case structure changed
            conn.execute(index_sql) # Create unique index (will fail harmlessly if exists)
        logger.info(f"Ensured database table '{DB_ANALYSIS_TABLE}' with required columns (including movement) and index exist.")
    except sqlite3.Error as e:
        logger.error(f"Database error ensuring analysis table/columns/index: {e}")
        raise

# (fetch_rmses unchanged)
def fetch_rmses(conn: sqlite3.Connection) -> float:
    """Fetches global in-sample RMSE from latest main model run (used as ultimate fallback)."""
    # This remains the same - it's the fallback if the sigma calculation within the
    # standard or meta predictors completely fails.
    logger.info("Fetching global in-sample RMSE from latest model run (fallback).")
    global_rmse = DEFAULT_RMSE_FALLBACK
    try:
        cursor_check = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (DB_TRAINING_PREDS_TABLE,))
        if cursor_check.fetchone() is None:
            logger.warning(f"Table '{DB_TRAINING_PREDS_TABLE}' not found. Cannot fetch global RMSE. Using default: {global_rmse}")
            return global_rmse
        with conn:
            cursor = conn.execute(
                f"""
                SELECT AVG(residual*residual)
                FROM {DB_TRAINING_PREDS_TABLE}
                WHERE model_run = (
                    SELECT MAX(model_run) FROM {DB_TRAINING_PREDS_TABLE}
                    WHERE model_run NOT LIKE '%playoffs%' -- Ensure we use main model runs
                    AND model_run NOT LIKE '%xgb%' -- Exclude XGB runs if they have separate IDs
                    AND model_run NOT LIKE '%meta%' -- Exclude Meta runs if logged here
                )
                """
            )
            result = cursor.fetchone()
            if result and result[0] is not None:
                global_rmse = float(np.sqrt(result[0]))
                logger.info(f"Using Global Fallback RMSE: {global_rmse:.3f}")
            else:
                logger.warning(f"No residuals found in {DB_TRAINING_PREDS_TABLE} for main runs. Using default RMSE: {global_rmse}")
    except sqlite3.Error as e:
        logger.error(f"Database error fetching global RMSE: {e}. Using default: {global_rmse}")
    except Exception as e:
        logger.error(f"Unexpected error fetching global RMSE: {e}. Using default: {global_rmse}")
    return global_rmse


def log_analysis_results_db(conn: sqlite3.Connection, analysis_results: List[Dict]):
    """Inserts analysis results into the database, including movement columns."""
    if not analysis_results:
        logger.info("No analysis results to log to database.")
        return

    # --- MODIFIED required keys include movement columns ---
    required_keys = {
        "game_id", "odds_api_game_id", "game_date", "player_id", "player_name",
        "team_id", "team_name", "opponent_team_id", "opponent_name", "bookmakers",
        "market", "line", "price", "bet_type",
        "predicted_points", # Stores standard pred OR meta pred
        "prediction_source", # 'standard' or 'playoff_meta_lgbm_optuna'
        "predicted_points_lgbm", # Base LGBM pred for meta source
        "predicted_points_xgb", # Base XGB pred for meta source
        "model_win_pct", "rmse", "z_score", "monte_carlo_win_pct",
        "implied_book_win_pct", "kelly_fraction", "edge", "alert_text",
        # Movement keys (can be None)
        "open_line", "open_price_over", "open_price_under", "line_movement",
        "price_movement_over", "price_movement_under", "model_edge_vs_open",
        "movement_context"
    }

    valid_rows = []
    for row in analysis_results:
        # Check only essential keys that MUST exist for a valid row
        essential_keys = {
            "game_id", "game_date", "player_id", "bookmakers", "market",
            "line", "price", "bet_type", "prediction_source"
        }
        if essential_keys.issubset(row.keys()) and row.get("prediction_source") is not None:
            try:
                # Ensure types match DB schema where possible, using get with default None for optional fields
                row_prepared = {
                    "game_id": str(row["game_id"]),
                    "odds_api_game_id": row.get("odds_api_game_id"),
                    "game_date": str(row["game_date"]),
                    "player_id": str(row["player_id"]),
                    "player_name": row.get("player_name"),
                    "team_id": str(row["team_id"]) if row.get("team_id") is not None else None,
                    "team_name": row.get("team_name"),
                    "opponent_team_id": str(row["opponent_team_id"]) if row.get("opponent_team_id") is not None else None,
                    "opponent_name": row.get("opponent_name"),
                    "bookmakers": str(row["bookmakers"]),
                    "market": str(row["market"]),
                    "line": float(row["line"]),
                    "price": int(row["price"]),
                    "bet_type": str(row["bet_type"]),
                    "predicted_points": float(row["predicted_points"]) if row.get("predicted_points") is not None else None,
                    "prediction_source": str(row["prediction_source"]), # Stores 'playoff_meta_lgbm_optuna' etc.
                    "predicted_points_lgbm": float(row["predicted_points_lgbm"]) if row.get("predicted_points_lgbm") is not None else None,
                    "predicted_points_xgb": float(row["predicted_points_xgb"]) if row.get("predicted_points_xgb") is not None else None,
                    "model_win_pct": float(row["model_win_pct"]) if row.get("model_win_pct") is not None else None,
                    "rmse": float(row["rmse"]) if row.get("rmse") is not None else None,
                    "z_score": float(row["z_score"]) if row.get("z_score") is not None else None,
                    "monte_carlo_win_pct": float(row["monte_carlo_win_pct"]) if row.get("monte_carlo_win_pct") is not None else None,
                    "implied_book_win_pct": float(row["implied_book_win_pct"]) if row.get("implied_book_win_pct") is not None else None,
                    "kelly_fraction": float(row["kelly_fraction"]) if row.get("kelly_fraction") is not None else None,
                    "edge": float(row["edge"]) if row.get("edge") is not None else None,
                    "alert_text": row.get("alert_text"),
                    "open_line": float(row["open_line"]) if row.get("open_line") is not None else None,
                    "open_price_over": int(row["open_price_over"]) if row.get("open_price_over") is not None else None,
                    "open_price_under": int(row["open_price_under"]) if row.get("open_price_under") is not None else None,
                    "line_movement": float(row["line_movement"]) if row.get("line_movement") is not None else None,
                    "price_movement_over": int(row["price_movement_over"]) if row.get("price_movement_over") is not None else None,
                    "price_movement_under": int(row["price_movement_under"]) if row.get("price_movement_under") is not None else None,
                    "model_edge_vs_open": float(row["model_edge_vs_open"]) if row.get("model_edge_vs_open") is not None else None,
                    "movement_context": row.get("movement_context")
                }

                # <<< CHANGE 11: Update check for meta prediction source before nulling base predictions >>>
                # Set base predictions to None if source isn't meta
                if row_prepared['prediction_source'] != 'playoff_meta_lgbm_optuna':
                    row_prepared['predicted_points_lgbm'] = None
                    row_prepared['predicted_points_xgb'] = None

                valid_rows.append(row_prepared)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping row due to type conversion error: {e}. Row keys: {list(row.keys())}")
        else:
            missing = essential_keys - row.keys()
            if row.get("prediction_source") is None: missing.add("prediction_source (None)")
            logger.warning(f"Skipping row due to missing essential keys/values: {missing}. Row keys: {list(row.keys())}")

    if not valid_rows:
        logger.warning("No valid rows remaining after key/type validation for DB insertion.")
        return

    # Use all required keys to define columns, ensuring all are present (even if None)
    cols = sorted(list(required_keys))
    placeholders = ", ".join(f":{col}" for col in cols)
    conflict_key_cols = ["game_id", "player_id", "bookmakers", "market", "line", "bet_type", "game_date", "prediction_source"]
    update_setters = ", ".join(f"{col}=excluded.{col}" for col in cols if col not in conflict_key_cols)
    update_setters += ", created_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'now', 'utc')"

    sql = f"""
        INSERT INTO {DB_ANALYSIS_TABLE} ({", ".join(cols)})
        VALUES ({placeholders})
        ON CONFLICT(game_id, player_id, bookmakers, market, line, bet_type, game_date, prediction_source)
        DO UPDATE SET {update_setters}
        """

    try:
        with conn:
            # Use list of dicts directly with executemany
            conn.executemany(sql, valid_rows)
        logger.info(f"Logged/Updated {len(valid_rows)} analysis results to database.")
    except sqlite3.Error as e:
        logger.error(f"Database error inserting/updating analysis results: {e}")
        logger.debug(f"SQL attempted: {sql}")
        if valid_rows: logger.debug(f"First row data keys: {list(valid_rows[0].keys())}")
    except Exception as e:
        logger.error(f"Unexpected error during DB insertion/update: {e}")


# ─────────────────── Betting Analysis Logic (MODIFIED slightly for context) ─────────────────
# (analyse_side unchanged)
def analyse_side(
    *, side: str, line: float, price: int,
    prediction: float, # This will be the STANDARD prediction or the META prediction
    rmse: float, # This will be the sigma from the standard or meta result
    mc_draws: Optional[np.ndarray],
    prediction_source: str # Keep for context ('standard' or 'playoff_meta_lgbm_optuna')
) -> Optional[dict]:
    """Analyzes a single side (over/under) of a prop bet using the provided prediction and RMSE."""
    # <<< CHANGE 12: Log prefix will automatically update based on prediction_source >>>
    log_prefix = f"[{prediction_source.upper()}]" # e.g., [STANDARD] or [PLAYOFF_META_LGBM_OPTUNA]
    if rmse is None or rmse <= 0 or not np.isfinite(rmse):
        logger.warning(f"{log_prefix} Invalid RMSE ({rmse}) for analysis: Pred={prediction:.2f}, Line={line}, Side={side}")
        return None
    if price is None:
        logger.warning(f"{log_prefix} Missing price for analysis: Pred={prediction:.2f}, Line={line}, Side={side}")
        return None
    if prediction is None or not np.isfinite(prediction):
        logger.warning(f"{log_prefix} Missing/invalid prediction ({prediction}) for analysis: Line={line}, Side={side}")
        return None

    assert side in {"over", "under"}

    try:
        # Z-score calculation based on the final prediction vs the line, using the provided RMSE
        z_std = (line - prediction) / rmse
        model_win = max(0.0, min(1.0, 1.0 - norm.cdf(z_std) if side == "over" else norm.cdf(z_std)))

        mc_win: Optional[float] = None
        if mc_draws is not None and len(mc_draws) > 0:
            mc_win = float(np.mean(mc_draws > line) if side == "over" else np.mean(mc_draws < line))
            mc_win = max(0.0, min(mc_win, 1.0))

        dec_odds = american_to_decimal(price)
        book_implied_win = implied_prob(price)
        edge = model_win - book_implied_win
        kelly = kelly_fraction(model_win, dec_odds)

        return {
            "bet_type": side, "line": float(line), "price": int(price),
            "z_score": float(z_std), "model_win_pct": float(model_win),
            "implied_book_win_pct": float(book_implied_win),
            "monte_carlo_win_pct": mc_win,
            "kelly_fraction": float(kelly),
            "edge": float(edge),
        }
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
        logger.error(f"{log_prefix} Error analyzing {side} {line} @{price} (Pred:{prediction:.2f} RMSE:{rmse:.2f}): {e}")
        return None


# ─────────────────── NEW Odds Movement Helpers ──────────────────

def call_odds_movement_script(script_path: Path, player_id: int, verbose: bool) -> Tuple[Optional[Dict[str, Any]], bool]: # <<< MODIFIED RETURN TYPE
    """
    Calls the player_odds_movement.py script and returns the parsed JSON output
    along with a boolean indicating if a timeout occurred.
    Returns: (JSON_data | None, timeout_occurred_bool)
    """
    timeout_occurred = False # <<< NEW: Initialize timeout flag
    if not script_path.exists():
        logger.error(f"Odds movement script not found at: {script_path}")
        return None, timeout_occurred # <<< MODIFIED RETURN

    command = [
        sys.executable, # Use the current Python interpreter
        str(script_path),
        "--player-id",
        str(player_id)
    ]
    if verbose:
        command.extend(["--log-level", "DEBUG"])
    else:
        command.extend(["--log-level", "WARNING"]) # Keep output clean unless verbose

    logger.debug(f"Calling odds movement script: {' '.join(command)}")
    try:
        # Timeout added for safety
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=45)

        if result.returncode != 0:
            logger.warning(f"Odds movement script for player {player_id} failed (code {result.returncode}): {result.stderr[:500]}...")
            return None, timeout_occurred # <<< MODIFIED RETURN

        try:
            open_odds_data = json.loads(result.stdout)
            if isinstance(open_odds_data, dict) and open_odds_data.get("status"): # Check basic structure
                logger.debug(f"Successfully received open odds data for player {player_id}. Status: {open_odds_data['status']}")
                # Only return if status indicates success or at least partial data
                if "Success" in open_odds_data['status'] or "partial" in open_odds_data.get('status', '').lower():
                    return open_odds_data, timeout_occurred # <<< MODIFIED RETURN (Success case)
                else:
                    logger.warning(f"Odds movement script for player {player_id} returned non-success status: {open_odds_data['status']}")
                    return None, timeout_occurred # <<< MODIFIED RETURN (Treat non-success status as failure)
            else:
                logger.warning(f"Failed to parse valid JSON or status from odds movement script output for player {player_id}. Output: {result.stdout[:200]}...")
                return None, timeout_occurred # <<< MODIFIED RETURN
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON from odds movement script output for player {player_id}. Output: {result.stdout[:200]}...")
            return None, timeout_occurred # <<< MODIFIED RETURN

    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout calling odds movement script for player {player_id}.")
        timeout_occurred = True # <<< NEW: Set timeout flag
        return None, timeout_occurred # <<< MODIFIED RETURN (Timeout case)
    except Exception as e:
        logger.error(f"Error calling odds movement script for player {player_id}: {e}", exc_info=True)
        return None, timeout_occurred # <<< MODIFIED RETURN (Other exception case)


def calculate_movement(
    current_line: float, current_over_price: int, current_under_price: int,
    open_odds_data: Optional[Dict[str, Any]]
) -> Dict[str, Optional[Union[float, int]]]:
    """Calculates line and price movement based on current and open odds."""
    movement = {
        "open_line": None,
        "open_price_over": None,
        "open_price_under": None,
        "line_movement": None,
        "price_movement_over": None,
        "price_movement_under": None,
    }
    if not open_odds_data:
        logger.debug("No open odds data provided, cannot calculate movement.")
        return movement

    # Use pts_over as the canonical open line, assume pts_under is the same
    open_line = open_odds_data.get("pts_over")
    open_price_over = open_odds_data.get("odds_over")
    open_price_under = open_odds_data.get("odds_under")

    movement["open_line"] = open_line
    movement["open_price_over"] = open_price_over
    movement["open_price_under"] = open_price_under

    if open_line is not None:
        try:
            movement["line_movement"] = float(current_line) - float(open_line)
        except (TypeError, ValueError): pass # Keep as None if conversion fails

    if open_price_over is not None:
        try:
            movement["price_movement_over"] = int(current_over_price) - int(open_price_over)
        except (TypeError, ValueError): pass

    if open_price_under is not None:
        try:
            movement["price_movement_under"] = int(current_under_price) - int(open_price_under)
        except (TypeError, ValueError): pass

    logger.debug(f"Calculated Movement: Line={movement['line_movement']}, PriceO={movement['price_movement_over']}, PriceU={movement['price_movement_under']}")
    return movement


def interpret_movement_context(
    model_edge_vs_open_over: Optional[float],
    model_edge_vs_open_under: Optional[float],
    line_movement: Optional[float]
) -> str:
    """Interprets the line movement relative to the initial value suggested by the model."""
    if line_movement is None:
        return "Movement Unknown"

    initial_value_side = None
    if model_edge_vs_open_over is not None and model_edge_vs_open_over > KELLY_EDGE_THRESHOLD:
        initial_value_side = "Over"
    elif model_edge_vs_open_under is not None and model_edge_vs_open_under > KELLY_EDGE_THRESHOLD:
        initial_value_side = "Under"

    if initial_value_side is None:
        return "No Clear Initial Edge"

    if abs(line_movement) < MOVEMENT_LINE_STABLE_THRESHOLD:
        return "Stable Line"

    if initial_value_side == "Over":
        if line_movement > 0:
            return "Supports Model (Over)" # Line moved in direction of value
        else: # line_movement < 0
            return "Contradicts Model (Over)" # Line moved against direction of value
    elif initial_value_side == "Under":
        if line_movement < 0:
            return "Supports Model (Under)" # Line moved in direction of value
        else: # line_movement > 0
            return "Contradicts Model (Under)" # Line moved against direction of value
    else:
        # Should not be reached if initial_value_side is determined
        return "Context Unknown"


# ─────────────────── Console Output (MODIFIED) ──────────────────

def _print_console(
    player: str, team: str, opp: str, game_date: str,
    analysis_result: Dict[str, Any], # Contains processed analysis info INCLUDING movement
    rmse_used: float, rmse_reason: str
) -> None:
    """Prints the analysis results neatly to the console, including movement."""

    final_pred = analysis_result.get('final_pred')
    base_lgbm_pred = analysis_result.get('base_lgbm_pred')
    base_xgb_pred = analysis_result.get('base_xgb_pred')
    # <<< CHANGE 13: Convert prediction_source to upper for display >>>
    prediction_source = analysis_result.get('prediction_source', 'unknown').upper()

    # Analysis vs CURRENT odds
    over_analysis = analysis_result.get('over')
    under_analysis = analysis_result.get('under')
    current_line = over_analysis.get('line') if over_analysis else (under_analysis.get('line') if under_analysis else None)

    # Analysis vs OPEN odds
    open_over_analysis = analysis_result.get('open_over_analysis')
    open_under_analysis = analysis_result.get('open_under_analysis')

    # Movement data
    open_line = analysis_result.get('open_line')
    open_price_over = analysis_result.get('open_price_over')
    open_price_under = analysis_result.get('open_price_under')
    line_movement = analysis_result.get('line_movement')
    price_movement_over = analysis_result.get('price_movement_over')
    price_movement_under = analysis_result.get('price_movement_under')
    movement_context = analysis_result.get('movement_context', 'N/A')

    # Determine overall sharpness based on the analysis performed vs CURRENT odds
    sharp = False
    alert_text = ""
    if over_analysis and over_analysis.get('kelly_fraction', 0) >= KELLY_EDGE_THRESHOLD and over_analysis.get('edge', -1) >= KELLY_EDGE_THRESHOLD:
        sharp = True
        alert_text = f"SHARP OVER vs Current (Context: {movement_context})"
    if under_analysis and under_analysis.get('kelly_fraction', 0) >= KELLY_EDGE_THRESHOLD and under_analysis.get('edge', -1) >= KELLY_EDGE_THRESHOLD:
        sharp = True
        # Append if both are sharp (unlikely but possible)
        alert_text += f"{'; ' if alert_text else ''}SHARP UNDER vs Current (Context: {movement_context})"

    banner = f"\n🔥🔥 {alert_text} 🔥🔥\n" if sharp else "\n"
    game_date_str = game_date.split('T')[0] if isinstance(game_date, str) and 'T' in game_date else str(game_date)
    header = f"{banner}===== {player} ({team} vs {opp} — {game_date_str}) =====\n"
    rmse_line_str = f"RMSE Used: {rmse_used:.3f} ({rmse_reason})\n"
    separator = "-"*70 + "\n"
    output = header + rmse_line_str + separator

    # --- Print the prediction results ---
    pred_final_str = f"{final_pred:.2f}" if final_pred is not None else "N/A"
    pred_base_lgbm_str = f"{base_lgbm_pred:.2f}" if base_lgbm_pred is not None else "N/A"
    pred_base_xgb_str = f"{base_xgb_pred:.2f}" if base_xgb_pred is not None else "N/A"

    # <<< CHANGE 14: Update check for the specific meta prediction source >>>
    if prediction_source == "PLAYOFF_META_LGBM_OPTUNA":
        output += f"Meta Playoff Model (LGBM) = {pred_final_str} <- Used for Analysis\n" # Updated label
        output += f"  (Base LGBM Pred = {pred_base_lgbm_str})\n"
        output += f"  (Base XGB Pred  = {pred_base_xgb_str})\n"
    elif prediction_source == "STANDARD":
        output += f"Standard Model         = {pred_final_str} <- Used for Analysis\n"
    else: # Fallback/Error case
        output += f"Prediction             = {pred_final_str} ({prediction_source})\n"
    output += separator

    # --- Print OPEN Odds Analysis ---
    output += f"--- Analysis vs OPEN Odds (Line: {open_line if open_line is not None else 'N/A'}) ---\n"
    if open_over_analysis:
        mc_pct = open_over_analysis.get('monte_carlo_win_pct')
        mc_pct_str = f" MC {mc_pct:.1%}" if mc_pct is not None else ""
        output += (f"  OVER  {open_over_analysis.get('line', 'N/A'):<4.1f} @ {open_over_analysis.get('price', 'N/A'):>5}  "
                   f"Win {open_over_analysis.get('model_win_pct', 0):.1%} Edge {open_over_analysis.get('edge', 0):+.1%}{mc_pct_str}\n")
    else: output += f"  OVER  (Open data unavailable)\n"

    if open_under_analysis:
        mc_pct = open_under_analysis.get('monte_carlo_win_pct')
        mc_pct_str = f" MC {mc_pct:.1%}" if mc_pct is not None else ""
        output += (f"  UNDER {open_under_analysis.get('line', 'N/A'):<4.1f} @ {open_under_analysis.get('price', 'N/A'):>5}  "
                   f"Win {open_under_analysis.get('model_win_pct', 0):.1%} Edge {open_under_analysis.get('edge', 0):+.1%}{mc_pct_str}\n")
    else: output += f"  UNDER (Open data unavailable)\n"
    output += separator

    # --- Print Movement Analysis ---
    output += f"--- Odds Movement (Current vs Open) ---\n"
    lm_str = f"{line_movement:+.1f}" if line_movement is not None else "N/A"
    po_str = f"{price_movement_over:+d}" if price_movement_over is not None else "N/A"
    pu_str = f"{price_movement_under:+d}" if price_movement_under is not None else "N/A"
    output += f"Line Movement:  {lm_str} pts\n"
    output += f"Price Movement: O {po_str} / U {pu_str} cents\n"
    output += f"Movement Context: {movement_context}\n" # Display calculated context
    output += separator

    # --- Print CURRENT Odds Analysis ---
    output += f"--- Analysis vs CURRENT Odds (Line: {current_line if current_line is not None else 'N/A'}) ---\n"
    if over_analysis:
        mc_pct = over_analysis.get('monte_carlo_win_pct')
        mc_pct_str = f" MC {mc_pct:.1%}" if mc_pct is not None else ""
        output += (f"  OVER  {over_analysis.get('line', 'N/A'):<4.1f} @ {over_analysis.get('price', 'N/A'):>5}  "
                   f"Win {over_analysis.get('model_win_pct', 0):.1%} Edge {over_analysis.get('edge', 0):+.1%} Kelly {over_analysis.get('kelly_fraction', 0):.1%}{mc_pct_str}\n")
    else: output += f"  OVER  (Current data unavailable)\n"

    if under_analysis:
        mc_pct = under_analysis.get('monte_carlo_win_pct')
        mc_pct_str = f" MC {mc_pct:.1%}" if mc_pct is not None else ""
        output += (f"  UNDER {under_analysis.get('line', 'N/A'):<4.1f} @ {under_analysis.get('price', 'N/A'):>5}  "
                   f"Win {under_analysis.get('model_win_pct', 0):.1%} Edge {under_analysis.get('edge', 0):+.1%} Kelly {under_analysis.get('kelly_fraction', 0):.1%}{mc_pct_str}\n")
    else: output += f"  UNDER (Current data unavailable)\n"
    output += "="*70 + "\n" # End marker

    print(output)


# ─────────────────── Refactored Main Logic Functions (MODIFIED) ────────────────────
# (get_schedule_data unchanged)
def get_schedule_data(conn: sqlite3.Connection, args: argparse.Namespace) -> Dict[str, dict]:
    """
    Fetches schedule data from the DB for the specified game(s).
    Includes team names and is_playoffs status. Keys for teams dict will be team_id (likely numeric).
    """
    # (Unchanged from previous ensemble version, noted team_id key type)
    schedule: Dict[str, dict] = {}
    where_clause, params = (("gs.game_id = ?", (args.game_id,)), ("DATE(gs.game_date) = DATE(?)", (args.game_date,)))[args.game_id is None]
    sql = f"""
        SELECT gs.game_id, gs.game_date, gs.team_id, gs.opponent_team_id,
               t1.team_city || ' ' || t1.team_name AS team_full,
               t2.team_city || ' ' || t2.team_name AS opp_full,
               COALESCE(gs.is_playoffs, 0) as is_playoffs
        FROM {DB_GAMESCHEDULES_TABLE} gs
        JOIN {DB_TEAMS_TABLE} t1 ON t1.team_id = gs.team_id
        JOIN {DB_TEAMS_TABLE} t2 ON t2.team_id = gs.opponent_team_id
        WHERE {where_clause} ORDER BY gs.game_date, gs.game_id
        """
    try:
        with conn: cursor = conn.execute(sql, params)
        cols = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        if not rows:
            logger.error(f"No schedule rows found for criteria: {where_clause}, {params}")
            return {}

        for row_values in rows:
            row_dict = dict(zip(cols, row_values))
            gid = row_dict['game_id']
            tid = row_dict['team_id'] # Assume this is the numeric ID from DB
            oid = row_dict['opponent_team_id'] # Assume numeric ID
            gdate_str = row_dict['game_date'] if isinstance(row_dict['game_date'], str) else pd.to_datetime(row_dict['game_date']).isoformat()
            game_is_playoff = bool(row_dict.get('is_playoffs', 0))

            if gid not in schedule:
                schedule[gid] = {
                    "game_id": gid, "game_date": gdate_str, "is_playoffs": game_is_playoff,
                    "teams": {}, "api_event_info": None, "players": {}
                }
            if schedule[gid]["is_playoffs"] != game_is_playoff:
                logger.warning(f"Conflicting playoff status for game_id {gid}. Using first encountered value ({schedule[gid]['is_playoffs']}).")
            # Use numeric team IDs as keys
            if tid not in schedule[gid]["teams"]:
                schedule[gid]["teams"][tid] = { "team_id": tid, "team_name": normalise_team(row_dict['team_full']), "opp_id": oid, "opp_name": normalise_team(row_dict['opp_full']) }
            if oid not in schedule[gid]["teams"]:
                 schedule[gid]["teams"][oid] = { "team_id": oid, "team_name": normalise_team(row_dict['opp_full']), "opp_id": tid, "opp_name": normalise_team(row_dict['team_full']) }

        logger.info(f"Fetched schedule data for {len(schedule)} game(s).")
        return schedule
    except sqlite3.Error as e:
        logger.error(f"DB error fetching schedule: {e}")
        return {}

# (fetch_market_data_and_map_events unchanged)
def fetch_market_data_and_map_events(
    schedule: Dict[str, dict],
    api_key_live: Optional[str], # Only live key needed now
    bookmakers: Optional[str]
) -> Tuple[Dict[str, list], Dict[str, str]]: # event_id_map returns str (API Event ID)
    """Fetches LIVE market data from Odds API and maps DB games to API events."""
    # (Removed historical API fetching logic)
    if not schedule: return {}, {}
    market_data: Dict[str, list] = {}
    event_id_map: Dict[str, str] = {} # Maps DB game_id -> API event_id
    processed_api_event_ids: Set[str] = set()

    if api_key_live:
        logger.info("Fetching live game IDs from Odds API...")
        # Fetch only live/upcoming games
        live_games = fetch_odds_api_games(api_key=api_key_live, bookmakers=bookmakers)
        logger.info(f"Found {len(live_games)} live events from API.")

        for db_gid, game_info in schedule.items():
            # Use normalized team names from the schedule for matching
            team_names_norm = {td["team_name"] for td in game_info.get("teams", {}).values()}
            if not team_names_norm or len(team_names_norm) < 2:
                logger.warning(f"Skipping live game matching for DB game {db_gid}: Missing team names.")
                continue

            matched_api_event = None
            for api_evt in live_games:
                api_home = normalise_team(api_evt.get("home_team", ""))
                api_away = normalise_team(api_evt.get("away_team", ""))
                api_teams_norm = {api_home, api_away}
                # Also check date loosely? API might return games slightly outside requested date range?
                # api_date = pd.to_datetime(api_evt.get("commence_time", "")).tz_localize(None).date()
                # db_date = pd.to_datetime(game_info.get("game_date", "")).tz_localize(None).date()
                # if team_names_norm == api_teams_norm and api_date == db_date:
                if team_names_norm == api_teams_norm: # Rely on team name match for now
                    event_id = api_evt.get("id")
                    if event_id:
                        logger.debug(f"Matched DB game {db_gid} to LIVE API event {event_id}")
                        event_id_map[db_gid] = event_id
                        schedule[db_gid]["api_event_info"] = event_id # Store just the ID
                        matched_api_event = True
                        break # Move to next DB game once matched
            if not matched_api_event:
                 logger.warning(f"Could not find matching LIVE Odds API event for DB game_id: {db_gid} ({schedule[db_gid]['game_date']})")
    else:
        logger.warning("No live API key provided. Cannot fetch market data.")
        return {}, {}

    # --- Fetch Props (Only for matched live events) ---
    logger.info("Fetching player props from Odds API for matched live events...")
    for db_gid, event_id in event_id_map.items():
        if event_id in processed_api_event_ids: continue

        # Always use the live key, no need to check is_historical
        if not api_key_live:
            logger.warning(f"Skipping props fetch for event {event_id}: Missing live API key.")
            continue

        props = fetch_odds_api_props(
            api_key=api_key_live,
            event_id=event_id,
            bookmakers=bookmakers,
            market=ODDS_API_MARKETS
        )
        market_data[event_id] = props or [] # Store results (or empty list)
        if props: logger.debug(f"Fetched {len(props)} prop market group(s) for API event {event_id}")
        else: logger.debug(f"No prop data found for market '{ODDS_API_MARKETS}' for API event {event_id}")
        processed_api_event_ids.add(event_id)

    logger.info(f"Fetched market data for {len(processed_api_event_ids)} unique API events.")
    return market_data, event_id_map

# (extract_and_match_players_from_api unchanged)
def extract_and_match_players_from_api(
    market_data: Dict[str, list],
    event_id_map: Dict[str, str], # Map DB GID -> API Event ID
    schedule: Dict[str, dict],
    conn: sqlite3.Connection
) -> Dict[str, Dict[Any, Dict[str, Any]]]: # Player ID key type might be int or str from DB
    """Extracts player names from API market data and matches to DB players."""
    # (Unchanged from previous ensemble version, noted player ID type)
    players_found_in_api: DefaultDict[str, Set[str]] = defaultdict(set)
    db_gid_map: Dict[str, str] = {v: k for k, v in event_id_map.items()} # Invert map: API Event ID -> DB GID

    logger.info("Extracting player names from API market data...")
    for event_id, props_list in market_data.items():
        db_gid = db_gid_map.get(event_id)
        if not db_gid:
            logger.warning(f"Could not map API event_id {event_id} back to DB game_id. Skipping player extraction for this event.")
            continue
        for prop_event_bookmaker_group in props_list:
            if not isinstance(prop_event_bookmaker_group, dict): continue
            for bookmaker in prop_event_bookmaker_group.get("bookmakers", []):
                if not isinstance(bookmaker, dict): continue
                for market in bookmaker.get("markets", []):
                    if not isinstance(market, dict): continue
                    if market.get("key") != ODDS_API_MARKETS: continue
                    for outcome in market.get("outcomes", []):
                        if not isinstance(outcome, dict): continue
                        api_pname_raw = outcome.get("description")
                        if api_pname_raw and isinstance(api_pname_raw, str):
                            api_pname_norm = remove_diacritics(api_pname_raw).lower().strip()
                            if api_pname_norm: players_found_in_api[db_gid].add(api_pname_norm)

    logger.info(f"Found player names in API props for {len(players_found_in_api)} games.")
    if not players_found_in_api: return {}

    # Player ID type depends on DB schema (TEXT or INTEGER)
    matched_players_by_game: Dict[str, Dict[Any, Dict[str, Any]]] = defaultdict(dict)
    logger.info("Matching API player names to DB players...")
    all_team_ids: Set[Any] = set() # Team IDs are keys in schedule's 'teams' dict (likely numeric)
    for db_gid in players_found_in_api.keys():
        if db_gid in schedule:
             # Team IDs are keys in schedule[db_gid]['teams'] (likely numeric)
             all_team_ids.update(schedule[db_gid].get("teams", {}).keys())
    if not all_team_ids:
        logger.warning("No team IDs found in schedule for games with API players. Cannot match players.")
        return {}

    # Ensure placeholders match the type of all_team_ids (likely numeric)
    placeholders = ','.join('?' * len(all_team_ids))
    # player_id type depends on DB, team_id type depends on DB (assume numeric for team_id to match gameschedules)
    sql = f"SELECT player_id, name, team_id FROM {DB_PLAYERS_TABLE} WHERE team_id IN ({placeholders}) AND roster_status = 1"
    potential_db_players: List[Tuple[Any, str, Any]] = [] # player_id and team_id types depend on DB
    try:
        with conn:
            # Convert team IDs to list for parameter binding
            cursor = conn.execute(sql, list(all_team_ids))
            potential_db_players = cursor.fetchall()
        logger.debug(f"Fetched {len(potential_db_players)} potential active players from DB for matching.")
    except sqlite3.Error as e:
        logger.error(f"Database error fetching potential players for matching: {e}")
        return {}

    # Store as {normalized_name: [player_info_dict, ...]}
    db_player_lookup: DefaultDict[str, List[Dict]] = defaultdict(list)
    for pid, name, tid in potential_db_players:
        db_name_norm = remove_diacritics(name).lower().strip()
        if db_name_norm:
            db_player_lookup[db_name_norm].append(
                {'player_id': pid, 'player_name': remove_diacritics(name), 'team_id': tid} # Store original types
            )

    match_count = 0
    unmatched_names_by_game: DefaultDict[str, Set[str]] = defaultdict(set)
    for db_gid, api_names_set in players_found_in_api.items():
        if db_gid not in schedule: continue
        game_team_ids = set(schedule[db_gid].get("teams", {}).keys()) # Keys are likely numeric team IDs
        if not game_team_ids:
            logger.warning(f"No teams associated with game {db_gid} in schedule. Cannot match players.")
            continue
        for api_name_norm in api_names_set:
            matched = False
            if api_name_norm in db_player_lookup:
                # Filter matches by team_id for the current game (compare potentially numeric types)
                possible_matches = [p for p in db_player_lookup[api_name_norm] if p['team_id'] in game_team_ids]
                if len(possible_matches) == 1:
                    player_info = possible_matches[0]
                    # Use original player_id (type depends on DB) as key
                    pid_key = player_info['player_id']
                    matched_players_by_game[db_gid][pid_key] = player_info
                    match_count += 1
                    matched = True
                elif len(possible_matches) > 1:
                    logger.warning(f"Ambiguous match for '{api_name_norm}' in game {db_gid}. Multiple players on roster match: {[p['player_id'] for p in possible_matches]}. Skipping.")
            if not matched: unmatched_names_by_game[db_gid].add(api_name_norm)

    logger.info(f"Matched {match_count} API players to unique DB players across all games.")
    if unmatched_names_by_game:
        for db_gid, names in unmatched_names_by_game.items():
            logger.warning(f"Could not match {len(names)} API player names for game {db_gid}: {list(names)}") # Show list

    return dict(matched_players_by_game)

# (enrich_schedule_with_predictions MODIFIED for prediction source)
def enrich_schedule_with_predictions(
    schedule: Dict[str, dict],
    players_to_predict: Dict[str, Dict[Any, Dict[str, Any]]], # Key is player_id (type depends on DB)
    predict_standard_func: Optional[callable],
    predict_meta_playoff_func: Optional[callable], # Meta predictor function
    cached_predictions: Optional[Dict[str, Dict[str, Any]]], # Cache format: {key: {player_id_str: {"standard": {}, "playoff_meta": {}}}}
    db_path: Path, # Pass DB path to meta predictor
    meta_artifact_dir: Path, # Pass artifact paths to meta predictor
    lgbm_playoff_artifact_dir: Path,
    xgb_playoff_artifact_dir: Path
) -> Dict[str, Dict[str, Any]]:
    """
    Adds model predictions ONLY for the players specified in players_to_predict.
    Uses cache if available. Calls standard model OR the meta playoff model.
    Updates the 'players' dict within the schedule with prediction results.

    Returns newly generated predictions for caching.
    """
    if not schedule or not players_to_predict: return {}

    stats = defaultdict(int)
    newly_generated_predictions: Dict[str, Dict[str, Any]] = {} # {pid_str: {"standard": ...}} OR {pid_str: {"playoff_meta": ...}}
    cache_for_this_key = cached_predictions if cached_predictions else {}

    for game_id, game_data in schedule.items():
        if game_id not in players_to_predict:
            logger.debug(f"No players from API found for game {game_id}. Skipping predictions.")
            continue

        game_is_playoffs = game_data.get("is_playoffs", False)
        logger.debug(f"Processing predictions for {len(players_to_predict[game_id])} players in game: {game_id} (Is Playoff Game: {game_is_playoffs})")
        final_player_data_for_game: Dict[Any, Dict[str, Any]] = {} # Key is player_id

        for pid_key, player_info in players_to_predict[game_id].items():
            pid_str = str(pid_key) # Use string representation for cache key consistency
            # Convert pid_key to int if possible for prediction functions (assuming they expect int)
            try:
                pid_int = int(pid_key)
            except (ValueError, TypeError):
                logger.error(f"Player ID '{pid_key}' cannot be converted to int. Skipping prediction.")
                stats['pid_conversion_failed'] += 1
                continue

            pname = player_info.get("player_name", f"Unknown ID {pid_str}")
            player_cache_entry = cache_for_this_key.get(pid_str)

            # --- Base player data for final storage ---
            current_player_data = player_info.copy()
            current_player_data["prediction_result"] = None # Store the raw result dict here
            current_player_data["prediction_source"] = None # 'standard' or 'playoff_meta_lgbm_optuna'
            current_player_data["final_prediction"] = None # The value used for analysis
            current_player_data["model_sigma"] = None # Sigma from prediction result
            current_player_data["sigma_method"] = None
            current_player_data["sigma_n"] = None
            current_player_data["base_lgbm_pred"] = None # Base preds if meta
            current_player_data["base_xgb_pred"] = None


            if game_is_playoffs:
                # --- Playoff Prediction (Meta Model - LGBM Optuna) ---
                pred_meta = None
                cache_hit_meta = False
                generated_meta = False
                # <<< CHANGE 15: Set prediction source string correctly >>>
                pred_source = "playoff_meta_lgbm_optuna"

                if predict_meta_playoff_func:
                    # Check Cache
                    if player_cache_entry and 'playoff_meta' in player_cache_entry:
                        pred_meta = player_cache_entry['playoff_meta']
                        # <<< CHANGE 16: Basic validation of cached meta result type >>>
                        if isinstance(pred_meta, dict) and pred_meta.get("prediction_type") == "playoffs_meta_lgbm_optuna":
                            cache_hit_meta = True
                            stats['meta_cached'] += 1
                            logger.debug(f"Using cached META Playoff (LGBM) prediction result for {pname}")
                        else:
                            logger.warning(f"Invalid playoff_meta (LGBM) cache structure/type for {pname}. Regenerating.")
                            pred_meta = None # Force regeneration
                    # Generate if no valid cache hit
                    if not pred_meta:
                        logger.debug(f"Generating META PLAYOFF (LGBM) prediction for {pname}")
                        try:
                            # Call the meta prediction function, passing necessary paths
                            pred_meta = predict_meta_playoff_func(
                                player_id=pid_int, # Pass integer player ID
                                target=DEFAULT_PLAYOFF_TARGET_STAT,
                                db_path=db_path,
                                meta_artifact_dir=meta_artifact_dir,
                                lgbm_artifact_dir=lgbm_playoff_artifact_dir,
                                xgb_artifact_dir=xgb_playoff_artifact_dir
                                # Pass verbosity if needed: base_verbose=logger.level <= logging.DEBUG
                            )
                            generated_meta = True
                        except Exception as ex:
                            logger.error(f"META PLAYOFF (LGBM) prediction failed for player {pname} (ID: {pid_int}): {ex}", exc_info=False)
                            stats['meta_failed'] += 1
                else:
                    stats['meta_func_missing'] += 1

                # Store result if valid prediction was obtained
                if isinstance(pred_meta, dict) and pred_meta.get("predicted_value") is not None:
                    current_player_data["prediction_result"] = pred_meta # Store full result
                    current_player_data["final_prediction"] = float(pred_meta["predicted_value"])
                    current_player_data["prediction_source"] = pred_source
                    current_player_data["model_sigma"] = float(pred_meta["model_sigma"]) if pred_meta.get("model_sigma") is not None else None
                    current_player_data["sigma_method"] = pred_meta.get("sigma_method", "unknown")
                    current_player_data["sigma_n"] = int(pred_meta.get("sigma_n", 0))
                    # Extract base predictions if available
                    base_preds = pred_meta.get("base_predictions", {})
                    current_player_data["base_lgbm_pred"] = float(base_preds.get("lgbm")) if base_preds.get("lgbm") is not None else None
                    current_player_data["base_xgb_pred"] = float(base_preds.get("xgb")) if base_preds.get("xgb") is not None else None

                    stats['meta_processed'] += 1
                    if generated_meta:
                        newly_generated_predictions.setdefault(pid_str, {})["playoff_meta"] = pred_meta

            else: # Standard game
                # --- Standard Prediction ONLY ---
                pred_std = None
                cache_hit_std = False
                generated_std = False
                pred_source = "standard"

                if predict_standard_func:
                    # Check Cache
                    if player_cache_entry and 'standard' in player_cache_entry:
                        pred_std = player_cache_entry['standard']
                        if isinstance(pred_std, dict) and pred_std.get("predicted_value") is not None:
                            cache_hit_std = True
                            stats['std_cached'] += 1
                            logger.debug(f"Using cached STANDARD prediction for {pname}")
                        else:
                            logger.warning(f"Invalid standard cache structure for {pname}. Regenerating.")
                            pred_std = None # Force regeneration
                    # Generate if no valid cache hit
                    if not pred_std:
                        logger.debug(f"Generating STANDARD prediction for {pname}")
                        try:
                            pred_std = predict_standard_func(player_id=pid_int) # Pass integer player ID
                            generated_std = True
                        except Exception as ex:
                            logger.error(f"STANDARD prediction failed for player {pname} (ID: {pid_int}): {ex}", exc_info=False)
                            stats['std_failed'] += 1
                else: stats['std_func_missing'] += 1

                # Store result if valid
                if isinstance(pred_std, dict) and pred_std.get("predicted_value") is not None:
                    current_player_data["prediction_result"] = pred_std # Store full result
                    current_player_data["final_prediction"] = float(pred_std["predicted_value"])
                    current_player_data["prediction_source"] = pred_source
                    current_player_data["model_sigma"] = float(pred_std["model_sigma"]) if pred_std.get("model_sigma") is not None else None
                    current_player_data["sigma_method"] = pred_std.get("sigma_method", "unknown")
                    current_player_data["sigma_n"] = int(pred_std.get("sigma_n", 0))
                    # Base predictions are not relevant for standard model
                    current_player_data["base_lgbm_pred"] = None
                    current_player_data["base_xgb_pred"] = None

                    stats['std_processed'] += 1
                    if generated_std:
                        newly_generated_predictions.setdefault(pid_str, {})["standard"] = pred_std


            # Add player data to the game only if a usable prediction source was determined
            if current_player_data["prediction_source"]:
                final_player_data_for_game[pid_key] = current_player_data # Use original pid_key
            else:
                logger.warning(f"No valid prediction determined for {pname} in game {game_id}. Player will be excluded from analysis.")


        schedule[game_id]["players"] = final_player_data_for_game
        logger.debug(f"Stored predictions for {len(final_player_data_for_game)} players in game {game_id}")

    # Summarize results
    total_players_from_api = sum(len(pids) for pids in players_to_predict.values())
    logger.info(
        f"Prediction Summary (for {total_players_from_api} players found in API):\n"
        f"  Standard Games: {stats['std_processed']} processed ({stats['std_cached']} cached), {stats['std_failed']} failed, {stats['std_func_missing']} func missing.\n"
        # <<< CHANGE 17: Update meta log summary label >>>
        f"  Playoff Games (Meta LGBM): {stats['meta_processed']} processed ({stats['meta_cached']} cached), {stats['meta_failed']} failed, {stats['meta_func_missing']} func missing."
    )
    if stats['pid_conversion_failed'] > 0:
        logger.warning(f"  {stats['pid_conversion_failed']} players skipped due to ID conversion errors.")
    return newly_generated_predictions


# --- MODIFIED perform_betting_analysis ---
# --- MODIFIED perform_betting_analysis ---
def perform_betting_analysis(
    schedule: Dict[str, dict],
    market_data: Dict[str, list],
    global_rmse: float, # Ultimate fallback
    sample_size: int,
    odds_movement_script_path: Path, # Path to the odds movement script
    verbose_logging: bool # To pass verbosity to odds movement script
) -> List[Dict]:
    """
    Performs betting analysis using predictions, CURRENT odds, and OPEN odds.
    Includes odds movement calculation and context interpretation.
    Skips calling the odds movement script for remaining players after the first timeout.
    """
    analysis_results = []
    if not schedule or not market_data:
        logger.warning("Skipping analysis: Missing schedule or market data.")
        return analysis_results

    logger.info(f"Starting betting analysis including odds movement. Global Fallback RMSE: {global_rmse:.3f}.")

    # <<< NEW: Flag to track if we should skip the odds movement script call >>>
    skip_odds_movement_script = False

    for db_gid, game_info in schedule.items():
        api_event_id = game_info.get("api_event_info")
        players_with_predictions = game_info.get("players", {})

        if not api_event_id: continue
        if not players_with_predictions: continue

        props_for_event = market_data.get(api_event_id, [])
        if not props_for_event: continue

        game_date_str = game_info.get('game_date', 'Unknown Date')
        game_teams = game_info.get('teams', {})

        for player_id_key, player_data in players_with_predictions.items():
            # ... (existing code for prediction source, final prediction, sigma, etc.) ...
            prediction_source = player_data.get("prediction_source") # Will be 'standard' or 'playoff_meta_lgbm_optuna'
            prediction_for_analysis = player_data.get("final_prediction")
            base_lgbm_pred_val = player_data.get("base_lgbm_pred")
            base_xgb_pred_val = player_data.get("base_xgb_pred")
            sigma_val = player_data.get("model_sigma")
            sigma_method = player_data.get("sigma_method")
            sigma_n = player_data.get("sigma_n")

            if prediction_for_analysis is None or prediction_source is None:
                logger.error(f"Programming Error: Player {player_id_key} in analysis list missing final_prediction/source. Skipping.")
                continue

            try:
                pid_int = int(player_id_key)
            except (ValueError, TypeError):
                logger.warning(f"Cannot convert player ID '{player_id_key}' to int for odds movement script call. Skipping movement analysis.")
                pid_int = None

            pid_str = str(player_id_key)
            pname = player_data.get("player_name", f"Unknown ID {pid_str}")
            team_id = player_data.get("team_id")

            # ... (existing code for Team/Opponent Name Lookup) ...
            team_name, opp_id, opp_name = "Unknown Team", "N/A", "Unknown Opp"
            if team_id is not None and game_teams:
                team_info = game_teams.get(team_id)
                if team_info:
                    team_name = team_info.get("team_name", f"TeamID {team_id}")
                    opp_id = team_info.get("opp_id", "N/A")
                    opp_info = game_teams.get(opp_id) if opp_id != "N/A" else None
                    if opp_info: opp_name = opp_info.get("team_name", "Unknown Opp")
                    else: opp_name = team_info.get("opp_name", "Unknown Opp")
                else: logger.warning(f"Could not find team info for team_id {team_id} in game {db_gid}'s team data.")

            # ... (existing code for RMSE Selection Logic) ...
            rmse_to_use = None
            reason = ""
            if sigma_val is not None and sigma_val > 0 and np.isfinite(sigma_val):
                rmse_to_use = sigma_val
                reason = f"Sigma from {prediction_source.capitalize()} Result (Method='{sigma_method}', N={sigma_n})"
            else:
                logger.warning(f"Sigma from {prediction_source} result invalid ({sigma_val}). Falling back to global RMSE for player {pname}.")
                rmse_to_use = global_rmse
                reason = f"Global Fallback ({prediction_source.capitalize()} Game - Sigma Invalid)"

            if rmse_to_use is None or rmse_to_use <= 0 or not np.isfinite(rmse_to_use):
                logger.error(f"Invalid final RMSE ({rmse_to_use}) for player {pname} ({prediction_source}). Skipping analysis.")
                continue

            logger.debug(f"Analysis for {pname} (ID: {pid_str}): Using Pred={prediction_for_analysis:.2f}, RMSE={rmse_to_use:.3f} ({reason})")

            # ... (existing code for Monte Carlo Draws) ...
            mc_draws = None
            if sample_size > 0 and rmse_to_use > 0:
                try:
                    mc_draws = np.random.normal(prediction_for_analysis, rmse_to_use, sample_size)
                except Exception as mc_err:
                    logger.error(f"Monte Carlo generation failed for {pname} ({prediction_source}): {mc_err}")

            player_found_in_market = False
            analysis_output_for_console = None

            # --- *** MODIFIED: Call Odds Movement Script (Conditional) *** ---
            open_odds_data = None
            timeout_happened = False # Ensure it's reset for each player unless skipped

            if skip_odds_movement_script:
                 logger.debug(f"Skipping odds movement script call for {pname} due to previous timeout.")
            elif pid_int is not None:
                # <<< Call the MODIFIED function, unpack both return values >>>
                open_odds_data, timeout_happened = call_odds_movement_script(
                    script_path=odds_movement_script_path,
                    player_id=pid_int,
                    verbose=verbose_logging
                )
                # <<< If timeout occurred, set the flag to skip future calls >>>
                if timeout_happened:
                    skip_odds_movement_script = True
                    logger.warning(f"Timeout detected calling odds movement script for player {pid_int}. Subsequent calls will be skipped.")
            else:
                logger.debug(f"Skipping open odds fetch for player {pname} due to invalid integer ID.")
            # --- *** END MODIFIED SECTION *** ---


            # --- Iterate through CURRENT market data from API ---
            for prop_event_bookmaker_group in props_for_event:
                # ... (existing inner loops for bookmakers, markets, outcomes) ...
                if not isinstance(prop_event_bookmaker_group, dict): continue
                for bookmaker in prop_event_bookmaker_group.get("bookmakers", []):
                    if not isinstance(bookmaker, dict): continue
                    bkm_title = bookmaker.get("title", bookmaker.get("key", "unknown"))
                    for market in bookmaker.get("markets", []):
                        if not isinstance(market, dict): continue
                        if market.get("key") != ODDS_API_MARKETS: continue

                        current_over_outcome, current_under_outcome = None, None
                        for outcome in market.get("outcomes", []):
                            if not isinstance(outcome, dict): continue
                            api_pname_raw = outcome.get("description")
                            if not api_pname_raw or not isinstance(api_pname_raw, str): continue
                            api_pname_norm = remove_diacritics(api_pname_raw).lower().strip()
                            db_pname_norm = remove_diacritics(pname).lower().strip()
                            if api_pname_norm == db_pname_norm:
                                player_found_in_market = True
                                outcome_name = outcome.get("name")
                                if outcome_name == "Over": current_over_outcome = outcome
                                elif outcome_name == "Under": current_under_outcome = outcome

                        # --- Perform analysis if we have both sides for CURRENT odds ---
                        if current_over_outcome and current_under_outcome:
                            # ... (existing code to extract current line/prices) ...
                            current_line_raw = current_over_outcome.get("point")
                            current_over_price_raw = current_over_outcome.get("price")
                            current_under_price_raw = current_under_outcome.get("price")

                            if current_line_raw is not None and current_over_price_raw is not None and current_under_price_raw is not None:
                                try:
                                    current_line = float(current_line_raw)
                                    current_over_price = int(current_over_price_raw)
                                    current_under_price = int(current_under_price_raw)

                                    # 1. Analyze vs CURRENT odds (Existing Logic)
                                    over_analysis = analyse_side(side="over", line=current_line, price=current_over_price, prediction=prediction_for_analysis, rmse=rmse_to_use, mc_draws=mc_draws, prediction_source=prediction_source)
                                    under_analysis = analyse_side(side="under", line=current_line, price=current_under_price, prediction=prediction_for_analysis, rmse=rmse_to_use, mc_draws=mc_draws, prediction_source=prediction_source)

                                    # --- *** NEW: Movement and Open Odds Analysis (uses open_odds_data which might be None) *** ---
                                    movement_results = calculate_movement(current_line, current_over_price, current_under_price, open_odds_data)
                                    open_line = movement_results.get("open_line")
                                    open_price_over = movement_results.get("open_price_over")
                                    open_price_under = movement_results.get("open_price_under")
                                    line_movement = movement_results.get("line_movement")
                                    price_movement_over = movement_results.get("price_movement_over")
                                    price_movement_under = movement_results.get("price_movement_under")

                                    # 2. Analyze vs OPEN odds (if available)
                                    # ... (existing logic for open_over/under_analysis, model_edge_vs_open_over/under) ...
                                    open_over_analysis = None
                                    open_under_analysis = None
                                    model_edge_vs_open_over = None
                                    model_edge_vs_open_under = None

                                    if open_line is not None and open_price_over is not None:
                                        open_over_analysis = analyse_side(side="over", line=open_line, price=open_price_over, prediction=prediction_for_analysis, rmse=rmse_to_use, mc_draws=mc_draws, prediction_source=prediction_source)
                                        if open_over_analysis: model_edge_vs_open_over = open_over_analysis.get('edge')

                                    if open_line is not None and open_price_under is not None:
                                        open_under_analysis = analyse_side(side="under", line=open_line, price=open_price_under, prediction=prediction_for_analysis, rmse=rmse_to_use, mc_draws=mc_draws, prediction_source=prediction_source)
                                        if open_under_analysis: model_edge_vs_open_under = open_under_analysis.get('edge')


                                    # 3. Interpret Movement Context
                                    movement_context = interpret_movement_context(model_edge_vs_open_over, model_edge_vs_open_under, line_movement)

                                    # 4. Prepare final results including movement data
                                    if over_analysis or under_analysis:
                                        # ... (existing logic to prepare analysis_output_for_console) ...
                                        if analysis_output_for_console is None:
                                             analysis_output_for_console = {
                                                 'final_pred': prediction_for_analysis,
                                                 'base_lgbm_pred': base_lgbm_pred_val,
                                                 'base_xgb_pred': base_xgb_pred_val,
                                                 'prediction_source': prediction_source, # Pass correct source
                                                 'over': over_analysis, # Vs Current
                                                 'under': under_analysis, # Vs Current
                                                 'open_over_analysis': open_over_analysis,
                                                 'open_under_analysis': open_under_analysis,
                                                 'open_line': open_line,
                                                 'open_price_over': open_price_over,
                                                 'open_price_under': open_price_under,
                                                 'line_movement': line_movement,
                                                 'price_movement_over': price_movement_over,
                                                 'price_movement_under': price_movement_under,
                                                 'movement_context': movement_context
                                             }

                                        # --- Log Enhanced Results to DB (existing logic) ---
                                        base_info = {
                                            "game_id": db_gid, "odds_api_game_id": api_event_id, "game_date": game_date_str,
                                            "player_id": pid_str, "player_name": pname, "team_id": str(team_id) if team_id is not None else None, "team_name": team_name,
                                            "opponent_team_id": str(opp_id) if opp_id != "N/A" else None, "opponent_name": opp_name, "bookmakers": bkm_title,
                                            "market": ODDS_API_MARKETS, "rmse": float(rmse_to_use) if rmse_to_use is not None else None,
                                            "prediction_source": prediction_source,
                                            "predicted_points": float(prediction_for_analysis),
                                            "predicted_points_lgbm": float(base_lgbm_pred_val) if base_lgbm_pred_val is not None else None,
                                            "predicted_points_xgb": float(base_xgb_pred_val) if base_xgb_pred_val is not None else None,
                                            "open_line": open_line,
                                            "open_price_over": open_price_over,
                                            "open_price_under": open_price_under,
                                            "line_movement": line_movement,
                                            "price_movement_over": price_movement_over,
                                            "price_movement_under": price_movement_under,
                                            "movement_context": movement_context
                                        }
                                        # ... (existing logic to append over/under analysis_results) ...
                                        if over_analysis:
                                            is_sharp_side = over_analysis['kelly_fraction'] >= KELLY_EDGE_THRESHOLD and over_analysis['edge'] >= KELLY_EDGE_THRESHOLD
                                            alert_text_db = f"SHARP OVER vs Current (Context: {movement_context})" if is_sharp_side else ""
                                            analysis_results.append({
                                                **base_info,
                                                "line": over_analysis["line"], "price": over_analysis["price"], "bet_type": "over",
                                                "model_win_pct": over_analysis["model_win_pct"], "z_score": over_analysis["z_score"],
                                                "monte_carlo_win_pct": over_analysis["monte_carlo_win_pct"], "implied_book_win_pct": over_analysis["implied_book_win_pct"],
                                                "kelly_fraction": over_analysis["kelly_fraction"], "edge": over_analysis["edge"],
                                                "model_edge_vs_open": model_edge_vs_open_over,
                                                "alert_text": alert_text_db,
                                              })
                                        if under_analysis:
                                            is_sharp_side = under_analysis['kelly_fraction'] >= KELLY_EDGE_THRESHOLD and under_analysis['edge'] >= KELLY_EDGE_THRESHOLD
                                            alert_text_db = f"SHARP UNDER vs Current (Context: {movement_context})" if is_sharp_side else ""
                                            analysis_results.append({
                                                **base_info,
                                                "line": under_analysis["line"], "price": under_analysis["price"], "bet_type": "under",
                                                "model_win_pct": under_analysis["model_win_pct"], "z_score": under_analysis["z_score"],
                                                "monte_carlo_win_pct": under_analysis["monte_carlo_win_pct"], "implied_book_win_pct": under_analysis["implied_book_win_pct"],
                                                "kelly_fraction": under_analysis["kelly_fraction"], "edge": under_analysis["edge"],
                                                "model_edge_vs_open": model_edge_vs_open_under,
                                                "alert_text": alert_text_db,
                                              })

                                except (ValueError, TypeError) as format_err:
                                    logger.warning(f"Could not format line/price for {pname} at {bkm_title}: line={current_line_raw}, O={current_over_price_raw}, U={current_under_price_raw}. Error: {format_err}")
                                # Reset outcomes after processing a market entry
                                current_over_outcome, current_under_outcome = None, None

            # --- Print Console Output (once per player, includes movement) ---
            if analysis_output_for_console:
                _print_console(
                    player=pname, team=team_name, opp=opp_name, game_date=game_date_str,
                    analysis_result=analysis_output_for_console,
                    rmse_used=rmse_to_use,
                    rmse_reason=reason
                )
            elif prediction_source:
                if not player_found_in_market:
                     logger.warning(f"Player {pname} (ID: {pid_str}) had prediction ({prediction_source}) but was not found in any market outcomes for event {api_event_id}.")


    logger.info(f"Completed betting analysis with movement context. Generated {len(analysis_results)} potential log entries.")
    return analysis_results


# (print_meta_prediction_summary MODIFIED for prediction source)
def print_meta_prediction_summary(schedule: Dict[str, dict]):
    """Prints a summary of meta model predictions for players in each game."""
    print("\n" + "="*30 + " META PREDICTION SUMMARY (LGBM Optuna) " + "="*30) # Updated Title
    summary_found = False

    sorted_game_ids = sorted(schedule.keys(), key=lambda gid: schedule[gid].get('game_date', ''))

    for game_id in sorted_game_ids:
        game_info = schedule[game_id]
        players_with_meta_predictions = {} # {team_id: [(player_name, prediction), ...]}
        team_names = {} # {team_id: team_name}
        game_title = f"GAME {game_id}"
        team_ids_in_game = list(game_info.get("teams", {}).keys())

        if len(team_ids_in_game) == 2:
            team1_id = team_ids_in_game[0]
            team2_id = team_ids_in_game[1]
            team1_name = game_info["teams"][team1_id].get("team_name", f"Team {team1_id}")
            team2_name = game_info["teams"][team2_id].get("team_name", f"Team {team2_id}")
            game_title = f"{game_id} - {team1_name.upper()} vs {team2_name.upper()}"
            team_names[team1_id] = team1_name
            team_names[team2_id] = team2_name
            players_with_meta_predictions[team1_id] = []
            players_with_meta_predictions[team2_id] = []
        else:
            # Fallback if teams aren't structured as expected
             for tid, tinfo in game_info.get("teams", {}).items():
                 team_names[tid] = tinfo.get("team_name", f"Team {tid}")
                 players_with_meta_predictions[tid] = []


        game_has_meta_preds = False
        for player_id, player_data in game_info.get("players", {}).items():
            # <<< CHANGE 18: Check for the specific meta prediction source >>>
            if player_data.get("prediction_source") == 'playoff_meta_lgbm_optuna':
                pred_val = player_data.get("final_prediction")
                team_id = player_data.get("team_id")
                pname = player_data.get("player_name", f"Unknown {player_id}")
                if pred_val is not None and team_id in players_with_meta_predictions:
                    players_with_meta_predictions[team_id].append((pname, pred_val))
                    game_has_meta_preds = True
                    summary_found = True

        if game_has_meta_preds:
            print(f"\n--- {game_title} ---")
            for team_id, player_list in players_with_meta_predictions.items():
                if player_list: # Only print team if they have meta-predicted players
                    print(f"\n{team_names.get(team_id, f'Team {team_id}')}")
                    # Sort players alphabetically
                    player_list.sort(key=lambda x: x[0])
                    for pname, pred_val in player_list:
                        print(f"  {pname} - {pred_val:.2f} pts")

    if not summary_found:
        print("\nNo players with Meta Model (LGBM Optuna) predictions found in the processed games.") # Updated Message
    print("\n" + "="*80)


# ───────────────────────── Main Orchestration (MODIFIED) ─────────────────────────
def main() -> None:
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        # <<< CHANGE 19: Update logger name for verbosity setting >>>
        logging.getLogger("playoff_predict_meta_lgbm_optuna").setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        # <<< CHANGE 20: Update logger name for verbosity setting >>>
        logging.getLogger("playoff_predict_meta_lgbm_optuna").setLevel(logging.INFO)

    start_time = time.time()

    # --- Prediction Function Import (Standard + Meta Playoff) ---
    # predict_player_meta_func will now be the function from predict_points_meta_model_lightgbm.py
    predict_player_std_func, predict_player_meta_func = import_prediction_functions()

    # --- API Key (Live Only) ---
    api_key_live = resolve_api_key(args)
    if not api_key_live:
        logger.error("Live Odds API key is required for fetching market data. Exiting.")
        sys.exit(1)

    bookmakers_param = None if not args.bookmakers or args.bookmakers.lower() == 'all' else args.bookmakers

    # --- Resolve Paths (including odds movement script) ---
    try:
        db_path = Path(args.db_file).resolve(strict=True)
        meta_artifact_dir = Path(args.meta_artifact_dir).resolve()
        lgbm_playoff_artifact_dir = Path(args.lgbm_playoff_artifact_dir).resolve()
        xgb_playoff_artifact_dir = Path(args.xgb_playoff_artifact_dir).resolve()
        odds_movement_script_path = Path(args.odds_script_path).resolve() # Resolve odds script path

        if not meta_artifact_dir.is_dir(): raise NotADirectoryError(f"Meta artifact path not found or not a directory: {meta_artifact_dir}")
        if not lgbm_playoff_artifact_dir.is_dir(): raise NotADirectoryError(f"LGBM playoff artifact path not found or not a directory: {lgbm_playoff_artifact_dir}")
        if not xgb_playoff_artifact_dir.is_dir(): raise NotADirectoryError(f"XGBoost playoff artifact path not found or not a directory: {xgb_playoff_artifact_dir}")
        if not odds_movement_script_path.is_file(): raise FileNotFoundError(f"Player odds movement script not found: {odds_movement_script_path}") # Check if it's a file

    except (FileNotFoundError, NotADirectoryError) as path_err:
            logger.error(f"Path Error: {path_err}. Please check the --db-file, artifact directory, and --odds-script-path arguments.")
            sys.exit(1)


    # --- Cache Key & Rebuild Logic (Unchanged) ---
    cache_key = str(args.game_id if args.game_id else args.game_date)
    logger.info(f"Using cache key: {cache_key}")
    cache_file = Path(PREDICTION_CACHE_FILE)
    if args.rebuild_cached:
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f: full_cache_data = json.load(f)
                if isinstance(full_cache_data, dict) and cache_key in full_cache_data:
                    del full_cache_data[cache_key]
                    with open(cache_file, 'w') as f: json.dump(full_cache_data, f, indent=2, default=str)
                    logger.info(f"Removed key '{cache_key}' from cache file due to --rebuild-cached: {cache_file}")
                elif isinstance(full_cache_data, dict): logger.info(f"--rebuild-cached specified, but key '{cache_key}' not found in cache file.")
                else: logger.warning(f"Cache file {cache_file} is invalid format. Deleting entirely due to --rebuild-cached."); cache_file.unlink(missing_ok=True)
            except (json.JSONDecodeError, IOError, OSError) as e:
                logger.error(f"Error modifying cache file {cache_file} for rebuild: {e}. Attempting to delete.")
                cache_file.unlink(missing_ok=True)
        else: logger.info("--rebuild-cached specified, but no cache file found.")

    conn = None
    schedule = {}
    analysis_results = []

    try:
        # --- Load Cache ---
        logger.info(f"Attempting to load predictions from cache file: {cache_file}")
        loaded_cache_for_key = load_prediction_cache(cache_file, cache_key)

        # --- Database Connection and Setup ---
        logger.info(f"Connecting to database: {db_path}")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        ensure_analysis_table(conn) # Ensures table and columns (incl. movement) exist

        # --- Fetch Global RMSE (Fallback) ---
        global_rmse = fetch_rmses(conn)

        # --- Fetch Schedule Data (DB) ---
        schedule = get_schedule_data(conn, args)
        if not schedule:
            logger.warning("No schedule data found for the specified criteria, exiting.")
            return

        # --- Fetch Market Data (API - Live Only) & Map Events ---
        market_data, event_id_map = fetch_market_data_and_map_events(schedule, api_key_live, bookmakers_param)

        # --- Extract Players from API Data & Match to DB ---
        players_to_predict = extract_and_match_players_from_api(market_data, event_id_map, schedule, conn)
        if not players_to_predict:
               logger.warning("No players found in API market data matched to DB players. No predictions will run. Exiting.")
               return

        # --- Enrich Schedule with Predictions (Handles Standard OR Meta Playoff) ---
        # This now uses the imported function from predict_points_meta_model_lightgbm.py
        newly_generated_predictions = enrich_schedule_with_predictions(
            schedule, # Modified in place
            players_to_predict,
            predict_standard_func=predict_player_std_func,
            predict_meta_playoff_func=predict_player_meta_func, # Pass meta func
            cached_predictions=loaded_cache_for_key,
            db_path=db_path,
            meta_artifact_dir=meta_artifact_dir,
            lgbm_playoff_artifact_dir=lgbm_playoff_artifact_dir,
            xgb_playoff_artifact_dir=xgb_playoff_artifact_dir
        )

        # --- Save Cache (Handles new structure) ---
        if newly_generated_predictions:
            logger.info(f"Saving {len(newly_generated_predictions)} newly generated/updated predictions to cache key '{cache_key}'...")
            save_prediction_cache(cache_file, cache_key, newly_generated_predictions)
        elif loaded_cache_for_key is not None:
             logger.info("Using existing cache; no new predictions generated for API players.")
        else:
             logger.info("No predictions generated (or cache miss and no new predictions).")

        # --- Perform Analysis (Now includes movement analysis) ---
        analysis_results = perform_betting_analysis(
            schedule,
            market_data,
            global_rmse,
            args.sample_size,
            odds_movement_script_path, # Pass the script path
            args.verbose # Pass verbosity flag
        )

        # --- Output/Log Results ---
        if args.dry_run:
            logger.info("Dry run requested. Analysis results printed to console. NOT logged to DB.")
            # Detailed console output already happened in perform_betting_analysis
        elif analysis_results:
            logger.info("Logging analysis results (including movement data) to database...")
            log_analysis_results_db(conn, analysis_results) # Logs enhanced results
        else:
            logger.info("No valid analysis results generated to log.")

        # --- Print Meta Prediction Summary (Unchanged) ---
        if schedule:
            print_meta_prediction_summary(schedule)

    except sqlite3.Error as db_err: logger.critical(f"Database critical error: {db_err}", exc_info=True)
    except ImportError as imp_err: logger.critical(f"Failed to import required module: {imp_err}", exc_info=True)
    except FileNotFoundError as fnf_err: logger.critical(f"File not found error: {fnf_err}", exc_info=True)
    except NotADirectoryError as nd_err: logger.critical(f"Path is not a directory: {nd_err}", exc_info=True)
    except Exception as e: logger.critical(f"An unexpected critical error occurred in main: {e}", exc_info=True)
    finally:
        if conn:
            try: conn.close(); logger.debug("Database connection closed.")
            except sqlite3.Error as e: logger.error(f"Error closing database connection: {e}")

    end_time = time.time()
    logger.info(f"Script finished in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()