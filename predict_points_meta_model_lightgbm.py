#!/usr/bin/env python3
"""
predict_points_meta_model_lightgbm.py
===========================================

Standalone prediction script for the *Optuna-tuned LGBM Pipeline MetaPlayoffModel*.

Predicts player stats for upcoming PLAYOFF games only using a pre-trained
meta-model pipeline (StandardScaler + LGBMRegressor). It obtains predictions
from the underlying base models (LGBM, XGBoost) for the specific game first,
then calculates meta-features, and finally feeds these into the loaded meta-model pipeline.

Key principles
--------------
1. **Calls Base Predictors:** Imports and calls the prediction functions from the
   standalone LightGBM and XGBoost playoff prediction scripts to get their
   predictions for the specific upcoming playoff game.
2. **Checks Playoff Game:** Only proceeds if the player's next game is a playoff game (handled by base predictors).
3. **Loads Meta-Model Pipeline Artifacts:** Uses target-specific meta-model artifacts
   (Pipeline containing Scaler + LGBM model, feature list) saved by
   train_playoffs_meta_model_optuna.py.
4. **Calculates Meta-Features:** Replicates the meta-feature calculation logic from
   the meta-model training script based on the base model predictions.
5. **Predicts with Meta-Model Pipeline:** Uses the loaded Pipeline (which handles scaling
   internally) to make the final prediction.
6. **Schema Parity:** Relies on base predictors for underlying data fetching.
7. **Sigma Reuse:** Uses the same sigma calculation logic (based on main model runs)
   as the base predictors for betting analysis.
"""

from __future__ import annotations

# ───────────────────────────── stdlib ──────────────────────────────
import argparse
import json
import logging
import math
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ───────────────────────────── 3rd‑party ───────────────────────────
import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.pipeline import Pipeline # Import Pipeline for loading

# --- Import the BASE prediction functions ---
# ASSUMPTION: These scripts have the core logic in these functions
#             and these functions return a dictionary with results.
try:
    from predict_points_lightgbm_playoffs import predict_player_playoffs as predict_lgbm
    logger_lgbm = logging.getLogger("playoff_predict_standalone") # Get base logger if needed
except ImportError as ie_lgbm:
    print(f"ERROR: Failed to import LightGBM prediction function from "
          f"'predict_points_lightgbm_playoffs.py'. Check script existence. Details: {ie_lgbm}", file=sys.stderr)
    sys.exit(1)

try:
    from predict_points_xgboost_playoffs import predict_player_playoffs_xgboost as predict_xgb
    logger_xgb = logging.getLogger("playoff_predict_xgboost") # Get base logger if needed
except ImportError as ie_xgb:
     print(f"ERROR: Failed to import XGBoost prediction function from "
           f"'predict_points_xgboost_playoffs.py'. Check script existence. Details: {ie_xgb}", file=sys.stderr)
     sys.exit(1)


# ───────────────────────────── logging ─────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s [PlayoffPredictMetaLGBMOptuna]", # Updated identifier
    datefmt="%Y‑%m‑%d %H:%M:%S",
)
# Updated logger name
logger = logging.getLogger("playoff_predict_meta_lgbm_optuna")

# ──────────────────────────── constants ────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()

# --- Define Database Path ---
try:
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    DB_FILE = PROJECT_ROOT / "nba.db"
    if not DB_FILE.exists():
        logger.warning(f"Default database file not found at: {DB_FILE}. Using relative 'nba.db'. Use --db-file argument.")
        DB_FILE = Path("nba.db")
except Exception:
     logger.warning("Could not determine project root reliably. Defaulting DB path to 'nba.db'. Use --db-file argument.")
     DB_FILE = Path("nba.db")

# --- Define Default Artifact Dirs ---
# Default location for the meta-model artifacts trained by train_playoffs_meta_model_optuna.py
DEFAULT_META_ARTIFACTS_DIR = SCRIPT_DIR / "meta_model_lgbm_optuna_playoff_artifacts" # UPDATED default path
# Defaults for BASE model artifacts (adjust if necessary or rely on CLI args)
DEFAULT_LGBM_ARTIFACTS_DIR = SCRIPT_DIR
DEFAULT_XGB_ARTIFACTS_DIR = SCRIPT_DIR / "xgboost_playoff_artifacts_v7" # Example from XGB script

# Small constant from meta-training script
EPSILON = 1e-6

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

# ─────────────────── META-MODEL PIPELINE artefact loading ───────────
# Updated cache type hint
_META_MODEL_CACHE: Dict[str, Tuple[Pipeline, List[str]]] = {}

def _load_meta_model(target: str, artifact_dir: Path) -> Tuple[Pipeline, List[str]]:
    """
    Loads the fitted meta-model PIPELINE (Scaler+LGBM) and its feature list
    for a specific target.
    """
    global _META_MODEL_CACHE
    cache_key = f"{target}_{artifact_dir}"
    if cache_key in _META_MODEL_CACHE:
        logger.debug(f"Returning cached meta-model pipeline artefacts for target '{target}' from {artifact_dir}.")
        return _META_MODEL_CACHE[cache_key]

    logger.info(f"Loading META-MODEL PIPELINE artefacts for target '{target}' from: {artifact_dir} …")

    # Updated filename to match Optuna training script output
    model_filename = f"meta_model_lgbm_optuna_pipeline_playoffs_{target}.joblib"
    features_filename = f"meta_model_features_playoffs_{target}.joblib" # Feature list name remains same
    model_file = artifact_dir / model_filename
    features_file = artifact_dir / features_filename

    logger.debug(f"Attempting to load meta-model pipeline: {model_file}")
    logger.debug(f"Attempting to load meta-model features: {features_file}")

    if not model_file.exists():
        raise FileNotFoundError(f"Required META-MODEL PIPELINE artifact file missing: {model_file}")
    if not features_file.exists():
        raise FileNotFoundError(f"Required META-MODEL features file missing: {features_file}")

    try:
        # Load the pipeline object
        model_pipeline: Pipeline = joblib.load(model_file)
        feature_names: List[str] = joblib.load(features_file)

        logger.info(
            f"META-MODEL PIPELINE artefacts loaded for target '{target}' — Pipeline using {len(feature_names)} features."
        )

        _META_MODEL_CACHE[cache_key] = (model_pipeline, feature_names)
        return model_pipeline, feature_names
    except Exception as e:
        logger.error(f"Failed to load meta-model pipeline artifacts for target '{target}': {e}", exc_info=True)
        raise

# ───────────────────────── SQL helpers (Copied/Minimal) ────────────
# Only need DB connection and sigma calculation here, as base predictors handle data fetching.
def _connect_db(db_path: Path) -> sqlite3.Connection:
    """Helper to connect to the database."""
    try:
        logger.debug(f"Attempting to connect to database: {db_path}")
        if not db_path.exists():
             raise FileNotFoundError(f"Database file does not exist at: {db_path}")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout = 5000;") # Optional: 5 seconds timeout
        conn.row_factory = sqlite3.Row # Optional
        logger.debug(f"Database connection successful: {db_path}")
        return conn
    except (sqlite3.Error, FileNotFoundError) as e:
        logger.error(f"Error connecting to database at {db_path}: {e}")
        raise ConnectionError(f"Failed to connect to database: {db_path}") from e

# --- Sigma Calculation Helpers (Reused from base predictors) ---
def _latest_two_runs(conn: sqlite3.Connection) -> Tuple[str, str]:
    """Gets identifiers for the two most recent MAIN training runs (copied)."""
    logger.debug("Fetching latest two MAIN model run identifiers for sigma...")
    try:
        # Assumes main runs don't have specific playoff identifiers like _playoffs_
        # Adjust WHERE clause if main runs have a different pattern
        runs_df = pd.read_sql_query(
            """SELECT DISTINCT model_run
               FROM training_predictions
               WHERE model_run NOT LIKE '%playoffs%' -- Exclude playoff runs if needed
               ORDER BY model_run DESC LIMIT 2""",
            conn
        )
        main_runs = runs_df["model_run"].tolist()
        if len(main_runs) < 2:
            logger.error("Need >= 2 MAIN 'model_run' batches in 'training_predictions' for sigma.")
            raise ValueError("Insufficient MAIN model runs found for sigma calculation.")
        logger.debug(f"Found latest MAIN model runs: {main_runs[0]}, {main_runs[1]}")
        return main_runs[0], main_runs[1]
    except (pd.errors.DatabaseError, sqlite3.Error, ValueError) as e:
        logger.error(f"Error fetching latest MAIN model runs: {e}")
        raise ValueError("Failed to fetch latest MAIN model runs.") from e

def _player_sigma(conn: sqlite3.Connection, player_id: int) -> Tuple[float | None, str, int]:
    """Calculates sigma based on MAIN model residuals (copied)."""
    logger.debug(f"Calculating sigma for player_id: {player_id} using MAIN model residuals...")
    try:
        run_new, run_prev = _latest_two_runs(conn)
    except ValueError as e:
        logger.error(f"Cannot calculate sigma, failed to get latest main model runs: {e}")
        return None, "error_no_runs", 0

    method = "pooled"; n_points = 0; sigma = None
    PLAYER_SIGMA_MIN_N = 75
    try:
        player_res_df = pd.read_sql_query(
            "SELECT residual FROM training_predictions WHERE player_id = ? AND model_run IN (?, ?)",
            conn, params=(player_id, run_new, run_prev),
        )
        player_res = player_res_df["residual"].dropna()
        n_player_points = len(player_res)

        if n_player_points >= PLAYER_SIGMA_MIN_N:
            variance = np.mean(player_res**2)
            if variance >= 0 and np.isfinite(variance):
                sigma = float(math.sqrt(variance))
            method = "player_specific"; n_points = n_player_points
            if sigma is not None and np.isfinite(sigma):
                logger.info("Using player-specific σ (N=%d, main runs %s/%s): %.4f", n_points, run_new[:6], run_prev[:6], sigma)
            else:
                logger.warning("Player-specific sigma calculation resulted in invalid value (Variance: %s). Falling back.", variance); sigma = None
        else:
            logger.info("Insufficient player data (N=%d, main runs %s/%s) for specific σ, using pooled.", n_player_points, run_new[:6], run_prev[:6])

        if sigma is None: # Fallback to pooled
            pooled_res_df = pd.read_sql_query(
                "SELECT residual FROM training_predictions WHERE model_run IN (?, ?)",
                conn, params=(run_new, run_prev),
            )
            pooled_res = pooled_res_df["residual"].dropna()
            if pooled_res.empty:
                logger.warning("No residuals found in main model runs for pooled sigma."); return None, "error_pooled_empty", 0
            n_pooled_points = len(pooled_res)
            variance = np.mean(pooled_res**2)
            if variance >= 0 and np.isfinite(variance):
                sigma = float(math.sqrt(variance))
            method = "pooled"; n_points = n_pooled_points
            if sigma is not None and np.isfinite(sigma):
                logger.info("Using pooled σ (N=%d, main runs %s/%s): %.4f", n_points, run_new[:6], run_prev[:6], sigma)
            else:
                logger.error("Pooled sigma calculation resulted in invalid value (Variance: %s). Sigma unavailable.", variance); return None, "error_pooled_calc", n_pooled_points

        if sigma is None or not np.isfinite(sigma) or sigma <= 0:
            logger.error(f"Final sigma value is invalid or non-positive ({sigma})."); return None, method, n_points
        return sigma, method, n_points
    except (pd.errors.DatabaseError, sqlite3.Error) as e:
        logger.error(f"Database error calculating sigma for player {player_id}: {e}"); return None, f"error_{method}_db", n_points
    except Exception as e:
        logger.error(f"Unexpected error calculating sigma for player {player_id}: {e}", exc_info=True); return None, f"error_{method}_unexpected", n_points


# ─────────────────── META-FEATURE Calculation ──────────────────────
def _calculate_meta_features(lgbm_pred: float, xgb_pred: float) -> Dict[str, float]:
    """
    Calculates derived meta-features based on the base model predictions.
    Matches the logic in train_playoffs_meta_model_optuna.py.
    """
    meta_features = {}
    lgbm = lgbm_pred
    xgb = xgb_pred

    # --- Replicate features from training ---
    meta_features['lgbm_pred'] = lgbm_pred # Base prediction 1
    meta_features['xgb_pred'] = xgb_pred   # Base prediction 2
    meta_features['diff_pred'] = lgbm - xgb
    meta_features['abs_diff'] = abs(meta_features['diff_pred'])
    meta_features['avg_pred'] = 0.5 * (lgbm + xgb)
    meta_features['rel_diff'] = meta_features['diff_pred'] / (meta_features['avg_pred'] + EPSILON)
    meta_features['ratio_pred'] = lgbm / (xgb + EPSILON)
    meta_features['min_pred'] = min(lgbm, xgb)
    meta_features['max_pred'] = max(lgbm, xgb)
    meta_features['prod_pred'] = lgbm * xgb
    meta_features['lgbm_sq'] = lgbm ** 2
    meta_features['xgb_sq'] = xgb ** 2
    lgbm_diff_sq = (lgbm - meta_features['avg_pred']) ** 2
    xgb_diff_sq = (xgb - meta_features['avg_pred']) ** 2
    # Handle potential division by zero or sqrt of negative if diffs are zero
    pair_std_val = 0.5 * (lgbm_diff_sq + xgb_diff_sq)
    meta_features['pair_std'] = math.sqrt(pair_std_val) if pair_std_val >= 0 else 0
    meta_features['model_order'] = 1.0 if lgbm > xgb else 0.0
    # --- End feature replication ---

    return meta_features

# ───────────────── META-MODEL LGBM PIPELINE PLAYOFF prediction core ───────────────
def predict_player_playoffs_meta(
    player_id: int,
    target: str, # Target stat required
    *,
    line: Optional[float] = None,
    over_odds: Optional[int] = None,
    under_odds: Optional[int] = None,
    db_path: Path = DB_FILE,
    meta_artifact_dir: Path = DEFAULT_META_ARTIFACTS_DIR, # Uses updated default
    lgbm_artifact_dir: Path = DEFAULT_LGBM_ARTIFACTS_DIR, # Path for LGBM base model artifacts
    xgb_artifact_dir: Path = DEFAULT_XGB_ARTIFACTS_DIR,  # Path for XGBoost base model artifacts
    # Add args to control base predictor verbosity if needed
    # base_verbose: bool = False
) -> Dict[str, Any]:
    """
    Predict the **next scheduled PLAYOFF** game stat for *player_id* using the
    Optuna-tuned LGBM Meta-Model Pipeline.

    Requires predictions from the underlying LightGBM and XGBoost playoff models.
    """
    conn: Optional[sqlite3.Connection] = None
    logger.info(f"Attempting LGBM META-PIPELINE PLAYOFF prediction for player {player_id}, target '{target}'...")

    # --- Suppress base model logging unless debug is enabled (Optional) ---
    # ... (logging level adjustment code if needed) ...

    try:
        # --- Step 1: Get Base Model Predictions ---
        # (Unchanged - relies on imported functions)
        logger.info("Running LightGBM base prediction...")
        lgbm_result = predict_lgbm(
            player_id=player_id, target=target, line=None, over_odds=None, under_odds=None,
            db_path=db_path, artifact_dir=lgbm_artifact_dir
        )
        lgbm_pred_value = lgbm_result['predicted_value']
        game_id = lgbm_result.get('game_id', 'N/A'); game_date = lgbm_result.get('game_date', 'N/A')
        team_id = lgbm_result.get('team_id', 'N/A'); opponent_team_id = lgbm_result.get('opponent_team_id', 'N/A')
        player_name = lgbm_result.get('player_name', f"ID {player_id}"); team_name = lgbm_result.get('team_name', f"ID {team_id}")
        logger.info(f"LGBM Base Prediction for {target}: {lgbm_pred_value:.3f} (Game: {game_id} on {game_date})")

        logger.info("Running XGBoost base prediction...")
        xgb_result = predict_xgb(
            player_id=player_id, target=target, line=None, over_odds=None, under_odds=None,
            db_path=db_path, artifact_dir=xgb_artifact_dir
        )
        xgb_pred_value = xgb_result['predicted_value']
        logger.info(f"XGBoost Base Prediction for {target}: {xgb_pred_value:.3f} (Game: {game_id} on {game_date})")

        if lgbm_result.get('game_id') != xgb_result.get('game_id'):
             logger.warning(f"Base predictors returned results for different games! Using LGBM game context.")

        # --- Step 2: Load Meta-Model Pipeline ---
        # Loads the pipeline (Scaler + LGBM) and the expected feature names
        meta_pipeline, meta_feature_names = _load_meta_model(target, meta_artifact_dir)

        # --- Step 3: Calculate Meta-Features ---
        logger.debug("Calculating meta-features...")
        meta_features_dict = _calculate_meta_features(lgbm_pred_value, xgb_pred_value)
        logger.debug(f"Calculated meta-features: {meta_features_dict}")

        # --- Step 4: Prepare Input for Meta-Model Pipeline ---
        # Create a single-row DataFrame with columns in the correct order
        # The pipeline's scaler will handle normalization.
        try:
            meta_input_df = pd.DataFrame([meta_features_dict])
            missing_meta_cols = [col for col in meta_feature_names if col not in meta_input_df.columns]
            if missing_meta_cols:
                 raise ValueError(f"Meta-feature calculation missing expected columns: {missing_meta_cols}")
            # Reorder columns to match the training feature order
            meta_input_df = meta_input_df[meta_feature_names]
        except Exception as e:
            logger.error(f"Failed to create input DataFrame for meta-model pipeline: {e}")
            raise ValueError("Error preparing data for meta-prediction.") from e

        logger.debug(f"Meta-model pipeline input DataFrame shape: {meta_input_df.shape}")
        logger.debug(f"Input columns: {meta_input_df.columns.tolist()}")
        logger.debug(f"Input data (pre-scaling): \n{meta_input_df.head().to_string()}")

        # --- Step 5: Calculate Prediction Sigma ---
        # (Unchanged - uses main model runs)
        conn = _connect_db(db_path)
        sigma, sigma_method, sigma_n = _player_sigma(conn, player_id)
        if conn: conn.close(); logger.debug("Sigma DB connection closed.")
        if sigma is not None: logger.info(f"Sigma calculation (main runs): Value={sigma:.4f}, Method='{sigma_method}', N={sigma_n}")
        else: logger.warning(f"Sigma calculation failed (Method: {sigma_method}). Proceeding without sigma.")

        # --- Step 6: Make Meta-Model Pipeline Prediction ---
        logger.debug("Making LGBM Meta-Model PIPELINE PLAYOFF prediction...")
        # Predict using the loaded pipeline object. It handles scaling internally.
        meta_prediction_value = meta_pipeline.predict(meta_input_df)
        predicted_target_value = float(meta_prediction_value[0])

        # Optional: Ensure prediction is non-negative (LGBM can sometimes predict slightly negative)
        predicted_target_value = max(0, predicted_target_value)

        logger.info(f"Predicted LGBM META-PIPELINE PLAYOFF {target}: {predicted_target_value:.3f}")

        # Format Results
        result = {
            "prediction_type": "playoffs_meta_lgbm_optuna", # Updated model type identifier
            "player_id": player_id,
            "player_name": player_name,
            "team_name": team_name,
            "team_id": team_id,
            "opponent_team_id": opponent_team_id,
            "game_id": game_id,
            "game_date": game_date,
            "is_playoffs": True, # Confirmed by base predictors
            "target_stat": target,
            "predicted_value": round(predicted_target_value, 3),
            "base_predictions": {
                 "lgbm": round(lgbm_pred_value, 3),
                 "xgb": round(xgb_pred_value, 3),
             },
            "model_sigma": round(sigma, 3) if sigma is not None else None,
            "sigma_method": sigma_method,
            "sigma_n": sigma_n,
        }

        # Optional Betting Edge Calculation (Unchanged)
        if line is not None and over_odds is not None and under_odds is not None:
            if sigma is not None and sigma > EPSILON:
                logger.debug("Calculating betting analysis for LGBM meta-pipeline playoff prediction...")
                z = (line - predicted_target_value) / sigma; p_over = 1.0 - norm.cdf(z); p_under = 1.0 - p_over
                d_over = _american_to_decimal(over_odds); d_under = _american_to_decimal(under_odds)
                result["betting_analysis"] = {
                    "line": line, "z_score": round(z, 3),
                    "over": {"am_odds": over_odds, "dec_odds": round(d_over, 3), "prob": round(p_over, 4), "imp_prob": round(_implied_prob(d_over), 4), "kelly": round(_kelly_fraction(p_over, d_over), 4)},
                    "under": {"am_odds": under_odds, "dec_odds": round(d_under, 3), "prob": round(p_under, 4), "imp_prob": round(_implied_prob(d_under), 4), "kelly": round(_kelly_fraction(p_under, d_under), 4)},
                }
                logger.debug("LGBM meta-pipeline playoff betting analysis complete.")
            else:
                logger.warning("Cannot calculate betting analysis due to missing/invalid sigma.")
                result["betting_analysis"] = None

        return result

    except (ValueError, FileNotFoundError, ConnectionError, RuntimeError) as e:
         logger.error(f"LGBM META-PIPELINE PLAYOFF prediction failed for player {player_id}, target '{target}': {e}")
         raise
    except Exception as e:
        logger.exception(f"Unexpected error during LGBM META-PIPELINE PLAYOFF prediction for player {player_id}, target '{target}': {e}")
        raise
    # finally:
        # ... (restore logging levels if needed) ...
        # ... (ensure DB connection closed if error occurred before close) ...


# ───────────────────── META-MODEL LGBM PIPELINE PLAYOFF CLI glue ──────────────────
def _cli_meta() -> None:
    """Command Line Interface setup and execution for Optuna-tuned LGBM Meta-Model PLAYOFF predictions."""
    parser = argparse.ArgumentParser(
        # Updated prog name and description
        prog="predict-playoffs-meta-lgbm-optuna",
        description=f"Predict NBA player stats for upcoming PLAYOFF games using the standalone Optuna-tuned LGBM Meta-Model Pipeline.\n"
                    f"Requires calling base LGBM and XGBoost prediction scripts.\n"
                    f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--player-id", type=int, required=True, help="NBA player_id (e.g., 201939 for Stephen Curry)")
    parser.add_argument(
        "--target", type=str, default="pts", required=False,
        help="Target stat to predict (default: pts) - MUST match trained meta-model artifacts."
    )
    parser.add_argument("--line", type=float, help="Prop bet line for the target stat (e.g., 28.5)")
    parser.add_argument("--over-odds", type=int, help="American odds for the OVER (e.g., -110)")
    parser.add_argument("--under-odds", type=int, help="American odds for the UNDER (e.g., -110)")
    parser.add_argument("--db-file", type=str, default=str(DB_FILE), help="Path to the SQLite database file.")
    parser.add_argument(
        "--meta-artifact-dir", type=str, default=str(DEFAULT_META_ARTIFACTS_DIR), # Uses updated default
        # Updated help text
        help="Directory containing META-MODEL PIPELINE artifacts (LGBM Optuna Pipeline, features)."
    )
    parser.add_argument(
        "--lgbm-artifact-dir", type=str, default=str(DEFAULT_LGBM_ARTIFACTS_DIR),
        help="Directory containing LightGBM PLAYOFF model artifacts (for base prediction)."
    )
    parser.add_argument(
        "--xgb-artifact-dir", type=str, default=str(DEFAULT_XGB_ARTIFACTS_DIR),
        help="Directory containing XGBoost PLAYOFF model artifacts (for base prediction)."
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging for meta-predictor.")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("playoff_predict_meta_lgbm_optuna").setLevel(logging.DEBUG) # Use updated logger name
        # Optionally set base predictor loggers to DEBUG too if needed
        logger.debug("Debug logging enabled.")
    # else:
        # ... (ensure base loggers are less verbose if needed) ...

    if args.line is not None and (args.over_odds is None or args.under_odds is None):
        parser.error("--line requires --over-odds and --under-odds.")
    if (args.over_odds is not None or args.under_odds is not None) and args.line is None:
         parser.error("--over-odds/--under-odds require --line.")

    try:
        db_path_to_use = Path(args.db_file).resolve()
        meta_artifact_path = Path(args.meta_artifact_dir).resolve()
        lgbm_artifact_path = Path(args.lgbm_artifact_dir).resolve()
        xgb_artifact_path = Path(args.xgb_artifact_dir).resolve()

        logger.info(f"Using database: {db_path_to_use}")
        logger.info(f"Using LGBM Meta-Pipeline artifact directory: {meta_artifact_path}") # Updated log
        logger.info(f"Using LGBM base artifact directory: {lgbm_artifact_path}")
        logger.info(f"Using XGBoost base artifact directory: {xgb_artifact_path}")

        # Validate paths
        if not db_path_to_use.exists(): raise FileNotFoundError(f"Database file not found: {db_path_to_use}")
        if not meta_artifact_path.is_dir(): raise NotADirectoryError(f"Meta artifact path not found or not a directory: {meta_artifact_path}")
        if not lgbm_artifact_path.is_dir(): raise NotADirectoryError(f"LGBM base artifact path not found or not a directory: {lgbm_artifact_path}")
        if not xgb_artifact_path.is_dir(): raise NotADirectoryError(f"XGBoost base artifact path not found or not a directory: {xgb_artifact_path}")


        # Call the main meta-model playoff prediction function
        prediction_result = predict_player_playoffs_meta(
            player_id=args.player_id,
            target=args.target,
            line=args.line,
            over_odds=args.over_odds,
            under_odds=args.under_odds,
            db_path=db_path_to_use,
            meta_artifact_dir=meta_artifact_path,
            lgbm_artifact_dir=lgbm_artifact_path,
            xgb_artifact_dir=xgb_artifact_path,
            # base_verbose=args.verbose # Pass verbosity control if implemented
        )
        print(json.dumps(prediction_result, indent=2, default=str))
        sys.exit(0) # Success

    except (ValueError, FileNotFoundError, ConnectionError, RuntimeError, NotADirectoryError) as e:
         logger.error(f"LGBM Meta-Pipeline Playoff prediction failed: {e}") # Updated message
         print(f"\nERROR: {e}", file=sys.stderr)
         sys.exit(1) # Failure exit code
    except ImportError as e:
         logger.error(f"Import error, likely missing base predictor script: {e}")
         print(f"\nERROR: Failed to import base prediction script. Ensure predict_points_lightgbm_playoffs.py and predict_points_xgboost_playoffs.py are accessible. Details: {e}", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
         logger.exception("An unexpected critical error occurred during LGBM meta-pipeline playoff prediction.") # Updated message
         print(f"\nERROR: An unexpected critical error occurred. Check logs for details. ({type(e).__name__})", file=sys.stderr)
         sys.exit(1) # Failure exit code

if __name__ == "__main__":
    _cli_meta()