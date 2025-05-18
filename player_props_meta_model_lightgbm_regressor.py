#!/usr/bin/env python3
"""
train_playoffs_meta_model_optuna.py

Trains a meta-model (LGBMRegressor) using out-of-fold predictions
from pre-trained base playoff models (LGBM, XGBoost) as input features.
Includes derived meta-features based on base model prediction interactions.
Uses Optuna with TimeSeriesSplit for hyperparameter tuning.

Reads predictions from the 'training_predictions' table in the specified SQLite DB.
Uses TimeSeriesSplit for robust cross-validation of the meta-model during optimization.
"""

from __future__ import annotations

# ───────────────────────────── stdlib ──────────────────────────────
import argparse
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
import warnings

# ─────────────────────────── 3rd‑party ─────────────────────────────
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna # Import Optuna
from sklearn.model_selection import TimeSeriesSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import DataConversionWarning
from sklearn.base import clone # Import clone for pipelines within objective
from sklearn.model_selection import GroupKFold     # add this


# Import set_config if using sklearn >= 1.2
try:
    from sklearn import set_config
except ImportError:
    set_config = None # Define as None if import fails

# ────────────────────────────── logging ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Updated logger name to reflect Optuna usage
logger = logging.getLogger("PlayoffMetaModelOptuna")
optuna.logging.set_verbosity(optuna.logging.WARNING) # Reduce Optuna's default verbosity

# Suppress specific warnings if needed
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message=".*verbosity is set.*")
warnings.filterwarnings("ignore", message=".*Parameter.*will be ignored.*")


# --- Configure sklearn output ---
if set_config:
    try:
        set_config(transform_output="pandas")
        logger.info("Set scikit-learn transform_output to 'pandas'.")
    except TypeError:
        logger.warning("Could not set transform_output='pandas'. Sklearn version might be < 1.2. Feature name warnings may persist.")
    except Exception as e:
        logger.warning(f"An error occurred setting transform_output: {e}. Feature name warnings may persist.")
else:
    logger.warning("sklearn.set_config not found. Sklearn version might be < 1.2. Feature name warnings may persist.")
# --- End sklearn output configuration ---


# ──────────────────────────── Constants ────────────────────────────
DEFAULT_DB_FILE = "nba.db"
DEFAULT_TARGET = "pts"
SCRIPT_DIR = Path(__file__).parent.resolve()
# Adjusted default artifacts directory name for Optuna version
DEFAULT_META_ARTIFACTS_DIR = SCRIPT_DIR / "meta_model_lgbm_optuna_playoff_artifacts"
N_CV_SPLITS = 5 # Number of splits for TimeSeriesSplit CV within Optuna objective
N_OPTUNA_TRIALS = 15 # Number of hyperparameter optimization trials for Optuna
EPSILON = 1e-6
GLOBAL_RANDOM_STATE = 1337 # For reproducibility in LGBM and Optuna sampler

# Note: LGBM_PARAM_GRID removed, search space defined in objective function


class GroupTimeSeriesSplit:
    """
    Same idea as sklearn.model_selection.TimeSeriesSplit, but splits on an
    *ordered group* key instead of individual rows.  We feed in the
    season-round-series string created below.
    """
    def __init__(self, n_splits: int):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("`groups` array is required.")
        # groups is array-like shaped (n_rows,)
        _, first_idx = np.unique(groups, return_index=True)
        ordered_groups = np.array(groups)[np.sort(first_idx)]
        n_groups = len(ordered_groups)

        test_size = max(1, n_groups // (self.n_splits + 1))
        for i in range(self.n_splits):
            test_start = n_groups - (i + 1) * test_size
            test_groups = ordered_groups[test_start : test_start + test_size]
            train_groups = ordered_groups[:test_start]

            train_idx = np.flatnonzero(np.isin(groups, train_groups))
            test_idx  = np.flatnonzero(np.isin(groups, test_groups))
            yield train_idx, test_idx


# ────────────────────────── Helper Functions ───────────────────────
# find_latest_model_run and load_predictions remain unchanged
def find_latest_model_run(db_conn: sqlite3.Connection, model_pattern: str) -> str | None:
    """
    Finds the most recent model_run ID matching the pattern in the database.
    Assumes model_run starts with a datetime string for sorting.
    """
    cursor = db_conn.cursor()
    try:
        cursor.execute(
            f"SELECT DISTINCT model_run FROM training_predictions WHERE model_run LIKE ?",
            (model_pattern,)
        )
        runs = [row[0] for row in cursor.fetchall()]
        if not runs:
            logger.warning(f"No model runs found matching pattern: {model_pattern}")
            return None
        runs.sort(reverse=True)
        latest_run = runs[0]
        logger.info(f"Found latest run for pattern '{model_pattern}': {latest_run}")
        return latest_run
    except sqlite3.Error as e:
        logger.error(f"Database error finding latest run for {model_pattern}: {e}")
        return None
    finally:
        if cursor: cursor.close()

def load_predictions(db_conn: sqlite3.Connection, model_run_id: str, model_prefix: str) -> pd.DataFrame:
    """Loads predictions for a specific model run."""
    query = f"""
    SELECT
        game_id,
        player_id,
        game_date,
        actual_points,
        predicted_points AS {model_prefix}_pred -- Rename prediction column
    FROM training_predictions
    WHERE model_run = ?
    """
    try:
        df = pd.read_sql_query(query, db_conn, params=(model_run_id,), parse_dates=["game_date"])
        logger.info(f"Loaded {len(df)} predictions for model run: {model_run_id}")
        if df.empty:
            logger.warning(f"No predictions found for model run: {model_run_id}")
        return df
    except Exception as e:
        logger.error(f"Failed to load predictions for {model_run_id}: {e}")
        return pd.DataFrame() # Return empty dataframe on error

# Custom scorer function (can be used inside objective if needed, but not required by Optuna)
def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ─────────────────────────── Main Logic ────────────────────────────

def train_meta_model(db_file: str, target_stat: str, artifacts_dir: Path):
    """Loads base predictions, creates meta-features, trains LGBM meta-model with Optuna."""
    logger.info(f"Starting Optuna-tuned LGBM meta-model training for target: {target_stat}")
    logger.info(f"Using database: {db_file}")

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    conn = None
    try:
        conn = sqlite3.connect(db_file)

        # ─────────────────────────────────────────────────────────────────────────────
        # 1. Locate the most-recent prediction runs for each base model
        # ─────────────────────────────────────────────────────────────────────────────
        lgbm_pattern = f"%_playoffs_lgbm_{target_stat}"
        xgb_pattern  = f"%_xgb_playoffs_{target_stat}"

        latest_lgbm_run = find_latest_model_run(conn, lgbm_pattern)
        latest_xgb_run  = find_latest_model_run(conn, xgb_pattern)

        if not latest_lgbm_run or not latest_xgb_run:
            raise ValueError("Could not find the latest prediction runs for both base models.")

        # ─────────────────────────────────────────────────────────────────────────────
        # 2. Load the player-level predictions for those runs
        # ─────────────────────────────────────────────────────────────────────────────
        df_lgbm = load_predictions(conn, latest_lgbm_run, "lgbm")
        df_xgb  = load_predictions(conn, latest_xgb_run,  "xgb")

        if df_lgbm.empty or df_xgb.empty:
            raise ValueError("Failed to load predictions for one or both base models.")

        # keep only the columns we need before merging
        df_lgbm_merge = df_lgbm[['game_id', 'player_id', 'game_date',
                                'actual_points', 'lgbm_pred']]
        df_xgb_merge  = df_xgb[['game_id', 'player_id', 'xgb_pred']]

        # ─────────────────────────────────────────────────────────────────────────────
        # 3. Build a **game-level** playoff-metadata table
        #
        #    • `teams_game_features` is team-level, so we aggregate by game_id.
        #    • We keep every row with `is_playoffs = 1` – that includes play-in games.
        #    • Play-ins have NULL `series_number`; we COALESCE it to 0 so that
        #      downstream logic sees an integer everywhere.
        # ─────────────────────────────────────────────────────────────────────────────
        games_meta_sql = """
        SELECT
            tgf.game_id,
            MAX(tgf.season)                              AS season,
            MAX(tgf.playoff_round)                       AS playoff_round,
            MAX(COALESCE(tgf.series_number, 0))          AS series_number   -- play-in → 0
        FROM   teams_game_features tgf
        WHERE  tgf.is_playoffs = 1                       -- keep series + play-in
        GROUP  BY tgf.game_id;                           -- one row per actual game
        """
        df_games_meta = pd.read_sql_query(games_meta_sql, conn)

        # ─────────────────────────────────────────────────────────────────────────────
        # 4. Merge the two prediction sets and attach the playoff metadata
        # ─────────────────────────────────────────────────────────────────────────────
        df_meta = (
            df_lgbm_merge
            .merge(df_xgb_merge,
                    on=['game_id', 'player_id'],
                    how='inner',
                    validate='many_to_many')             # same player rows in both dfs
            .merge(df_games_meta,
                    on='game_id',
                    how='left',
                    validate='many_to_one')              # one meta row per game
        )

        # ─────────────────────────────────────────────────────────────────────────────
        # 5. Remove rows whose game_id did not match any playoff metadata
        #    (i.e. regular-season games that slipped through)
        # ─────────────────────────────────────────────────────────────────────────────
        rows_missing = df_meta[['season', 'playoff_round']].isnull().any(axis=1)
        if rows_missing.any():
            n_drop = int(rows_missing.sum())
            logger.warning(
                "Dropping %d rows whose game_id could not be matched to playoff "
                "metadata (likely regular-season games).",
                n_drop,
            )
            df_meta = df_meta.loc[~rows_missing].copy()

        if df_meta.empty:
            raise ValueError("No rows left after filtering out games without playoff metadata.")

        # ─────────────────────────────────────────────────────────────────────────────
        # 6. Create the series_key and run a few final sanity checks
        # ─────────────────────────────────────────────────────────────────────────────
        df_meta['series_key'] = (
            df_meta['season'].astype(str)
            + '_' + df_meta['playoff_round'].astype(str)
            + '_' + df_meta['series_number'].astype(str)
        )
        logger.info("Distinct series_key count: %d", df_meta['series_key'].nunique())

        logger.info("Merged predictions. Meta-dataset size: %d rows.", len(df_meta))
        if df_meta[['lgbm_pred', 'xgb_pred']].isnull().any().any():
            logger.warning("NaNs found in base prediction columns after merge. Dropping.")
            df_meta.dropna(subset=['lgbm_pred', 'xgb_pred'], inplace=True)
            if df_meta.empty:
                raise ValueError("Meta-dataset empty after dropping NaN base predictions.")


        # 4. Add Meta-Features (Unchanged)
        logger.info("Calculating derived meta-features from base predictions...")
        lgbm_p = df_meta['lgbm_pred']
        xgb_p = df_meta['xgb_pred']
        df_meta['diff_pred'] = lgbm_p - xgb_p
        df_meta['abs_diff'] = df_meta['diff_pred'].abs()
        df_meta['avg_pred'] = 0.5 * (lgbm_p + xgb_p)
        df_meta['rel_diff'] = df_meta['diff_pred'] / (df_meta['avg_pred'] + EPSILON)
        df_meta['ratio_pred'] = lgbm_p / (xgb_p + EPSILON)
        df_meta['min_pred'] = np.minimum(lgbm_p, xgb_p)
        df_meta['max_pred'] = np.maximum(lgbm_p, xgb_p)
        df_meta['prod_pred'] = lgbm_p * xgb_p
        df_meta['lgbm_sq'] = np.square(lgbm_p)
        df_meta['xgb_sq'] = np.square(xgb_p)
        lgbm_diff_sq = np.square(lgbm_p - df_meta['avg_pred'])
        xgb_diff_sq = np.square(xgb_p - df_meta['avg_pred'])
        df_meta['pair_std'] = np.sqrt(0.5 * (lgbm_diff_sq + xgb_diff_sq)).fillna(0)
        df_meta['model_order'] = np.where(lgbm_p > xgb_p, 1, 0)
        df_meta['model_order'] = np.where(lgbm_p > xgb_p, 1, 0)
        # ───────────────── Ensure every meta-feature column is numeric ───────────────
        meta_cols = df_meta.columns.difference(['game_id', 'player_id', 'game_date', 'series_key'])
        df_meta[meta_cols] = df_meta[meta_cols].apply(
            lambda s: pd.to_numeric(s, errors='coerce')
        )
        # ─────────────────────────────────────────────────────────────────────────────
        non_derived_cols = ['game_id', 'player_id', 'game_date', 'actual_points', 'lgbm_pred', 'xgb_pred']
        num_derived_features = len(df_meta.columns) - len(non_derived_cols)
        logger.info(f"Finished calculating {num_derived_features} derived meta-features.")

        # Handle potential NaNs/Infs from calculations (Unchanged)
        cols_to_check = df_meta.columns.drop(non_derived_cols)
        if df_meta[cols_to_check].isnull().any().any() or np.isinf(df_meta[cols_to_check].values).any():
            logger.warning("NaNs or Infs found after calculating meta-features. Replacing Inf with NaN and filling NaN with 0.")
            df_meta.fillna(0, inplace=True)

        # 5. Prepare data for TimeSeriesSplit (Unchanged)
        df_meta.sort_values('game_date', inplace=True)
        df_meta.reset_index(drop=True, inplace=True)
        feature_columns = [
            'lgbm_pred', 'xgb_pred', 'diff_pred', 'abs_diff', 'avg_pred',
            'rel_diff', 'ratio_pred', 'min_pred', 'max_pred', 'prod_pred',
            'lgbm_sq', 'xgb_sq', 'pair_std', 'model_order'
        ]
        # -----------------------------------------------------------------
        # Clean up the derived features that can explode to ±Inf.
        # We do this *before* slicing X_meta, so the original integer index
        # stays intact for TimeSeriesSplit.
        # -----------------------------------------------------------------
        for col in ('ratio_pred', 'rel_diff', 'prod_pred'):
            if col in df_meta.columns:
                df_meta[col] = (
                    df_meta[col]
                    .replace([np.inf, -np.inf], np.nan)   # turn ±Inf → NaN
                    .fillna(0)                            # turn NaN → 0
                )
        missing_cols = [col for col in feature_columns if col not in df_meta.columns]
        if missing_cols: raise ValueError(f"Missing expected feature columns: {missing_cols}")
        X_meta = df_meta[feature_columns].copy()
        y_meta = df_meta['actual_points']
        logger.info(f"Prepared meta-model data: X_meta shape {X_meta.shape}, y_meta shape {y_meta.shape}")

        # 6. Define Optuna Objective Function
        # This function will be called by Optuna for each trial
        def objective(trial: optuna.trial.Trial) -> float:
            """Objective function for Optuna hyperparameter optimization."""

            # Define hyperparameters to search using trial object
            lgbm_params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 250, step=25),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 8, 64),
                'max_depth': trial.suggest_int('max_depth', 3, 10), # Explicit max_depth can help
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True), # L1
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True), # L2
                'random_state': GLOBAL_RANDOM_STATE,
                'n_jobs': 1, # Run single LGBM instance per trial job
                'verbose': -1 # Suppress verbose output from LGBM itself
            }

            # Create the base estimator with suggested params
            lgbm_meta_estimator = lgb.LGBMRegressor(**lgbm_params)

            # Define the pipeline for this trial
            pipeline_trial = Pipeline([
                ('scaler', StandardScaler()),
                ('lgbm', lgbm_meta_estimator)
            ])

            # Cross-validation using TimeSeriesSplit
            groups_meta = df_meta["series_key"].values
            n_groups = len(np.unique(groups_meta))
            logger.info(f"[DEBUG] unique series_key groups: {n_groups}")
            if n_groups < 2:
                return float("inf")

            # chronological leave-one-series-out splitter
            tscv = GroupTimeSeriesSplit(n_splits=n_groups - 1)
            fold_rmses = []

            for fold_no, (tr_idx, te_idx) in enumerate(
                tscv.split(X_meta, groups=groups_meta), 1):
                X_train_fold, X_test_fold = X_meta.iloc[tr_idx], X_meta.iloc[te_idx]
                y_train_fold, y_test_fold = y_meta.iloc[tr_idx], y_meta.iloc[te_idx]

                if X_train_fold.empty or X_test_fold.empty:
                    logger.warning(f"Skipping Optuna CV fold {fold_no} in trial {trial.number} because train or test set is empty.")
                    continue

                try:
                    # --- Crucial: Clone the pipeline for each fold ---
                    # This ensures that each fold fits a fresh model state
                    # without affecting other folds or the base pipeline_trial object.
                    pipeline_fold = clone(pipeline_trial)

                    # Fit the cloned pipeline on the training data for this fold
                    pipeline_fold.fit(X_train_fold, y_train_fold)

                    # Predict on the test fold
                    y_pred_fold = pipeline_fold.predict(X_test_fold)
                    y_pred_fold = np.maximum(0, y_pred_fold) # Ensure non-negative predictions

                    # Calculate RMSE for this fold
                    rmse = rmse_score(y_test_fold, y_pred_fold)
                    fold_rmses.append(rmse)

                except Exception as e:
                    # Log error and potentially return a high RMSE or let Optuna handle
                    logger.error(f"Error during Optuna CV fold {fold_no+1} in trial {trial.number}: {e}", exc_info=False) # Less verbose logging inside objective
                    # Return a large value to penalize this trial if a fold fails badly
                    # Or re-raise if the error should stop the trial (Optuna catches it)
                    # For simplicity, we'll skip the fold if it errors. If all folds error,
                    # nanmean below will handle it (returning NaN, Optuna treats as failure).
                    # Consider Optuna's trial.report() and trial.should_prune() for more advanced handling.
                    pass # Skip fold on error

            # Calculate the average RMSE across successful folds for this trial
            if not fold_rmses:
                logger.warning(f"Trial {trial.number} had no successful CV folds.")
                return float('inf') # Return high value if no folds succeeded

            average_rmse = np.mean(fold_rmses) # Use mean, nanmean might hide issues
            logger.debug(f"Trial {trial.number}: Avg RMSE = {average_rmse:.4f} with params {trial.params}")
            return average_rmse # Optuna minimizes this value

        # 7. Run Optuna Optimization
        
        # Add sampler for reproducibility
        sampler = optuna.samplers.TPESampler(seed=GLOBAL_RANDOM_STATE)
        study = optuna.create_study(direction='minimize', sampler=sampler)

        # Pass data implicitly via scope, or explicitly using functools.partial if preferred
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

        logger.info("Optuna optimization finished.")
        logger.info(f"Best Trial Number: {study.best_trial.number}")
        logger.info(f"Best Value (RMSE): {study.best_value:.4f}")
        logger.info(f"Best Parameters: {study.best_params}")

        # 8. Train final meta-model pipeline on all data using best parameters
        logger.info("Training final LGBM meta-model pipeline on all data using best Optuna parameters...")

        # Get the best hyperparameters found by Optuna
        best_params = study.best_params

        # Create the final pipeline with the base LGBM estimator
        # We need to add the fixed parameters back in
        final_lgbm_params = best_params.copy()
        final_lgbm_params['random_state'] = GLOBAL_RANDOM_STATE
        final_lgbm_params['n_jobs'] = -1 # Use all cores for final fit
        final_lgbm_params['verbose'] = -1 # Keep it quiet

        final_meta_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lgbm', lgb.LGBMRegressor(**final_lgbm_params)) # Use best params
        ])

        # Fit the final pipeline on the entire dataset
        final_meta_pipeline.fit(X_meta, y_meta)
        logger.info("Final meta-model pipeline trained successfully.")

        # Optional: Evaluate final model on the full dataset (in-sample) - use with caution
        y_pred_final = final_meta_pipeline.predict(X_meta)
        y_pred_final = np.maximum(0, y_pred_final)
        final_rmse = rmse_score(y_meta, y_pred_final)
        final_mae = mean_absolute_error(y_meta, y_pred_final)
        final_r2 = r2_score(y_meta, y_pred_final)
        logger.info("--- Final Model In-Sample Evaluation ---")
        logger.info(f"Final RMSE: {final_rmse:.4f}")
        logger.info(f"Final MAE:  {final_mae:.4f}")
        logger.info(f"Final R²:   {final_r2:.4f}")
        logger.info("---------------------------------------")
        # Note: The best CV score (study.best_value) is a better estimate of generalization performance.

        # Log feature importances from the final model
        try:
            final_lgbm_model = final_meta_pipeline.named_steps['lgbm']
            if hasattr(final_lgbm_model, 'feature_importances_'):
                f_names = getattr(final_lgbm_model, 'feature_name_', feature_columns)
                importances = final_lgbm_model.feature_importances_
                if len(f_names) == len(importances):
                     importance_info = ", ".join([f"{name}={imp}" for name, imp in sorted(zip(f_names, importances), key=lambda item: item[1], reverse=True)])
                     logger.info(f"Feature Importances: {importance_info}")
                else:
                    logger.warning("Mismatch between feature names and importance values count.")
        except Exception as fe_err:
            logger.warning(f"Could not retrieve feature importances: {fe_err}")

        # 9. Save the final meta-model PIPELINE and feature list (Unchanged)
        model_filename = f"meta_model_lgbm_optuna_pipeline_playoffs_{target_stat}.joblib"
        model_path = artifacts_dir / model_filename
        joblib.dump(final_meta_pipeline, model_path)
        logger.info(f"Saved trained meta-model pipeline to: {model_path}")

        features_filename = f"meta_model_features_playoffs_{target_stat}.joblib"
        features_path = artifacts_dir / features_filename
        joblib.dump(feature_columns, features_path)
        logger.info(f"Saved meta-model feature list to: {features_path}")

        # Return the best CV score found by Optuna
        return {'best_cv_rmse': study.best_value}

    except sqlite3.Error as db_err:
        logger.exception("Database error during meta-model training: %s", db_err)
        sys.exit(1)
    except ValueError as val_err:
        logger.error("Value error during meta-model training: %s", val_err, exc_info=True)
        sys.exit(1)
    except optuna.exceptions.OptunaError as opt_err:
         logger.exception("Optuna optimization error: %s", opt_err)
         sys.exit(1)
    except Exception as e:
        logger.exception("An unexpected error occurred during meta-model training: %s", e)
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

# ─────────────────────────── CLI Entry Point ───────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train-playoffs-meta-model-optuna", # Updated name
        description="Train an LGBM meta-model with Optuna using base model OOF predictions and derived meta-features for NBA playoffs.", # Updated description
    )

    parser.add_argument(
        "--db-file",
        type=str,
        default=DEFAULT_DB_FILE,
        help=f"Path to the SQLite database file (default: {DEFAULT_DB_FILE})."
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help=f"Target variable (e.g., 'pts') used by base models (default: {DEFAULT_TARGET})."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=str(DEFAULT_META_ARTIFACTS_DIR),
        help=f"Directory to save the trained Optuna-tuned LGBM meta-model artifact and features (default: {DEFAULT_META_ARTIFACTS_DIR})." # Updated help text
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=N_OPTUNA_TRIALS,
        help=f"Number of Optuna optimization trials (default: {N_OPTUNA_TRIALS})."
    )

    args = parser.parse_args()

    # Update global constant if provided via CLI
    N_OPTUNA_TRIALS = args.trials

    # Convert artifacts_dir string to Path object
    artifacts_path = Path(args.artifacts_dir)

    # Run the training process
    results = train_meta_model(
        db_file=args.db_file,
        target_stat=args.target,
        artifacts_dir=artifacts_path
    )
    if results:
      logger.info(f"Optuna optimization completed. Best Cross-Validated RMSE found: {results.get('best_cv_rmse', 'N/A'):.4f}")

    logger.info("Optuna-tuned LGBM meta-model training script finished.") # Updated message
    sys.exit(0)