#!/usr/bin/env python3
"""
player_props_playoffs_xgboost_from_cache_v2.py

Loads pre-computed playoff features from the LightGBM script's cache
and trains a standalone XGBoost playoff player prop model.
Uses XGBoost's internal feature importance for pruning instead of SHAP.

Assumes the feature cache corresponding to LGBM_PLAYOFF_FEATURE_CACHE_VERSION exists.

MODIFIED (v2): Implements a white-list approach for feature selection in
prepare_model_data, mimicking the original LightGBM script's feature
name generation, to prevent data leakage from unexpected columns in the cache
(like the raw target variable).
"""

from __future__ import annotations

# ───────────────────────────── stdlib ──────────────────────────────
import argparse
import hashlib
import logging
import random
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union # Use Optional and Union
import warnings # To suppress specific warnings cleanly
import glob # To find the cache file
import importlib.util # For checking torch

# ─────────────────────────── 3rd‑party ─────────────────────────────
import joblib
import xgboost as xgb # Import XGBoost
import numpy as np
import optuna
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sql_player_props_playoffs import VIEW_SETUP_SQL, PLAYER_GAMES_SQL
from features_player_props import BASE_STATS, TEAM_STATS, OPP_STATS, PLAYER_ADVANCED_TRACKING, INTERACTION_NAMES, EXTRA_FEATURES, CAT_FLAGS    # <- new line


# ---> ADD THIS SNIPPET HERE <---
# Suppress the specific Optuna UserWarning about reporting the same step multiple times per trial
# This occurs due to the pruning callback being used inside the inner CV loop of the objective function
warnings.filterwarnings(
    'ignore',
    message="The reported value is ignored because this `step`.*is already reported.",
    category=UserWarning,
    module='optuna\\.trial\\._trial' # Be specific about the source module
)
# ------------------------------->

# ────────────────────────────── logging ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Use a distinct logger name for this XGBoost script
logger = logging.getLogger("PlayoffModelXGBoost_v8") # Updated logger name

# ──────────────────────────── reproducibility ──────────────────────
GLOBAL_RANDOM_STATE = 1337
np.random.seed(GLOBAL_RANDOM_STATE)
random.seed(GLOBAL_RANDOM_STATE)

# ───────────────── FEATURE CACHE (FROM LIGHTGBM SCRIPT) ───────────
# Hardcoded path and version from the LightGBM script
# NOTE: Ensure this matches the version used to generate the cache you want to load.
LGBM_PLAYOFF_FEATURE_CACHE_VERSION = "v17_playoffs_with_prior_avgs"
LGBM_PLAYOFF_FEATURE_CACHE_DIR = Path(".fe_cache_playoffs_only_v17")

# ──────────────── XGBOOST-SPECIFIC Artifact Paths & Constants ─────
SCRIPT_DIR_XGBOOST = Path(__file__).parent.resolve()
DEFAULT_ARTIFACTS_DIR_XGBOOST = SCRIPT_DIR_XGBOOST / "xgboost_playoff_artifacts_v8" # Separate dir v2
DEFAULT_DB_FILE = "nba.db" # Default DB name (same as source)

# --- Path Generation Functions (XGBoost Model Only) ---
# (Functions remain the same, but base dir might change if needed)
def get_xgboost_model_path(target: str, base_dir: Path = DEFAULT_ARTIFACTS_DIR_XGBOOST) -> Path:
    return base_dir / f"modelXGBoost_playoffs_{target}.joblib"
def get_xgboost_features_path(target: str, base_dir: Path = DEFAULT_ARTIFACTS_DIR_XGBOOST) -> Path:
    return base_dir / f"featuresXGBoost_playoffs_{target}.joblib"
def get_xgboost_cat_features_path(target: str, base_dir: Path = DEFAULT_ARTIFACTS_DIR_XGBOOST) -> Path:
    return base_dir / f"categorical_featuresXGBoost_playoffs_{target}.joblib"
def get_xgboost_study_path(target: str, base_dir: Path = DEFAULT_ARTIFACTS_DIR_XGBOOST) -> Path:
    return base_dir / f"optuna_xgb_study_playoffs_{target}.db"
def get_xgboost_pruned_features_csv_path(target: str, base_dir: Path = DEFAULT_ARTIFACTS_DIR_XGBOOST) -> Path:
    return base_dir / f"xgb_kept_features_importances_playoffs_{target}.csv"

# Constants
TOP_MI_FEATURES = 800
TOP_IMPORTANCE_FEATURES = 85
# Consistent constant from LGBM FE logic
MIN_PLAYOFF_GAMES_FOR_CAREER_AVG = 5
# Define rolling windows here to match feature name generation
DEFAULT_ROLLING_WINDOWS = [2, 5, 10]


# ─────────────────────── Helper Utilities (Unchanged) ───────────────────
def _hash_raw_df(df: pd.DataFrame) -> str:
    """Deterministic 12‑char SHA‑1 prefix for raw dataframe content."""
    # Sort columns before hashing for consistency
    df_sorted = df.sort_index(axis=1)
    digest = hashlib.sha1(pd.util.hash_pandas_object(df_sorted, index=True).values).hexdigest()
    return digest[:12]

def _bulk_concat(df: pd.DataFrame, new_cols: Dict[str, pd.Series]) -> pd.DataFrame:
    """Concatenate many Series to *df* in a single, defragmented pass."""
    if not new_cols:
        return df
    # Filter out any new columns that already exist in df
    new_cols_filtered = {k: v for k, v in new_cols.items() if k not in df.columns}
    if not new_cols_filtered:
        return df
    block = pd.DataFrame(new_cols_filtered, index=df.index)
    # Ensure indices align perfectly before concat
    if not df.index.equals(block.index):
        logger.warning("_bulk_concat: Aligning indices before concatenation.")
        block = block.reindex(df.index)
    return pd.concat([df, block], axis=1, copy=False).copy() # Final copy for defragmentation

# ───────────────────── Identity Scaler Shim (Unchanged) ───────────────────
class _NoOpScaler:
    """A drop‑in replacement for StandardScaler that does nothing."""
    def fit(self, X, y=None): return self
    def transform(self, X): return X
    def fit_transform(self, X, y=None): return X
    def inverse_transform(self, X): return X

class GroupTimeSeriesSplit:
    """
    Same idea as sklearn.model_selection.TimeSeriesSplit, but the atomic unit
    is an ordered *group* (our series_key) instead of individual rows.
    """
    def __init__(self, n_splits: int):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("`groups` array is required.")

        if 'game_date' not in X.columns:
            raise ValueError("X must contain 'game_date' for chronological ordering")

        # --- safety casts ---
        dates   = pd.to_datetime(X['game_date'].values, errors='raise')
        grp_idx = pd.Index(groups, name='series_key')

        # earliest date each group appears
        group_first_date = (
            pd.Series(dates, index=grp_idx)
            .groupby(level=0, observed=True)
            .min()
            .sort_values()
        )

        ordered_groups = group_first_date.index.to_numpy()

        n_groups = len(ordered_groups)
        test_size = max(1, n_groups // (self.n_splits + 1))

        for i in range(self.n_splits):
            test_start  = n_groups - (i + 1) * test_size
            test_groups = ordered_groups[test_start : test_start + test_size]
            train_groups = ordered_groups[:test_start]

            train_idx = np.flatnonzero(np.isin(groups, train_groups))
            test_idx  = np.flatnonzero(np.isin(groups, test_groups))
            yield train_idx, test_idx

# ─────────────────── XGBoost Playoff Model Class (v2) ────────────────────
class XGBoostPlayoffModel:
    """
    Loads pre-computed playoff features and trains an XGBoost model.
    Uses XGBoost feature importance for pruning.
    Implements white-list feature selection to prevent leakage.
    """
    def __init__(
        self,
        db_file: str = DEFAULT_DB_FILE,
        target: str = "pts",
        xgboost_artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR_XGBOOST,
        random_state: int = GLOBAL_RANDOM_STATE,
        use_gpu: bool = False,
        rolling_windows: Optional[List[int]] = None, # Added for feature name generation
    ):
        self.db_file = db_file
        self.target = target
        self.random_state = random_state
        self.use_gpu = use_gpu
        # Store rolling windows used during FE (must match LGBM script)
        self.rolling_windows = rolling_windows or DEFAULT_ROLLING_WINDOWS

        # Convert string paths to Path objects
        self.xgboost_artifacts_dir = Path(xgboost_artifacts_dir).resolve()
        self.xgboost_artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Set instance-specific paths based on target and base directories
        self.model_path = get_xgboost_model_path(self.target, self.xgboost_artifacts_dir)
        self.features_path = get_xgboost_features_path(self.target, self.xgboost_artifacts_dir)
        self.cat_features_path = get_xgboost_cat_features_path(self.target, self.xgboost_artifacts_dir)
        self.study_path = get_xgboost_study_path(self.target, self.xgboost_artifacts_dir)
        self.kept_features_csv_path = get_xgboost_pruned_features_csv_path(self.target, self.xgboost_artifacts_dir)

        # Model and feature attributes
        self.model: Optional[xgb.XGBRegressor] = None
        self.scaler: _NoOpScaler = _NoOpScaler() # Keep NoOp scaler for consistency
        self.feature_names: List[str] = [] # Final features after pruning
        self.initial_feature_names: List[str] = [] # Features from white-list validation
        self.categorical_features: List[str] = [] # Potential categorical features identified from white-list
        self.final_categorical_features: List[str] = [] # Actual categoricals in final feature set (names only)
        self.model_run_id: Optional[str] = None

        logger.info(f"Initialized XGBoost Playoff Model v2 for target '{self.target}'.")
        logger.info(f"XGBoost artifact directory: {self.xgboost_artifacts_dir}")
        logger.info(f"XGBoost Optuna study path: {self.study_path}")
        logger.info(f"Expecting feature cache from: {LGBM_PLAYOFF_FEATURE_CACHE_DIR} (Version: {LGBM_PLAYOFF_FEATURE_CACHE_VERSION})")
        logger.info(f"Using rolling windows for feature name generation: {self.rolling_windows}")

    # ---------------------------------------------------------------------#
    # Data Retrieval (Needed for averages and metadata - Unchanged)
    # ---------------------------------------------------------------------#
    def retrieve_data(self) -> pd.DataFrame:
        """Load the joined raw data from SQLite. Loads ALL data (reg + playoff)."""
        logger.info("Reading ALL game data (including regular season) from %s", self.db_file)
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)

            conn.executescript(VIEW_SETUP_SQL)

            df = pd.read_sql_query(PLAYER_GAMES_SQL, conn, parse_dates=["game_date"])
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            raise
        finally:
            if conn: conn.close()

        if df.columns.duplicated().any():
            dupes = df.columns[df.columns.duplicated()].unique().tolist()
            logger.warning("Dropping duplicate columns: %s", ", ".join(dupes))
            df = df.loc[:, ~df.columns.duplicated()]

        # ─────────────── Vegas-odds derived features ───────────────
        # Convert raw odds columns to numeric and create probability / script fields
        odds_numeric_cols = [
            "moneyline_odds", "spread_line", "total_line",
            "over_odds", "under_odds"
        ]
        for col in odds_numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                logger.warning("Vegas odds column '%s' missing after join.", col)

        def _implied_prob(usd_odds: pd.Series) -> pd.Series:
            """American odds → implied probability (vectorised)."""
            return np.where(
                usd_odds < 0,
                -usd_odds / (-usd_odds + 100),
                100 / (usd_odds + 100),
            )

        # money-line based features
        if "moneyline_odds" in df.columns:
            df["p_ml"]     = _implied_prob(df["moneyline_odds"])
            df["fav_flag"] = (df["moneyline_odds"] < 0).astype("int8")
        else:
            df["p_ml"]     = 0.0
            df["fav_flag"] = 0

        # spread / total based features
        if {"spread_line", "total_line"}.issubset(df.columns):
            df["abs_spread"] = np.abs(df["spread_line"])
            df["teim"]       = (df["total_line"] / 2) - (df["spread_line"] / 2)  # team implied score
            df["opim"]       = (df["total_line"] / 2) + (df["spread_line"] / 2)  # opponent implied score
        else:
            df["abs_spread"] = 0.0
            df["teim"]       = 0.0
            df["opim"]       = 0.0
        
        # Fill any remaining NaNs in the odds-related columns
        odds_cols = ["p_ml", "abs_spread", "teim", "opim", "over_odds", "under_odds"]
        existing_odds = [c for c in odds_cols if c in df.columns]
        df[existing_odds] = df[existing_odds].fillna(0)

        if df.empty:
            raise ValueError("No rows returned from the database.")

        # --- Playoff flag defaults & Type Conversion ---
        playoff_cols = [
            "is_playoffs", "playoff_round", "series_number",
            "series_record", "is_elimination_game", "has_home_court",
        ]
        for col in playoff_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype("int16")
            else:
                logger.warning(f"Expected playoff column '{col}' not found. Adding with default 0.")
                df[col] = 0

        # --- Derive 'win' column ---
        if 'team_wl' in df.columns:
            df['win'] = df['team_wl'].apply(lambda x: 1 if isinstance(x, str) and x.upper() == 'W' else 0).astype(int)
        else:
            logger.warning("'team_wl' column not found. Creating 'win' column with default 0.")
            df['win'] = 0

        # --- Ensure 'season' column ---
        if 'season' in df.columns:
            df['season'] = df['season'].astype(str)
        else:
            logger.error("'season' column not found in data. Season average calculation WILL FAIL.")
            raise ValueError("'season' column is required but missing.")

        # --- Ensure opponent_vs_player columns ---
        ovp_cols = ["opponent_vs_player_fgm_allowed", "opponent_vs_player_fga_allowed", "opponent_vs_player_pts_allowed"]
        for col in ovp_cols:
            if col not in df.columns:
                logger.warning(f"Column '{col}' missing. Adding with default 0.")
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
        logger.info("Loaded %d total game rows (before playoff filter).", len(df))
        return df

    # ---------------------------------------------------------------------#
    # Feature Loading (Replaces Feature Engineering - Unchanged)
    # ---------------------------------------------------------------------#
    def load_cached_features(self, raw_df_playoffs: pd.DataFrame, averages_df: pd.DataFrame) -> pd.DataFrame:
        """
        Loads engineered playoff features from the cache generated by the LGBM script.
        Finds the appropriate cache file based on version and merges pre-calculated averages.
        """
        logger.info(f"Attempting to load cached playoff features (Version: {LGBM_PLAYOFF_FEATURE_CACHE_VERSION})...")

        cache_pattern = f"features_playoffs_*_{LGBM_PLAYOFF_FEATURE_CACHE_VERSION}.parquet"
        cache_files = list(LGBM_PLAYOFF_FEATURE_CACHE_DIR.glob(cache_pattern))

        if not cache_files:
            logger.error(f"No cache file found matching pattern '{cache_pattern}' in {LGBM_PLAYOFF_FEATURE_CACHE_DIR}")
            raise FileNotFoundError(f"Playoff feature cache not found for version {LGBM_PLAYOFF_FEATURE_CACHE_VERSION}")

        cache_path = cache_files[0]
        logger.info(f"Found cache file: {cache_path}")

        try:
            df_engineered_playoffs_cached = pd.read_parquet(cache_path)
            logger.info(f"Loaded {len(df_engineered_playoffs_cached)} rows from cache.")
        except Exception as e:
            logger.error(f"Failed to read cache file {cache_path}: {e}")
            raise

        # Ensure correct dtypes after loading
        if 'season' in df_engineered_playoffs_cached.columns: df_engineered_playoffs_cached['season'] = df_engineered_playoffs_cached['season'].astype(str)
        if 'game_date' in df_engineered_playoffs_cached.columns: df_engineered_playoffs_cached['game_date'] = pd.to_datetime(df_engineered_playoffs_cached['game_date'])

        # Merge the averages calculated externally
        logger.info("Merging pre-calculated averages onto CACHED engineered playoff data...")
        career_avg_col = f"{self.target}_career_avg"
        season_avg_col = f"{self.target}_season_avg"
        avg_cols_to_merge = [career_avg_col, season_avg_col]

        if not df_engineered_playoffs_cached.index.equals(averages_df.index):
            logger.warning("Cached playoff features index differs from averages index. Reindexing cached data.")
            df_engineered_playoffs_cached = df_engineered_playoffs_cached.reindex(averages_df.index)

        # Drop average columns if they exist in cache (shouldn't) and join
        df_engineered_playoffs_cached = df_engineered_playoffs_cached.drop(columns=avg_cols_to_merge, errors='ignore')
        df_engineered_playoffs = df_engineered_playoffs_cached.join(averages_df[avg_cols_to_merge], how='left')

        # Impute NaNs in merged averages
        if df_engineered_playoffs[career_avg_col].isnull().any() or df_engineered_playoffs[season_avg_col].isnull().any():
            nan_career = df_engineered_playoffs[career_avg_col].isnull().sum()
            nan_season = df_engineered_playoffs[season_avg_col].isnull().sum()
            logger.warning(f"NaNs found/introduced after merging averages onto cached data: {nan_career} in career_avg, {nan_season} in season_avg. Imputing with 0.")
            df_engineered_playoffs[career_avg_col] = df_engineered_playoffs[career_avg_col].fillna(0)
            df_engineered_playoffs[season_avg_col] = df_engineered_playoffs[season_avg_col].fillna(0)

        logger.info("Finished loading cached features and merging averages.")
        # IMPORTANT: The raw target column (e.g., 'pts') might still be in df_engineered_playoffs here!
        return df_engineered_playoffs

    # ---------------------------------------------------------------------#
    # Data Prep for Modelling (MAJOR CHANGE: White-list approach)
    # ---------------------------------------------------------------------#
    def prepare_model_data(self, df_engineered: pd.DataFrame
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Build the modelling matrix (X, y) from the loaded engineered playoff dataframe.
        Uses a WHITE-LIST approach based on expected feature name patterns from the
        original LGBM FE script to prevent data leakage. Determines the initial
        feature set and potential categorical features. Converts categoricals to codes.

        Args:
            df_engineered: DataFrame containing loaded features + merged averages.

        Returns:
            Tuple: (df_model, X_initial, y)
                df_model: The dataframe subset used for modeling (target non-NaN).
                X_initial: DataFrame of initial features (categoricals as codes).
                y: Series of the target variable.
        """
        logger.info("Preparing XGBoost model data matrix using WHITE-LIST feature selection...")
        if df_engineered.empty:
            logger.warning("Input DataFrame to prepare_model_data is empty.")
            return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)

        # --- Define expected feature name patterns (mirroring LGBM FE) ---
        # Copied/adapted from LGBM script's prepare_model_data
        base_stats = BASE_STATS
        player_advanced_tracking = PLAYER_ADVANCED_TRACKING
        team_stats = TEAM_STATS
        opp_stats = OPP_STATS
        interactions = INTERACTION_NAMES
        # Combine stats for generating feature names based on patterns
        stats_for_fe_naming = sorted(list(set(
             [s for s in (base_stats + player_advanced_tracking + team_stats + opp_stats)]
        )))
        # Explicitly add target to the list for *target-specific* features (home/away, hot/cold)
        stats_for_fe_naming_with_target = sorted(list(set(stats_for_fe_naming + [self.target])))


        # Helper for polynomial feature names (base, sq) - used in LGBM FE
        def _poly_sq(name: str) -> list[str]:
            return [name, f"{name}_sq"]

        # --- Generate WHITE-LIST of potential feature names ---
        features_potential: list[str] = []

        # 1. Pre-calculated Averages (merged onto cached data) - ADD THESE DIRECTLY
        features_potential += _poly_sq(f"{self.target}_career_avg")
        features_potential += _poly_sq(f"{self.target}_season_avg")

        # 2. Rolling / Trend / Playoff Career Avg Features (generated by LGBM FE)
        for stat in stats_for_fe_naming_with_target: # Use list that includes target here
            # Add defaulted playoff career avg for target, standard for others
            if stat == self.target:
                features_potential += _poly_sq(f"{stat}_playoff_career_avg_defaulted")
            else:
                # Only add non-defaulted if the stat is NOT the target
                 if stat in stats_for_fe_naming: # Ensure it's not the target itself
                     features_potential += _poly_sq(f"{stat}_playoff_career_avg")

            # Add standard rolling features (for all stats including target's derived)
            for w in self.rolling_windows: features_potential += _poly_sq(f"{stat}_rolling_{w}")
            trend_suffixes = ["trend", "trend_ewm", "std", "acceleration", "trend_std"]
            for suff in trend_suffixes: features_potential += _poly_sq(f"{stat}_{suff}")

        # 3. Interactions: Base term, Rolling / Trend / Playoff Career Avg (generated by LGBM FE)
        interaction_trend_suffixes = ["trend", "trend_ewm", "std"]
        for inter in interactions:
            features_potential.append(inter) # Keep shifted raw interaction term
            # Non-defaulted playoff avg for interactions
            features_potential += _poly_sq(f"{inter}_playoff_career_avg")
            for w in self.rolling_windows: features_potential += _poly_sq(f"{inter}_rolling_{w}")
            for suff in interaction_trend_suffixes: features_potential += _poly_sq(f"{inter}_{suff}")

        # 4. Home/Away Features (TARGET ONLY - generated by LGBM FE)
        for loc in ("home", "away"):
            for w in self.rolling_windows: features_potential += _poly_sq(f"{self.target}_{loc}_rolling_{w}")
            # Add the defaulted home/away playoff average name for the target
            features_potential += _poly_sq(f"{self.target}_{loc}_playoff_career_avg_defaulted")

        # 5. Playoff Context Features (generated by LGBM FE)
        playoff_context_features = [
            "series_tied"
        ]
        features_potential.extend(playoff_context_features)

        # 6. Extra Context Features (generated by LGBM FE)
        
        extra_features = EXTRA_FEATURES
        # Note: use_extra_features flag isn't available here, assume they were generated if present
        features_potential += extra_features

        # --- De-duplicate generated potential list ---
        features_potential = sorted(list(set(features_potential)))
        logger.info(f"Generated white-list of {len(features_potential)} potential feature names.")

        # --- Validate white-list against actual columns in the loaded DataFrame ---
        validated_features = []
        missing_potential = []
        unexpected_columns = [] # Columns in data not in our generated white-list

        df_columns_set = set(df_engineered.columns)
        potential_features_set = set(features_potential)

        # Check which potential features actually exist in the loaded data
        for f in features_potential:
            if f in df_columns_set:
                validated_features.append(f)
            else:
                missing_potential.append(f)

        # Check for columns in the data that weren't in our expected list
        # Exclude known non-feature columns explicitly here
        known_non_features = {
            self.target, 'game_id', 'player_id', 'team_id', 'opponent_team_id',
            'game_date', 'season', 'team_wl'
            # Add any other known identifiers or raw columns present in cache but not features
        }
        for col in df_engineered.columns:
             # If a column is not a known non-feature AND not in our generated potential list
             if col not in known_non_features and col not in potential_features_set:
                 unexpected_columns.append(col)


        if missing_potential:
            head = ", ".join(missing_potential[:20])
            more = " …" if len(missing_potential) > 20 else ""
            logger.warning(
                f"Found {len(validated_features)} expected features in loaded data. "
                f"{len(missing_potential)} potential features from white-list generation "
                f"were MISSING: {head}{more}"
            )
        else:
             logger.info(f"Found all {len(validated_features)} generated potential features in loaded data.")

        if unexpected_columns:
            head = ", ".join(unexpected_columns[:20])
            more = " …" if len(unexpected_columns) > 20 else ""
            logger.warning(
                f"Found {len(unexpected_columns)} UNEXPECTED columns in loaded data "
                f"(not in white-list or known non-features): {head}{more}"
            )
            # CRITICAL CHECK: Ensure the raw target is not in the unexpected list if it shouldn't be
            if self.target in unexpected_columns:
                 logger.error(f"FATAL: Raw target '{self.target}' was found as an UNEXPECTED column. Leakage likely occurred before this point or white-list is wrong.")
                 # Raise error here or let it fail later, but this is a sign something is wrong upstream

        # --- Persist INITIAL lists (based on white-list validation) ---
        self.initial_feature_names = sorted(validated_features)
        if not self.initial_feature_names:
            logger.error("No features available after white-list validation against loaded DataFrame columns.")
            raise ValueError("White-list feature validation resulted in empty feature set.")

        # --- Identify potential categorical features (Check against validated features) ---
        # Use the same potential list as before, but filter against the validated initial_feature_names
        cat_flags_potential = CAT_FLAGS
        self.categorical_features = [c for c in cat_flags_potential if c in self.initial_feature_names]
        logger.info(f"Identified {len(self.categorical_features)} potential categorical features from validated white-list.")


        # --- Build final model dataframe ---
        if self.target not in df_engineered.columns:
            raise ValueError(f"Target column '{self.target}' not found in engineered DataFrame for final prep.")

        # Drop rows with NaN target using the original df_engineered
        df_model = df_engineered.dropna(subset=[self.target]).copy()

        upper = df_model[self.target].quantile(0.995)        # top 0.5 %
        df_model[self.target] = df_model[self.target].clip(upper=upper)

        df_model = df_model.sort_values(
            ["season", "game_date", "playoff_round", "series_number"]
        ).reset_index(drop=True)

        # ---- build season/round/series grouping column identical to LGBM ----
        df_model["series_key"] = (
            df_model["season"].astype(str) + "_" +
            df_model["playoff_round"].astype(str) + "_" +
            df_model["series_number"].astype(str)
        )
        groups = df_model["series_key"].copy()

        # Select the validated features + target
        cols_to_select = ["game_date"] + self.initial_feature_names + [self.target]
        missing_in_model_df = [c for c in cols_to_select if c not in df_model.columns]
        if missing_in_model_df:
             # This shouldn't happen if df_model is just df_engineered with NaNs dropped
             logger.error(f"Columns missing after dropna, this indicates an issue: {missing_in_model_df}")
             raise ValueError(f"Data inconsistency: Columns missing after dropping NaN target: {missing_in_model_df}")

        df_model = df_model[cols_to_select]

        if df_model.empty:
            logger.warning("DataFrame is empty after dropping NaN target values.")
            return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)

        # --- Convert categorical columns to integer codes for XGBoost ---
        logger.info(f"Converting {len(self.categorical_features)} categoricals to integer codes...")
        X_initial = df_model[self.initial_feature_names].copy() # Use the validated white-list

        for col in self.categorical_features:
            if col not in X_initial.columns: continue # Skip if somehow missing

            if X_initial[col].isnull().any():
                fill_val = -1 # Use -1 for NaN codes
                try:
                    is_numeric_like = pd.api.types.is_numeric_dtype(X_initial[col]) or \
                                      pd.to_numeric(X_initial[col], errors='coerce').notna().all()
                    if is_numeric_like:
                          X_initial[col] = X_initial[col].fillna(fill_val)
                    else:
                          X_initial[col] = X_initial[col].astype(str).fillna(str(fill_val))
                except Exception:
                     X_initial[col] = X_initial[col].astype(str).fillna(str(fill_val))

            try:
                if not pd.api.types.is_numeric_dtype(X_initial[col]):
                    X_initial[col] = X_initial[col].astype(str)
                X_initial[col] = X_initial[col].astype('category').cat.codes.astype("int32")
                if (X_initial[col] == -1).any():
                    logger.debug(f"NaNs in '{col}' converted to code -1.")
            except Exception as e:
                logger.warning(f"Could not convert column '{col}' to category codes: {e}. Filling with 0.")
                X_initial[col] = 0

        y: pd.Series = df_model[self.target]

        logger.info(
            "Prepared initial XGBoost modelling dataset: %d rows | %d features | %d categoricals (as codes)",
            len(df_model), len(self.initial_feature_names), len(self.categorical_features),
        )
        # Final check for leakage BEFORE returning X_initial
        if self.target in X_initial.columns:
             logger.error(f"LEAKAGE CONFIRMED: Target '{self.target}' is present in the final X_initial columns!")
             raise ValueError(f"Data leakage detected: Target column '{self.target}' found in X_initial.")

        non_numeric_cols = X_initial.select_dtypes(exclude=np.number).columns
        if not non_numeric_cols.empty:
            logger.error(f"Non-numeric columns remain after categorical conversion: {non_numeric_cols.tolist()}. Check conversion logic.")
            for col in non_numeric_cols:
                X_initial[col] = pd.to_numeric(X_initial[col], errors='coerce').fillna(0)
            logger.warning("Forced remaining non-numeric columns to numeric (errors filled with 0).")

        return df_model, X_initial, y, groups


    # ---------------------------------------------------------------------#
    # Optuna Tuner (XGBoost Version - Unchanged from previous XGBoost script)
    # ---------------------------------------------------------------------#
    def _tune_with_optuna_xgboost(
            self, X: pd.DataFrame, y: pd.Series, groups: pd.Series
    ) -> tuple[dict[str, Any], xgb.XGBRegressor]: # Return fitted XGBRegressor for importance
        """
        Hyper‑parameter optimisation for XGBoost with chronological CV using xgb.train.
        Uses XGBoost study path. Operates on MI-filtered data.
        Returns best params and a *temporary* XGBRegressor model fitted on X with those params (for importance).
        Assumes X contains numerical features (categoricals converted to codes).
        """
        min_required_samples = 7
        n_samples = len(X)
        max_splits_possible_cv = n_samples - min_required_samples
        n_cv_splits_optuna = min(7, max(1, max_splits_possible_cv))

        if n_cv_splits_optuna < 2:
            logger.warning(f"Playoff data size ({n_samples}) allows only {n_cv_splits_optuna} split(s) for Optuna CV. Results may be less stable.")
        if max_splits_possible_cv < 1:
            raise ValueError(f"Not enough playoff data ({n_samples}, need >= {min_required_samples+1}) for even one Optuna CV split.")

        logger.info(f"Using {n_cv_splits_optuna} splits for XGBoost Optuna's internal CV.")
        tscv_outer = GroupTimeSeriesSplit(n_splits=n_cv_splits_optuna)


        X.columns = [str(c) for c in X.columns] # Ensure string column names

        def objective(trial: optuna.Trial) -> float:
            params = {
                # ───────── loss & metric ─────────
                # let Optuna pick between squared-error and smoothed Huber
                "objective": trial.suggest_categorical(
                    "objective", ["reg:squarederror", "reg:pseudohubererror"]
                ),
                "eval_metric": "rmse",

                # ───────── model type ─────────
                "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),

                # ───────── regularisation ─────────
                "lambda": trial.suggest_float("lambda", 1e-4, 10.0, log=True),   # L2
                "alpha":  trial.suggest_float("alpha",  1e-4, 10.0, log=True),   # L1
                "gamma":  trial.suggest_float("gamma",  1e-4,  1.0, log=True),   # min-split loss
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
                "max_delta_step":   trial.suggest_int("max_delta_step",   0,  5),# stabilises gradients

                # ───────── tree complexity ─────────
                "max_depth":  trial.suggest_int("max_depth", 3, 10),
                "max_leaves": trial.suggest_int("max_leaves", 8, 64),         # works with grow_policy=lossguide
                "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),

                # ───────── learning rate & rounds ─────────
                "eta": trial.suggest_float("eta", 0.015, 0.3, log=True),

                # ───────── column / row sampling ─────────
                "subsample":        trial.suggest_float("subsample",        0.30, 0.90),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.30, 0.90),

                # ───────── execution backend ─────────
                "tree_method": "gpu_hist" if self.use_gpu else "hist",
                "device":      "cuda"     if self.use_gpu else "cpu",
                "seed": self.random_state,
                "nthread": -1,
            }

            if params['booster'] == 'dart':
                params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
                params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
                params['rate_drop'] = trial.suggest_float('rate_drop', 1e-8, 0.5, log=True)
                params['skip_drop'] = trial.suggest_float('skip_drop', 1e-8, 0.5, log=True)

            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-rmse")

            fold_scores: list[float] = []
            fold_iters: list[int] = []
            num_boost_round_optuna = 50
            early_stopping_rounds_optuna = 20

            for fold_num, (tr_idx, va_idx) in enumerate(tscv_outer.split(X, groups=groups), 1):
                X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
                X_tr = X_tr.drop(columns=["game_date"])
                X_va = X_va.drop(columns=["game_date"])

                if X_tr.empty or X_va.empty:
                    logger.warning(f"Optuna CV fold {fold_num+1} resulted in empty train/val set. Skipping.")
                    continue

                X_tr.columns = [str(c) for c in X_tr.columns]
                X_va.columns = [str(c) for c in X_va.columns]
                dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=X_tr.columns.tolist())
                dvalid = xgb.DMatrix(X_va, label=y_va, feature_names=X_va.columns.tolist())
                evals = [(dtrain, 'train'), (dvalid, 'validation')]

                try:
                    bst = xgb.train(
                        params,
                        dtrain,
                        num_boost_round=num_boost_round_optuna,
                        evals=evals,
                        early_stopping_rounds=early_stopping_rounds_optuna,
                        callbacks=[pruning_callback],
                        verbose_eval=False
                    )
                    best_score = bst.best_score
                    best_iter = bst.best_iteration + 1
                    fold_scores.append(best_score)
                    fold_iters.append(best_iter)
                except optuna.TrialPruned as e:
                    logger.debug(f"Optuna Trial Pruned: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Error during XGBoost Optuna fold {fold_num+1} using xgb.train: {e}", exc_info=True)
                    fold_scores.append(np.nan)
                    fold_iters.append(num_boost_round_optuna)
                    continue

            valid_scores = [s for s in fold_scores if not np.isnan(s)]
            if not valid_scores: return float('inf')

            avg_score = float(np.mean(valid_scores))
            avg_iter = int(np.median(fold_iters)) if fold_iters else 100

            trial.set_user_attr("best_iter", avg_iter)
            return avg_score

        # --- Optuna Study Setup ---
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        if self.use_gpu:
            try:
                xgb.XGBRegressor(device='cuda').fit(X.iloc[:1], y.iloc[:1]) # Quick check
                logger.info("GPU detected and available for XGBoost.")
            except Exception as gpu_err:
                logger.warning(f"GPU requested but unavailable/error ({gpu_err}). Falling back to CPU.")
                self.use_gpu = False # Ensure subsequent steps use CPU

        self.study_path.parent.mkdir(parents=True, exist_ok=True)
        storage_name = f"sqlite:///{self.study_path}"
        study_name = f"xgb-playoffs-tuning-{self.target}-{datetime.now():%Y%m%d}"
        logger.info(f"Using XGBoost Playoff Optuna study '{study_name}' with storage: {storage_name}")
        study = optuna.create_study(
            study_name=study_name, storage=storage_name,
            direction="minimize", sampler=sampler, load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )

        # --- Optimize ---
        n_optuna_trials = 5 # Increased trials back
        timeout_optuna = 1200
        logger.info(f"Running XGBoost Optuna optimization for {n_optuna_trials} trials (timeout: {timeout_optuna}s)...")
        study.optimize(objective, n_trials=n_optuna_trials, timeout=timeout_optuna)

        # --- Get Best Params ---
        best_params_from_study = {
            **study.best_trial.params,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'seed': self.random_state,
            'tree_method': 'hist',
            'device': 'cuda' if self.use_gpu else 'cpu',
            'nthread': -1
        }
        n_estimators = study.best_trial.user_attrs.get("best_iter", 100)
        if n_estimators <= 0: n_estimators = 100
        best_params_for_regressor = best_params_from_study.copy()
        best_params_for_regressor["n_estimators"] = n_estimators

        logger.info(f"XGBoost Playoff Optuna found best params (RMSE: {study.best_value:.4f}) with {n_estimators} estimators.")

        # --- Train temporary XGBRegressor model for importance ---
        logger.info("Training temporary XGBoost model with best params for importance analysis...")
        X_numeric_fit = X.select_dtypes(include=np.number)
        if X_numeric_fit.shape[1] != X.shape[1]:
             logger.warning("Non-numeric columns detected before fitting importance probe model.")
             X_numeric_fit = X.copy()
             for col in X_numeric_fit.select_dtypes(exclude=np.number).columns:
                 X_numeric_fit[col] = pd.to_numeric(X_numeric_fit[col], errors='coerce').fillna(0)

        # Ensure columns are strings
        X_numeric_fit.columns = [str(c) for c in X_numeric_fit.columns]
        importance_probe_model = xgb.XGBRegressor(**best_params_for_regressor, enable_categorical=False)
        try:
            importance_probe_model.fit(X_numeric_fit, y, verbose=False)
        except Exception as e:
            logger.error(f"Error fitting temporary XGBoost model for importance: {e}")
            raise

        return best_params_for_regressor, importance_probe_model

    # ---------------------------------------------------------------------#
    # Training Pipeline (XGBoost Version - Using White-list Prep)
    # ---------------------------------------------------------------------#
    def train(self) -> dict:
        """
        End-to-end training routine for the XGBoost playoff model using cached features.

        Steps
        -----
        1.  Load all raw data from SQLite (needed for averages / metadata).
        2.  Compute shifted career & season averages for the target on the full set.
        3.  Filter rows to PLAYOFF games only.
        4.  Load engineered playoff features from the LightGBM cache.
        5.  Merge the pre-computed averages onto those cached features.
        6.  Build the modelling matrix with white-list feature selection +
            categorical encoding (`prepare_model_data`).
        7.  Mutual-information feature filter.
        8.  Nested-CV Optuna hyper-parameter search (XGBoost).
        9.  Feature pruning based on XGBoost gain importances.
        10. Re-fit a final model on the pruned feature set.
        11. Chronological CV for out-of-fold metrics + OOF predictions.
        12. Save artifacts and (optionally) export OOF predictions.
        """
        logger.info(
            f"=== XGBoost Standalone Playoff Training pipeline v2 started "
            f"for target: {self.target} ==="
        )

        # ───────────────────────────── 1-5. Load + averages + cache ─────────────────────────────
        raw_df_all = self.retrieve_data()

        target_col      = self.target
        career_avg_col  = f"{target_col}_career_avg"
        season_avg_col  = f"{target_col}_season_avg"

        if target_col not in raw_df_all.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
        if "season" not in raw_df_all.columns:
            raise ValueError("'season' column required.")

        logger.info("Calculating shifted career and season averages for '%s'...", target_col)

        career_avg_shifted = (
            raw_df_all.groupby("player_id")[target_col]
            .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        )
        season_avg_shifted = (
            raw_df_all.groupby(["player_id", "season"], observed=True, sort=False)[target_col]
            .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        )
        averages_df_all = pd.DataFrame(
            {career_avg_col: career_avg_shifted, season_avg_col: season_avg_shifted},
            index=raw_df_all.index,
        )

        logger.info("Filtering raw data for PLAYOFFS ONLY…")

        if "is_playoffs" not in raw_df_all.columns:
            raise ValueError("'is_playoffs' column missing.")
        if not pd.api.types.is_numeric_dtype(raw_df_all["is_playoffs"]):
            raw_df_all["is_playoffs"] = pd.to_numeric(
                raw_df_all["is_playoffs"], errors="coerce"
            ).fillna(0)

        raw_df_playoffs = raw_df_all[raw_df_all["is_playoffs"] == 1].copy()
        if raw_df_playoffs.empty:
            logger.warning("No playoff games found in the DB. Cannot train.")
            return {"error": "No playoff data found in source DB"}

        averages_df_playoffs = averages_df_all.loc[raw_df_playoffs.index]

        try:
            df_engineered_playoffs = self.load_cached_features(
                raw_df_playoffs, averages_df_playoffs
            )
        except Exception as e:
            logger.error("Cache load / merge failed: %s", e, exc_info=True)
            return {"error": f"Failed to load/process feature cache: {e}"}

        if df_engineered_playoffs.empty:
            logger.error("Cached playoff feature frame is empty. Abort.")
            return {"error": "Loading feature cache produced no data"}

        # ───────────────────────────── 6. Prep model matrix ─────────────────────────────
        df_model, X_initial, y, groups = self.prepare_model_data(df_engineered_playoffs)

        if X_initial.empty or y.empty:
            logger.warning("No data left after prepare_model_data.")
            return {"error": "No data post-prep (playoffs)"}

        # ───────────────────────────── 7. Mutual-information filter ─────────────────────
        logger.info("Computing mutual-information filter (top %d)…", TOP_MI_FEATURES)

        X_processed_mi = X_initial.copy()

        # drop completely-NaN columns
        all_nan_cols = X_processed_mi.columns[X_processed_mi.isna().all()]
        if len(all_nan_cols):
            logger.warning("Dropping %d all-NaN columns before MI: %s",
                           len(all_nan_cols), all_nan_cols[:5].tolist())
            X_processed_mi = X_processed_mi.drop(columns=all_nan_cols)

        if X_processed_mi.empty:
            raise ValueError("No features remain after dropping all-NaN columns.")

        # sample rows for MI calculation
        mi_sample_size = min(10_000, len(X_processed_mi))
        rng            = np.random.RandomState(self.random_state)
        sample_idx     = rng.choice(X_processed_mi.index, size=mi_sample_size, replace=False)

        X_numeric_mi_sample = X_processed_mi.loc[sample_idx]
        y_mi_sample         = y.loc[sample_idx]

        if X_numeric_mi_sample.isna().any().any():
            imp = SimpleImputer(strategy="median")
            X_numeric_mi_sample = pd.DataFrame(
                imp.fit_transform(X_numeric_mi_sample),
                columns=X_numeric_mi_sample.columns,
                index=X_numeric_mi_sample.index,
            ).fillna(0)

        X_mi_input = X_numeric_mi_sample.replace([np.inf, -np.inf], np.nan).fillna(0)

        def _choose_sq(mi_scores: pd.Series) -> list[str]:
            """Keep whichever of {base, base_sq} has the higher MI."""
            keep: list[str] = []
            visited: set[str] = set()
            for col in mi_scores.index:                      # already sorted
                base = col[:-3] if col.endswith("_sq") else col
                if base in visited:
                    continue
                twin = f"{base}_sq"
                if twin in mi_scores.index and twin != col:
                    better = col if mi_scores[col] >= mi_scores[twin] else twin
                    keep.append(better)
                else:
                    keep.append(col)
                visited.add(base)
            return keep

        mi_scores   = mutual_info_regression(X_mi_input, y_mi_sample,
                                             random_state=self.random_state)
        mi_series   = pd.Series(mi_scores, index=X_mi_input.columns).fillna(0)
        mi_series   = mi_series.loc[_choose_sq(mi_series)]            # de-dup base/_sq
        mi_kept     = mi_series[mi_series > 1e-7].nlargest(TOP_MI_FEATURES).index.tolist()

        X_filtered = X_processed_mi[mi_kept].copy()
        cats_after_mi = [c for c in self.categorical_features if c in X_filtered.columns]

        logger.info("MI filter kept %d features.", len(mi_kept))

        # ───────────────────────────── 8. Optuna tuning ─────────────────────────────
        logger.info("Starting Optuna hyper-parameter tuning with %d features…",
                    len(mi_kept))

        X_split = pd.concat(
            [df_model.loc[X_filtered.index, ["game_date"]], X_filtered],
            axis=1,
        )

        best_params, probe_model = self._tune_with_optuna_xgboost(
            X_split, y, groups.loc[X_filtered.index]
        )

        # ───────────────────────────── 9. Importance pruning ─────────────────────────
        logger.info("Pruning features by XGBoost gain importance…")

        importances = probe_model.feature_importances_
        feat_names  = probe_model.feature_names_in_
        importance_series = (
            pd.Series(importances, index=feat_names, name="xgboost_gain")
            .fillna(0)
            .sort_values(ascending=False)
        )
        kept_importance_series = importance_series.head(TOP_IMPORTANCE_FEATURES)
        self.feature_names = kept_importance_series.index.tolist()

        logger.info("Kept %d features after importance pruning (hard cap %d).",
                    len(self.feature_names), TOP_IMPORTANCE_FEATURES)

        # save CSV of kept features
        try:
            self.kept_features_csv_path.parent.mkdir(parents=True, exist_ok=True)
            kept_importance_series.to_frame().to_csv(self.kept_features_csv_path,
                                                     index_label="feature_name")
        except Exception as e:
            logger.error("Could not write kept-feature CSV: %s", e)

        # ───────────────────────────── 10. Final refit ──────────────────────────────
        logger.info("Refitting final XGBoost playoff model on pruned features…")

        date_frame    = df_model.loc[X_filtered.index, ["game_date"]]
        feature_frame = X_filtered[self.feature_names]
        final_X       = pd.concat([date_frame, feature_frame], axis=1)

        X_final_train = final_X.select_dtypes(include=np.number)
        X_final_train.columns = [str(c) for c in X_final_train.columns]

        self.model = xgb.XGBRegressor(**best_params, enable_categorical=False)
        self.model.fit(X_final_train, y, verbose=False)
        logger.info("Final XGBoost playoff model fitted.")

        # ───────────────────────────── 11. Chronological CV ─────────────────────────
        logger.info("Starting final chronological CV for OOF metrics…")

        X_cv = final_X.copy()
        y_cv = y.copy()

        min_train_samples_cv = 6
        n_samples_cv         = len(X_cv)
        max_splits_cv        = n_samples_cv - min_train_samples_cv
        n_cv_splits          = min(6, max(1, max_splits_cv))

        avg_metrics = {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
        oof_preds   = pd.Series(index=X_cv.index, dtype=float)

        if n_cv_splits >= 2:
            logger.info("Using %d splits.", n_cv_splits)
            tscv     = GroupTimeSeriesSplit(n_splits=n_cv_splits)
            groups_cv = groups.loc[X_cv.index].values
            metrics = {"rmse": [], "mae": [], "r2": []}

            for split_no, (tr_idx, te_idx) in enumerate(
                tscv.split(X_cv, groups=groups_cv), 1
            ):
                X_train, X_test = X_cv.iloc[tr_idx].copy(), X_cv.iloc[te_idx].copy()
                y_train, y_test = y_cv.iloc[tr_idx], y_cv.iloc[te_idx]

                # drop helper column so XGBoost never sees it
                for df_ in (X_train, X_test):
                    if "game_date" in df_.columns:
                        df_.drop(columns=["game_date"], inplace=True)

                if X_train.empty or X_test.empty:
                    logger.warning("Skipping CV fold %d: empty train/test.", split_no)
                    metrics["rmse"].append(np.nan)
                    metrics["mae"].append(np.nan)
                    metrics["r2"].append(np.nan)
                    continue

                X_train_num = X_train.select_dtypes(include=np.number)
                X_test_num  = X_test.select_dtypes(include=np.number)
                X_train_num.columns = [str(c) for c in X_train_num.columns]
                X_test_num.columns  = [str(c) for c in X_test_num.columns]

                fold_model = xgb.XGBRegressor(**best_params, enable_categorical=False)
                fold_model.fit(X_train_num, y_train, verbose=False)
                y_pred = fold_model.predict(X_test_num)

                oof_preds.loc[X_test.index] = y_pred
                metrics["rmse"].append(float(np.sqrt(mean_squared_error(y_test, y_pred))))
                metrics["mae"].append(float(mean_absolute_error(y_test, y_pred)))
                metrics["r2"].append(float(r2_score(y_test, y_pred)))

            avg_metrics = {
                k: float(np.nanmean(v)) if not np.all(np.isnan(v)) else np.nan
                for k, v in metrics.items()
            }

            logger.info("CV scores → RMSE %.4f | MAE %.4f | R² %.4f",
                        avg_metrics["rmse"], avg_metrics["mae"], avg_metrics["r2"])
        else:
            logger.warning(
                "Only %d split(s) possible with %d playoff rows. Skipping final CV.",
                n_cv_splits, n_samples_cv,
            )

        # ───────────────────────────── 12. Save + export OOF ────────────────────────
        logger.info("Saving playoff artifacts…")
        self.save_artifacts()

        self.model_run_id = (
            datetime.now().strftime("%Y%m%d%H%M%S%f") + f"_xgb_playoffs_{self.target}"
        )

        # -------- OOF prediction export (mirror LightGBM logic) --------------------
        if n_cv_splits >= 2 and not oof_preds.isnull().all():
            meta_cols = ["game_id", "player_id", "game_date"]
            export_df = pd.DataFrame()                                    # will stay empty if alignment fails

            # make sure metadata columns exist in the raw frame
            if all(col in raw_df_all.columns for col in meta_cols):

                # ─── fast path ─── unique index allows direct .loc
                if y_cv.index.is_unique:
                    meta_df   = raw_df_all.loc[y_cv.index, meta_cols]
                    oof_df    = pd.DataFrame({"actual": y_cv, "predicted": oof_preds})
                    export_df = (
                        meta_df.join(oof_df, how="inner")
                               .dropna(subset=["predicted"])
                    )

                # ─── fallback ─── non-unique index → merge on preserved index
                else:
                    df_meta_temp = raw_df_all[meta_cols].copy()
                    df_meta_temp["original_index"] = raw_df_all.index

                    oof_df = pd.DataFrame({"actual": y_cv, "predicted": oof_preds})
                    oof_df["original_index"] = y_cv.index

                    export_df = (
                        pd.merge(df_meta_temp, oof_df,
                                 on="original_index", how="inner")
                          .drop(columns=["original_index"])
                          .dropna(subset=["predicted"])
                    )
            else:
                logger.warning(
                    "Skipping OOF export: required metadata columns %s missing in raw "
                    "data frame.", meta_cols
                )

            # ─── final export or skip ───────────────────────────────────────────────
            if export_df.empty:
                logger.warning(
                    "Skipping OOF export (could not align OOF rows with metadata)."
                )
            else:
                logger.info(
                    "Exporting %d OOF predictions for model run %s…",
                    len(export_df), self.model_run_id,
                )
                self.export_training_predictions(
                    export_df["game_id"],
                    export_df["player_id"],
                    export_df["game_date"],
                    export_df["actual"],
                    export_df["predicted"],
                    self.model_run_id,
                )

        else:
            logger.info(
                "Skipping OOF export (CV not run or no predictions produced)."
            )

        # ───────────────────────────── pipeline finished ───────────────────────────
        logger.info(
            "=== XGBoost Standalone Playoff Training pipeline v2 finished "
            "for target: %s ===",
            self.target,
        )
        return avg_metrics


    # ---------------------------------------------------------------------#
    # Export Predictions (Unchanged Logic)
    # ---------------------------------------------------------------------#
    def export_training_predictions(
        self, game_ids: pd.Series, player_ids: pd.Series, game_dates: pd.Series,
        y_actual: pd.Series, y_predicted: pd.Series, model_run: str
    ):
        """Persist OOF‑style training predictions to SQLite."""
        # This function remains identical
        if not all(len(s) == len(game_ids) for s in [player_ids, game_dates, y_actual, y_predicted]):
             logger.error("Prediction export failed: Input Series lengths mismatch.")
             return
        if game_ids.empty:
             logger.warning("No training predictions to export.")
             return

        common_index = y_actual.index.intersection(y_predicted.index)
        if len(common_index) != len(y_actual) or len(common_index) != len(y_predicted):
             logger.warning("Aligning actual and predicted indices before calculating residual.")
             y_actual = y_actual.loc[common_index]
             y_predicted = y_predicted.loc[common_index]
             game_ids = game_ids.loc[common_index]
             player_ids = player_ids.loc[common_index]
             game_dates = game_dates.loc[common_index]

        resid = y_actual - y_predicted
        cols = ["game_id", "player_id", "game_date", "actual_points", "predicted_points", "residual", "model_run"]
        res_df = pd.DataFrame({
             "game_id": game_ids, "player_id": player_ids,
             "game_date": pd.to_datetime(game_dates).dt.strftime("%Y-%m-%d %H:%M:%S"),
             "actual_points": y_actual, "predicted_points": y_predicted,
             "residual": resid, "model_run": model_run,
             }, columns=cols)

        initial_count = len(res_df)
        res_df = res_df.dropna(subset=['residual'])
        dropped_count = initial_count - len(res_df)
        if dropped_count > 0: logger.warning(f"Dropped {dropped_count} rows with NaN residuals before exporting predictions.")
        if res_df.empty:
             logger.warning("No valid predictions remained after dropping NaNs. Nothing to export.")
             return

        conn = None
        try:
            db_path = Path(self.db_file)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_file, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout = 30000;")
            table_name = "training_predictions" # Use the same table name
            conn.execute(f"""
                  CREATE TABLE IF NOT EXISTS {table_name} (
                      game_id TEXT NOT NULL, player_id TEXT NOT NULL, game_date TEXT NOT NULL,
                      actual_points REAL, predicted_points REAL, residual REAL,
                      model_run TEXT NOT NULL,
                      PRIMARY KEY (game_id, player_id, model_run) );
            """)
            placeholders = ", ".join(["?"] * len(cols))
            insert_sql = f"INSERT OR REPLACE INTO {table_name} ({', '.join(cols)}) VALUES ({placeholders});"
            chunk_size = 5000
            for i in range(0, len(res_df), chunk_size):
                chunk = res_df.iloc[i:i + chunk_size]
                with conn:
                     conn.executemany(insert_sql, chunk.itertuples(index=False, name=None))
            logger.info("Exported %d XGBoost training predictions for model run %s.", len(res_df), model_run)
        except sqlite3.Error as e:
             logger.error(f"SQLite error during prediction export: {e}")
        finally:
             if conn: conn.close()

    # ---------------------------------------------------------------------#
    # Artifact I/O (XGBoost Paths - Unchanged)
    # ---------------------------------------------------------------------#
    def save_artifacts(self) -> None:
        """Persist XGBoost playoff model and metadata using instance paths."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.feature_names, self.features_path)
        joblib.dump(self.final_categorical_features, self.cat_features_path) # Save names list
        logger.info(f"Saved XGBoost playoff artifacts (model, features, cats) to {self.model_path.parent}")

    def load_artifacts(self) -> None:
        """Load XGBoost playoff model and metadata using instance paths."""
        try:
            if not self.model_path.is_file(): raise FileNotFoundError(f"XGBoost Model file not found: {self.model_path}")
            if not self.features_path.is_file(): raise FileNotFoundError(f"XGBoost Features file not found: {self.features_path}")
            if not self.cat_features_path.is_file(): raise FileNotFoundError(f"XGBoost Categorical features file not found: {self.cat_features_path}")

            self.model = joblib.load(self.model_path)
            self.feature_names = joblib.load(self.features_path)
            self.final_categorical_features = joblib.load(self.cat_features_path) # Load names list
            self.scaler = _NoOpScaler()
            logger.info(
                "Loaded XGBoost playoff model and metadata from %s (features: %d, categoricals: %d)",
                self.model_path.parent, len(self.feature_names), len(self.final_categorical_features)
            )
            # Optional sanity check
            model_features = None
            if hasattr(self.model, 'n_features_in_'):
                 model_features = self.model.n_features_in_
            elif hasattr(self.model, 'feature_names_in_'):
                 model_features = len(self.model.feature_names_in_)
            if model_features is not None and model_features != len(self.feature_names):
                 logger.warning(f"XGBoost playoff model expects {model_features} features but loaded list has {len(self.feature_names)}.")
        except FileNotFoundError as err:
            logger.error(f"XGBoost playoff artifact file not found (target: {self.target}): {err}")
            raise
        except Exception as err:
            logger.error(f"Unexpected error loading XGBoost playoff artifacts (target: {self.target}): {err}", exc_info=True)
            raise

    # ---------------------------------------------------------------------#
    # Prediction (XGBoost Version - Simplified Placeholder - Unchanged)
    # ---------------------------------------------------------------------#
    def predict(self, new_data_prepared: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on prepared new playoff data (assumes data is already processed
        with correct features, codes, and order). Basic implementation.
        """
        if self.model is None:
            logger.warning("Model not loaded. Attempting to load artifacts...")
            self.load_artifacts()
            if self.model is None: raise ValueError("XGBoost model must be loaded before predicting.")
        if not isinstance(new_data_prepared, pd.DataFrame): raise TypeError("Input must be a pandas DataFrame.")

        # Ensure columns match feature_names
        if list(new_data_prepared.columns) != self.feature_names:
            logger.warning("Input data columns mismatch expected features. Attempting reorder/selection.")
            try:
                 predict_data = new_data_prepared[self.feature_names].copy()
            except KeyError as e:
                 missing = [f for f in self.feature_names if f not in new_data_prepared.columns]
                 raise ValueError(f"Prediction failed: Input data missing required features: {missing}") from e
        else:
            predict_data = new_data_prepared.copy()

        # Ensure all data is numeric
        non_numeric = predict_data.select_dtypes(exclude=np.number).columns
        if not non_numeric.empty:
            logger.warning(f"Non-numeric columns found before prediction: {non_numeric.tolist()}. Forcing numeric (NaN->0).")
            for col in non_numeric:
                 predict_data[col] = pd.to_numeric(predict_data[col], errors='coerce').fillna(0)

        if predict_data.isna().any().any():
            logger.warning("NaNs detected before prediction. Filling with 0.")
            predict_data.fillna(0, inplace=True)

        logger.info(f"Predicting with XGBoost model on {len(predict_data)} rows.")
        try:
            predict_data.columns = [str(c) for c in predict_data.columns]
            predictions = self.model.predict(predict_data)
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}", exc_info=True)
            logger.error(f"Data shape: {predict_data.shape}, Columns: {predict_data.columns.tolist()[:10]}")
            raise
        return predictions

# ─────────────────────────── CLI entry‑point (v2) ────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="player-prop-playoffs-xgb-train-v2", # Updated program name
        description="Train the STANDALONE PLAYOFF NBA XGBoost player-prop model FROM CACHED FEATURES (v2 - White-list features).", # Updated description
    )

    parser.add_argument("--db-file", type=str, default=DEFAULT_DB_FILE, help="Path to SQLite DB (needed for averages and OOF metadata).")
    parser.add_argument("--target", type=str, default="pts", help="Target variable (default: pts). Must match the target used for cache generation.")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU usage for XGBoost.")
    parser.add_argument(
        "--xgboost-artifacts-dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR_XGBOOST),
        help="Directory to save XGBoost playoff model artifacts."
    )

    args = parser.parse_args()

    logger.info("Starting XGBoost playoff model training script v2 (from cache, white-list).")
    logger.info(f"Database file: {args.db_file}")
    logger.info(f"Target variable: {args.target}")
    logger.info(f"Use GPU: {args.use_gpu}")
    logger.info(f"XGBoost Artifacts Dir: {args.xgboost_artifacts_dir}")
    logger.info(f"Loading features from cache version: {LGBM_PLAYOFF_FEATURE_CACHE_VERSION}")

    # Instantiate the XGBoost playoff model class
    try:
        xgb_playoff_model = XGBoostPlayoffModel(
            db_file=args.db_file,
            target=args.target,
            xgboost_artifacts_dir=args.xgboost_artifacts_dir,
            use_gpu=args.use_gpu,
            random_state=GLOBAL_RANDOM_STATE
            # rolling_windows will use default unless specified otherwise
        )
    except (ImportError, FileNotFoundError, ValueError) as init_err:
        logger.exception("Fatal error during model initialization: %s", init_err)
        print(f"\n--- XGBoost Playoff Training FAILED during initialization ---")
        print(f"Error: {init_err}")
        sys.exit(1)
    except Exception as init_err:
        logger.exception("Unexpected fatal error during model initialization: %s", init_err)
        print(f"\n--- XGBoost Playoff Training FAILED during initialization ---")
        print(f"Error: {init_err}")
        sys.exit(1)

    # Run Training
    try:
        metrics_results = xgb_playoff_model.train()
        print("\n--- XGBoost Standalone Playoff Training complete ---")
        if "error" in metrics_results:
            print(f"Training failed: {metrics_results['error']}")
            sys.exit(1)
        else:
            print(f"Target: {args.target}")
            print("Cross-Validation Metrics (Avg over folds, Playoffs Only):")
            for metric_name, metric_value in metrics_results.items():
                value_str = f"{metric_value:.4f}" if not pd.isna(metric_value) else "NaN"
                print(f"  {metric_name}: {value_str}")
            print(f"\nXGBoost artifacts saved to: {xgb_playoff_model.model_path.parent}")
            print(f"XGBoost model: {xgb_playoff_model.model_path.name}")
            print(f"XGBoost features: {xgb_playoff_model.features_path.name}")
            print(f"XGBoost categoricals: {xgb_playoff_model.cat_features_path.name}")
            print(f"XGBoost KEPT features/importances list: {xgb_playoff_model.kept_features_csv_path.name}")

    except (FileNotFoundError, ValueError, RuntimeError) as train_err:
        logger.error("Training execution failed: %s", train_err, exc_info=True)
        print(f"\n--- XGBoost Playoff Training FAILED ---")
        print(f"Error: {train_err}")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected fatal error during XGBoost playoff training execution: %s", exc)
        print(f"\n--- XGBoost Playoff Training FAILED ---")
        print(f"Error: {exc}")
        sys.exit(1)

    logger.info("XGBoost playoff model training script v2 finished successfully.")
    sys.exit(0)