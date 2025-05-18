#!/usr/bin/env python3
"""
player_props_playoffs_lightgbm_alt.py

Trains a standalone LightGBM playoff player prop model using ONLY playoff data
and derived features. This version does NOT use predictions from any
other model as a feature.

MODIFIED: Includes pre-calculated career & season averages (shifted) for the target stat,
and uses the season average as a default for the playoff career average when
playoff history is insufficient (< 5 games). Adds career/season averages as features.
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
import importlib.util # For checking torch for Optuna sampler
import warnings # To suppress specific warnings cleanly
import re



# ─────────────────────────── 3rd‑party ─────────────────────────────
import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from optuna.integration import LightGBMPruningCallback
from optuna.exceptions import TrialPruned   # add near other imports
import itertools, collections, numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_regression


from sql_player_props_playoffs import VIEW_SETUP_SQL, PLAYER_GAMES_SQL
from features_player_props import BASE_STATS, TEAM_STATS, OPP_STATS, PLAYER_ADVANCED_TRACKING, INTERACTIONS, INTERACTION_NAMES, EXTRA_FEATURES, CAT_FLAGS, ODD_COLS_NUMERIC, ODD_COLS, PLAYOFF_COLS    # <- new line


# ─────────────────────────── Project Imports ───────────────────────
# No external model class import needed for this version.

# ────────────────────────────── logging ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s", # Added logger name
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Use a distinct logger name for this standalone script
logger = logging.getLogger("PlayoffModelStandalone")

# ──────────────────────────── reproducibility ──────────────────────
GLOBAL_RANDOM_STATE = 42
np.random.seed(GLOBAL_RANDOM_STATE)
random.seed(GLOBAL_RANDOM_STATE)

# ──────────────── PLAYOFF-SPECIFIC feature cache ─────────────────
# Use a distinct cache version/directory for playoff features
# NOTE: Cache version should be incremented if feature logic changes significantly.
# Cache needs rebuild after this change due to merged averages affecting FE input.
PLAYOFF_FEATURE_CACHE_VERSION = "v17_playoffs_with_prior_avgs" # Updated version name
PLAYOFF_FEATURE_CACHE_DIR = Path(".fe_cache_playoffs_only_v17") # Distinct directory
PLAYOFF_FEATURE_CACHE_DIR.mkdir(exist_ok=True)


# ──────────────── PLAYOFF-SPECIFIC Artifact Paths & Constants ─────
SCRIPT_DIR_PLAYOFFS = Path(__file__).parent.resolve()
DEFAULT_ARTIFACTS_DIR_PLAYOFFS = SCRIPT_DIR_PLAYOFFS # Store playoff model artifacts here
DEFAULT_DB_FILE = "nba.db" # Default DB name

# --- Path Generation Functions (Playoff Model Only) ---
def get_playoff_model_path(target: str, base_dir: Path = DEFAULT_ARTIFACTS_DIR_PLAYOFFS) -> Path:
    return base_dir / f"modelLightgbm_playoffs_{target}.joblib"
def get_playoff_features_path(target: str, base_dir: Path = DEFAULT_ARTIFACTS_DIR_PLAYOFFS) -> Path:
    return base_dir / f"featuresLightgbm_playoffs_{target}.joblib"
def get_playoff_cat_features_path(target: str, base_dir: Path = DEFAULT_ARTIFACTS_DIR_PLAYOFFS) -> Path:
    return base_dir / f"categorical_featuresLightgbm_playoffs_{target}.joblib"
def get_playoff_study_path(target: str, base_dir: Path = DEFAULT_ARTIFACTS_DIR_PLAYOFFS) -> Path:
    return base_dir / f"optuna_lgbm_study_playoffs_{target}.db"
def get_playoff_pruned_features_csv_path(target: str, base_dir: Path = DEFAULT_ARTIFACTS_DIR_PLAYOFFS) -> Path:
    return base_dir / f"shap_kept_features_playoffs_{target}.csv"

# Constants


TOP_MI_FEATURES = 700
MIN_PLAYOFF_GAMES_FOR_CAREER_AVG = 5 # Min games needed for playoff career avg feature


def _greedy_diverse(df_pairs: pd.DataFrame, k: int, max_per_feat: int) -> pd.DataFrame:
    """
    df_pairs  – columns = ['f1','f2','score'] already sorted DESC by score.
    Keeps at most `max_per_feat` appearances of any single feature and
    returns the first `k` surviving rows.
    """
    used = collections.Counter()
    keep = []
    for _, r in df_pairs.iterrows():
        if len(keep) == k:
            break
        if used[r.f1] >= max_per_feat or used[r.f2] >= max_per_feat:
            continue
        keep.append(r)
        used[r.f1] += 1
        used[r.f2] += 1
    return pd.DataFrame(keep)

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

def _exp_weighted_slope(arr: np.ndarray) -> float:
    """Slope of first‑order fit to *arr* with exponential decay (half‑life len/2)."""
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return np.nan
    y = arr[mask]
    x = np.arange(len(arr))[mask]
    # Increase weight calculation stability: use larger value if len(arr) is small
    half_life = max(1, len(arr) / 2)
    weights = np.exp(-(len(arr) - 1 - x) / half_life)
    try:
        # Add small epsilon to weights to prevent all-zero weights? Unlikely needed.
        return np.polyfit(x, y, 1, w=weights)[0]
    except (np.linalg.LinAlgError, ValueError):
        # Fallback to unweighted fit if weighted fails
        try:
            return np.polyfit(x, y, 1)[0]
        except (np.linalg.LinAlgError, ValueError):
            return np.nan
        
def _log_non_numeric(df: pd.DataFrame, cols: list[str], context: str) -> None:
    """
    Emit a DEBUG/ERROR message for every column in *cols* whose dtype is not
    numeric **or** that still contains non-numeric objects after coercion.

    Args
    ----
    df : DataFrame under inspection
    cols : list of column names that are expected to be numeric
    context : short tag that will appear in the log message
    """
    import numbers

    offenders = []
    for c in cols:
        if c not in df.columns:
            continue
        series = df[c]
        if pd.api.types.is_numeric_dtype(series):
            # mixture?  e.g. float plus stray strings/None
            mask_bad = ~series.map(
                lambda x: (isinstance(x, numbers.Number) or pd.isna(x))
            )
            if mask_bad.any():
                offenders.append(
                    (c, series.dtype, series[mask_bad].head(3).tolist())
                )
        else:
            offenders.append(
                (c, series.dtype, series.dropna().head(3).tolist())
            )

    if offenders:
        msg = "; ".join(
            f"{col} dtype={dtype} sample={sample}"
            for col, dtype, sample in offenders
        )
        logger.error("NON-NUMERIC DETECTED [%s] → %s", context, msg)


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
        # groups is an array-like of shape (n_rows,)
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

# ─────────────────── Standalone Playoff Model Class ────────────────────
class LightGBMPlayoffModel:
    """
    Predict a box‑score stat (e.g., 'pts') using ONLY playoff data and
    derived features. Does not rely on external model predictions.
    Includes career/season averages and defaults playoff avg to season avg.
    """
    def __init__(
        self,
        db_file: str = DEFAULT_DB_FILE,
        target: str = "pts",
        playoff_artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR_PLAYOFFS,
        rolling_windows: Optional[List[int]] = None, # Used in FE
        random_state: int = GLOBAL_RANDOM_STATE,
        use_extra_features: bool = True, # Controls inclusion in prepare_model_data
        use_gpu: bool = False,
    ):
        self.db_file = db_file
        self.target = target
        self.rolling_windows = rolling_windows or [2, 5, 10] # Adjusted defaults
        self.random_state = random_state
        self.use_extra_features = use_extra_features
        self.use_gpu = use_gpu

        # Convert string paths to Path objects
        self.playoff_artifacts_dir = Path(playoff_artifacts_dir).resolve()
        self.playoff_artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Set instance-specific paths based on target and base directories
        self.model_path = get_playoff_model_path(self.target, self.playoff_artifacts_dir)
        self.features_path = get_playoff_features_path(self.target, self.playoff_artifacts_dir)
        self.cat_features_path = get_playoff_cat_features_path(self.target, self.playoff_artifacts_dir)
        self.study_path = get_playoff_study_path(self.target, self.playoff_artifacts_dir)
        self.kept_features_csv_path = get_playoff_pruned_features_csv_path(self.target, self.playoff_artifacts_dir)

        # Model and feature attributes
        self.model: Optional[lgb.LGBMRegressor] = None
        self.scaler: _NoOpScaler = _NoOpScaler() # Playoff model doesn't use scaling
        self.feature_names: List[str] = [] # Final features after pruning
        self.initial_feature_names: List[str] = [] # Features before MI/SHAP
        self.categorical_features: List[str] = [] # Potential categorical features identified pre-filtering
        self.final_categorical_features: List[str] = [] # Actual categoricals in final feature set
        self.model_run_id: Optional[str] = None

        logger.info(f"Initialized Standalone Playoff Model for target '{self.target}'.")
        logger.info(f"Playoff artifact directory: {self.playoff_artifacts_dir}")
        logger.info(f"Playoff Optuna study path: {self.study_path}")
        logger.info(f"Playoff-only feature cache directory: {PLAYOFF_FEATURE_CACHE_DIR}")

    # ---------------------------------------------------------------------#
    # Data Retrieval (Unchanged from previous - Still retrieves all data initially)
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
        odds_numeric_cols = ODD_COLS_NUMERIC
        
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
        odds_cols = ODD_COLS
        existing_odds = [c for c in odds_cols if c in df.columns]
        df[existing_odds] = df[existing_odds].fillna(0)

        if df.empty:
            raise ValueError("No rows returned from the database.")

        # --- Playoff flag defaults & Type Conversion ---
        playoff_cols = PLAYOFF_COLS
        for col in playoff_cols:
            if col in df.columns:
                 # Handle potential non-numeric values before converting
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype("int16")
            else:
                logger.warning(f"Expected playoff column '{col}' not found. Adding with default 0.")
                df[col] = 0 # Add as int

        # --- Derive 'win' column from 'team_wl' ---
        if 'team_wl' in df.columns:
            df['win'] = df['team_wl'].apply(lambda x: 1 if isinstance(x, str) and x.upper() == 'W' else 0).astype(int)
        else:
            logger.warning("'team_wl' column not found. Creating 'win' column with default 0.")
            df['win'] = 0

        # Ensure 'season' column exists and is string type if present
        if 'season' in df.columns:
            # Ensure season is in a consistent format (e.g., YYYY-YY) if possible, then string
            # Example: Convert 2023 to '2023-24'. Requires logic based on game_date.
            # For simplicity here, just ensure it's string.
            df['season'] = df['season'].astype(str)
        else:
            logger.error("'season' column not found in data. Season average calculation WILL FAIL.")
            # Optionally, derive season from game_date if necessary
            # Example: df['season'] = df['game_date'].apply(lambda d: f"{d.year - 1}-{str(d.year)[-2:]}" if d.month < 9 else f"{d.year}-{str(d.year + 1)[-2:]}")
            raise ValueError("'season' column is required but missing.")


        # Ensure opponent_vs_player columns exist, fill with 0 if missing after LEFT JOIN
        ovp_cols = ["opponent_vs_player_fgm_allowed", "opponent_vs_player_fga_allowed", "opponent_vs_player_pts_allowed"]
        for col in ovp_cols:
            if col not in df.columns:
                logger.warning(f"Column '{col}' missing. Adding with default 0.")
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)


        df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True) # Sort crucial for shift/rolling
        logger.info("Loaded %d total game rows (before playoff filter).", len(df))
        return df

    # ---------------------------------------------------------------------#
    # Feature Engineering - Base & Rolling (MODIFIED)
    # ---------------------------------------------------------------------#
    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling, trend, variability, fatigue/load, schedule‑density,
        opponent‑adjusted, pace‑gap and hot/cold features. **All features are
        leak‑free**.

        MODIFIED:
        - Expects df to contain pre-calculated, shifted '{target}_career_avg' and '{target}_season_avg'.
        - Calculates playoff career avg for the target, defaults to season avg if insufficient playoff history.
        - Adds squared versions of career avg and season avg.
        - Other rolling features are calculated based ONLY on playoff data within df.
        """
        df = df.copy()
        logger.info("Adding rolling/context features (expecting playoff data + prior avgs)...")
        if df.empty:
            logger.warning("Input DataFrame to add_rolling_features is empty. Skipping.")
            return df

        # Stat definitions (copied)
        # Ensure target stat exists in df
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in input DataFrame for rolling features.")

        # Check for pre-calculated averages (added in train method)
        career_avg_col = f"{self.target}_career_avg"
        season_avg_col = f"{self.target}_season_avg"
        if career_avg_col not in df.columns or season_avg_col not in df.columns:
            raise ValueError(f"Missing pre-calculated '{career_avg_col}' or '{season_avg_col}' in input DataFrame.")

        base_stats = BASE_STATS
        player_advanced_tracking = PLAYER_ADVANCED_TRACKING
        team_stats = TEAM_STATS
        opp_stats = OPP_STATS


        # ---- quick diagnostic: list any would-be numeric columns that
        #      are still objects/strings for this particular dataframe ----
        numeric_cols = (
            base_stats
            + team_stats
            + opp_stats
            + player_advanced_tracking
            + [c for pair in INTERACTIONS.values() for c in pair]
        )
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        _log_non_numeric(df, numeric_cols, "add_rolling_features-pre")

        # Combined stats list for FE (use only stats present in df)
        combined_stats_for_rolling = sorted(list(set(
             [s for s in (base_stats + team_stats + opp_stats + player_advanced_tracking) if s in df.columns]
        )))
        # Ensure target is included if present
        if self.target in df.columns and self.target not in combined_stats_for_rolling:
             combined_stats_for_rolling.append(self.target)

        # Grouping handles (assuming df contains ONLY playoff data now)
        player_group = df.groupby("player_id")
        team_group = df.groupby("team_id")
        opp_team_group = df.groupby("opponent_team_id")
        new_cols: Dict[str, pd.Series] = {}

        # --- Add Squares for Pre-calculated Averages ---
        # These columns were added *before* this function in train()
        logger.info(f"Adding squared features for pre-calculated '{self.target}' career and season averages.")
        new_cols[f"{career_avg_col}_sq"] = df[career_avg_col].pow(2)
        new_cols[f"{season_avg_col}_sq"] = df[season_avg_col].pow(2)

        # 1. Scalar stats — rolling means, trends, variability
        logger.info(f"Calculating rolling/trend/playoff_avg features for {len(combined_stats_for_rolling)} base stats...")
        for stat_idx, stat in enumerate(combined_stats_for_rolling):
            if stat not in df.columns:
                logger.debug(f"Skipping rolling features for missing stat: {stat}")
                continue

            # Shift AFTER grouping by player_id to ensure correct alignment within each player's history
            shifted = player_group[stat].shift(1) # leak‑free series within playoffs

            # --- Playoff Career Expanding Average (using default) ---
            if stat == self.target:
                # Need to group the shifted series again for transform
                playoff_career_avg_raw = shifted.groupby(df["player_id"]).transform(
                    lambda s: s.expanding(min_periods=MIN_PLAYOFF_GAMES_FOR_CAREER_AVG).mean()
                )
                # Use pre-calculated season average (already shifted) as default
                # Ensure alignment before fillna - should be okay if merge in train() used index
                # If df index was reset, df[season_avg_col] index might mismatch playoff_career_avg_raw index
                # Reindex season_avg_col just in case (though join in train should handle this)
                playoff_career_avg_defaulted = playoff_career_avg_raw.fillna(df[season_avg_col].reindex(playoff_career_avg_raw.index))

                base_pc_avg_def = f"{stat}_playoff_career_avg_defaulted"
                new_cols[base_pc_avg_def] = playoff_career_avg_defaulted
                new_cols[f"{base_pc_avg_def}_sq"] = playoff_career_avg_defaulted.pow(2)
                # logger.debug(f"Calculated defaulted playoff career avg for target: {stat}") # Less verbose

            else:
                # For non-target stats, calculate the standard playoff career avg without default
                 playoff_career_avg = shifted.groupby(df["player_id"]).transform(
                     lambda s: s.expanding(min_periods=MIN_PLAYOFF_GAMES_FOR_CAREER_AVG).mean()
                 )
                 base_pc_avg = f"{stat}_playoff_career_avg"
                 new_cols[base_pc_avg] = playoff_career_avg
                 new_cols[f"{base_pc_avg}_sq"] = playoff_career_avg.pow(2)


            # Rolling means (w ∈ rolling_windows) + square polynomial
            for w in self.rolling_windows:
                 # Use transform within the group to handle NaNs correctly at start of series per player
                 roll = shifted.groupby(df["player_id"]).transform(lambda s: s.rolling(w, min_periods=1).mean())
                 base = f"{stat}_rolling_{w}"
                 new_cols[base] = roll
                 new_cols[f"{base}_sq"] = roll.pow(2)

            # 5‑game OLS trend + square polynomial
            # Use transform to apply rolling operation per group
            trend = shifted.groupby(df["player_id"]).transform(
                 lambda s: s.rolling(5, min_periods=max(2, 5-1)).apply(
                     lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y[~np.isnan(y)]) >= 2 else np.nan,
                     raw=False # Must be False for apply with custom lambda using len(y)
                 )
            )
            base = f"{stat}_trend"
            new_cols[base] = trend
            new_cols[f"{base}_sq"] = trend.pow(2)

            # 5‑game exponentially‑weighted trend + square polynomial
            ew_trend = shifted.groupby(df["player_id"]).transform(
                 lambda s: s.rolling(5, min_periods=max(2, 5-1)).apply(_exp_weighted_slope, raw=True)
            )
            base = f"{stat}_trend_ewm"
            new_cols[base] = ew_trend
            new_cols[f"{base}_sq"] = ew_trend.pow(2)

            # 10‑game standard deviation + square polynomial
            std10 = shifted.groupby(df["player_id"]).transform(lambda s: s.rolling(10, min_periods=1).std())
            base = f"{stat}_std"
            new_cols[base] = std10
            new_cols[f"{base}_sq"] = std10.pow(2)

            # Acceleration (Δ 5‑game trend − 10‑game trend) + square polynomial
            trend10 = shifted.groupby(df["player_id"]).transform(
                 lambda s: s.rolling(10, min_periods=max(2, 10-1)).apply(
                     lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y[~np.isnan(y)]) >= 2 else np.nan,
                     raw=False
                 )
            )
            accel = trend - trend10 # Relies on trend calculated above
            base = f"{stat}_acceleration"
            new_cols[base] = accel
            new_cols[f"{base}_sq"] = accel.pow(2)

            # Std-of-trend over trailing 10 + square polynomial
            # Trend is already calculated per group, need to shift and roll within group again
            trend_std = trend.groupby(df["player_id"]).transform(
                 lambda s: s.shift(1).rolling(10, min_periods=1).std()
            )
            base = f"{stat}_trend_std"
            new_cols[base] = trend_std
            new_cols[f"{base}_sq"] = trend_std.pow(2)

            if (stat_idx + 1) % 10 == 0: # Log progress periodically
                logger.debug(f"  ({stat_idx+1}/{len(combined_stats_for_rolling)}) Calculated rolling features for: {stat}")


        # 2. Interaction terms
        interactions = INTERACTIONS

        # Filter interactions based on columns present in df
        valid_interactions = {
            name: pair for name, pair in interactions.items()
            if pair[0] in df.columns and pair[1] in df.columns
        }
        logger.info(f"Calculating features for {len(valid_interactions)} valid interaction terms...")

        for interaction_idx, (name, (a, b)) in enumerate(valid_interactions.items()):
            base_raw = (
                pd.to_numeric(df[a], errors="coerce") *
                pd.to_numeric(df[b], errors="coerce")
            )
            # Shift AFTER grouping
            shifted_interaction = base_raw.groupby(df["player_id"]).shift(1)
            new_cols[name] = shifted_interaction # Store shifted raw interaction
            grp = shifted_interaction.groupby(df["player_id"]) # Group the shifted series

            # --- Playoff Career Expanding Average for Interaction ---
            # Use transform for expanding mean within group
            inter_playoff_career_avg = grp.transform(
                lambda s: s.expanding(min_periods=MIN_PLAYOFF_GAMES_FOR_CAREER_AVG).mean()
            )
            base_inter_pc_avg = f"{name}_playoff_career_avg"
            new_cols[base_inter_pc_avg] = inter_playoff_career_avg
            new_cols[f"{base_inter_pc_avg}_sq"] = inter_playoff_career_avg.pow(2)

            # Rolling means + square
            for w in self.rolling_windows:
                 # Use transform within group for rolling mean
                 roll = grp.transform(lambda s: s.rolling(w, min_periods=1).mean())
                 base = f"{name}_rolling_{w}"
                 new_cols[base] = roll
                 new_cols[f"{base}_sq"] = roll.pow(2)

            # Trends (OLS and EW) + square
            # Use transform within group for rolling apply
            trend = grp.transform(
                 lambda s: s.rolling(5, min_periods=max(2, 5-1)).apply(
                     lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y[~np.isnan(y)]) >= 2 else np.nan,
                     raw=False
                 )
            )
            base_tr = f"{name}_trend"
            new_cols[base_tr] = trend
            new_cols[f"{base_tr}_sq"] = trend.pow(2)

            ew_trend = grp.transform(
                 lambda s: s.rolling(5, min_periods=max(2, 5-1)).apply(_exp_weighted_slope, raw=True)
            )
            base = f"{name}_trend_ewm"
            new_cols[base] = ew_trend
            new_cols[f"{base}_sq"] = ew_trend.pow(2)

            # 10‑game std + square
            # Use transform within group for rolling std
            std = grp.transform(lambda s: s.rolling(10, min_periods=1).std())
            base = f"{name}_std"
            new_cols[base] = std
            new_cols[f"{base}_sq"] = std.pow(2)

            if (interaction_idx + 1) % 5 == 0:
                logger.debug(f"  ({interaction_idx+1}/{len(valid_interactions)}) Completed interaction features for: {name}")

        # 3. Fatigue / load signals
        fatigue_cols = ["min", "usage_rate", "team_travel_distance", "team_is_back_to_back"]
        if all(c in df.columns for c in fatigue_cols):
            # Calculate rolling means/std within player groups
            min5 = player_group["min"].transform(lambda s: s.shift(1).rolling(5, 1).mean())
            usage5 = player_group["usage_rate"].transform(lambda s: s.shift(1).rolling(5, 1).mean())
            min_std = player_group["min"].transform(lambda s: s.shift(1).rolling(5, 1).std())

            new_cols["min5_travel_distance"] = min5 * df["team_travel_distance"]
            new_cols["usage5_b2b"] = usage5 * df["team_is_back_to_back"]
            new_cols["rolling_minutes_std"] = min_std
            new_cols["rolling_minutes_std_sq"] = min_std.pow(2)
        else:
            logger.warning("Skipping fatigue/load signals due to missing source columns (%s).", fatigue_cols)

        # 4. Schedule‑density signals
        if 'game_date' in df.columns:
            # Calculate rest days within player/team groups
            player_rest = (df["game_date"] - player_group["game_date"].shift(1)).dt.days
            team_rest = (df["game_date"] - team_group["game_date"].shift(1)).dt.days
            # Opponent rest days: This requires a more complex lookup or pre-calculation.
            # The simple groupby opponent_team_id might not be correct if opponent data isn't fully present.
            # Using the existing potentially flawed logic for now.
            try:
                opp_rest = (df["game_date"] - opp_team_group["game_date"].shift(1)).dt.days
                new_cols["opponent_team_rest_days"] = opp_rest
                new_cols["rest_days_diff"] = player_rest - opp_rest
            except Exception as e:
                 logger.warning(f"Could not calculate opponent rest days reliably: {e}. Skipping.")
                 new_cols["opponent_team_rest_days"] = np.nan
                 new_cols["rest_days_diff"] = np.nan


            new_cols["player_rest_days"] = player_rest
            new_cols["team_rest_days"] = team_rest

        else:
            logger.warning("Skipping schedule density features due to missing 'game_date'.")

        # 5. Opponent‑adjusted form & pace gap (check dependencies in new_cols)
        opp_adj_cols = [
            self.target, "opponent_def_rating", "team_pace", "opponent_pace",
            "opponent_vs_player_pts_allowed"
        ]
        target_roll_5_feat = f"{self.target}_rolling_5"
        fgm_trend_feat = "fgm_trend"
        fgm_fg3m_trend_feat = "fgm_fg3m_trend"

        # Check if necessary base columns and *generated* features exist in new_cols or df
        if (all(c in df.columns or c in new_cols for c in opp_adj_cols) and
              target_roll_5_feat in new_cols): # Check against new_cols where rolling features are added

            pts5 = new_cols[target_roll_5_feat] # Use already computed rolling mean

            # Shift opponent rating within opponent group
            opp_def_prev = df.groupby("opponent_team_id")["opponent_def_rating"].shift(1).reindex(df.index) # Reindex to align
            new_cols["pts5_per_opp_def_rating"] = pts5 / opp_def_prev.replace(0, np.nan).ffill().bfill() # Handle 0 and NaN

            # Shift pace within team/opponent groups
            team_pace_prev = df.groupby("team_id")["team_pace"].shift(1).reindex(df.index)
            opp_pace_prev = df.groupby("opponent_team_id")["opponent_pace"].shift(1).reindex(df.index)
            pace_gap = team_pace_prev - opp_pace_prev
            new_cols["pace_gap"] = pace_gap

            if fgm_trend_feat in new_cols:
                new_cols["fgm_trend_pace_gap"] = new_cols[fgm_trend_feat] * pace_gap

            opp_pts_allowed_prev = df.groupby("opponent_team_id")["opponent_vs_player_pts_allowed"].shift(1).reindex(df.index)
            if fgm_fg3m_trend_feat in new_cols:
                new_cols["fgm_fg3m_trend_adj_opp_pts_allowed"] = (
                        new_cols[fgm_fg3m_trend_feat] * opp_pts_allowed_prev
                )
        else:
            # Log which specific required feature might be missing
            missing_reqs = [f for f in opp_adj_cols + [target_roll_5_feat] if (f not in df.columns and f not in new_cols)]
            logger.warning(f"Skipping opponent-adjusted/pace gap features due to missing dependencies (e.g., {missing_reqs}).")


        # 6. Hot / cold streak flags (Uses defaulted playoff_career_avg)
        pts_trend_feat = f"{self.target}_trend"
        pts_roll5_feat = f"{self.target}_rolling_5"
        # Use the defaulted playoff career average feature name created earlier
        pts_playoff_career_avg_feat = f"{self.target}_playoff_career_avg_defaulted"

        if (pts_trend_feat in new_cols and
              pts_roll5_feat in new_cols and
              pts_playoff_career_avg_feat in new_cols): # Check against new_cols

            # Handle potential NaNs in comparisons gracefully
            hot  = (
                 (new_cols[pts_trend_feat].fillna(0) > 0) &
                 # Compare rolling 5 to defaulted playoff career average
                 (new_cols[pts_roll5_feat] > new_cols[pts_playoff_career_avg_feat])
                ).astype("int8")
            cold = (
                 (new_cols[pts_trend_feat].fillna(0) < 0) &
                 # Compare rolling 5 to defaulted playoff career average
                 (new_cols[pts_roll5_feat] < new_cols[pts_playoff_career_avg_feat])
                ).astype("int8")
            new_cols["is_hot_streak"]  = hot
            new_cols["is_cold_streak"] = cold
        else:
            missing_reqs = [f for f in [pts_trend_feat, pts_roll5_feat, pts_playoff_career_avg_feat] if f not in new_cols]
            logger.warning("Skipping hot/cold streak flags due to missing source features (%s).", missing_reqs)


        # Attach engineered columns & return
        df_out = _bulk_concat(df, new_cols)
        logger.info("Finished rolling/context features for playoff data.")
        return df_out

    # ---------------------------------------------------------------------#
    # Feature Engineering - Home/Away (MODIFIED)
    # ---------------------------------------------------------------------#
    def add_home_away_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Leak‑free home/away aggregates.
        MODIFIED: Uses defaulted playoff career avg for the target stat.
        """
        df = df.copy()
        if df.empty: return df
        logger.info("Adding home/away features (using defaulted playoff career avg)...")
        new_cols: Dict[str, pd.Series] = {}

        # Check if is_home column exists and is suitable type
        if "is_home" not in df.columns:
            logger.warning("Skipping home/away features: 'is_home' column not found.")
            return df
        if not pd.api.types.is_numeric_dtype(df['is_home']) and not pd.api.types.is_bool_dtype(df['is_home']):
            logger.warning("Converting 'is_home' to numeric for home/away split.")
            df['is_home'] = pd.to_numeric(df['is_home'], errors='coerce').fillna(0).astype(int)

        # Pre-calculated season average column name
        season_avg_col = f"{self.target}_season_avg"
        if season_avg_col not in df.columns:
             raise ValueError(f"Missing pre-calculated '{season_avg_col}' in input DataFrame for home/away features.")

        def _make_home_away(ctx_df: pd.DataFrame, prefix: str) -> Dict[str, pd.Series]:
            local: Dict[str, pd.Series] = {}
            if ctx_df.empty: return local
            # Group within the context (home or away)
            g_ctx = ctx_df.groupby("player_id")

            # Check if target exists in this subset
            if self.target not in ctx_df.columns:
                logger.warning(f"Target '{self.target}' not found in {prefix} subset for home/away features.")
                return local

            # Shift target within the group/context
            shifted_target_ctx = g_ctx[self.target].shift(1)

            # Rolling means (standard windows) + square
            for w in self.rolling_windows:
                 # Use transform within group
                 rolled = shifted_target_ctx.groupby(ctx_df["player_id"]).transform(lambda s: s.rolling(w, min_periods=1).mean())
                 col = f"{self.target}_{prefix}_rolling_{w}"
                 local[col] = rolled
                 local[f"{col}_sq"] = rolled.pow(2)

            # Playoff career average specific to home/away context (with default) + square
            # Calculate raw expanding mean within the context group
            playoff_career_avg_raw_ctx = shifted_target_ctx.groupby(ctx_df["player_id"]).transform(
                lambda s: s.expanding(min_periods=MIN_PLAYOFF_GAMES_FOR_CAREER_AVG).mean()
            )

            # Default using the overall season average (needs alignment)
            # Reindex the main df's season_avg_col to match the context df index before filling
            season_avg_ctx = df[season_avg_col].reindex(ctx_df.index) # Get season avg for rows in this context
            career_avg_defaulted_ctx = playoff_career_avg_raw_ctx.fillna(season_avg_ctx)

            career_avg_col_def = f"{self.target}_{prefix}_playoff_career_avg_defaulted"
            local[career_avg_col_def] = career_avg_defaulted_ctx
            local[f"{career_avg_col_def}_sq"] = career_avg_defaulted_ctx.pow(2)

            return local

        # Calculate features for home and away contexts
        new_cols.update(_make_home_away(df[df["is_home"] == 1].copy(), "home"))
        new_cols.update(_make_home_away(df[df["is_home"] == 0].copy(), "away"))

        df_out = _bulk_concat(df, new_cols)
        logger.info("Finished home/away features.")
        return df_out


    # ---------------------------------------------------------------------#
    # Feature Engineering - Playoff Context Specific (Unchanged)
    # ---------------------------------------------------------------------#
    def add_playoff_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds features derived purely from playoff context.
        MODIFIED: Calculates ONLY 'series_tied' and its prerequisites.
        Assumes input df is already filtered for playoffs.
        """
        df = df.copy()
        if df.empty: return df
        logger.info("Adding playoff context features (series_tied only)...") # Updated log

        # Pre-checks for necessary columns (still need win, game_date etc. for calculation)
        required_context_cols = ["season", "playoff_round", "series_number", "team_id", "player_id", "win", "game_date"]
        missing_cols = [col for col in required_context_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Skipping playoff context features: Missing columns {missing_cols}.")
            return df

        # Ensure 'win' is integer
        if not pd.api.types.is_integer_dtype(df['win']):
            df['win'] = pd.to_numeric(df['win'], errors='coerce').fillna(0).astype(int)

        # Setup for Playoff Calculations
        series_group_cols = ["season", "playoff_round", "series_number", "team_id"]
        # Ensure correct sorting for cumsum/cumcount within series
        df = df.sort_values(series_group_cols + ["game_date"])
        series_group = df.groupby(series_group_cols, observed=True, sort=False)

        new_cols: Dict[str, pd.Series] = {} # Initialize empty dictionary

        # 1. Calculate prerequisites for 'series_tied'
        try:
            # Calculate game number within the series group (chronological) - NEEDED for opp_wins_so_far
            game_in_series = series_group.cumcount() + 1

            # Calculate wins so far within the series group (shifted) - NEEDED
            team_wins_so_far = series_group['win'].transform(lambda x: x.shift(1).cumsum().fillna(0)).astype(int)

            # Games played before this one - NEEDED
            games_played_so_far = game_in_series - 1

            # Opponent wins: Total games played before this one - team wins before this one - NEEDED
            opp_wins_so_far = (games_played_so_far - team_wins_so_far).clip(lower=0).astype(int)

            # Calculate 'series_tied' flag
            is_series_tied = ((team_wins_so_far == opp_wins_so_far) & (games_played_so_far > 0)).astype("int8")

            # Add ONLY 'series_tied' to the dictionary of new columns
            new_cols["series_tied"] = is_series_tied


        except Exception as e:
            logger.error(f"Error calculating series_tied feature: {e}", exc_info=True) # Show traceback
            # Clear new_cols if error occurred
            new_cols = {}

        # Write back into the master frame
        if new_cols:
            logger.info(f"Adding {len(new_cols)} new playoff context feature(s): {list(new_cols.keys())}") # Updated log
            df = _bulk_concat(df, new_cols)
        else:
            logger.info("No new playoff context features were added (or calculation failed).")

        logger.info("Finished adding playoff context features (series_tied only).") # Updated log
        return df

    # ---------------------------------------------------------------------#
    # Feature Engineering - Main Pipeline (Unchanged Structure)
    # ---------------------------------------------------------------------#
    def feature_engineering(self, df_playoffs_only: pd.DataFrame) -> pd.DataFrame:
        """
        Master feature engineering pipeline for the standalone playoff model.
        Expects DataFrame containing ONLY playoff games, potentially with
        pre-merged career/season averages.
        Includes base rolling (with defaulted playoff career avg), home/away, and playoff context.
        """
        if df_playoffs_only.empty:
            logger.warning("Input to feature_engineering is empty. Returning empty DataFrame.")
            return df_playoffs_only

        logger.info("Starting playoff feature engineering pipeline (on playoff data + prior avgs)...")
        # Ensure required columns exist before calling sub-functions
        # These averages are expected to be merged *before* calling this function now
        required_cols = [
            'player_id', 'team_id', 'opponent_team_id', 'game_date', 'is_home',
            'season', 'playoff_round', 'series_number', 'win', self.target,
            f"{self.target}_career_avg", f"{self.target}_season_avg" # Check for pre-merged cols
        ]
        missing = [c for c in required_cols if c not in df_playoffs_only.columns]
        if missing:
            raise ValueError(f"Missing required columns for playoff feature engineering: {missing}")

        # Calls the MODIFIED versions of add_rolling_features and add_home_away_features
        df_playoffs_only = self.add_rolling_features(df_playoffs_only)
        df_playoffs_only = self.add_home_away_features(df_playoffs_only)
        df_playoffs_only = self.add_playoff_context_features(df_playoffs_only) # Unchanged call
        logger.info("Playoff feature engineering pipeline finished.")
        return df_playoffs_only

    # ---------------------------------------------------------------------#
    # Data Prep for Modelling (MODIFIED to include new avg feature names)
    # ---------------------------------------------------------------------#
    def prepare_model_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Build the modelling matrix (X, y) from an *engineered playoff* dataframe.
        Determines the initial feature set and potential categorical features
        *before* any filtering/pruning for the playoff model.
        Does NOT include any external model baseline prediction.

        MODIFIED: Includes names for career avg, season avg, and defaulted playoff avg features.
        """
        logger.info("Preparing playoff model data matrix...")
        if df.empty:
            logger.warning("Input DataFrame to prepare_model_data is empty.")
            return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)

        # --- Define base lists needed for name generation ---
        base_stats = BASE_STATS
        player_advanced_tracking = PLAYER_ADVANCED_TRACKING
        team_stats = TEAM_STATS
        opp_stats = OPP_STATS

        interactions = INTERACTION_NAMES


        # Combine stats for generating feature names based on patterns
        stats_for_fe_naming = sorted(list(set(
             [s for s in (base_stats + team_stats + player_advanced_tracking + opp_stats)]
        )))

        # Helper for polynomial feature names (base, sq)
        def _poly_sq(name: str) -> list[str]:
            return [name, f"{name}_sq"]

        # --- Generate feature names ---
        features_potential: list[str] = []

        # --- NEW: Add pre-calculated career and season averages for target ---
        features_potential += _poly_sq(f"{self.target}_career_avg")
        features_potential += _poly_sq(f"{self.target}_season_avg")

        # 1. Rolling / Trend / Playoff Career Avg Features
        for stat in stats_for_fe_naming:
            # Add defaulted playoff career avg for target, standard for others
            if stat == self.target:
                features_potential += _poly_sq(f"{stat}_playoff_career_avg_defaulted")
            else:
                features_potential += _poly_sq(f"{stat}_playoff_career_avg") # Non-defaulted for other stats

            # Add standard rolling features
            for w in self.rolling_windows: features_potential += _poly_sq(f"{stat}_rolling_{w}")
            trend_suffixes = ["trend", "trend_ewm", "std", "acceleration", "trend_std"]
            for suff in trend_suffixes: features_potential += _poly_sq(f"{stat}_{suff}")

        # Interactions: Rolling / Trend / Playoff Career Avg
        interaction_trend_suffixes = ["trend", "trend_ewm", "std"]
        for inter in interactions:
            features_potential.append(inter) # Keep raw interaction term
            # Non-defaulted playoff avg for interactions
            features_potential += _poly_sq(f"{inter}_playoff_career_avg")
            for w in self.rolling_windows: features_potential += _poly_sq(f"{inter}_rolling_{w}")
            for suff in interaction_trend_suffixes: features_potential += _poly_sq(f"{inter}_{suff}")

        # 2. Home/Away Features (Using Defaulted Playoff Avg for Target)
        for loc in ("home", "away"):
            for w in self.rolling_windows: features_potential += _poly_sq(f"{self.target}_{loc}_rolling_{w}")
            # Add the defaulted home/away playoff average name
            features_potential += _poly_sq(f"{self.target}_{loc}_playoff_career_avg_defaulted")

        # 3. Playoff Context Features (from add_playoff_context_features)
        playoff_context_features = [
           "series_tied"
        ]
        features_potential.extend(playoff_context_features)

        # 4. Extra Context Features (from add_rolling_features)
        extra_features = EXTRA_FEATURES

        if self.use_extra_features:
            features_potential += extra_features

        # --- De-duplicate and Validate against DataFrame ---
        seen: set[str] = set()
        validated_features_potential = []
        potential_but_missing = []

        for f in features_potential:
            if f in df.columns: # Check if the feature actually exists in the engineered df
                if f not in seen:
                    validated_features_potential.append(f)
                    seen.add(f)
            elif f not in seen: # Only track missing if not already seen
                 potential_but_missing.append(f)
                 seen.add(f) # Add to seen even if missing

        if potential_but_missing:
            head = ", ".join(potential_but_missing[:30])
            more = " …" if len(potential_but_missing) > 30 else ""
            logger.warning(
                f"Dropping {len(potential_but_missing)} potential features not found in engineered playoff data: {head}{more}"
            )

        # --- Persist INITIAL lists ---
        self.initial_feature_names = validated_features_potential
        if not self.initial_feature_names:
            logger.error("No features available after validation against Playoff DataFrame columns.")
            return df, pd.DataFrame(index=df.index), df[self.target].copy() if self.target in df else pd.Series(dtype=float)

        # --- Identify potential categorical features (Check against validated features) ---
        cat_flags_potential = CAT_FLAGS

        # Filter based on validated features
        self.categorical_features = [c for c in cat_flags_potential if c in self.initial_feature_names]

        # --- Build final model dataframe ---
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in Playoff DataFrame for final prep.")

        # Drop rows with NaN target, select final initial features
        df_model = df.dropna(subset=[self.target]).copy()

        upper = df_model[self.target].quantile(0.995)      # 99.5-percentile
        df_model[self.target] = df_model[self.target].clip(upper=upper)

        df_model = (
            df_model
            .sort_values(["season",
                            "playoff_round",       # 1 < 2 < 3 < 4
                            "series_number",       # 1 … n
                            "game_date"])          # actual date
        )

        # ─── series-level CV key  ───────────────────────────────────────────
        df_model["series_key"] = (
            df_model["season"].astype(str)
            + "_" + df_model["playoff_round"].astype(str)
            + "_" + df_model["series_number"].astype(str)
        )
        # ────────────────────────────────────────────────────────────────────
        
        # Ensure all initial features are present before selecting
        missing_initial = [f for f in self.initial_feature_names if f not in df_model.columns]
        if missing_initial:
             logger.error(f"Initial features missing from df_model after dropna: {missing_initial}")
             self.initial_feature_names = [f for f in self.initial_feature_names if f not in missing_initial]
             if not self.initial_feature_names:
                  raise ValueError("No initial features left after removing missing ones.")
             
        groups = df_model["series_key"].copy()

        # Select initial features + target
        df_model = df_model[self.initial_feature_names + [self.target]]

        # --- Convert categorical columns to 'category' dtype (Improved Robustness - Unchanged logic) ---
        logger.info(f"Converting {len(self.categorical_features)} potential categoricals...")
        for col in self.categorical_features:
            if col not in df_model.columns: continue # Skip if somehow missing

            # Handle NaNs before conversion
            if df_model[col].isnull().any():
                fill_val = -1 # Default for numeric-like
                try: # Attempt to infer type for better fill value
                    numeric_col = pd.to_numeric(df_model[col].dropna(), errors='raise')
                    if pd.api.types.is_float_dtype(numeric_col): fill_val = -1.0
                    else: fill_val = -1 # Integer-like
                except (ValueError, TypeError):
                    fill_val = "missing" # Use string for non-numeric

                # Fill based on inferred type
                if isinstance(fill_val, str):
                     # Ensure column is object/string type before filling with string
                     if not pd.api.types.is_object_dtype(df_model[col]) and not pd.api.types.is_string_dtype(df_model[col]):
                          df_model[col] = df_model[col].astype(str)
                     df_model[col] = df_model[col].fillna(fill_val)
                else:
                     df_model[col] = df_model[col].fillna(fill_val)

            # Convert to category
            try:
                 # Ensure compatible type before category conversion if possible
                 if pd.api.types.is_numeric_dtype(df_model[col]):
                     # Check if all values are integer-like
                     is_integer_like = df_model[col].apply(lambda x: isinstance(x, (int, np.integer)) or (isinstance(x, (float, np.floating)) and float(x).is_integer())).all()
                     if is_integer_like:
                          try: df_model[col] = df_model[col].astype(int)
                          except (ValueError, TypeError): pass # Keep as float if conversion fails
                 elif col == 'season' or col == 'series_record':
                      df_model[col] = df_model[col].astype(str) # Ensure string type

                 # Final conversion attempt
                 df_model[col] = df_model[col].astype("category")

            except Exception as e:
                 logger.warning(f"Could not convert column '{col}' to category dtype: {e}. Attempting string fallback.")
                 try: df_model[col] = df_model[col].astype(str).astype("category")
                 except Exception as e2: logger.error(f"Failed converting '{col}' to category after fallback: {e2}")


        X: pd.DataFrame = df_model[self.initial_feature_names]
        y: pd.Series = df_model[self.target]

        logger.info(
            "Prepared initial playoff modelling dataset: %d rows | %d features | %d potential categorical",
            len(df_model), len(self.initial_feature_names), len(self.categorical_features),
        )

        return df_model, X, y, groups
        

    # ---------------------------------------------------------------------#
    # Optuna Tuner (Unchanged Logic - operates on provided X, y)
    # ---------------------------------------------------------------------#
    def _tune_with_optuna_gpsampler(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        categorical_features_subset: list[str],
        groups: pd.Series,        # <-- make sure you pass this in the call
    ) -> tuple[dict[str, Any], lgb.LGBMRegressor]:

        # ---------- decide how many splits we can afford ----------
        min_required_samples = 7          # leave one game for test
        n_samples = len(X)
        max_splits_possible = n_samples - min_required_samples
        n_cv_splits_optuna = min(7, max(1, max_splits_possible))

        if n_cv_splits_optuna < 2:
            logger.warning(
                "Playoff data (%d rows) allows only %d split(s) for Optuna CV; "
                "results may be unstable.",
                n_samples, n_cv_splits_optuna,
            )
        if max_splits_possible < 1:
            raise ValueError(
                f"Not enough playoff rows ({n_samples}) for even one CV split."
            )

        logger.info("Using %d splits for Optuna’s internal CV.", n_cv_splits_optuna)

        # ---------- create the group-aware splitter ----------
        tscv_outer = GroupTimeSeriesSplit(n_splits=n_cv_splits_optuna)

           # ── add **median pruner** right HERE  ───────────────────────────
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials = 15,   # run at least 10 full trials before pruning
            n_warmup_steps   = 50,   # ignore the first 30 boosting rounds
            interval_steps   = 10,   # then check every 10 rounds
        )

        def objective(trial: optuna.Trial) -> float:
            params: dict[str, Any] = {
                "objective": "regression", "metric": "rmse", "verbosity": -1,
                "seed": self.random_state, "deterministic": True, "feature_pre_filter": False,
                # Adjust ranges slightly for potentially smaller playoff data
                "max_depth": trial.suggest_int("max_depth", 2, 32), # Slightly smaller range?
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 8, 256), # Slightly smaller range
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 200), # Range based on expected fold size
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 200), # Range based on expected fold size
                "subsample": trial.suggest_float("subsample", 0.5, 0.9), # Smaller range?
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0), # Smaller range?
                "bagging_freq": 1, # Typically 1 or 0
                "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 5.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1.0, 10.0, log=True),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1), # Smaller range
                "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
                "max_bin": trial.suggest_int("max_bin", 64, 255), # Smaller max_bin might help generalization
            }
            if params["boosting_type"] == "dart":
                params["drop_rate"] = trial.suggest_float("drop_rate", 0.05, 0.3) # Smaller rate?

            if self.use_gpu:
                 params.update({"device_type": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})

            fold_scores: list[float] = []
            fold_iters: list[int] = []
            # Use the specific subset of categoricals passed to this function
            current_cats_in_X = [c for c in categorical_features_subset if c in X.columns]

            for fold_num, (tr_idx, va_idx) in enumerate(
                tscv_outer.split(X, y, groups=groups), 1):
                # Use iloc for safety with potential index resets
                X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

                if X_tr.empty or X_va.empty:
                    logger.warning(f"Optuna CV fold {fold_num+1} resulted in empty train ({len(X_tr)}) or val ({len(X_va)}) set. Skipping fold.")
                    continue

                # Ensure categorical features are type 'category' for this fold's data
                X_tr_fold, X_va_fold = X_tr.copy(), X_va.copy() # Work on copies
                fold_cats_present = []
                for col in current_cats_in_X:
                    # Check if column exists in this specific fold's data
                    if col in X_tr_fold.columns:
                         if X_tr_fold[col].dtype.name != 'category': X_tr_fold[col] = X_tr_fold[col].astype('category')
                         if X_va_fold[col].dtype.name != 'category': X_va_fold[col] = X_va_fold[col].astype('category')
                         fold_cats_present.append(col)

                dtrain = lgb.Dataset(X_tr_fold, label=y_tr, categorical_feature=fold_cats_present or 'auto', free_raw_data=False)
                dvalid = lgb.Dataset(X_va_fold, label=y_va, reference=dtrain, categorical_feature=fold_cats_present or 'auto', free_raw_data=False)

                evals_result = {}
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning) # Suppress specific LGBM warnings
                    
                    # ▼▼▼  ADD THESE THREE LINES  ▼▼▼
                    prune_cb = LightGBMPruningCallback(
                        trial, metric="rmse", valid_name="valid_0"
                    )
                    try:
                        booster = lgb.train(
                            params,
                            dtrain,
                            num_boost_round=800,
                            valid_sets=[dvalid],
                            valid_names=['valid_0'],
                            callbacks=[
                                lgb.early_stopping(20, verbose=False),
                                lgb.log_evaluation(period=-1),
                                lgb.record_evaluation(evals_result),
                                prune_cb,                      # pruning callback
                            ],
                        )

                    # ---- let Optuna handle pruned trials --------------------
                    except TrialPruned:
                        raise  # bubble up; *do not* mark this fold as an error

                    # ---- handle genuine LightGBM errors --------------------
                    except Exception as e:
                        logger.error(
                            f"Error during Optuna fold {fold_num+1} training: {e}"
                        )
                        fold_scores.append(np.nan)
                        fold_iters.append(300)
                        continue

                # Score retrieval logic (robust check)
                best_score = np.nan
                best_iter = 1500 # Default max

                if booster.best_score and 'valid_0' in booster.best_score and 'rmse' in booster.best_score['valid_0']:
                    best_score = booster.best_score["valid_0"]["rmse"]
                    best_iter = booster.best_iteration if booster.best_iteration > 0 else 1
                elif evals_result and 'valid_0' in evals_result and 'rmse' in evals_result['valid_0']:
                    rmse_list = evals_result['valid_0']['rmse']
                    if rmse_list: # Ensure list is not empty
                         best_iter_idx = np.argmin(rmse_list)
                         best_score = rmse_list[best_iter_idx]
                         best_iter = best_iter_idx + 1 # Iteration is 1-based index
                    else: # Handle empty eval result list case
                         logger.warning("Eval results empty for Optuna fold. Using NaN.")
                else:
                    logger.warning("Could not retrieve score/iteration for Optuna fold. Using NaN.")

                fold_scores.append(best_score)
                fold_iters.append(best_iter)

            valid_scores = [s for s in fold_scores if not np.isnan(s)]
            if not valid_scores: return float('inf') # Return high value if all folds failed

            avg_score = float(np.mean(valid_scores))
            # Use median iteration to be robust to outliers if one fold runs long/short
            avg_iter = int(np.median(fold_iters)) if fold_iters else 100 # Default if list is empty

            trial.set_user_attr("best_iter", avg_iter) # Store median iter
            return avg_score

        # Choose sampler
        try:
            if importlib.util.find_spec("torch") is not None:
                sampler: optuna.samplers.BaseSampler = optuna.samplers.GPSampler(seed=self.random_state)
                logger.info("Using Optuna GPSampler (requires PyTorch).")
            else: raise ImportError
        except (ImportError, AttributeError):
            logger.warning("PyTorch not found or GPSampler issue — using Optuna TPESampler.")
            sampler = optuna.samplers.TPESampler(seed=self.random_state)

        # Setup playoff study
        self.study_path.parent.mkdir(parents=True, exist_ok=True)
        storage_name = f"sqlite:///{self.study_path}"
        study_name = f"lgbm-playoffs-tuning-{self.target}-{datetime.now():%Y%m%d}" # Add date to study name
        logger.info(f"Using Playoff Optuna study '{study_name}' with storage: {storage_name}")
        study = optuna.create_study(
             study_name=study_name, storage=storage_name,
             direction="minimize", sampler=sampler, pruner = pruner, load_if_exists=True # Load allows continuation
        )

        # Optimize (adjust trials/timeout as needed for playoffs)
        n_optuna_trials = 100
        timeout_optuna = 1200 # 20 minutes
        logger.info(f"Running Optuna optimization for {n_optuna_trials} trials (timeout: {timeout_optuna}s)...")
        study.optimize(objective, n_trials=n_optuna_trials, timeout=timeout_optuna)

        # Get best params and iteration
        best_params = {
             **study.best_trial.params, "objective": "regression", "metric": "rmse",
             "seed": self.random_state, "deterministic": True, "feature_pre_filter": False,
        }
        if self.use_gpu:
            best_params.update({"device_type": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})

        n_estimators = study.best_trial.user_attrs.get("best_iter", 100) # Use stored median iter, default 100
        if n_estimators <= 0: n_estimators = 100
        best_params["n_estimators"] = n_estimators

        logger.info(f"Playoff Optuna found best params (RMSE: {study.best_value:.4f}) with {n_estimators} estimators.")

        # Train a model on the full input data (X) using these best params for SHAP step
        logger.info("Training temporary model with best params for SHAP analysis...")
        shap_probe_model = lgb.LGBMRegressor(**best_params)
        final_cats_in_X = [c for c in categorical_features_subset if c in X.columns]

        X_copy_for_fit = X.copy() # Avoid modifying original X
        fit_cats = []
        for col in final_cats_in_X:
            if X_copy_for_fit[col].dtype.name != 'category':
                 try:
                      X_copy_for_fit[col] = X_copy_for_fit[col].astype('category')
                      fit_cats.append(col)
                 except Exception as e: logger.warning(f"Could not convert '{col}' to category for SHAP model fit: {e}")
            else:
                 fit_cats.append(col) # Already category

        try:
            shap_probe_model.fit(X_copy_for_fit, y, categorical_feature=fit_cats or 'auto')
        except Exception as e:
            logger.error(f"Error fitting temporary model for SHAP: {e}")
            raise # Re-raise error as SHAP cannot proceed

        return best_params, shap_probe_model # Return params AND the fitted model

    # ---------------------------------------------------------------------#
    # Training Pipeline (MODIFIED to calculate averages BEFORE FE)
    # ---------------------------------------------------------------------#
    def train(self, *, force_rebuild_cache: bool = False) -> dict:
        """
        End‑to‑end training routine for the standalone playoff model:
          1. Raw‑data ingest (SQLite - all data)
          2. Calculate Career/Season PTS Averages (Shifted) on ALL data.
          3. Filter for Playoff Games
          4. Merge Career/Season PTS Averages onto playoff data.
          5. Playoff Feature‑engineering & caching (using playoff data + merged avgs)
          6. Data prep (Initial X, y, potential cats from engineered playoff data)
          7. MI filter (on playoff data)
          8. Nested‑CV Optuna tuning (on MI-filtered playoff data)
          9. SHAP pruning (using tuned playoff model) -> Export KEPT Features CSV
         10. Final fit + CV (on SHAP-pruned playoff data, NO seasonal weight)
         11. Artifact export (Playoff Paths) + OOF preds export
        """
        logger.info(f"=== Standalone Playoff Training pipeline started for target: {self.target} ===")

        # 1. Raw data (Get all data first)
        raw_df_all = self.retrieve_data() # Sorted by player_id, game_date

        # 2. Calculate Career/Season PTS Averages (Shifted) on ALL data
        target_col = self.target
        career_avg_col = f"{target_col}_career_avg"
        season_avg_col = f"{target_col}_season_avg"

        if target_col not in raw_df_all.columns:
             raise ValueError(f"Target column '{target_col}' not found in retrieved data.")
        if 'season' not in raw_df_all.columns:
            # Already checked in retrieve_data, but double-check
            raise ValueError("'season' column required for season average calculation is missing.")

        logger.info(f"Calculating shifted career and season averages for '{target_col}' on all data...")
        # Group by player for career average
        player_group_all = raw_df_all.groupby('player_id')
        # Use transform to broadcast result back to original shape
        career_avg_shifted = player_group_all[target_col].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean() # Use expanding mean over all past games
        )

        # Group by player and season for season average
        player_season_group_all = raw_df_all.groupby(['player_id', 'season'], observed=True, sort=False)
        season_avg_shifted = player_season_group_all[target_col].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean() # Use expanding mean over past games *within* season
        )

        # Store these averages with the index of raw_df_all
        averages_df = pd.DataFrame({
            career_avg_col: career_avg_shifted,
            season_avg_col: season_avg_shifted
        }, index=raw_df_all.index)

        # 3. Filter for Playoff Games
        logger.info("Filtering raw data for PLAYOFFS ONLY...")
        if 'is_playoffs' not in raw_df_all.columns:
            raise ValueError("'is_playoffs' column missing from raw data.")
        if not pd.api.types.is_numeric_dtype(raw_df_all['is_playoffs']): # Ensure numeric for filter
             raw_df_all['is_playoffs'] = pd.to_numeric(raw_df_all['is_playoffs'], errors='coerce').fillna(0)

        # Keep original index when filtering for merging averages later
        raw_df_playoffs = raw_df_all[raw_df_all['is_playoffs'] == 1].copy()
        playoff_raw_rows = len(raw_df_playoffs)
        if playoff_raw_rows == 0:
            logger.warning("No playoff games found in the raw data. Cannot train.")
            return {"error": "No playoff data found in source DB"}
        logger.info(f"Using {playoff_raw_rows} playoff game rows for feature engineering and training.")

        # 4. Merge Career/Season PTS Averages onto playoff data
        # Use the index preserved from raw_df_all to merge the calculated averages
        logger.info(f"Merging pre-calculated '{target_col}' averages onto playoff data...")
        # Ensure indices align - should be okay as raw_df_playoffs is a slice of raw_df_all
        df_playoffs_with_avgs = raw_df_playoffs.join(averages_df, how='left')

        # Check merge success
        if df_playoffs_with_avgs[career_avg_col].isnull().any() or df_playoffs_with_avgs[season_avg_col].isnull().any():
             nan_career = df_playoffs_with_avgs[career_avg_col].isnull().sum()
             nan_season = df_playoffs_with_avgs[season_avg_col].isnull().sum()
             logger.warning(f"NaNs found after merging averages: {nan_career} in career_avg, {nan_season} in season_avg. Consider imputation if problematic.")
             # Impute with 0 for now if needed (e.g., first game of career/season)
             df_playoffs_with_avgs[career_avg_col] = df_playoffs_with_avgs[career_avg_col].fillna(0) # <-- CODE LINE 1
             df_playoffs_with_avgs[season_avg_col] = df_playoffs_with_avgs[season_avg_col].fillna(0) # <-- CODE LINE 2

        # 5. Playoff Feature cache (Operates ONLY on playoff data, now with merged averages)
        # Cache key based on the raw playoff data hash (might not reflect merged cols - consider this)
        # For simplicity, keep existing hash method. Rebuild cache if issues arise.
        # The INPUT to feature_engineering is now df_playoffs_with_avgs
        cache_key = f"{_hash_raw_df(raw_df_playoffs)}_{PLAYOFF_FEATURE_CACHE_VERSION}" # Base key on original playoff raw data
        cache_path = PLAYOFF_FEATURE_CACHE_DIR / f"features_playoffs_{cache_key}.parquet"

        if cache_path.exists() and not force_rebuild_cache:
            logger.info("Loading engineered PLAYOFF features from cache: %s", cache_path)
            df_engineered_playoffs = pd.read_parquet(cache_path)
            # Ensure correct dtypes after loading
            if 'season' in df_engineered_playoffs.columns: df_engineered_playoffs['season'] = df_engineered_playoffs['season'].astype(str)
            if 'game_date' in df_engineered_playoffs.columns: df_engineered_playoffs['game_date'] = pd.to_datetime(df_engineered_playoffs['game_date'])

            # *** Crucial: Merge the averages AGAIN after loading from cache ***
            # Because the cached version doesn't contain the averages calculated in step 2.
            logger.info("Merging pre-calculated averages onto CACHED engineered playoff data...")
            # Ensure indices align - cache should store index
            if not df_engineered_playoffs.index.equals(df_playoffs_with_avgs.index):
                 logger.warning("Cached playoff features index differs from playoff data index. Reindexing cached data.")
                 # Reindex based on the index of the data that HAS the averages
                 df_engineered_playoffs = df_engineered_playoffs.reindex(df_playoffs_with_avgs.index)

            # Select only the average columns to avoid duplication if FE somehow created them
            avg_cols_to_merge = [career_avg_col, season_avg_col]
            df_engineered_playoffs = df_engineered_playoffs.drop(columns=avg_cols_to_merge, errors='ignore') # Drop if exist
            df_engineered_playoffs = df_engineered_playoffs.join(df_playoffs_with_avgs[avg_cols_to_merge], how='left')
            # Impute any NaNs potentially introduced by join/reindex mismatch
            if df_engineered_playoffs[career_avg_col].isnull().any() or df_engineered_playoffs[season_avg_col].isnull().any():
                 logger.warning("NaNs found in averages after merging onto cached data. Imputing with 0.")
                 # --- FIX: Use assignment instead of inplace=True ---
                 df_engineered_playoffs[career_avg_col] = df_engineered_playoffs[career_avg_col].fillna(0)
                 df_engineered_playoffs[season_avg_col] = df_engineered_playoffs[season_avg_col].fillna(0)
                 # --- END FIX ---

        else:
            logger.info("Playoff cache miss/rebuild — running feature engineering (on playoff data + merged avgs)...")
            # Pass playoff data WITH merged averages to the feature engineering pipeline
            df_engineered_playoffs = self.feature_engineering(df_playoffs_with_avgs) # This preserves index

            logger.info("Writing engineered PLAYOFF features (excl. pre-calced avgs) to cache: %s", cache_path)
            try:
                # Save WITHOUT the dynamically added averages, as they are recalculated/merged anyway
                cols_to_cache = [c for c in df_engineered_playoffs.columns if c not in [career_avg_col, season_avg_col]]
                if not cols_to_cache:
                     logger.warning("No columns left to cache after excluding averages.")
                else:
                     df_engineered_playoffs[cols_to_cache].to_parquet(cache_path, index=True) # Save index
            except Exception as e:
                 logger.error(f"Failed to write playoff feature cache: {e}", exc_info=True)


        # --- Subsequent steps operate on df_engineered_playoffs (which now includes the averages) ---
        if df_engineered_playoffs.empty:
            logger.error("Playoff feature engineering resulted in an empty DataFrame. Cannot proceed.")
            return {"error": "Playoff feature engineering produced no data"}

        # 6. Prepare model data (using engineered playoff FE results + merged avgs)
        # This calls the MODIFIED prepare_model_data which expects the avg columns
        df_model, X_initial, y, groups = self.prepare_model_data(df_engineered_playoffs)

        if X_initial.empty or y.empty:
            logger.warning("No data left after prepare_model_data (playoffs).")
            return {"error": "No data post-prep (playoffs)"}

        # 7. MI Filter (on playoff data X_initial, y)
        logger.info(f"Computing mutual‑information filter (k={TOP_MI_FEATURES}) for playoffs...")
        X_processed = X_initial.copy() # Start with features from prepare_model_data

        # 7.1 Numeric frame for MI (includes new avg features)
        X_numeric_mi = X_processed.copy()
        potential_categorical = self.categorical_features # Get the list from prepare_model_data
        cats_in_initial = [c for c in potential_categorical if c in X_numeric_mi.columns]

        for c in cats_in_initial:
            # Convert to category if not already
            if X_numeric_mi[c].dtype.name != 'category':
                try: X_numeric_mi[c] = X_numeric_mi[c].astype('category')
                except Exception as e:
                     logger.warning(f"Could not convert '{c}' to category for MI. Using raw values if numeric, else skipping.")
                     if not pd.api.types.is_numeric_dtype(X_numeric_mi[c]):
                          X_numeric_mi = X_numeric_mi.drop(columns=[c]) # Drop non-numeric that failed conversion
                     continue

            # Handle NaNs before getting codes
            if X_numeric_mi[c].isnull().any():
                # Using simpler NaN handling for codes: fill after getting codes if needed
                pass # Let .cat.codes assign -1 for NaN

            # Get codes
            try:
                X_numeric_mi[c] = X_numeric_mi[c].cat.codes.astype("int32")
                # Handle potential -1 codes from NaNs explicitly after conversion
                if (X_numeric_mi[c] == -1).any():
                     logger.debug(f"NaNs resulted in -1 codes for '{c}'. Will be treated as a separate category value.")
                     # Optional: Replace -1 with another value if desired, e.g., X_numeric_mi[c].replace(-1, -999, inplace=True)
            except Exception as e:
                logger.error(f"Failed to get category codes for {c}: {e}. Dropping column for MI.")
                X_numeric_mi = X_numeric_mi.drop(columns=[c], errors='ignore')


        # 7.2 Drop all‑NaN columns
        all_nan_cols = X_numeric_mi.columns[X_numeric_mi.isna().all()]
        if len(all_nan_cols):
            logger.warning(f"Dropping {len(all_nan_cols)} all‑NaN columns before MI (playoffs): {list(all_nan_cols[:5])}...")
            X_numeric_mi = X_numeric_mi.drop(columns=all_nan_cols)

        if X_numeric_mi.empty: raise ValueError("No features remain after dropping all-NaN columns before MI.")

        # 7.3 Sample + Impute for MI calculation
        mi_sample_size = min(10_000, len(X_numeric_mi))
        if mi_sample_size <= 0: raise ValueError("Cannot sample for MI, no playoff data rows available.")

        if not X_numeric_mi.index.is_unique:
            logger.warning("Resetting non-unique index before MI sampling.")
            X_numeric_mi = X_numeric_mi.reset_index(drop=True)
            y = y.set_axis(X_numeric_mi.index) # Align y index

        rng = np.random.RandomState(self.random_state)
        sample_idx = rng.choice(X_numeric_mi.index, size=mi_sample_size, replace=False)
        X_numeric_mi_sample = X_numeric_mi.loc[sample_idx]
        y_mi_sample = y.loc[sample_idx]

        # Impute only the sample for MI calculation speed
        if X_numeric_mi_sample.isna().any().any():
            logger.info("Imputing missing values (median) for MI calculation sample...")
            imp = SimpleImputer(strategy="median")
            numeric_cols_mi_sample = X_numeric_mi_sample.select_dtypes(include=np.number).columns
            if not numeric_cols_mi_sample.empty:
                 # Impute only numeric columns
                 X_mi_imputed_numeric = imp.fit_transform(X_numeric_mi_sample[numeric_cols_mi_sample])
                 X_mi_imputed = pd.DataFrame(X_mi_imputed_numeric, columns=numeric_cols_mi_sample, index=X_numeric_mi_sample.index)
                 # Add back non-numeric columns (category codes already handled NaNs)
                 non_numeric_cols = X_numeric_mi_sample.columns.difference(numeric_cols_mi_sample)
                 if not non_numeric_cols.empty:
                      X_mi_imputed = pd.concat([X_mi_imputed, X_numeric_mi_sample[non_numeric_cols]], axis=1)[X_numeric_mi_sample.columns] # Keep order
                 # Fill any remaining NaNs (e.g., if entire numeric col was NaN) with 0
                 X_mi_imputed = X_mi_imputed.fillna(0)
            else: # No numeric columns to impute
                 X_mi_imputed = X_numeric_mi_sample.fillna(0) # Fill any Nans (e.g., failed cat codes) with 0
        else:
            X_mi_imputed = X_numeric_mi_sample # No imputation needed

        # ───────────────────────────────────────────────────────────────
        # 7.4  Mutual-information filter WITH “keep-one-of {base, base_sq}”
        # ───────────────────────────────────────────────────────────────
        # 1) Numeric frame for MI
        X_mi_input = X_mi_imputed.select_dtypes(include=np.number)
        X_mi_input = X_mi_input.replace([np.inf, -np.inf], np.nan).fillna(0)

        if X_mi_input.empty:
            raise ValueError("No numeric features left for MI calculation.")

        # 2) MI scores
        mi_scores = mutual_info_regression(
            X_mi_input, y_mi_sample, random_state=self.random_state
        )
        mi_series = (
            pd.Series(mi_scores, index=X_mi_input.columns, name="mi")
            .fillna(0)
            .sort_values(ascending=False)
        )

        # 3) Choose the better twin for each {base, base_sq}
        def _choose_sq(mi: pd.Series) -> list[str]:
            keep, visited = [], set()
            for col in mi.index:                         # already sorted desc
                base = col[:-3] if col.endswith("_sq") else col
                if base in visited:
                    continue
                twin = f"{base}_sq"
                if twin in mi.index and twin != col:
                    better = col if mi[col] >= mi[twin] else twin
                    keep.append(better)
                else:
                    keep.append(col)
                visited.add(base)
            return keep

        mi_series = mi_series.loc[_choose_sq(mi_series)]

        # 4) Top-K after de-duplication
        n_keep = min(TOP_MI_FEATURES,
                    len(mi_series[mi_series > 1e-5]) or len(mi_series))
        if n_keep == 0:
            raise ValueError("MI calculation resulted in zero features to keep.")

        mi_kept_features = mi_series.head(n_keep).index.tolist()

        # 5) Slice the working frame
        mi_kept_features_present = [f for f in mi_kept_features if f in X_processed.columns]
        if not mi_kept_features_present:
            raise ValueError("None of the MI-kept features are in X_processed!")

        X_filtered = X_processed[mi_kept_features_present].copy()

        # 6) Book-keeping
        self.initial_feature_names = mi_kept_features_present
        self.categorical_features  = [
            c for c in self.categorical_features if c in mi_kept_features_present
        ]
        cats_after_mi = self.categorical_features  # <- will be passed to Optuna

        # 7) Realign y if the index changed (rare, but safe)
        if not y.index.equals(X_filtered.index):
            logger.warning("Realigning y to MI-filtered X index.")
            y = y.reindex(X_filtered.index)
            valid_idx = y.dropna().index
            X_filtered = X_filtered.loc[valid_idx]
            y = y.loc[valid_idx]
            if X_filtered.empty:
                raise ValueError("No data left after realigning y.")
        # ───────────────────────────────────────────────────────────────
        # 8. Optuna Tuning (on playoff data X_filtered, y, cats_after_mi)
        logger.info(f"Starting Optuna hyperparameter tuning (playoffs) on {len(mi_kept_features_present)} features...")
        best_params, shap_probe_model = self._tune_with_optuna_gpsampler(
            X_filtered, y, cats_after_mi,  # existing args
            groups.loc[X_filtered.index]   # NEW fourth arg
        )

        # 9. SHAP Pruning (using shap_probe_model trained by Optuna)
        logger.info("Starting SHAP-based feature pruning (playoffs)...")
        n_shap_samples = min(5000, len(X_filtered))
        if n_shap_samples <= 0: raise ValueError("Cannot calculate SHAP, no data rows available.")

        if not X_filtered.index.is_unique: logger.warning(
            "Non-unique index detected in X_filtered; keeping it intact to preserve "
            "the link back to raw_df_all."
        )
        shap_sample_indices = np.random.choice(X_filtered.index.unique(), size=n_shap_samples, replace=False)
        X_shap_sample = X_filtered.loc[shap_sample_indices].copy()

        # Ensure categories are set for SHAP sample
        cats_for_shap_probe = [c for c in cats_after_mi if c in X_shap_sample.columns]
        for col in cats_for_shap_probe:
            if X_shap_sample[col].dtype.name != 'category':
                try: X_shap_sample[col] = X_shap_sample[col].astype('category')
                except Exception as e: logger.warning(f"Could not convert SHAP sample col '{col}' to category: {e}")

        logger.info(f"Calculating SHAP values on {n_shap_samples} playoff samples...")
        try:
            # Ensure the probe model uses correct categorical feature list if needed by predict method
            shap_mat = shap_probe_model.predict(X_shap_sample, pred_contrib=True)
        except Exception as e:
            logger.error(f"SHAP value calculation failed: {e}", exc_info=True)
            raise ValueError("Could not calculate SHAP values.") from e

        # Prune based on SHAP (logic unchanged)
        expected_cols = len(X_filtered.columns) + 1 # Features + Bias term
        if shap_mat.shape[1] != expected_cols:
             logger.error(f"SHAP matrix columns ({shap_mat.shape[1]}) mismatch expected ({expected_cols}). Features in probe model: {len(shap_probe_model.feature_name_)}.")
             raise ValueError(f"SHAP matrix column count ({shap_mat.shape[1]}) != expected ({expected_cols}). Check model features.")

        shap_val_features = np.abs(shap_mat[:, :-1]) # Exclude bias term
        mean_abs_shap = np.mean(shap_val_features, axis=0)
        shap_feature_names = shap_probe_model.feature_name_ # Get feature names from the fitted probe model
        if len(mean_abs_shap) != len(shap_feature_names):
             raise ValueError(f"Mismatch between SHAP values count ({len(mean_abs_shap)}) and feature name count ({len(shap_feature_names)}).")

        # ────────────────── SHAP-based feature pruning (hard cap) ──────────────────
        TOP_SHAP_FEATURES = 100  # <-- hard limit on # of features to keep

        shap_series = pd.Series(mean_abs_shap, index=shap_feature_names).fillna(0)

        best_per_family: dict[str, tuple[str, float]] = {}

        for feat, val in shap_series.items():
            #   A family is “foo” for both “foo” and “foo_sq”
            base = re.sub(r'_sq$', '', feat)

            # If we haven't seen this family OR we just found a bigger SHAP value,
            # record (winning_feature_name, shap_value)
            if base not in best_per_family or val > best_per_family[base][1]:
                best_per_family[base] = (feat, val)

        # Re-create a Series with only the winners
        shap_series_uniqued = pd.Series(
            {feat: val for feat, val in best_per_family.values()},
            name="mean_abs_shap_value",
        ).sort_values(ascending=False)
        
        shap_series.name = "mean_abs_shap_value"

        if shap_series.empty:
            logger.warning("SHAP pruning produced no values; falling back to MI-filtered set.")
            self.feature_names = list(X_filtered.columns)

        else:
            # keep, at most, the TOP_SHAP_FEATURES highest-impact features
            kept_shap_series = shap_series.sort_values(ascending=False).head(TOP_SHAP_FEATURES)
            self.feature_names = kept_shap_series.index.tolist()

            logger.info(
                f"SHAP pruning (playoffs) kept {len(self.feature_names)} / "
                f"{len(X_filtered.columns)} features (hard cap = {TOP_SHAP_FEATURES})."
            )

            # ── export kept features & SHAP values ────────────────────────────────
            try:
                self.kept_features_csv_path.parent.mkdir(parents=True, exist_ok=True)
                kept_df = kept_shap_series.to_frame()
                kept_df.index.name = "feature_name"
                kept_df.to_csv(self.kept_features_csv_path)
                logger.info(
                    f"Exported {len(kept_df)} KEPT playoff features/SHAP values to: "
                    f"{self.kept_features_csv_path}"
                )
            except Exception as e:
                logger.error(f"Failed to export kept playoff SHAP features: {e}")

        if not self.feature_names: raise RuntimeError("No features remained after SHAP pruning (and fallback).")

        self.final_categorical_features = [c for c in cats_after_mi if c in self.feature_names]
        logger.info(f"Final playoff model using {len(self.feature_names)} features ({len(self.final_categorical_features)} categorical).")
        # Log presence of new avg features if kept
        kept_avg_feats = [f for f in [career_avg_col, season_avg_col, f"{target_col}_playoff_career_avg_defaulted"] if f in self.feature_names]
        logger.info(f"Avg features kept in final set: {kept_avg_feats}")


        # 10. Final Refit on SHAP-pruned playoff data
        logger.info("Refitting final playoff model on SHAP-pruned features...")
        final_X = X_filtered[self.feature_names].copy()

        # Align index with y if necessary (e.g., if SHAP sampling reset index)
        if not final_X.index.equals(y.index):
            logger.warning("Realigning final_X index to y index before final fit.")
            # Use intersection to handle potential drops during SHAP sampling/index reset
            common_index = final_X.index.intersection(y.index)
            if common_index.empty: raise ValueError("No common index between final_X and y after filtering/SHAP.")
            final_X = final_X.loc[common_index]
            y = y.loc[common_index]
            if final_X.empty: raise ValueError("No data left after final index alignment for refit.")


        # Ensure categorical dtypes for final fit
        X_final_train = final_X.copy()
        fit_cats_final = []
        for col in self.final_categorical_features:
            if col not in X_final_train.columns: continue
            if X_final_train[col].dtype.name != 'category':
                try:
                     X_final_train[col] = X_final_train[col].astype('category')
                     fit_cats_final.append(col)
                except Exception as e: logger.warning(f"Could not convert final fit col '{col}' to category: {e}")
            else:
                 fit_cats_final.append(col) # Already category

        # Create and fit the final model (stored in self.model)
        self.model = lgb.LGBMRegressor(**best_params) # Use best params from Optuna
        self.model.fit(X_final_train, y, categorical_feature=fit_cats_final or 'auto')
        logger.info("Final playoff model fitted successfully.")

        # 11. Final CV (on SHAP-pruned playoff data, NO seasonal weight)
        logger.info("Starting final cross-validation (playoffs)...")
        X_cv = final_X.copy() # Use data prepared for final fit
        y_cv = y.copy()

        min_train_samples_cv = 6
        n_samples_cv = len(X_cv)
        max_splits_cv = n_samples_cv - min_train_samples_cv
        n_cv_splits = min(6, max(1, max_splits_cv)) # At least 1, max 6

        if n_cv_splits < 2:
            logger.warning(f"Playoff data size ({n_samples_cv}) allows only {n_cv_splits} split(s) for final CV. Skipping CV.")
            avg_metrics = {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
            oof_preds = pd.Series(index=X_cv.index, dtype=float) # Empty preds
        else:
            logger.info(f"Using {n_cv_splits} splits for playoff final CV.")
            groups_cv = groups.loc[X_cv.index].values   # align series_key to this frame
            tscv = GroupTimeSeriesSplit(n_splits=n_cv_splits)
            metrics = {"rmse": [], "mae": [], "r2": []}
            oof_preds = pd.Series(index=X_cv.index, dtype=float) # Store out-of-fold preds

            for split_no, (tr_idx, te_idx) in enumerate(
                tscv.split(X_cv, y_cv, groups=groups_cv), 1):
                logger.info(f"Playoff CV split {split_no}/{n_cv_splits}")
                X_train_fold, X_test_fold = X_cv.iloc[tr_idx, :], X_cv.iloc[te_idx, :]
                y_train_fold, y_test_fold = y_cv.iloc[tr_idx], y_cv.iloc[te_idx]

                if X_train_fold.empty or X_test_fold.empty:
                     logger.warning(f"Skipping CV fold {split_no} due to empty train ({len(X_train_fold)}) or test ({len(X_test_fold)}) data.")
                     metrics["rmse"].append(np.nan); metrics["mae"].append(np.nan); metrics["r2"].append(np.nan)
                     continue

                # Prep fold data (categoricals) - use self.final_categorical_features
                X_train_fold_fit = X_train_fold.copy()
                current_fold_cats = []
                for col in self.final_categorical_features: # Use final list here
                    if col in X_train_fold_fit.columns:
                        if X_train_fold_fit[col].dtype.name != 'category':
                            try: X_train_fold_fit[col] = X_train_fold_fit[col].astype('category'); current_fold_cats.append(col)
                            except Exception as e: logger.warning(f"CV Fold {split_no}: Failed category conversion for '{col}': {e}")
                        else: current_fold_cats.append(col)

                # Train fold model (using best params)
                fold_model = lgb.LGBMRegressor(**best_params)
                try:
                    fold_model.fit(X_train_fold_fit, y_train_fold, categorical_feature=current_fold_cats or 'auto')
                except Exception as e:
                     logger.error(f"Error fitting model for CV fold {split_no}: {e}")
                     metrics["rmse"].append(np.nan); metrics["mae"].append(np.nan); metrics["r2"].append(np.nan)
                     continue

                # Predict and evaluate
                X_test_fold_pred = X_test_fold.copy()
                for col in self.final_categorical_features: # Use final list here
                    if col in X_test_fold_pred.columns and X_test_fold_pred[col].dtype.name != 'category':
                         try: X_test_fold_pred[col] = X_test_fold_pred[col].astype('category')
                         except Exception as e: logger.warning(f"CV Fold {split_no} Test: Failed category conversion for '{col}': {e}")

                try:
                     y_pred = fold_model.predict(X_test_fold_pred)
                     oof_indices = X_cv.iloc[te_idx].index # Get original index for OOF storage
                     oof_preds.loc[oof_indices] = y_pred

                     metrics["rmse"].append(float(np.sqrt(mean_squared_error(y_test_fold, y_pred))))
                     metrics["mae"].append(float(mean_absolute_error(y_test_fold, y_pred)))
                     metrics["r2"].append(float(r2_score(y_test_fold, y_pred)))
                except Exception as e:
                     logger.error(f"Error predicting/evaluating CV fold {split_no}: {e}")
                     metrics["rmse"].append(np.nan); metrics["mae"].append(np.nan); metrics["r2"].append(np.nan)


            # Calculate average metrics (ignoring NaNs)
            avg_metrics = {
                k: float(np.nanmean(v)) if not np.all(np.isnan(v)) else np.nan
                for k, v in metrics.items()
            }
            logger.info(
                "Playoff CV scores: RMSE %.4f | MAE %.4f | R² %.4f",
                avg_metrics["rmse"], avg_metrics["mae"], avg_metrics["r2"],
            )

        # 12. Persist Playoff Artifacts & Export OOF Predictions
        logger.info("Saving playoff artifacts...")
        self.save_artifacts() # Uses playoff paths defined in __init__

        self.model_run_id = datetime.now().strftime("%Y%m%d%H%M%S%f") + f"_playoffs_lgbm_{self.target}"

        # Export OOF preds if CV ran and produced results
        if n_cv_splits >= 2 and not oof_preds.isnull().all():
            logger.info("Preparing to export playoff training (OOF) predictions...")
            # Use the *original* raw_df_all to get metadata, align with y_cv index
            meta_cols = ["game_id", "player_id", "game_date"]
            if all(c in raw_df_all.columns for c in meta_cols):
                 # Align metadata with the y_cv index (which matches oof_preds index)
                 # Use y_cv.index as the definitive index for OOF results
                 if y_cv.index.is_unique:
                      df_meta = raw_df_all.set_index(raw_df_all.index).loc[y_cv.index, meta_cols].copy() # Set index temp for loc
                 else: # Handle non-unique index from raw_df_all if necessary
                      logger.warning("Raw data index not unique, attempting merge for OOF export.")
                      df_meta_temp = raw_df_all[meta_cols].copy()
                      df_meta_temp['original_index'] = raw_df_all.index
                      oof_results_temp = pd.DataFrame({'actual': y_cv, 'predicted': oof_preds})
                      oof_results_temp['original_index'] = y_cv.index
                      # Merge based on the original index
                      export_df = pd.merge(df_meta_temp, oof_results_temp, on='original_index', how='inner')
                      export_df = export_df.drop(columns=['original_index'])
                      # Filter NaNs added by merge/join
                      export_df = export_df.dropna(subset=['predicted'])
                      if not export_df.empty:
                          logger.info(f"Exporting {len(export_df)} playoff training predictions...")
                          self.export_training_predictions(
                              export_df['game_id'], export_df['player_id'], export_df['game_date'],
                              export_df['actual'], export_df['predicted'], self.model_run_id
                          )
                      else: logger.warning("No OOF predictions available to export (after merge).")
                      export_df = None # Signal that export happened or failed within this block

                 # If index was unique and loc worked:
                 if 'export_df' not in locals() or export_df is not None: # Check if not handled by merge block
                     oof_results = pd.DataFrame({'actual': y_cv, 'predicted': oof_preds}).dropna(subset=['predicted'])
                     export_df = df_meta.join(oof_results, how='inner') # Join on index

                     if not export_df.empty:
                          logger.info(f"Exporting {len(export_df)} playoff training predictions...")
                          self.export_training_predictions(
                              export_df['game_id'], export_df['player_id'], export_df['game_date'],
                              export_df['actual'], export_df['predicted'], self.model_run_id
                          )
                     else: logger.warning("No OOF predictions available to export (after joining metadata).")
            else: logger.warning("Metadata columns missing in raw data. Cannot export OOF.")
        else: logger.info("Skipping playoff OOF prediction export (CV skipped or no preds generated).")


        logger.info(f"=== Standalone Playoff Training pipeline finished for target: {self.target} ===")
        return avg_metrics

    # ---------------------------------------------------------------------#
    # Export Predictions (Unchanged)
    # ---------------------------------------------------------------------#
    def export_training_predictions(
        self, game_ids: pd.Series, player_ids: pd.Series, game_dates: pd.Series,
        y_actual: pd.Series, y_predicted: pd.Series, model_run: str
    ):
        """Persist OOF‑style training predictions to SQLite."""
        if not all(len(s) == len(game_ids) for s in [player_ids, game_dates, y_actual, y_predicted]):
            logger.error("Prediction export failed: Input Series lengths mismatch.")
            return
        if game_ids.empty:
            logger.warning("No training predictions to export.")
            return

        # Ensure indices align before calculation (though join should handle this)
        common_index = y_actual.index.intersection(y_predicted.index)
        if len(common_index) != len(y_actual) or len(common_index) != len(y_predicted):
             logger.warning("Aligning actual and predicted indices before calculating residual.")
             y_actual = y_actual.loc[common_index]
             y_predicted = y_predicted.loc[common_index]
             # Also align metadata Series
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
             }, columns=cols) # Enforce column order

        # Filter out rows where prediction failed (NaN residual)
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

            conn = sqlite3.connect(self.db_file, timeout=30.0) # Increased timeout
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout = 30000;") # 30 seconds busy timeout
            table_name = "training_predictions"
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    game_id TEXT NOT NULL, player_id TEXT NOT NULL, game_date TEXT NOT NULL,
                    actual_points REAL, predicted_points REAL, residual REAL,
                    model_run TEXT NOT NULL,
                    PRIMARY KEY (game_id, player_id, model_run) );
            """)
            placeholders = ", ".join(["?"] * len(cols))
            insert_sql = f"INSERT OR REPLACE INTO {table_name} ({', '.join(cols)}) VALUES ({placeholders});"

            # Insert in chunks
            chunk_size = 5000
            for i in range(0, len(res_df), chunk_size):
                chunk = res_df.iloc[i:i + chunk_size]
                with conn:
                     conn.executemany(insert_sql, chunk.itertuples(index=False, name=None))

            logger.info("Exported %d training predictions for model run %s.", len(res_df), model_run)
        except sqlite3.Error as e:
            logger.error(f"SQLite error during prediction export: {e}")
        finally:
            if conn: conn.close()

    # ---------------------------------------------------------------------#
    # Artifact I/O (Unchanged - Uses playoff instance paths)
    # ---------------------------------------------------------------------#
    def save_artifacts(self) -> None:
        """Persist playoff model and metadata using instance paths."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.feature_names, self.features_path)
        joblib.dump(self.final_categorical_features, self.cat_features_path)
        logger.info(f"Saved playoff artifacts (model, features, cats) to {self.model_path.parent}")

    def load_artifacts(self) -> None:
        """Load playoff model and metadata using instance paths."""
        try:
            if not self.model_path.is_file(): raise FileNotFoundError(f"Model file not found: {self.model_path}")
            if not self.features_path.is_file(): raise FileNotFoundError(f"Features file not found: {self.features_path}")
            if not self.cat_features_path.is_file(): raise FileNotFoundError(f"Categorical features file not found: {self.cat_features_path}")

            self.model = joblib.load(self.model_path)
            self.feature_names = joblib.load(self.features_path)
            self.final_categorical_features = joblib.load(self.cat_features_path)
            self.scaler = _NoOpScaler() # Recreate scaler
            logger.info(
                "Loaded playoff model and metadata from %s (features: %d, categoricals: %d)",
                self.model_path.parent, len(self.feature_names), len(self.final_categorical_features)
            )
            # Optional sanity check
            model_features = getattr(self.model, 'n_features_', None) or \
                              (len(getattr(self.model, 'feature_name_', [])) if hasattr(self.model, 'feature_name_') else None)

            if model_features is not None and model_features != len(self.feature_names):
                logger.warning(f"Playoff model expects {model_features} features but loaded list has {len(self.feature_names)}.")

        except FileNotFoundError as err:
            logger.error(f"Playoff artifact file not found (target: {self.target}): {err}")
            raise
        except Exception as err:
            logger.error(f"Unexpected error loading playoff artifacts (target: {self.target}): {err}", exc_info=True)
            raise

 # ---------------------------------------------------------------------#
    # Prediction Helpers (REWRITTEN prepare_new_data)
    # ---------------------------------------------------------------------#
    def prepare_new_data(
        self, historical_df: pd.DataFrame, new_game_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepares feature matrix for a new playoff game prediction using historical context.
        Requires loaded playoff artifacts. Uses playoff FE pipeline ONLY, ensuring
        input to the FE pipeline mirrors the training process.

        REWRITTEN LOGIC:
        1. Combines history + new game.
        2. Calculates OVERALL shifted career/season averages on combined data.
        3. Filters combined data for PLAYOFF games only.
        4. Merges overall averages onto the filtered playoff data.
        5. Runs feature_engineering ONLY on this playoff data + merged averages.
        6. Selects the new game row(s) and final features.
        7. Performs final imputation and ordering.

        Args:
            historical_df: DataFrame containing historical game data rows (reg + playoff)
                           necessary to compute lag/rolling features and overall averages.
                           Should be sorted by player/date if possible, but will be sorted here.
            new_game_df: DataFrame containing the row(s) for the new playoff game(s).
                         Must contain an 'is_playoffs' flag == 1.

        Returns:
            DataFrame ready for the playoff model's predict method.
        """
        if not self.feature_names:
            try:
                self.load_artifacts() # Load if not already loaded
            except Exception as load_err:
                 logger.error(f"Failed to load artifacts required for prepare_new_data: {load_err}", exc_info=True)
                 raise ValueError("Playoff artifacts required for prediction (loading failed?).") from load_err
        if not self.feature_names: raise ValueError("Playoff feature names could not be loaded.")
        if not self.model: raise ValueError("Playoff model must be loaded for prediction.")

        logger.info("Preparing data for new playoff game prediction (mirroring training FE input)...")

        # --- 1. Basic Input Preparation & Validation ---
        if historical_df is None or new_game_df is None:
            raise ValueError("Both historical_df and new_game_df must be provided.")
        if new_game_df.empty:
             raise ValueError("new_game_df cannot be empty.")

        historical_df_copy = historical_df.copy()
        new_game_df_copy = new_game_df.copy()

        # ─── PATCH: guarantee an `is_home` column on the historical rows ───
        if 'is_home' not in historical_df_copy.columns and 'team_is_home' in historical_df_copy.columns:
            historical_df_copy['is_home'] = historical_df_copy['team_is_home']

        # (the new-game row already has is_home, but add this guard for safety)
        if 'is_home' not in new_game_df_copy.columns and 'team_is_home' in new_game_df_copy.columns:
            new_game_df_copy['is_home'] = new_game_df_copy['team_is_home']

        # Basic type/column checks & Ensure 'is_playoffs' exists and is valid
        for df_, name in [(historical_df_copy, 'historical_df'), (new_game_df_copy, 'new_game_df')]:
            if df_.empty and name == 'historical_df': continue # Allow empty history if player is new

            required_cols_pred = ['game_date', 'season', 'player_id', 'is_playoffs', self.target]
            missing_cols = [c for c in required_cols_pred if c not in df_.columns]
            if missing_cols:
                 # Allow target to be missing in new_game_df, fill later
                 if self.target in missing_cols and name == 'new_game_df':
                      missing_cols.remove(self.target)
                      df_[self.target] = np.nan # Add target as NaN
                 if missing_cols: # If others are still missing
                    raise ValueError(f"Missing required columns in {name}: {missing_cols}")

            df_['game_date'] = pd.to_datetime(df_['game_date'], errors='coerce')
            if df_['game_date'].isnull().any(): raise ValueError(f"Invalid or missing game_date found in {name}.")
            df_['game_date'] = df_['game_date'].dt.tz_localize(None) # Ensure naive datetime

            if 'season' in df_.columns: df_['season'] = df_['season'].astype(str)
            else: raise ValueError(f"'season' column missing from {name}.") # Required for avg calc

            if 'player_id' not in df_.columns: raise ValueError(f"'player_id' column missing from {name}.")

            # Ensure 'is_playoffs' is numeric and valid in new_game_df
            if 'is_playoffs' in df_.columns:
                 df_['is_playoffs'] = pd.to_numeric(df_['is_playoffs'], errors='coerce').fillna(0).astype(int)
                 if name == 'new_game_df' and not (df_['is_playoffs'] == 1).all():
                      raise ValueError("All rows in new_game_df must have 'is_playoffs' == 1 for playoff prediction.")
            else: raise ValueError(f"'is_playoffs' column missing from {name}.")

            # Ensure win column exists or is derived (needed for context features)
            if 'win' not in df_.columns:
                if 'team_wl' in df_.columns:
                    df_['win'] = df_['team_wl'].apply(lambda x: 1 if isinstance(x, str) and x.upper() == 'W' else 0).astype(int)
                else: df_['win'] = 0 # Default if WL is also missing


        # --- 2. Combine Historical and New Data, Sort ---
        historical_df_copy['is_new'] = 0
        new_game_df_copy['is_new'] = 1
        # Use concat instead of append, ensure indices are unique if overlapping
        # Preserve original index temporarily if needed, but reset after sort usually safer
        combo = pd.concat([historical_df_copy, new_game_df_copy], ignore_index=True)
        # Sort is CRUCIAL for correct shift/expanding calculation
        combo.sort_values(["player_id", "game_date"], inplace=True)
        combo.reset_index(drop=True, inplace=True) # Reset index after sort

        # Identify indices corresponding to the new game(s) within the sorted combo df
        new_game_indices = combo[combo['is_new'] == 1].index
        combo_original_index = combo.index # Keep track of indices before potential filtering

        if len(new_game_indices) != len(new_game_df):
            logger.error(f"Mismatch identifying new game rows after sorting/concat. Expected {len(new_game_df)}, found {len(new_game_indices)}.")
            # Attempt to find based on game_id if possible, otherwise raise error
            if 'game_id' in new_game_df_copy.columns and new_game_df_copy['game_id'].is_unique:
                 new_ids = set(new_game_df_copy['game_id'])
                 matched_indices = combo.index[combo['game_id'].isin(new_ids) & combo['is_new'] == 1]
                 if len(matched_indices) == len(new_ids):
                      logger.warning("Re-identified new game rows using game_id.")
                      new_game_indices = matched_indices
                 else: raise RuntimeError("Could not reliably identify new game rows after sorting.")
            else: raise RuntimeError("Could not reliably identify new game rows after sorting (no unique game_id).")

        combo = combo.drop(columns=['is_new']) # Remove temporary flag

        # --- 3. Calculate OVERALL Shifted Averages on Combined Data ---
        target_col = self.target
        career_avg_col = f"{target_col}_career_avg"
        season_avg_col = f"{target_col}_season_avg"
        logger.debug(f"Calculating overall shifted '{target_col}' averages on combined prediction data...")

        # Ensure target column is numeric for calculations, handle potential NaNs introduced
        combo[target_col] = pd.to_numeric(combo[target_col], errors='coerce')

        # Group by player for career average
        player_group_combo = combo.groupby('player_id', observed=True) # Use observed=True
        combo_career_avg_shifted = player_group_combo[target_col].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )

        # Group by player and season for season average
        player_season_group_combo = combo.groupby(['player_id', 'season'], observed=True, sort=False)
        combo_season_avg_shifted = player_season_group_combo[target_col].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )

        # Store these averages aligned with the combo DataFrame's index
        averages_df_pred = pd.DataFrame({
            career_avg_col: combo_career_avg_shifted,
            season_avg_col: combo_season_avg_shifted
        }, index=combo.index)

        # Impute NaNs that arise from the first game (shift(1)) or missing target values
        averages_df_pred[career_avg_col] = averages_df_pred[career_avg_col].fillna(0)
        averages_df_pred[season_avg_col] = averages_df_pred[season_avg_col].fillna(0)

        # --- 4. Filter Combined Data for PLAYOFF Games Only ---
        logger.debug("Filtering combined data for playoff games to prepare FE input...")
        # Ensure 'is_playoffs' column is present and numeric in combo df
        if 'is_playoffs' not in combo.columns:
             raise ValueError("'is_playoffs' column is unexpectedly missing from combined data.")
        combo['is_playoffs'] = pd.to_numeric(combo['is_playoffs'], errors='coerce').fillna(0).astype(int)

        # Filter, keeping the original index from combo
        combo_playoffs = combo[combo['is_playoffs'] == 1].copy()

        # Verify the new game row(s) are still present
        if not new_game_indices.isin(combo_playoffs.index).all():
            raise ValueError("The new game row(s) were filtered out - ensure they were correctly flagged as playoffs.")

        logger.debug(f"Filtered playoff data shape for FE: {combo_playoffs.shape}")

        # --- 5. Merge Overall Averages onto Playoff-Only Data ---
        logger.debug("Merging overall averages onto filtered playoff data...")
        # Join using the index, which was preserved from 'combo'
        df_for_fe = combo_playoffs.join(averages_df_pred, how='left')

        # Check merge success and handle potential NaNs (should be rare if indices matched)
        if df_for_fe[career_avg_col].isnull().any() or df_for_fe[season_avg_col].isnull().any():
            nan_count_c = df_for_fe[career_avg_col].isnull().sum()
            nan_count_s = df_for_fe[season_avg_col].isnull().sum()
            logger.warning(f"Found {nan_count_c} NaNs in career_avg, {nan_count_s} in season_avg after merge. Imputing with 0.")
            df_for_fe[career_avg_col] = df_for_fe[career_avg_col].fillna(0)
            df_for_fe[season_avg_col] = df_for_fe[season_avg_col].fillna(0)

        # --- 6. Run Feature Engineering on Playoff-Only Data (+ Merged Averages) ---
        logger.debug("Running playoff feature engineering pipeline (on playoff data + overall avgs)...")
        try:
            # This now receives data structured identically to training
            engineered_playoffs_df = self.feature_engineering(df_for_fe)

            # ── ENSURE THE MODEL’S FULL COLUMN LIST IS PRESENT ──
            missing_cols = [
                c for c in self.feature_names
                if c not in engineered_playoffs_df.columns
            ]
            if missing_cols:
                logger.debug(
                    "prepare_new_data: adding %d missing columns: %s",
                    len(missing_cols),
                    ", ".join(missing_cols[:15]) +
                    (" …" if len(missing_cols) > 15 else "")
                )
                for col in missing_cols:
                    engineered_playoffs_df[col] = np.nan

        except Exception as fe_err:
            logger.error(f"Error during playoff feature engineering for prediction: {fe_err}", exc_info=True)
            raise RuntimeError("Feature engineering failed during prediction preparation.") from fe_err

        # --- 7. Select Row(s) for the New Game(s) ---
        # Use .loc with the original indices identified after sorting 'combo'
        logger.debug(f"Selecting engineered row(s) for new game(s) using indices: {new_game_indices.tolist()}")
        try:
             # Ensure the index of engineered_playoffs_df still aligns
             if not engineered_playoffs_df.index.equals(df_for_fe.index):
                  logger.warning("Index mismatch after feature_engineering call. Reindexing engineered data.")
                  engineered_playoffs_df = engineered_playoffs_df.reindex(df_for_fe.index)

             new_rows_engineered = engineered_playoffs_df.loc[new_game_indices]
        except KeyError:
             logger.error(f"Failed to locate new game indices ({new_game_indices.tolist()}) in the engineered playoff dataframe.")
             logger.debug(f"Engineered DF index sample: {engineered_playoffs_df.index[:5].tolist()}...")
             raise RuntimeError("Could not select new game row after feature engineering.")
        except Exception as sel_err:
             logger.error(f"Unexpected error selecting new game row: {sel_err}", exc_info=True)
             raise RuntimeError("Error selecting new game row after feature engineering.") from sel_err


        if new_rows_engineered.empty:
            raise ValueError("Selection resulted in empty DataFrame for new game(s). Check indices and FE.")

        # --- 8. Select Final Features Required by Model ---
        logger.debug(f"Selecting final {len(self.feature_names)} features for model...")
        missing_features = [f for f in self.feature_names if f not in new_rows_engineered.columns]
        if missing_features:
            logger.warning(f"Prediction data missing required features after FE: {missing_features}. Adding with NaN.")
            for f in missing_features: new_rows_engineered[f] = np.nan # Add missing columns as NaN

        # Ensure correct column order and selection
        try:
             # Use reindex to select, order, and handle potential missing columns gracefully if warning above was missed
             final_feature_data = new_rows_engineered.reindex(columns=self.feature_names)
        except Exception as reindex_err:
             logger.error(f"Failed to select/reorder final features: {reindex_err}", exc_info=True)
             raise ValueError("Could not prepare final feature set.") from reindex_err


        # --- 9. Final Preparations (Categoricals, Scaling, Imputation) ---
        logger.debug("Applying final conversions, scaling (NoOp), and imputation...")
        final_data = final_feature_data.copy() # Work on a copy

        # Apply categorical conversions using FINAL playoff categoricals list
        for col in self.final_categorical_features:
            if col in final_data.columns:
                # Handle NaNs before attempting conversion -> fill with -1 for 'category' dtype compatibility
                if final_data[col].isnull().any():
                    fill_val = -1
                    # logger.debug(f"Filling NaN in prediction categorical '{col}' with value '{fill_val}'.")
                    final_data[col] = final_data[col].fillna(fill_val)

                # Convert to category dtype if not already
                if final_data[col].dtype.name != 'category':
                    try:
                        # Attempt base type conversion first for consistency
                        if pd.api.types.is_numeric_dtype(final_data[col]):
                             is_int_like = final_data[col].apply(lambda x: isinstance(x, (int, np.integer)) or (isinstance(x, (float, np.floating)) and float(x).is_integer())).all()
                             if is_int_like: final_data[col] = final_data[col].astype(int)
                        elif col == 'season' or col == 'series_record':
                             final_data[col] = final_data[col].astype(str)

                        final_data[col] = final_data[col].astype("category")
                    except Exception as e:
                        logger.warning(f"Could not convert prediction col '{col}' to category: {e}. Using raw values if numeric, else filling with 0.")
                        # Fallback: If conversion fails, try to keep numeric as is, fill others with 0
                        if not pd.api.types.is_numeric_dtype(final_data[col]):
                             final_data[col] = 0 # Replace non-numeric that failed conversion

        # Apply scaler (NoOp - returns DataFrame)
        final_data = self.scaler.transform(final_data)

        # Impute any remaining NaNs (e.g., from numerical features or failed category conversions)
        if final_data.isna().any().any():
            nan_cols = final_data.columns[final_data.isna().any()].tolist()
            logger.warning(f"Found NaNs in final prediction data ({len(nan_cols)} cols: {nan_cols[:5]}...). Imputing with median/0.")
            numeric_cols = final_data.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                 # Check for all-NaN columns before fitting imputer
                 numeric_data_to_impute = final_data[numeric_cols]
                 cols_to_impute = numeric_data_to_impute.columns[numeric_data_to_impute.notna().any()].tolist()
                 cols_all_nan = numeric_data_to_impute.columns[numeric_data_to_impute.isna().all()].tolist()

                 if cols_to_impute:
                      imp = SimpleImputer(strategy="median")
                      final_data[cols_to_impute] = imp.fit_transform(final_data[cols_to_impute])
                 if cols_all_nan:
                      logger.warning(f"Columns {cols_all_nan} were all NaN, filling with 0.")
                      final_data[cols_all_nan] = final_data[cols_all_nan].fillna(0)

            # Impute remaining non-numeric (should be rare) with 0
            final_data.fillna(0, inplace=True)

        # Ensure columns are in the exact order expected by the model (redundant check after reindex earlier, but safe)
        if list(final_data.columns) != self.feature_names:
            logger.error(f"Column mismatch just before returning from prepare_new_data. Got {len(final_data.columns)}, expected {len(self.feature_names)}.")
            logger.debug(f"Got: {final_data.columns.tolist()}")
            logger.debug(f"Expected: {self.feature_names}")
            # Attempt final reindex
            final_data = final_data.reindex(columns=self.feature_names, fill_value=0) # Fill missing with 0
            if list(final_data.columns) != self.feature_names:
                 raise ValueError("Final column order correction failed in prepare_new_data.")


        logger.info(f"Prepared prediction data shape: {final_data.shape}. Columns match expected features.")
        return final_data


    # File: player_props_playoffs_lightgbm_alt.py
    # Method: predict (within LightGBMPlayoffModel class)

    def predict(self, new_data_prepared: pd.DataFrame) -> np.ndarray:
        """Make predictions on prepared new playoff data."""
        if self.model is None: raise ValueError("Playoff model must be loaded before predicting.")
        if not isinstance(new_data_prepared, pd.DataFrame): raise TypeError("Input must be a pandas DataFrame.")

        # --- Load feature names if needed (should be loaded by prepare_new_data) ---
        if not self.feature_names:
             logger.warning("Feature names not found in predict, attempting to load artifacts...")
             self.load_artifacts() # Ensure artifacts are loaded
             if not self.feature_names: raise ValueError("Failed to load feature names required for prediction.")


        # --- Initial Check for required columns and attempt reorder/selection ---
        missing_req_features = [f for f in self.feature_names if f not in new_data_prepared.columns]
        extra_features = [f for f in new_data_prepared.columns if f not in self.feature_names]

        if missing_req_features or extra_features:
             logger.warning(f"Prediction data columns mismatch. Missing: {len(missing_req_features)}, Extra: {len(extra_features)}. Attempting selection.")
             logger.debug(f"Missing required: {missing_req_features[:5]}...")
             logger.debug(f"Extra found: {extra_features[:5]}...")
             try:
                 # Select only the columns that are in self.feature_names AND in new_data_prepared
                 cols_to_keep = [f for f in self.feature_names if f in new_data_prepared.columns]
                 if len(cols_to_keep) != len(self.feature_names):
                      missing_after_select = [f for f in self.feature_names if f not in cols_to_keep]
                      logger.error(f"CRITICAL: Cannot proceed with prediction. Required features missing even after selection attempt: {missing_after_select}")
                      raise ValueError(f"Required features missing from prediction data: {missing_after_select}")
                 # Select and reorder
                 predict_data = new_data_prepared[self.feature_names].copy()
             except KeyError as e:
                  logger.error(f"Prediction failed: Could not select/reorder required features. Error: {e}")
                  raise ValueError("Prediction data column selection failed.") from e
        else:
             # Columns already match exactly
             predict_data = new_data_prepared.copy()


        # --- Ensure Categorical Dtypes ---
        # (Keep the existing loop to convert columns in predict_data to 'category')
        predict_cats = []
        for col in self.final_categorical_features:
            if col in predict_data.columns: # Check column exists
                if predict_data[col].dtype.name != 'category':
                    # logger.warning(f"Column '{col}' in prediction data was not 'category'. Attempting conversion.") # Make less verbose if needed
                    try:
                        # Handle potential NaNs before conversion if not already done
                        if predict_data[col].isnull().any():
                            predict_data[col] = predict_data[col].fillna(-1) # Use consistent NaN fill
                        # Convert base type if needed
                        if pd.api.types.is_numeric_dtype(predict_data[col]):
                             is_int_like = predict_data[col].apply(lambda x: isinstance(x, (int, np.integer)) or (isinstance(x, (float, np.floating)) and float(x).is_integer())).all()
                             if is_int_like: predict_data[col] = predict_data[col].astype(int)
                        elif col == 'season' or col == 'series_record':
                             predict_data[col] = predict_data[col].astype(str)

                        predict_data[col] = predict_data[col].astype('category')
                        predict_cats.append(col)
                    except Exception as e: raise TypeError(f"Prediction failed: Cannot convert '{col}' to category: {e}") from e
                else:
                    predict_cats.append(col) # Already correct type

        # --- Final NaN Check & Fill ---
        if predict_data.isna().any().any():
             nan_cols = predict_data.columns[predict_data.isna().any()].tolist()
             logger.warning(f"NaNs detected in data ({len(nan_cols)} cols) immediately before prediction. Filling with 0 as final fallback.")
             logger.debug(f"NaN columns before final fill: {nan_cols[:10]}...")
             predict_data.fillna(0, inplace=True) # Final safety net

        # --- Predict using the playoff model ---
        try:
            # predict_data should now have exactly self.feature_names columns in the correct order
            if list(predict_data.columns) != self.feature_names:
                 logger.critical(f"FATAL: Column mismatch IMMEDIATELY before self.model.predict. Got: {predict_data.columns.tolist()}, Expected: {self.feature_names}")
                 # Attempt one last desperate reorder
                 try:
                      predict_data = predict_data[self.feature_names]
                 except Exception as final_reorder_err:
                      raise ValueError("Column mismatch before predict, final reorder failed.") from final_reorder_err

            predictions = self.model.predict(predict_data)
        except Exception as e:
             # Enhanced logging on failure
             logger.error(f"LightGBM prediction failed: {e}", exc_info=True)
             logger.error(f"Data shape passed to predict: {predict_data.shape}")
             logger.error(f"Expected features ({len(self.feature_names)}): {self.feature_names[:10]}...")
             logger.error(f"Actual columns ({len(predict_data.columns)}): {predict_data.columns.tolist()[:10]}...")
             model_feats = []
             try: model_feats = self.model.feature_name_
             except AttributeError: pass
             logger.error(f"Model internal features ({len(model_feats)}): {model_feats[:10]}...")
             if set(self.feature_names) != set(model_feats):
                 logger.error("MISMATCH DETECTED between loaded feature list and model's internal feature list!")
             raise

        return predictions

# ─────────────────────────── CLI entry‑point ────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="player-prop-playoffs-train", # Updated program name
        description="Train the STANDALONE PLAYOFF NBA LightGBM player-prop model.", # Updated description
    )

    parser.add_argument("--db-file", type=str, default=DEFAULT_DB_FILE, help="Path to SQLite DB.")
    parser.add_argument("--target", type=str, default="pts", help="Target variable (default: pts).")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild PLAYOFF feature cache (only uses playoff raw data).")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU usage for LightGBM.")
    parser.add_argument(
        "--playoff-artifacts-dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR_PLAYOFFS),
        help="Directory to save playoff model artifacts."
    )
    # Removed --regular-artifacts-dir argument

    args = parser.parse_args()

    logger.info("Starting standalone playoff model training script.") # Simplified message
    logger.info(f"Database file: {args.db_file}")
    logger.info(f"Target variable: {args.target}")
    logger.info(f"Rebuild Playoff FE Cache: {args.rebuild_cache}")
    logger.info(f"Use GPU: {args.use_gpu}")
    logger.info(f"Playoff Artifacts Dir: {args.playoff_artifacts_dir}")
    # Removed logging for regular season artifacts dir

    # Instantiate the standalone playoff model class
    try:
        playoff_model = LightGBMPlayoffModel(
            db_file=args.db_file,
            target=args.target,
            playoff_artifacts_dir=args.playoff_artifacts_dir,
            # Removed regular_season_artifacts_dir argument
            use_gpu=args.use_gpu,
            random_state=GLOBAL_RANDOM_STATE
        )
    except (ImportError, FileNotFoundError, ValueError) as init_err: # Added ValueError for missing required columns
        logger.exception("Fatal error during model initialization: %s", init_err)
        print(f"\n--- Standalone Playoff Training FAILED during initialization ---")
        print(f"Error: {init_err}")
        sys.exit(1)
    except Exception as init_err:
        logger.exception("Unexpected fatal error during model initialization: %s", init_err)
        print(f"\n--- Standalone Playoff Training FAILED during initialization ---")
        print(f"Error: {init_err}")
        sys.exit(1)

    # Run Training
    try:
        metrics_results = playoff_model.train(force_rebuild_cache=args.rebuild_cache)
        print("\n--- Standalone Playoff Training complete ---")
        if "error" in metrics_results:
            print(f"Training failed: {metrics_results['error']}")
            sys.exit(1)
        else:
            print(f"Target: {args.target}")
            print("Cross-Validation Metrics (Avg over folds, Playoffs Only):")
            for metric_name, metric_value in metrics_results.items():
                 value_str = f"{metric_value:.4f}" if not pd.isna(metric_value) else "NaN"
                 print(f"  {metric_name}: {value_str}")
            print(f"\nPlayoff artifacts saved to: {playoff_model.model_path.parent}")
            print(f"Playoff model: {playoff_model.model_path.name}")
            print(f"Playoff features: {playoff_model.features_path.name}")
            print(f"Playoff categoricals: {playoff_model.cat_features_path.name}")
            print(f"Playoff KEPT features/SHAP list: {playoff_model.kept_features_csv_path.name}")

    except Exception as exc:
        logger.exception("Fatal error during standalone playoff training execution: %s", exc)
        print(f"\n--- Standalone Playoff Training FAILED ---")
        print(f"Error: {exc}")
        sys.exit(1)

    logger.info("Standalone playoff model training script finished successfully.")
    sys.exit(0)