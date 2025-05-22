#!/usr/bin/env python3
"""LightGBM training on all player game rows.

This is a very small replacement for the original
``player_props_playoffs_lightgbm_alt.py`` script.  The old version only
trained on playoff specific rows.  This rewrite uses **all** available
rows from the database when fitting the model.

The script pulls the raw player game data using the SQL definitions from
``sql_player_props_playoffs.py`` (those queries already join everything we
need).  A few simple leak‑free features are added before fitting a basic
``lightgbm.LGBMRegressor`` model.

The goal of this rewrite is to keep the training pipeline easy to follow
while ensuring that both regular‑season and playoff games contribute to
the model.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
import lightgbm as lgb

from sql_player_props_playoffs import VIEW_SETUP_SQL, PLAYER_GAMES_SQL


def load_data(db_file: Path) -> pd.DataFrame:
    """Return all player game rows from the SQLite database."""

    conn = sqlite3.connect(db_file)
    try:
        conn.executescript(VIEW_SETUP_SQL)
        df = pd.read_sql_query(PLAYER_GAMES_SQL, conn, parse_dates=["game_date"])
    finally:
        conn.close()

    # Sort for deterministic expanding calculations
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    return df


def add_basic_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Add very small set of leak‑free features used for training."""

    df = df.copy()

    # Career and season averages up to the previous game
    df["career_avg"] = (
        df.groupby("player_id")[target]
        .shift(1)
        .expanding(min_periods=1)
        .mean()
    )

    df["season_avg"] = (
        df.groupby(["player_id", "season"])[target]
        .shift(1)
        .expanding(min_periods=1)
        .mean()
    )

    # Short term form
    df["roll5"] = (
        df.groupby("player_id")[target]
        .shift(1)
        .rolling(5, min_periods=1)
        .mean()
    )

    df["roll10"] = (
        df.groupby("player_id")[target]
        .shift(1)
        .rolling(10, min_periods=1)
        .mean()
    )

    df.fillna(0, inplace=True)
    return df


def prepare_training(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` for model fitting."""

    y = df[target].astype(float)

    drop_cols = {target, "game_date"}
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Keep only numeric features
    feature_df = feature_df.select_dtypes(include=["number"]).copy()

    return feature_df, y


def train_lgbm(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMRegressor:
    """Fit and return a LightGBM model."""

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        random_state=0,
    )
    model.fit(X, y)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM model on all games")
    parser.add_argument("--db-file", type=Path, required=True, help="SQLite database")
    parser.add_argument("--target", default="pts", help="Target stat")
    parser.add_argument("--model-path", type=Path, default=Path("model_all.joblib"))
    args = parser.parse_args()

    df = load_data(args.db_file)
    df = add_basic_features(df, args.target)
    X, y = prepare_training(df, args.target)

    model = train_lgbm(X, y)

    joblib.dump(model, args.model_path)
    print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    main()

