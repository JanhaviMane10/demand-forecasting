# ============================================================
# config.py — Central settings for the whole project
# Change things here and they update everywhere automatically
# ============================================================

import os

# --- Paths ---
DATA_DIR = "data"
MODELS_DIR = "models"
OUTPUT_DIR = "outputs"

# Create folders if they don't exist
for folder in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    os.makedirs(folder, exist_ok=True)

# --- Data Files ---
SALES_FILE      = os.path.join(DATA_DIR, "sales_train_validation.csv")
CALENDAR_FILE   = os.path.join(DATA_DIR, "calendar.csv")
PRICES_FILE     = os.path.join(DATA_DIR, "sell_prices.csv")

# --- Scope: we work with ONE store to keep things fast ---
# FOODS = grocery, HOBBIES = hobby items, HOUSEHOLD = home goods
TARGET_STORE    = "CA_1"       # California Store 1
TARGET_DEPT     = "FOODS_3"    # Food department 3 (most data)
N_ITEMS         = 50           # Use top 50 items for speed (change to None for all)

# --- Time Settings ---
# The dataset has 1913 days. We use last 28 days as final test set.
HORIZON         = 28           # Forecast 28 days ahead (same as M5 competition)
VAL_DAYS        = 28           # Validation window size
N_FOLDS         = 3            # Number of walk-forward validation folds

# --- Quantile Settings (for prediction intervals) ---
# We predict LOW, MID, HIGH estimates
QUANTILES       = [0.1, 0.5, 0.9]   # 10th, 50th (median), 90th percentile
COVERAGE_TARGET = 0.80               # Our interval should contain 80% of real values

# --- LightGBM Settings ---
LGBM_PARAMS = {
    "objective": "quantile",     # This tells LightGBM to do quantile regression
    "metric": "quantile",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "n_jobs": -1,
}

# --- Random Seed (for reproducibility) ---
SEED = 42
