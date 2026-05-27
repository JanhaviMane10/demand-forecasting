# ============================================================
# 02_features.py — Feature Engineering
#
# WHAT THIS FILE DOES:
# - Takes the cleaned data from Step 1
# - Creates new columns (features) that help the model predict
# - Features are clues we give the model: "hey, it's a weekend"
#   or "hey, sales were high last week" etc.
#
# WHY THIS MATTERS:
# A model is only as good as the features you give it.
# Raw dates mean nothing to a model. But "is it Saturday?"
# and "what were sales 7 days ago?" are very informative.
#
# HOW TO RUN: After running 01_eda.py
# ============================================================

# %% — Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from config import *

print("✅ Libraries loaded")

# %% — Load the data we saved in Step 1
print("Loading data...")
df = pd.read_csv(f"{OUTPUT_DIR}/sales_long.csv", parse_dates=["date"])
df = df.sort_values(["id", "date"]).reset_index(drop=True)

print(f"✅ Loaded {len(df):,} rows")
print(f"   Date range: {df['date'].min().date()} → {df['date'].max().date()}")

# %% — Group 1: Calendar features
# These tell the model WHEN something is happening
print("\n📅 Creating calendar features...")

df["day_of_week"]    = df["date"].dt.dayofweek       # 0=Monday, 6=Sunday
df["day_of_month"]   = df["date"].dt.day             # 1–31
df["week_of_year"]   = df["date"].dt.isocalendar().week.astype(int)
df["month"]          = df["date"].dt.month           # 1–12
df["year"]           = df["date"].dt.year            # 2011, 2012 etc.
df["quarter"]        = df["date"].dt.quarter         # 1–4
df["is_weekend"]     = (df["day_of_week"] >= 5).astype(int)  # 1 if Sat/Sun
df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)

# Days to end of month (useful for shopping patterns)
df["days_to_month_end"] = df["date"].dt.days_in_month - df["date"].dt.day

print("   ✅ Calendar features done")

# %% — Group 2: Event features
# These tell the model if there's a special event today
print("🎉 Creating event features...")

# Is there ANY event today? (1 = yes, 0 = no)
df["has_event"] = df["event_name_1"].notna().astype(int)

# What TYPE of event? (sporting, national, religious, cultural)
df["is_sporting_event"]  = (df["event_type_1"] == "Sporting").astype(int)
df["is_national_event"]  = (df["event_type_1"] == "National").astype(int)
df["is_religious_event"] = (df["event_type_1"] == "Religious").astype(int)
df["is_cultural_event"]  = (df["event_type_1"] == "Cultural").astype(int)

# SNAP days = government food assistance benefit days
# These cause big spikes in food sales
snap_col = f"snap_{TARGET_STORE.split('_')[0]}"  # snap_CA for CA stores
if snap_col in df.columns:
    df["is_snap_day"] = df[snap_col].astype(int)
else:
    df["is_snap_day"] = 0

print("   ✅ Event features done")

# %% — Group 3: Lag features
# These tell the model what happened RECENTLY
# "Lag 7" = what were sales exactly 7 days ago?
# This is the most powerful set of features for time series
print("⏪ Creating lag features (most important!)...")

# We must create lags PER ITEM (not across the whole dataset)
# Sort properly first
df = df.sort_values(["id", "date"]).reset_index(drop=True)

lag_days = [1, 2, 3, 7, 14, 21, 28]  # 1 day ago, 1 week ago, 2 weeks ago, etc.

for lag in lag_days:
    df[f"lag_{lag}"] = df.groupby("id")["sales"].shift(lag)
    print(f"   ✅ lag_{lag} created")

# %% — Group 4: Rolling average features
# These smooth out noise — "what was the average sales over the last 7 days?"
print("📊 Creating rolling average features...")

# Rolling averages at different window sizes
windows = [7, 14, 28]

for w in windows:
    # Shift by 1 first — we can't use TODAY's sales to predict TODAY
    df[f"rolling_mean_{w}"] = (
        df.groupby("id")["sales"]
        .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    )
    df[f"rolling_std_{w}"] = (
        df.groupby("id")["sales"]
        .transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
    )
    print(f"   ✅ rolling_mean_{w} and rolling_std_{w} created")

# Rolling max and min (captures demand spikes)
df["rolling_max_7"]  = df.groupby("id")["sales"].transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).max()
)
df["rolling_min_7"]  = df.groupby("id")["sales"].transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).min()
)

# %% — Group 5: Price features
print("💰 Creating price features...")

df["sell_price"] = df["sell_price"].ffill() # Fill missing prices

# Price relative to item's own average (is it on sale?)
df["price_mean"] = df.groupby("id")["sell_price"].transform("mean")
df["price_relative"] = df["sell_price"] / df["price_mean"]   # 0.9 = 10% below avg = on sale

# Price change from last week
df["price_change"] = df.groupby("id")["sell_price"].pct_change(periods=7)

print("   ✅ Price features done")

# %% — Group 6: Encode item/store identity
# The model needs to know which item is which (as numbers)
print("🏷️  Encoding item/store identity features...")

df["item_encoded"]  = df["item_id"].astype("category").cat.codes
df["store_encoded"] = df["store_id"].astype("category").cat.codes
df["dept_encoded"]  = df["dept_id"].astype("category").cat.codes
df["cat_encoded"]   = df["cat_id"].astype("category").cat.codes

print("   ✅ Encoding done")

# %% — Drop rows with NaN (from lag creation — first few rows will be NaN)
original_len = len(df)
df = df.dropna(subset=[f"lag_{max(lag_days)}"])  # Drop rows where longest lag is NaN
print(f"\n🧹 Dropped {original_len - len(df):,} rows with NaN lags (expected)")
print(f"   Remaining rows: {len(df):,}")

# %% — Show what features we created
feature_cols = [c for c in df.columns if c not in
                ["id", "item_id", "dept_id", "cat_id", "store_id",
                 "state_id", "d", "date", "wm_yr_wk",
                 "weekday", "event_name_1", "event_type_1",
                 "snap_CA", "snap_TX", "snap_WI", "sales"]]

print(f"\n📋 Features created ({len(feature_cols)} total):")
for i, f in enumerate(feature_cols):
    print(f"   {i+1:2d}. {f}")

# %% — Quick visualization: do lag features actually correlate with sales?
print("\n📊 Checking feature correlations with sales...")

correlations = df[feature_cols + ["sales"]].corr()["sales"].drop("sales").sort_values(ascending=False)

plt.figure(figsize=(10, 8))
correlations.head(15).plot(kind="barh", color="#2563EB", alpha=0.8)
plt.title("Top 15 Features Most Correlated with Sales", fontsize=13, fontweight="bold")
plt.xlabel("Correlation with Sales")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/chart7_feature_correlations.png", dpi=150)
plt.show()
print("💡 Lag features should be at the top — this confirms they're useful")

# %% — Save the processed data
print("\n💾 Saving processed data...")
df.to_csv(f"{OUTPUT_DIR}/sales_features.csv", index=False)
print(f"✅ Saved to {OUTPUT_DIR}/sales_features.csv")

# Also save the feature column list (we'll need this in the model file)
import json
with open(f"{OUTPUT_DIR}/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)
print(f"✅ Feature list saved to {OUTPUT_DIR}/feature_cols.json")

# %% — Summary
print("\n" + "="*60)
print("📋 FEATURE ENGINEERING SUMMARY")
print("="*60)
print(f"✅ Total features created: {len(feature_cols)}")
print(f"✅ Feature groups:")
print(f"   - Calendar features (day, month, weekend, etc.)")
print(f"   - Event features (has_event, snap_day, event_type)")
print(f"   - Lag features (what sold 1/7/14/28 days ago)")
print(f"   - Rolling averages (7/14/28 day windows)")
print(f"   - Price features (price level, price changes)")
print(f"   - Identity encoding (item, store, dept)")
print(f"\n👉 Next step: Run 03_train.py to train the models")
