# ============================================================
# 01_eda.py — Exploratory Data Analysis
# 
# WHAT THIS FILE DOES:
# - Loads the 3 M5 dataset files
# - Explores what the data looks like
# - Creates charts to understand patterns
# - Answers: What are we working with?
#
# HOW TO RUN: Open in VS Code as a Jupyter notebook
# Each section marked with # %% is one cell
# ============================================================

# %% [markdown]
# # Step 1: Exploratory Data Analysis
# Before building any model, we need to understand our data.
# Think of this like reading a book before writing a summary.

# %% — Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from config import *

# Make charts look clean
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (14, 5)
plt.rcParams["font.size"] = 12

print("✅ Libraries loaded successfully!")

# %% — Load the 3 data files
print("Loading data files... (this may take 30 seconds)")

sales    = pd.read_csv(SALES_FILE)
calendar = pd.read_csv(CALENDAR_FILE)
prices   = pd.read_csv(PRICES_FILE)

print(f"\n📦 Sales data:    {sales.shape[0]:,} rows × {sales.shape[1]:,} columns")
print(f"📅 Calendar data: {calendar.shape[0]:,} rows × {calendar.shape[1]:,} columns")
print(f"💰 Prices data:   {prices.shape[0]:,} rows × {prices.shape[1]:,} columns")

# %% — Understand the sales data structure
print("=" * 60)
print("SALES DATA — First 3 rows:")
print("=" * 60)
print(sales.head(3).to_string())
print("\n💡 Each ROW = one product in one store")
print("💡 Each COLUMN (d_1, d_2, ...) = sales on that day")
print("💡 d_1 = day 1, d_1913 = day 1913 (about 5.2 years)")

# %% — Understand the calendar data
print("=" * 60)
print("CALENDAR DATA — First 5 rows:")
print("=" * 60)
print(calendar.head(5).to_string())
print("\n💡 This maps day numbers (d_1, d_2) to actual dates")
print("💡 Also tells us about events like SuperBowl, Christmas etc.")

# %% — Understand the prices data
print("=" * 60)
print("PRICES DATA — First 5 rows:")
print("=" * 60)
print(prices.head(5).to_string())
print("\n💡 Price changes week by week for each product")

# %% — What stores and categories do we have?
print("\n🏪 STORES:", sales["store_id"].unique())
print("\n🛒 CATEGORIES:", sales["cat_id"].unique())
print("\n📊 DEPARTMENTS:", sales["dept_id"].unique())

print(f"\n📌 We will focus on Store: {TARGET_STORE}, Department: {TARGET_DEPT}")

# %% — Filter to our target store + department
mask = (sales["store_id"] == TARGET_STORE) & (sales["dept_id"] == TARGET_DEPT)
sales_filtered = sales[mask].reset_index(drop=True)

print(f"\n✅ After filtering: {len(sales_filtered)} products")

# Optionally limit to top N items (for speed)
if N_ITEMS:
    # Pick top N items by total sales volume
    day_cols = [c for c in sales_filtered.columns if c.startswith("d_")]
    sales_filtered["total_sales"] = sales_filtered[day_cols].sum(axis=1)
    sales_filtered = sales_filtered.nlargest(N_ITEMS, "total_sales").reset_index(drop=True)
    sales_filtered = sales_filtered.drop("total_sales", axis=1)
    print(f"✅ Using top {N_ITEMS} items by total sales volume")

print(f"\nItems we are forecasting:")
print(sales_filtered["item_id"].values[:10], "...")

# %% — Convert wide format to long format
# Right now: each day is a column (1913 columns)
# We want: each day is a row (much easier to work with)
print("\n🔄 Converting data format...")

day_cols = [c for c in sales_filtered.columns if c.startswith("d_")]
id_cols  = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

sales_long = sales_filtered.melt(
    id_vars=id_cols,
    value_vars=day_cols,
    var_name="d",
    value_name="sales"
)

# Add actual dates from calendar
sales_long = sales_long.merge(
    calendar[["d", "date", "wm_yr_wk", "weekday", "wday",
              "month", "year", "event_name_1", "event_type_1",
              "snap_CA", "snap_TX", "snap_WI"]],
    on="d", how="left"
)

# Add prices
sales_long = sales_long.merge(
    prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
    on=["store_id", "item_id", "wm_yr_wk"],
    how="left"
)

# Convert date column to datetime type
sales_long["date"] = pd.to_datetime(sales_long["date"])
sales_long = sales_long.sort_values(["id", "date"]).reset_index(drop=True)

print(f"✅ Long format shape: {sales_long.shape}")
print(f"\nFirst 3 rows of long format:")
print(sales_long.head(3)[["id", "date", "sales", "sell_price", "event_name_1"]].to_string())

# Save for next steps
sales_long.to_csv(f"{OUTPUT_DIR}/sales_long.csv", index=False)
print(f"\n💾 Saved to {OUTPUT_DIR}/sales_long.csv")

# %% — Chart 1: Overall daily sales trend
print("\n📊 Creating charts...")

daily_total = sales_long.groupby("date")["sales"].sum().reset_index()

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(daily_total["date"], daily_total["sales"], linewidth=0.8, color="#2563EB", alpha=0.8)
ax.set_title(f"Total Daily Sales — Store {TARGET_STORE}, Dept {TARGET_DEPT}", fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/chart1_overall_trend.png", dpi=150)
plt.show()
print("💡 You can see yearly patterns (seasonality) in this chart")

# %% — Chart 2: Average sales by day of week
dow_sales = sales_long.groupby("weekday")["sales"].mean().reset_index()
# Sort days properly
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_sales["weekday"] = pd.Categorical(dow_sales["weekday"], categories=day_order, ordered=True)
dow_sales = dow_sales.sort_values("weekday")

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(dow_sales["weekday"], dow_sales["sales"],
              color=["#93C5FD"]*5 + ["#2563EB"]*2)
ax.set_title("Average Sales by Day of Week", fontsize=14, fontweight="bold")
ax.set_xlabel("Day of Week")
ax.set_ylabel("Avg Units Sold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/chart2_day_of_week.png", dpi=150)
plt.show()
print("💡 Weekends (Sat/Sun) typically have higher sales — our model needs to know this")

# %% — Chart 3: Average sales by month
month_sales = sales_long.groupby("month")["sales"].mean().reset_index()
month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
month_sales["month_name"] = month_sales["month"].map(month_names)

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(month_sales["month_name"], month_sales["sales"], color="#2563EB", alpha=0.8)
ax.set_title("Average Sales by Month (Seasonality)", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Avg Units Sold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/chart3_monthly_seasonality.png", dpi=150)
plt.show()
print("💡 November-December spike = holiday season demand")

# %% — Chart 4: Effect of events on sales
event_effect = sales_long.groupby(
    sales_long["event_name_1"].fillna("No Event")
)["sales"].mean().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 5))
event_effect.plot(kind="bar", ax=ax, color="#2563EB", alpha=0.8)
ax.set_title("Average Sales During Events vs Normal Days", fontsize=14, fontweight="bold")
ax.set_xlabel("Event")
ax.set_ylabel("Avg Units Sold")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/chart4_event_effect.png", dpi=150)
plt.show()
print("💡 Some events significantly boost sales — we'll include these as features")

# %% — Chart 5: Sales distribution (are there zeros? outliers?)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(sales_long["sales"], bins=50, color="#2563EB", alpha=0.8, edgecolor="white")
axes[0].set_title("Sales Distribution", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Units Sold")
axes[0].set_ylabel("Frequency")

zero_pct = (sales_long["sales"] == 0).mean() * 100
axes[1].pie([zero_pct, 100-zero_pct],
            labels=[f"Zero sales ({zero_pct:.1f}%)", f"Positive sales ({100-zero_pct:.1f}%)"],
            colors=["#93C5FD", "#2563EB"], autopct="%1.1f%%")
axes[1].set_title("Zero vs Positive Sales Days", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/chart5_distribution.png", dpi=150)
plt.show()
print(f"💡 {zero_pct:.1f}% of days have zero sales — our model must handle this")

# %% — Chart 6: Individual product time series (pick 3 products)
sample_items = sales_long["item_id"].unique()[:3]

fig, axes = plt.subplots(3, 1, figsize=(16, 12))
for i, item in enumerate(sample_items):
    item_data = sales_long[sales_long["item_id"] == item]
    axes[i].plot(item_data["date"], item_data["sales"],
                 linewidth=0.8, color="#2563EB", alpha=0.8)
    axes[i].set_title(f"Product: {item}", fontsize=12, fontweight="bold")
    axes[i].set_ylabel("Units Sold")
    if i == 2:
        axes[i].set_xlabel("Date")

plt.suptitle("Individual Product Time Series", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/chart6_individual_products.png", dpi=150, bbox_inches="tight")
plt.show()
print("💡 Each product has its own unique pattern — some spiky, some smooth")

# %% — Summary
print("\n" + "="*60)
print("📋 EDA SUMMARY")
print("="*60)
print(f"✅ Dataset: {len(sales_filtered)} products, {len(day_cols)} days of history")
print(f"✅ Date range: {sales_long['date'].min().date()} → {sales_long['date'].max().date()}")
print(f"✅ Zero sales rate: {zero_pct:.1f}%")
print(f"✅ Key patterns found:")
print(f"   - Weekend sales spike (Sat/Sun higher)")
print(f"   - Holiday season spike (Nov/Dec)")
print(f"   - Event-driven spikes (SuperBowl, etc.)")
print(f"   - Weekly seasonality visible in charts")
print(f"\n✅ Data saved to {OUTPUT_DIR}/sales_long.csv")
print(f"\n👉 Next step: Run 02_features.py to engineer features for the model")
