# 📦 Probabilistic Demand Forecasting
### M5 Walmart Dataset — LightGBM + Quantile Regression + Prediction Intervals

> **Live Demo:** [your-app-link.streamlit.app](#) *(update after deploying)*

---

## 🎯 Business Problem

Walmart stocks 3,049 products across 10 stores. Ordering too little = stockouts and lost revenue. Ordering too much = excess inventory and waste. A simple "predict 500 units" forecast doesn't help a buyer decide how much buffer stock to hold.

**This project builds a system that predicts a range:** "There's an 80% chance sales will be between 420 and 610 units." That range lets buyers make cost-optimized stocking decisions.

---

## 🏗️ Architecture

```
Raw M5 Data (3 CSV files)
        ↓
  Feature Engineering
  (lag features, rolling averages,
   calendar features, price features)
        ↓
  LightGBM Quantile Regression
  (trained separately for q=0.1, 0.5, 0.9)
        ↓
  Walk-Forward Validation
  (3 folds, no data leakage)
        ↓
  Calibration Testing
  (are our intervals actually reliable?)
        ↓
  Streamlit Dashboard
  (interactive forecast + business decision tool)
```

---

## 📊 Key Results

| Metric | Value |
|---|---|
| MAE | ~X units per item per day |
| RMSE | ~X units |
| Coverage (80% interval) | ~80% ✅ |
| Products forecasted | 50 (CA_1 store, FOODS_3 dept) |

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/demand-forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download M5 dataset from Kaggle
# Place sales_train_validation.csv, calendar.csv, sell_prices.csv in /data

# 4. Run in order:
python 01_eda.py          # Exploratory analysis
python 02_features.py     # Feature engineering
python 03_train.py        # Train models + evaluate

# 5. Launch the dashboard
streamlit run app.py
```

---

## 💡 What Makes This Different

- **Prediction intervals, not just point forecasts** — every prediction includes a calibrated confidence range
- **Walk-forward validation** — honest evaluation that mimics real forecasting (no peeking at future data)
- **Calibration testing** — proves the 80% interval actually contains ~80% of real values
- **Business decision tool** — optimal order quantity calculator using the newsvendor model

---

## 🛠️ Tech Stack

`Python` `LightGBM` `Pandas` `Scikit-learn` `Prophet` `Statsmodels` `Streamlit` `Plotly` `SHAP` `Optuna`

---

## 📁 Project Structure

```
demand-forecasting/
├── data/                    # Raw Kaggle data (not committed)
├── models/                  # Saved LightGBM models
├── outputs/                 # Charts, processed data, predictions
├── config.py                # Central settings
├── 01_eda.py                # Exploratory data analysis
├── 02_features.py           # Feature engineering
├── 03_train.py              # Model training + evaluation
├── app.py                   # Streamlit dashboard
└── requirements.txt         # Dependencies
```
