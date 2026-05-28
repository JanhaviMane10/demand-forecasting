# 📦 Probabilistic Demand Forecasting
### LightGBM + Quantile Regression on M5 Walmart Dataset

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red)](https://janhavi-demand-forecasting.streamlit.app)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)]()

🔗 **[Live Demo](https://janhavi-demand-forecasting.streamlit.app)**

---

## 📋 Project Overview

Most demand forecasting systems give you a single number — "we'll sell 150 units." But businesses need to know the *range* of possible outcomes to make smart inventory decisions. This project builds a **probabilistic forecasting system** that predicts not just the expected demand, but calibrated confidence intervals.

> **Business Question:** How many units will we sell next week — and how uncertain are we about that prediction?

---

## 🔍 Key Features

- **Walk-forward validation** — simulates real deployment by training only on past data
- **Calibrated 80% prediction intervals** via quantile regression (Q10, Q50, Q90)
- **40+ engineered features** — lag features, rolling statistics, calendar effects
- **Interactive Streamlit dashboard** — select any product, adjust confidence level, see cost analysis

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Model | LightGBM (3 quantile models) |
| Validation | Walk-forward (time-series aware) |
| Coverage | ~80% of actual values fall inside 80% interval |
| Dataset | M5 Walmart Competition — 50 products, Store CA_1 |

---

## 🗂️ Repository Structure

```
demand-forecasting/
├── config.py              ← Settings and parameters
├── 01_eda.py              ← Exploratory data analysis
├── 02_features.py         ← Feature engineering (40+ features)
├── 03_train.py            ← LightGBM training with quantile regression
├── app.py                 ← Streamlit dashboard
├── requirements.txt
├── models/
│   ├── lgbm_q1.pkl        ← Q10 model (lower bound)
│   ├── lgbm_q5.pkl        ← Q50 model (median forecast)
│   ├── lgbm_q9.pkl        ← Q90 model (upper bound)
│   ├── feature_cols.json
│   └── results_summary.json
└── outputs/
    └── test_predictions.csv
```

---

## 🚀 How to Run

```bash
git clone https://github.com/JanhaviMane10/demand-forecasting.git
cd demand-forecasting
pip install -r requirements.txt

# Run in order
python 01_eda.py
python 02_features.py
python 03_train.py

# Launch dashboard
streamlit run app.py
```

---

## 🛠️ Methods

- **Dataset:** M5 Walmart Forecasting Competition (Kaggle) — 42,840 daily sales records
- **Features:** 40+ including lag-7, lag-14, lag-28, rolling means/stds, day-of-week, month, SNAP flags
- **Models:** 3 separate LightGBM models trained on Q10, Q50, Q90 loss functions
- **Validation:** Walk-forward cross-validation (no data leakage)
- **Calibration:** Verified that 80% prediction interval captures ~80% of actual values

---

## 💡 Business Impact

The dashboard allows supply chain managers to:
- See probabilistic forecasts for any product
- Adjust confidence level (50%-95%)
- Input stockout and overstock costs to find optimal order quantity
- Understand where forecasts are most uncertain

---

## 🔧 Technologies

`Python` `LightGBM` `pandas` `numpy` `scikit-learn` `Streamlit` `Plotly`
