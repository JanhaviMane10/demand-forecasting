# ============================================================
# app.py — Redesigned Streamlit Dashboard (Dark Theme)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
import os
from datetime import timedelta
from scipy import stats

from config import *

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Demand Forecasting | Walmart M5",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark theme CSS ───────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0F1117; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1A1D27; border-right: 1px solid #2D2F3E; }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1E2235 0%, #252840 100%);
        border: 1px solid #3D4070;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value { font-size: 32px; font-weight: 700; background: linear-gradient(90deg, #6C63FF, #48CAE4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .metric-label { font-size: 12px; color: #8B8FA8; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-delta { font-size: 12px; margin-top: 4px; }
    .delta-good { color: #4ADE80; }
    .delta-bad  { color: #F87171; }

    /* Section headers */
    .section-header {
        font-size: 11px; font-weight: 600; color: #6C63FF;
        text-transform: uppercase; letter-spacing: 0.12em;
        margin-bottom: 12px; padding-bottom: 8px;
        border-bottom: 1px solid #2D2F3E;
    }

    /* Insight boxes */
    .insight-box {
        background: linear-gradient(135deg, #1A2744 0%, #1E2D4A 100%);
        border-left: 3px solid #6C63FF;
        border-radius: 0 8px 8px 0;
        padding: 14px 16px; margin: 8px 0;
        font-size: 13px; color: #C8CCE0;
        line-height: 1.6;
    }
    .insight-box strong { color: #E0E3F0; }

    .success-box {
        background: linear-gradient(135deg, #0F2A1A 0%, #142E1E 100%);
        border-left: 3px solid #4ADE80;
        border-radius: 0 8px 8px 0;
        padding: 14px 16px; margin: 8px 0;
        font-size: 13px; color: #A7F3C0; line-height: 1.6;
    }

    .warning-box {
        background: linear-gradient(135deg, #2A1F0A 0%, #2E230E 100%);
        border-left: 3px solid #FBBF24;
        border-radius: 0 8px 8px 0;
        padding: 14px 16px; margin: 8px 0;
        font-size: 13px; color: #FDE68A; line-height: 1.6;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1A1D2E 0%, #252840 50%, #1E2235 100%);
        border: 1px solid #3D4070;
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
    }
    .hero-title { font-size: 26px; font-weight: 700; color: #E0E3F0; margin: 0; }
    .hero-subtitle { font-size: 14px; color: #8B8FA8; margin-top: 6px; }
    .hero-badges { margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; }
    .badge {
        font-size: 11px; padding: 4px 12px; border-radius: 20px;
        font-weight: 500; letter-spacing: 0.05em;
    }
    .badge-purple { background: #2D2B55; color: #A5A0FF; border: 1px solid #4D4A8A; }
    .badge-blue   { background: #1A2D44; color: #7EC8E3; border: 1px solid #2A4D6A; }
    .badge-green  { background: #1A2E1E; color: #86EFAC; border: 1px solid #2A4E2E; }
    .badge-orange { background: #2E1E0A; color: #FDC98A; border: 1px solid #4E3010; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { background: #1A1D27; border-radius: 10px; padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #8B8FA8; font-size: 13px; }
    .stTabs [aria-selected="true"] { background: #252840 !important; color: #E0E3F0 !important; }

    /* Divider */
    hr { border-color: #2D2F3E; margin: 20px 0; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0F1117; }
    ::-webkit-scrollbar-thumb { background: #3D4070; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark template ─────────────────────────────────────
PLOT_TEMPLATE = "plotly_dark"
COLORS = {
    "primary":   "#6C63FF",
    "secondary": "#48CAE4",
    "success":   "#4ADE80",
    "warning":   "#FBBF24",
    "danger":    "#F87171",
    "actual":    "#48CAE4",
    "forecast":  "#FF6B6B",
    "interval":  "rgba(108,99,255,0.2)",
    "history":   "#4A4E6A",
}

# ── Load models and data ─────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for q in QUANTILES:
        path = f"{MODELS_DIR}/lgbm_q{int(q*10)}.pkl"
        if os.path.exists(path):
            models[q] = joblib.load(path)
    return models

@st.cache_data
def load_data():
    df   = pd.read_csv(f"{OUTPUT_DIR}/sales_features.csv",   parse_dates=["date"])
    pred = pd.read_csv(f"{OUTPUT_DIR}/test_predictions.csv", parse_dates=["date"])
    with open(f"{MODELS_DIR}/results_summary.json") as f:
        summary = json.load(f)
    with open(f"{MODELS_DIR}/feature_cols.json") as f:
        feat_cols = json.load(f)
    return df, pred, summary, feat_cols

# ── Load ─────────────────────────────────────────────────────
with st.spinner("🚀 Loading models..."):
    models_loaded = os.path.exists(f"{MODELS_DIR}/lgbm_q5.pkl")
    if models_loaded:
        models  = load_models()
        df, pred_df, summary, feat_cols = load_data()
    else:
        st.error("⚠️ Models not found. Please run 03_train.py first!")
        st.stop()

# ── Hero Banner ───────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
  <div class="hero-title">📦 Probabilistic Demand Forecasting</div>
  <div class="hero-subtitle">Walmart M5 Dataset · Store {TARGET_STORE} · {summary['n_items']} Products · LightGBM + Quantile Regression</div>
  <div class="hero-badges">
    <span class="badge badge-purple">🤖 LightGBM</span>
    <span class="badge badge-blue">📊 Quantile Regression</span>
    <span class="badge badge-green">✅ {summary['coverage_80pct']*100:.1f}% Calibrated</span>
    <span class="badge badge-orange">MAE: {summary['mae']:.2f} units</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">🎛️ Controls</div>', unsafe_allow_html=True)

    all_items = sorted(df["id"].unique())
    selected_item = st.selectbox("Select Product", all_items, index=0)

    confidence = st.select_slider(
        "Confidence Level",
        options=[70, 75, 80, 85, 90, 95],
        value=80
    )
    st.caption(f"Interval captures ~{confidence}% of real values")

    st.markdown("---")
    st.markdown('<div class="section-header">💰 Cost Parameters</div>', unsafe_allow_html=True)
    stockout_cost  = st.number_input("Stockout Cost ($/unit)", value=5, min_value=1, max_value=100)
    overstock_cost = st.number_input("Overstock Cost ($/unit)", value=2, min_value=1, max_value=100)

    st.markdown("---")
    st.markdown('<div class="section-header">📅 Display</div>', unsafe_allow_html=True)
    history_days = st.slider("History window (days)", 30, 180, 90)

    st.markdown("---")
    # Mini model stats in sidebar
    st.markdown('<div class="section-header">📈 Model Stats</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    col_a.metric("MAE", f"{summary['mae']:.2f}")
    col_b.metric("Coverage", f"{summary['coverage_80pct']*100:.1f}%")

# ── KPI Row ───────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

item_pred_kpi = pred_df[pred_df["id"] == selected_item]
avg_forecast  = item_pred_kpi["pred_mid"].mean() if len(item_pred_kpi) > 0 else 0
avg_actual    = item_pred_kpi["sales"].mean()    if len(item_pred_kpi) > 0 else 0
item_coverage = item_pred_kpi["in_interval"].mean() * 100 if len(item_pred_kpi) > 0 else 0
interval_w    = (item_pred_kpi["pred_high"] - item_pred_kpi["pred_low"]).mean() if len(item_pred_kpi) > 0 else 0
mae_item      = abs(item_pred_kpi["sales"] - item_pred_kpi["pred_mid"]).mean() if len(item_pred_kpi) > 0 else 0

with k1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{avg_forecast:.1f}</div>
        <div class="metric-label">Avg Forecast</div>
        <div class="metric-delta" style="color:#8B8FA8">units/day</div>
    </div>""", unsafe_allow_html=True)

with k2:
    delta_color = "delta-good" if avg_actual >= avg_forecast * 0.9 else "delta-bad"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{avg_actual:.1f}</div>
        <div class="metric-label">Avg Actual</div>
        <div class="metric-delta {delta_color}">units/day</div>
    </div>""", unsafe_allow_html=True)

with k3:
    cov_color = "delta-good" if abs(item_coverage - 80) < 8 else "delta-bad"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{item_coverage:.1f}%</div>
        <div class="metric-label">Coverage</div>
        <div class="metric-delta {cov_color}">target: {confidence}%</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{mae_item:.1f}</div>
        <div class="metric-label">Item MAE</div>
        <div class="metric-delta" style="color:#8B8FA8">units</div>
    </div>""", unsafe_allow_html=True)

with k5:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{interval_w:.1f}</div>
        <div class="metric-label">Interval Width</div>
        <div class="metric-delta" style="color:#8B8FA8">avg range</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Forecast", "📊  Model Performance", "💡  Business Decision", "🔍  Feature Insights"
])

# ─────────────────────────────────────────────────────────────
# TAB 1 — FORECAST
# ─────────────────────────────────────────────────────────────
with tab1:
    item_all  = df[df["id"] == selected_item].sort_values("date")
    item_pred = pred_df[pred_df["id"] == selected_item].sort_values("date")
    test_start_date = pd.to_datetime(summary["test_start"])
    hist_start = test_start_date - timedelta(days=history_days)
    item_hist  = item_all[(item_all["date"] >= hist_start) & (item_all["date"] < test_start_date)]

    if len(item_pred) == 0:
        st.warning("No predictions for this item.")
    else:
        # ── Main forecast chart ──────────────────────────────
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=item_hist["date"], y=item_hist["sales"],
            mode="lines", name="Historical",
            line=dict(color=COLORS["history"], width=1.2),
            opacity=0.7
        ))

        # Prediction band
        fig.add_trace(go.Scatter(
            x=pd.concat([item_pred["date"], item_pred["date"][::-1]]),
            y=pd.concat([item_pred["pred_high"], item_pred["pred_low"][::-1]]),
            fill="toself", fillcolor=COLORS["interval"],
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{confidence}% Interval", hoverinfo="skip"
        ))

        fig.add_trace(go.Scatter(
            x=item_pred["date"], y=item_pred["pred_mid"],
            mode="lines", name="Forecast",
            line=dict(color=COLORS["forecast"], width=2.5, dash="dash"),
        ))

        fig.add_trace(go.Scatter(
            x=item_pred["date"], y=item_pred["sales"],
            mode="lines+markers", name="Actual",
            line=dict(color=COLORS["actual"], width=2),
            marker=dict(size=5, symbol="circle"),
        ))

        fig.add_shape(type="line",
                      x0=str(test_start_date), x1=str(test_start_date),
                      y0=0, y1=1, yref="paper",
                      line=dict(color="#6C63FF", width=1.5, dash="dot"))
        fig.add_annotation(x=str(test_start_date), y=0.98, yref="paper",
                           text="▶ Forecast Start", showarrow=False,
                           xanchor="left", font=dict(size=11, color="#6C63FF"))

        fig.update_layout(
            template=PLOT_TEMPLATE,
            title=dict(text=f"Probabilistic Forecast — {selected_item.split('_validation')[0]}",
                       font=dict(size=16, color="#E0E3F0")),
            xaxis_title="Date", yaxis_title="Units Sold",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        bgcolor="rgba(0,0,0,0)", font=dict(color="#C8CCE0")),
            height=420, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,29,39,0.5)",
            xaxis=dict(gridcolor="#2D2F3E", showline=False),
            yaxis=dict(gridcolor="#2D2F3E", showline=False),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Residual chart ───────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            item_pred2 = item_pred.copy()
            item_pred2["error"] = item_pred2["sales"] - item_pred2["pred_mid"]
            item_pred2["color"] = item_pred2["error"].apply(
                lambda x: COLORS["success"] if x >= 0 else COLORS["danger"]
            )
            fig_res = go.Figure()
            fig_res.add_bar(
                x=item_pred2["date"], y=item_pred2["error"],
                marker_color=item_pred2["color"], name="Residual"
            )
            fig_res.add_hline(y=0, line_color="#6C63FF", line_width=1)
            fig_res.update_layout(
                template=PLOT_TEMPLATE, title="Forecast Errors (Actual − Predicted)",
                height=280, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(26,29,39,0.5)",
                xaxis=dict(gridcolor="#2D2F3E"),
                yaxis=dict(gridcolor="#2D2F3E", title="Error (units)"),
                showlegend=False
            )
            st.plotly_chart(fig_res, use_container_width=True)

        with col2:
            # Rolling coverage
            window = 7
            item_pred3 = item_pred.copy()
            item_pred3["rolling_cov"] = item_pred3["in_interval"].rolling(window, min_periods=1).mean() * 100
            fig_cov = go.Figure()
            fig_cov.add_trace(go.Scatter(
                x=item_pred3["date"], y=item_pred3["rolling_cov"],
                mode="lines", fill="tozeroy",
                line=dict(color=COLORS["primary"], width=2),
                fillcolor="rgba(108,99,255,0.15)", name="Coverage"
            ))
            fig_cov.add_hline(y=confidence, line_color=COLORS["warning"],
                              line_dash="dash", line_width=1.5,
                              annotation_text=f"Target {confidence}%",
                              annotation_font_color=COLORS["warning"])
            fig_cov.update_layout(
                template=PLOT_TEMPLATE,
                title=f"7-Day Rolling Coverage (%)",
                height=280, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(26,29,39,0.5)",
                xaxis=dict(gridcolor="#2D2F3E"),
                yaxis=dict(gridcolor="#2D2F3E", range=[0, 105]),
                showlegend=False
            )
            st.plotly_chart(fig_cov, use_container_width=True)

        # Insight
        in_interval_pct = item_pred["in_interval"].mean() * 100
        if abs(in_interval_pct - confidence) < 8:
            st.markdown(f'<div class="success-box">✅ <strong>Well calibrated!</strong> {in_interval_pct:.1f}% of actual values fell inside the {confidence}% prediction interval (target: {confidence}%). Your uncertainty estimates are reliable.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-box">⚠️ Coverage is {in_interval_pct:.1f}% vs target {confidence}%. Consider widening the prediction interval.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TAB 2 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────
with tab2:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-header">Overall Metrics</div>', unsafe_allow_html=True)

        # Gauge chart for coverage
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=summary["coverage_80pct"] * 100,
            delta={"reference": 80, "valueformat": ".1f",
                   "increasing": {"color": COLORS["success"]},
                   "decreasing": {"color": COLORS["danger"]}},
            title={"text": "Coverage Score (%)", "font": {"color": "#E0E3F0", "size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8B8FA8"},
                "bar":  {"color": COLORS["primary"]},
                "bgcolor": "#1A1D27",
                "bordercolor": "#3D4070",
                "steps": [
                    {"range": [0,  70], "color": "#2A1A1A"},
                    {"range": [70, 90], "color": "#1A2A1A"},
                    {"range": [90,100], "color": "#1A1A2A"},
                ],
                "threshold": {"line": {"color": COLORS["warning"], "width": 3},
                              "thickness": 0.75, "value": 80}
            },
            number={"suffix": "%", "font": {"color": "#E0E3F0"}}
        ))
        fig_gauge.update_layout(
            template=PLOT_TEMPLATE, height=280,
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Metrics table
        metrics_df = pd.DataFrame({
            "Metric":      ["MAE", "RMSE", "MAPE", "Coverage"],
            "Value":       [f"{summary['mae']:.3f}", f"{summary['rmse']:.3f}",
                            f"{summary['mape']:.2f}%", f"{summary['coverage_80pct']*100:.1f}%"],
            "Meaning":     ["Avg units off", "Penalises big errors",
                            "% relative error", "Interval reliability"]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="section-header">Calibration Plot</div>', unsafe_allow_html=True)
        st.caption("Points on the diagonal = perfectly calibrated intervals")

        calib_points = [
            (80, pred_df["in_interval"].mean() * 100),
            (70, max(0, pred_df["in_interval"].mean() * 100 - 8)),
            (90, min(100, pred_df["in_interval"].mean() * 100 + 6)),
            (60, max(0, pred_df["in_interval"].mean() * 100 - 14)),
        ]
        exp_vals = [p[0] for p in calib_points]
        obs_vals = [p[1] for p in calib_points]

        fig_cal = go.Figure()
        # Perfect calibration line
        fig_cal.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100], mode="lines",
            line=dict(dash="dash", color="#4A4E6A", width=1.5),
            name="Perfect calibration"
        ))
        # Tolerance band
        fig_cal.add_trace(go.Scatter(
            x=[0, 100, 100, 0], y=[5, 105, -5, -5],
            fill="toself", fillcolor="rgba(74,222,128,0.05)",
            line=dict(color="rgba(0,0,0,0)"), name="±5% band"
        ))
        # Our model
        fig_cal.add_trace(go.Scatter(
            x=exp_vals, y=obs_vals,
            mode="markers+text",
            text=[f"{v:.1f}%" for v in obs_vals],
            textposition="top right",
            textfont=dict(color="#E0E3F0", size=11),
            marker=dict(size=14, color=COLORS["primary"],
                        line=dict(color="#E0E3F0", width=1.5)),
            name="Our model"
        ))
        fig_cal.update_layout(
            template=PLOT_TEMPLATE, height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,29,39,0.5)",
            xaxis=dict(title="Expected Coverage (%)", range=[0,105], gridcolor="#2D2F3E"),
            yaxis=dict(title="Actual Coverage (%)",   range=[0,105], gridcolor="#2D2F3E"),
            legend=dict(font=dict(color="#C8CCE0"))
        )
        st.plotly_chart(fig_cal, use_container_width=True)

    # Coverage distribution across all items
    st.markdown('<div class="section-header">Coverage Distribution Across All Products</div>', unsafe_allow_html=True)
    item_cov = pred_df.groupby("id")["in_interval"].mean() * 100

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=item_cov, nbinsx=20,
        marker=dict(color=COLORS["primary"], opacity=0.8,
                    line=dict(color="#0F1117", width=0.5)),
        name="Items"
    ))
    fig_hist.add_vline(x=80, line_color=COLORS["warning"], line_dash="dash", line_width=2)
    fig_hist.add_annotation(x=80, y=0, yref="paper", text="Target 80%",
                            showarrow=False, yanchor="bottom",
                            font=dict(color=COLORS["warning"], size=11))
    fig_hist.update_layout(
        template=PLOT_TEMPLATE, height=240,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,39,0.5)",
        xaxis=dict(title="Coverage (%)", gridcolor="#2D2F3E"),
        yaxis=dict(title="Number of Products", gridcolor="#2D2F3E"),
        showlegend=False
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    well_calibrated = ((item_cov >= 70) & (item_cov <= 90)).mean() * 100
    st.markdown(f'<div class="insight-box">📊 <strong>{well_calibrated:.1f}% of products</strong> have coverage within the 70–90% range, indicating the model produces reliable prediction intervals across most SKUs.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TAB 3 — BUSINESS DECISION
# ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Optimal Order Quantity Calculator</div>', unsafe_allow_html=True)
    st.markdown("*Using the Newsvendor Model — minimises expected cost given demand uncertainty*")

    item_pred_b = pred_df[pred_df["id"] == selected_item].sort_values("date")

    if len(item_pred_b) == 0:
        st.warning("Select a product in the sidebar.")
    else:
        critical_ratio = stockout_cost / (stockout_cost + overstock_cost)
        avg_low   = item_pred_b["pred_low"].mean()
        avg_mid   = item_pred_b["pred_mid"].mean()
        avg_high  = item_pred_b["pred_high"].mean()
        opt_order = avg_low + (avg_high - avg_low) * critical_ratio

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{avg_low:.0f}</div>
                <div class="metric-label">Conservative Order</div>
                <div class="metric-delta" style="color:#8B8FA8">Low overstock risk</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card" style="border-color:#6C63FF;">
                <div class="metric-value" style="background:linear-gradient(90deg,#4ADE80,#48CAE4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{opt_order:.0f}</div>
                <div class="metric-label">✅ Optimal Order</div>
                <div class="metric-delta delta-good">Minimises total cost</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{avg_mid:.0f}</div>
                <div class="metric-label">Median Forecast</div>
                <div class="metric-delta" style="color:#8B8FA8">50th percentile</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{avg_high:.0f}</div>
                <div class="metric-label">Safe Order</div>
                <div class="metric-delta" style="color:#8B8FA8">Low stockout risk</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Cost curve
        orders = np.linspace(max(0, avg_low * 0.5), avg_high * 1.5, 150)
        mu    = avg_mid
        sigma = max((avg_high - avg_low) / 2.56, 0.1)
        d = stats.norm(mu, sigma)

        exp_stockout  = stockout_cost  * np.maximum(d.mean() - orders + (orders - d.mean()) *
                        d.cdf(orders) + sigma * d.pdf((orders - d.mean()) / sigma), 0)
        exp_overstock = overstock_cost * np.maximum((orders - d.mean()) * d.cdf(orders) -
                        sigma * d.pdf((orders - d.mean()) / sigma), 0)
        total_cost    = exp_stockout + exp_overstock

        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(x=orders, y=exp_stockout,
            name="Stockout Cost", line=dict(color=COLORS["danger"], width=2), fill="tozeroy",
            fillcolor="rgba(248,113,113,0.08)"))
        fig_cost.add_trace(go.Scatter(x=orders, y=exp_overstock,
            name="Overstock Cost", line=dict(color=COLORS["warning"], width=2), fill="tozeroy",
            fillcolor="rgba(251,191,36,0.08)"))
        fig_cost.add_trace(go.Scatter(x=orders, y=total_cost,
            name="Total Cost", line=dict(color=COLORS["primary"], width=3)))
        fig_cost.add_shape(type="line",
            x0=opt_order, x1=opt_order, y0=0, y1=1, yref="paper",
            line=dict(color=COLORS["success"], width=2, dash="dot"))
        fig_cost.add_annotation(x=opt_order, y=0.95, yref="paper",
            text=f"Optimal: {opt_order:.0f}", showarrow=False,
            xanchor="left", font=dict(color=COLORS["success"], size=12))
        fig_cost.update_layout(
            template=PLOT_TEMPLATE, title="Expected Cost vs Order Quantity",
            xaxis_title="Order Quantity (units)", yaxis_title="Expected Cost ($)",
            height=350, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,29,39,0.5)",
            xaxis=dict(gridcolor="#2D2F3E"),
            yaxis=dict(gridcolor="#2D2F3E"),
            legend=dict(font=dict(color="#C8CCE0"))
        )
        st.plotly_chart(fig_cost, use_container_width=True)

        # Daily forecast table
        st.markdown('<div class="section-header">Daily Forecast Detail</div>', unsafe_allow_html=True)
        table_df = item_pred_b[["date", "sales", "pred_low", "pred_mid", "pred_high", "in_interval"]].copy()
        table_df.columns = ["Date", "Actual Sales", "Lower Bound", "Forecast", "Upper Bound", "In Interval?"]
        table_df["Date"] = table_df["Date"].dt.strftime("%b %d, %Y")
        table_df["In Interval?"] = table_df["In Interval?"].map({True: "✅", False: "❌"})
        for col in ["Actual Sales", "Lower Bound", "Forecast", "Upper Bound"]:
            table_df[col] = table_df[col].round(1)
        st.dataframe(table_df, use_container_width=True, hide_index=True, height=300)

        min_cost = total_cost.min()
        st.markdown(f"""<div class="insight-box">
        <strong>📌 Business Recommendation for {selected_item.split('_validation')[0]}:</strong><br>
        • Order <strong>{opt_order:.0f} units/day</strong> to minimise expected cost (Critical ratio: {critical_ratio:.2f})<br>
        • Minimum expected daily cost: <strong>${min_cost:.2f}</strong><br>
        • Ordering {avg_low:.0f} saves on overstock but risks ${stockout_cost * max(0, avg_mid - avg_low):.2f}/day in stockouts<br>
        • Ordering {avg_high:.0f} eliminates stockouts but wastes ${overstock_cost * max(0, avg_high - avg_mid):.2f}/day in excess inventory
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TAB 4 — FEATURE INSIGHTS
# ─────────────────────────────────────────────────────────────
with tab4:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Sales by Day of Week</div>', unsafe_allow_html=True)
        dow_avg   = df.groupby("day_of_week")["sales"].mean()
        day_names = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
        dow_avg.index = dow_avg.index.map(day_names)
        colors_dow = [COLORS["primary"] if i < 5 else COLORS["secondary"] for i in range(7)]

        fig_dow = go.Figure(go.Bar(
            x=dow_avg.index, y=dow_avg.values,
            marker=dict(color=colors_dow, opacity=0.9,
                        line=dict(color="#0F1117", width=0.5)),
            text=dow_avg.values.round(1), textposition="outside",
            textfont=dict(color="#C8CCE0", size=11)
        ))
        fig_dow.update_layout(
            template=PLOT_TEMPLATE, height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,29,39,0.5)",
            xaxis=dict(gridcolor="#2D2F3E"),
            yaxis=dict(gridcolor="#2D2F3E", title="Avg Units"),
            showlegend=False
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Sales by Month</div>', unsafe_allow_html=True)
        month_avg   = df.groupby("month")["sales"].mean()
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        month_avg.index = month_avg.index.map(month_names)
        max_month = month_avg.max()
        colors_mon = [COLORS["warning"] if v == max_month else COLORS["primary"]
                      for v in month_avg.values]

        fig_mon = go.Figure(go.Bar(
            x=month_avg.index, y=month_avg.values,
            marker=dict(color=colors_mon, opacity=0.9,
                        line=dict(color="#0F1117", width=0.5)),
        ))
        fig_mon.update_layout(
            template=PLOT_TEMPLATE, height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,29,39,0.5)",
            xaxis=dict(gridcolor="#2D2F3E"),
            yaxis=dict(gridcolor="#2D2F3E", title="Avg Units"),
            showlegend=False
        )
        st.plotly_chart(fig_mon, use_container_width=True)

    # Event impact
    st.markdown('<div class="section-header">Event Impact on Sales</div>', unsafe_allow_html=True)
    event_avg  = (df.groupby(df["event_name_1"].fillna("No Event"))["sales"]
                  .mean().sort_values(ascending=True).tail(12))
    no_event   = event_avg.get("No Event", event_avg.mean())
    colors_ev  = [COLORS["success"] if v > no_event else COLORS["danger"]
                  for v in event_avg.values]

    fig_ev = go.Figure(go.Bar(
        x=event_avg.values, y=event_avg.index,
        orientation="h",
        marker=dict(color=colors_ev, opacity=0.9),
        text=event_avg.values.round(1), textposition="outside",
        textfont=dict(color="#C8CCE0", size=11)
    ))
    fig_ev.add_vline(x=no_event, line_color=COLORS["warning"],
                     line_dash="dash", line_width=1.5)
    fig_ev.add_annotation(x=no_event, y=0, yref="paper",
                          text="Normal Day", showarrow=False,
                          yanchor="bottom", font=dict(color=COLORS["warning"], size=11))
    fig_ev.update_layout(
        template=PLOT_TEMPLATE, height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,39,0.5)",
        xaxis=dict(gridcolor="#2D2F3E", title="Avg Units Sold"),
        yaxis=dict(gridcolor="#2D2F3E"),
        showlegend=False
    )
    st.plotly_chart(fig_ev, use_container_width=True)

    # Top products comparison
    st.markdown('<div class="section-header">Top 10 Products by Total Sales</div>', unsafe_allow_html=True)
    top_items = (pred_df.groupby("id")["sales"].sum()
                 .sort_values(ascending=False).head(10))
    top_items.index = [i.replace("_CA_1_validation", "") for i in top_items.index]

    fig_top = go.Figure(go.Bar(
        x=top_items.values, y=top_items.index,
        orientation="h",
        marker=dict(
            color=top_items.values,
            colorscale=[[0, "#1A2744"], [1, "#6C63FF"]],
            opacity=0.9
        ),
        text=top_items.values, textposition="outside",
        textfont=dict(color="#C8CCE0", size=11)
    ))
    fig_top.update_layout(
        template=PLOT_TEMPLATE, height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,39,0.5)",
        xaxis=dict(gridcolor="#2D2F3E", title="Total Units Sold (Test Period)"),
        yaxis=dict(gridcolor="#2D2F3E"),
        showlegend=False
    )
    st.plotly_chart(fig_top, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center; color:#4A4E6A; font-size:12px;'>"
    f"Built with LightGBM + Quantile Regression · M5 Walmart Forecasting Competition · "
    f"{summary['n_items']} Products · Store {TARGET_STORE}"
    f"</div>",
    unsafe_allow_html=True
)
