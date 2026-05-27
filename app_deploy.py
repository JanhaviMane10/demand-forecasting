# ============================================================
# app_deploy.py — Streamlit Dashboard (Deployment Version)
# Works without large sales_features.csv
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import os
from datetime import timedelta
from scipy import stats

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
    .stApp { background-color: #0F1117; }
    [data-testid="stSidebar"] { background-color: #1A1D27; border-right: 1px solid #2D2F3E; }
    .metric-card {
        background: linear-gradient(135deg, #1E2235 0%, #252840 100%);
        border: 1px solid #3D4070; border-radius: 12px;
        padding: 20px; text-align: center;
    }
    .metric-value { font-size: 32px; font-weight: 700; background: linear-gradient(90deg, #6C63FF, #48CAE4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .metric-label { font-size: 12px; color: #8B8FA8; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-delta { font-size: 12px; margin-top: 4px; }
    .delta-good { color: #4ADE80; }
    .delta-bad  { color: #F87171; }
    .section-header { font-size: 11px; font-weight: 600; color: #6C63FF; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #2D2F3E; }
    .insight-box { background: linear-gradient(135deg, #1A2744 0%, #1E2D4A 100%); border-left: 3px solid #6C63FF; border-radius: 0 8px 8px 0; padding: 14px 16px; margin: 8px 0; font-size: 13px; color: #C8CCE0; line-height: 1.6; }
    .insight-box strong { color: #E0E3F0; }
    .success-box { background: linear-gradient(135deg, #0F2A1A 0%, #142E1E 100%); border-left: 3px solid #4ADE80; border-radius: 0 8px 8px 0; padding: 14px 16px; margin: 8px 0; font-size: 13px; color: #A7F3C0; line-height: 1.6; }
    .warning-box { background: linear-gradient(135deg, #2A1F0A 0%, #2E230E 100%); border-left: 3px solid #FBBF24; border-radius: 0 8px 8px 0; padding: 14px 16px; margin: 8px 0; font-size: 13px; color: #FDE68A; line-height: 1.6; }
    .hero-banner { background: linear-gradient(135deg, #1A1D2E 0%, #252840 50%, #1E2235 100%); border: 1px solid #3D4070; border-radius: 16px; padding: 28px 32px; margin-bottom: 24px; }
    .hero-title { font-size: 26px; font-weight: 700; color: #E0E3F0; margin: 0; }
    .hero-subtitle { font-size: 14px; color: #8B8FA8; margin-top: 6px; }
    .badge { font-size: 11px; padding: 4px 12px; border-radius: 20px; font-weight: 500; letter-spacing: 0.05em; display: inline-block; margin: 4px; }
    .badge-purple { background: #2D2B55; color: #A5A0FF; border: 1px solid #4D4A8A; }
    .badge-blue   { background: #1A2D44; color: #7EC8E3; border: 1px solid #2A4D6A; }
    .badge-green  { background: #1A2E1E; color: #86EFAC; border: 1px solid #2A4E2E; }
    .badge-orange { background: #2E1E0A; color: #FDC98A; border: 1px solid #4E3010; }
    .stTabs [data-baseweb="tab-list"] { background: #1A1D27; border-radius: 10px; padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #8B8FA8; font-size: 13px; }
    .stTabs [aria-selected="true"] { background: #252840 !important; color: #E0E3F0 !important; }
    hr { border-color: #2D2F3E; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

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

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    pred = pd.read_csv("outputs/test_predictions.csv", parse_dates=["date"])
    with open("models/results_summary.json") as f:
        summary = json.load(f)
    return pred, summary

with st.spinner("🚀 Loading data..."):
    try:
        pred_df, summary = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# ── Hero Banner ───────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
  <div class="hero-title">📦 Probabilistic Demand Forecasting</div>
  <div class="hero-subtitle">Walmart M5 Dataset · Store CA_1 · {summary['n_items']} Products · LightGBM + Quantile Regression</div>
  <div style="margin-top:14px;">
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
    all_items = sorted(pred_df["id"].unique())
    selected_item = st.selectbox("Select Product", all_items, index=0)
    confidence = st.select_slider("Confidence Level", options=[70, 75, 80, 85, 90, 95], value=80)
    st.caption(f"Interval captures ~{confidence}% of real values")
    st.markdown("---")
    st.markdown('<div class="section-header">💰 Cost Parameters</div>', unsafe_allow_html=True)
    stockout_cost  = st.number_input("Stockout Cost ($/unit)", value=5, min_value=1, max_value=100)
    overstock_cost = st.number_input("Overstock Cost ($/unit)", value=2, min_value=1, max_value=100)
    st.markdown("---")
    st.markdown('<div class="section-header">📈 Model Stats</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    col_a.metric("MAE", f"{summary['mae']:.2f}")
    col_b.metric("Coverage", f"{summary['coverage_80pct']*100:.1f}%")

# ── KPI Row ───────────────────────────────────────────────────
item_pred_kpi = pred_df[pred_df["id"] == selected_item]
avg_forecast  = item_pred_kpi["pred_mid"].mean() if len(item_pred_kpi) > 0 else 0
avg_actual    = item_pred_kpi["sales"].mean()    if len(item_pred_kpi) > 0 else 0
item_coverage = item_pred_kpi["in_interval"].mean() * 100 if len(item_pred_kpi) > 0 else 0
interval_w    = (item_pred_kpi["pred_high"] - item_pred_kpi["pred_low"]).mean() if len(item_pred_kpi) > 0 else 0
mae_item      = abs(item_pred_kpi["sales"] - item_pred_kpi["pred_mid"]).mean() if len(item_pred_kpi) > 0 else 0

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{avg_forecast:.1f}</div><div class="metric-label">Avg Forecast</div><div class="metric-delta" style="color:#8B8FA8">units/day</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{avg_actual:.1f}</div><div class="metric-label">Avg Actual</div><div class="metric-delta delta-good">units/day</div></div>""", unsafe_allow_html=True)
with k3:
    cov_color = "delta-good" if abs(item_coverage - 80) < 8 else "delta-bad"
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{item_coverage:.1f}%</div><div class="metric-label">Coverage</div><div class="metric-delta {cov_color}">target: {confidence}%</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{mae_item:.1f}</div><div class="metric-label">Item MAE</div><div class="metric-delta" style="color:#8B8FA8">units</div></div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{interval_w:.1f}</div><div class="metric-label">Interval Width</div><div class="metric-delta" style="color:#8B8FA8">avg range</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈  Forecast", "📊  Model Performance", "💡  Business Decision"])

# ── TAB 1: FORECAST ───────────────────────────────────────────
with tab1:
    item_pred = pred_df[pred_df["id"] == selected_item].sort_values("date")

    if len(item_pred) == 0:
        st.warning("No predictions for this item.")
    else:
        fig = go.Figure()
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
            marker=dict(size=5),
        ))
        fig.update_layout(
            template=PLOT_TEMPLATE,
            title=dict(text=f"Probabilistic Forecast — {selected_item.replace('_validation','').replace('_CA_1','')}",
                       font=dict(size=16, color="#E0E3F0")),
            xaxis_title="Date", yaxis_title="Units Sold",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
            height=420, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,29,39,0.5)",
            xaxis=dict(gridcolor="#2D2F3E"),
            yaxis=dict(gridcolor="#2D2F3E"),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            item_pred2 = item_pred.copy()
            item_pred2["error"] = item_pred2["sales"] - item_pred2["pred_mid"]
            colors_err = [COLORS["success"] if x >= 0 else COLORS["danger"] for x in item_pred2["error"]]
            fig_res = go.Figure()
            fig_res.add_bar(x=item_pred2["date"], y=item_pred2["error"], marker_color=colors_err)
            fig_res.add_hline(y=0, line_color="#6C63FF", line_width=1)
            fig_res.update_layout(template=PLOT_TEMPLATE, title="Forecast Errors",
                height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,39,0.5)",
                xaxis=dict(gridcolor="#2D2F3E"), yaxis=dict(gridcolor="#2D2F3E"), showlegend=False)
            st.plotly_chart(fig_res, use_container_width=True)

        with col2:
            item_pred3 = item_pred.copy()
            item_pred3["rolling_cov"] = item_pred3["in_interval"].rolling(7, min_periods=1).mean() * 100
            fig_cov = go.Figure()
            fig_cov.add_trace(go.Scatter(
                x=item_pred3["date"], y=item_pred3["rolling_cov"],
                mode="lines", fill="tozeroy",
                line=dict(color=COLORS["primary"], width=2),
                fillcolor="rgba(108,99,255,0.15)"
            ))
            fig_cov.add_hline(y=confidence, line_color=COLORS["warning"], line_dash="dash", line_width=1.5)
            fig_cov.update_layout(template=PLOT_TEMPLATE, title="7-Day Rolling Coverage (%)",
                height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,39,0.5)",
                xaxis=dict(gridcolor="#2D2F3E"), yaxis=dict(gridcolor="#2D2F3E", range=[0,105]),
                showlegend=False)
            st.plotly_chart(fig_cov, use_container_width=True)

        in_pct = item_pred["in_interval"].mean() * 100
        if abs(in_pct - confidence) < 8:
            st.markdown(f'<div class="success-box">✅ <strong>Well calibrated!</strong> {in_pct:.1f}% of actual values fell inside the {confidence}% prediction interval.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-box">⚠️ Coverage is {in_pct:.1f}% vs target {confidence}%.</div>', unsafe_allow_html=True)

# ── TAB 2: MODEL PERFORMANCE ──────────────────────────────────
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Overall Metrics</div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=summary["coverage_80pct"] * 100,
            delta={"reference": 80, "valueformat": ".1f"},
            title={"text": "Coverage Score (%)", "font": {"color": "#E0E3F0", "size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8B8FA8"},
                "bar":  {"color": COLORS["primary"]},
                "bgcolor": "#1A1D27", "bordercolor": "#3D4070",
                "steps": [{"range": [0,70], "color": "#2A1A1A"},
                           {"range": [70,90], "color": "#1A2A1A"},
                           {"range": [90,100], "color": "#1A1A2A"}],
                "threshold": {"line": {"color": COLORS["warning"], "width": 3}, "thickness": 0.75, "value": 80}
            },
            number={"suffix": "%", "font": {"color": "#E0E3F0"}}
        ))
        fig_gauge.update_layout(template=PLOT_TEMPLATE, height=280, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

        import pandas as pd
        metrics_df = pd.DataFrame({
            "Metric":  ["MAE", "RMSE", "MAPE", "Coverage"],
            "Value":   [f"{summary['mae']:.3f}", f"{summary['rmse']:.3f}",
                        f"{summary['mape']:.2f}%", f"{summary['coverage_80pct']*100:.1f}%"],
            "Meaning": ["Avg units off per day", "Penalises big errors more",
                        "% error vs actual", "Interval reliability"]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="section-header">Coverage Distribution</div>', unsafe_allow_html=True)
        item_cov = pred_df.groupby("id")["in_interval"].mean() * 100
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=item_cov, nbinsx=20,
            marker=dict(color=COLORS["primary"], opacity=0.8, line=dict(color="#0F1117", width=0.5))))
        fig_hist.add_vline(x=80, line_color=COLORS["warning"], line_dash="dash", line_width=2)
        fig_hist.update_layout(template=PLOT_TEMPLATE, height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,39,0.5)",
            xaxis=dict(title="Coverage (%)", gridcolor="#2D2F3E"),
            yaxis=dict(title="# Products", gridcolor="#2D2F3E"), showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

        well_cal = ((item_cov >= 70) & (item_cov <= 90)).mean() * 100
        st.markdown(f'<div class="insight-box">📊 <strong>{well_cal:.1f}% of products</strong> have coverage within the 70–90% range, showing reliable uncertainty estimates across most SKUs.</div>', unsafe_allow_html=True)

        # Top vs Bottom items
        st.markdown('<div class="section-header">Best & Worst Calibrated Products</div>', unsafe_allow_html=True)
        top5 = item_cov.nlargest(5).reset_index()
        top5.columns = ["Product", "Coverage %"]
        top5["Product"] = top5["Product"].str.replace("_CA_1_validation","").str.replace("_validation","")
        top5["Coverage %"] = top5["Coverage %"].round(1)
        st.dataframe(top5, use_container_width=True, hide_index=True)

# ── TAB 3: BUSINESS DECISION ──────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Optimal Order Quantity Calculator</div>', unsafe_allow_html=True)
    item_pred_b = pred_df[pred_df["id"] == selected_item].sort_values("date")

    if len(item_pred_b) == 0:
        st.warning("Select a product in the sidebar.")
    else:
        critical_ratio = stockout_cost / (stockout_cost + overstock_cost)
        avg_low  = item_pred_b["pred_low"].mean()
        avg_mid  = item_pred_b["pred_mid"].mean()
        avg_high = item_pred_b["pred_high"].mean()
        opt_order = avg_low + (avg_high - avg_low) * critical_ratio

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{avg_low:.0f}</div><div class="metric-label">Conservative</div><div class="metric-delta" style="color:#8B8FA8">Low overstock risk</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card" style="border-color:#6C63FF;"><div class="metric-value" style="background:linear-gradient(90deg,#4ADE80,#48CAE4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{opt_order:.0f}</div><div class="metric-label">✅ Optimal Order</div><div class="metric-delta delta-good">Minimises cost</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{avg_mid:.0f}</div><div class="metric-label">Median Forecast</div><div class="metric-delta" style="color:#8B8FA8">50th percentile</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{avg_high:.0f}</div><div class="metric-label">Safe Order</div><div class="metric-delta" style="color:#8B8FA8">Low stockout risk</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        orders = np.linspace(max(0, avg_low * 0.5), avg_high * 1.5, 150)
        mu    = avg_mid
        sigma = max((avg_high - avg_low) / 2.56, 0.1)
        d = stats.norm(mu, sigma)
        exp_stockout  = stockout_cost  * np.maximum(d.mean() - orders + (orders - d.mean()) * d.cdf(orders) + sigma * d.pdf((orders - d.mean()) / sigma), 0)
        exp_overstock = overstock_cost * np.maximum((orders - d.mean()) * d.cdf(orders) - sigma * d.pdf((orders - d.mean()) / sigma), 0)
        total_cost    = exp_stockout + exp_overstock

        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(x=orders, y=exp_stockout, name="Stockout Cost",
            line=dict(color=COLORS["danger"], width=2), fill="tozeroy", fillcolor="rgba(248,113,113,0.08)"))
        fig_cost.add_trace(go.Scatter(x=orders, y=exp_overstock, name="Overstock Cost",
            line=dict(color=COLORS["warning"], width=2), fill="tozeroy", fillcolor="rgba(251,191,36,0.08)"))
        fig_cost.add_trace(go.Scatter(x=orders, y=total_cost, name="Total Cost",
            line=dict(color=COLORS["primary"], width=3)))
        fig_cost.add_shape(type="line", x0=opt_order, x1=opt_order, y0=0, y1=1, yref="paper",
            line=dict(color=COLORS["success"], width=2, dash="dot"))
        fig_cost.add_annotation(x=opt_order, y=0.95, yref="paper",
            text=f"Optimal: {opt_order:.0f}", showarrow=False,
            xanchor="left", font=dict(color=COLORS["success"], size=12))
        fig_cost.update_layout(template=PLOT_TEMPLATE, title="Expected Cost vs Order Quantity",
            xaxis_title="Order Quantity (units)", yaxis_title="Expected Cost ($)",
            height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,39,0.5)",
            xaxis=dict(gridcolor="#2D2F3E"), yaxis=dict(gridcolor="#2D2F3E"),
            legend=dict(font=dict(color="#C8CCE0")))
        st.plotly_chart(fig_cost, use_container_width=True)

        table_df = item_pred_b[["date","sales","pred_low","pred_mid","pred_high","in_interval"]].copy()
        table_df.columns = ["Date","Actual Sales","Lower Bound","Forecast","Upper Bound","In Interval?"]
        table_df["Date"] = table_df["Date"].dt.strftime("%b %d, %Y")
        table_df["In Interval?"] = table_df["In Interval?"].map({True:"✅", False:"❌"})
        for col in ["Actual Sales","Lower Bound","Forecast","Upper Bound"]:
            table_df[col] = table_df[col].round(1)
        st.dataframe(table_df, use_container_width=True, hide_index=True, height=300)

        min_cost = total_cost.min()
        st.markdown(f"""<div class="insight-box">
        <strong>📌 Recommendation for {selected_item.replace('_CA_1_validation','').replace('_validation','')}:</strong><br>
        • Order <strong>{opt_order:.0f} units/day</strong> to minimise expected cost (Critical ratio: {critical_ratio:.2f})<br>
        • Minimum expected daily cost: <strong>${min_cost:.2f}</strong><br>
        • Stockout cost: ${stockout_cost}/unit | Overstock cost: ${overstock_cost}/unit
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#4A4E6A;font-size:12px;'>"
    f"Built with LightGBM + Quantile Regression · M5 Walmart Forecasting Competition · "
    f"{summary['n_items']} Products · Store CA_1"
    f"</div>", unsafe_allow_html=True
)
