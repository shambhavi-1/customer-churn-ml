"""
app.py  —  Churn Intelligence Dashboard

Pages
-----
1. 🎯 Predict          — single customer scoring with SHAP + retention
2. 📊 Model Insights   — feature importance + training metrics
3. 📂 Batch Prediction — CSV upload → bulk score → download
4. 📈 Analytics        — churn distributions & patterns
5. 🔍 Monitoring       — live risk tracking + alerts

Run
---
    streamlit run dashboard/app.py
"""

import json
import sys
import shutil
from pathlib import Path
from datetime import datetime

# ── force-clear stale bytecode cache on every startup ───────────────────────
for _cache in Path(__file__).parent.parent.rglob("__pycache__"):
    shutil.rmtree(_cache, ignore_errors=True)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── repo root ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.predict import ChurnPredictor

# ── paths ────────────────────────────────────────────────────────────────────
MODEL_PATH    = _ROOT / "models" / "best_model.pkl"
METADATA_PATH = _ROOT / "models" / "model_metadata.json"
FI_PATH       = _ROOT / "models" / "feature_importance.csv"
DATA_PATH     = _ROOT / "data"   / "churn_dataset.csv"

# ── constants ────────────────────────────────────────────────────────────────
HIGH_RISK_ALERT_THRESHOLD = 0.70   # alert if > 70% of scored customers are high risk

# ────────────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #1a1f2e; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# Cached resource loaders
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_predictor():
    if not MODEL_PATH.exists():
        return None
    return ChurnPredictor.load(model_path=MODEL_PATH, metadata_path=METADATA_PATH)

@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text())

@st.cache_data
def load_feature_importance():
    if not FI_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(FI_PATH)

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)

# ────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ────────────────────────────────────────────────────────────────────────────
def init_session_state():
    """Initialise all session_state keys used across pages."""
    defaults = {
        "last_result":       None,
        "last_input":        None,
        "batch_results_df":  None,
        "batch_filename":    None,
        "monitoring_log":    [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Validate cached batch df has all required columns — clear if stale
    required_batch_cols = {"churn_label", "risk_segment", "churn_probability", "expected_revenue_loss"}
    cached = st.session_state.get("batch_results_df")
    if cached is not None and not required_batch_cols.issubset(set(cached.columns)):
        st.session_state["batch_results_df"] = None
        st.session_state["batch_filename"]   = None

# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────
def render_sidebar(predictor, metadata):
    with st.sidebar:
        st.title("📉 Churn Intelligence")
        st.caption("Customer Churn Prediction System")
        st.divider()

        page = st.radio(
            "Navigation",
            options=[
                "🎯 Predict",
                "📊 Model Insights",
                "📂 Batch Prediction",
                "📈 Analytics",
                "🔍 Monitoring",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        if predictor:
            best = metadata.get("best_model", {})
            st.success("✅ Model Loaded")
            st.markdown(f"**Model:** <span style='color:white'>{best.get('run_id','–')}</span>", unsafe_allow_html=True)
            st.markdown(f"**AUC:** <span style='color:white'>{best.get('test_roc_auc', 0):.4f}</span>", unsafe_allow_html=True)
            st.markdown(f"**F1:** <span style='color:white'>{best.get('test_f1', 0):.4f}</span>", unsafe_allow_html=True)
            st.markdown(f"**SMOTE:** <span style='color:white'>{'Yes' if best.get('smote') else 'No'}</span>", unsafe_allow_html=True)
        else:
            st.error("❌ Model not found")
            st.caption("Run `python src/run_training.py` first")

        # monitoring badge — show live alert in sidebar if threshold breached
        log = st.session_state.get("monitoring_log", [])
        if log:
            latest = log[-1]
            if latest["high_risk_pct"] > HIGH_RISK_ALERT_THRESHOLD * 100:
                st.divider()
                st.error(f"🚨 Alert: {latest['high_risk_pct']:.1f}% high-risk")

    return page

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
RISK_EMOJI = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}

def risk_label(segment):
    return f"{RISK_EMOJI.get(segment, '⚪')} {segment}"

def gauge_fig(probability):
    color = (
        "#ef4444" if probability >= 0.75 else
        "#f97316" if probability >= 0.50 else
        "#eab308" if probability >= 0.25 else
        "#22c55e"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        gauge={
            "axis":  {"range": [0, 100]},
            "bar":   {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0,  25], "color": "rgba(34,197,94,0.15)"},
                {"range": [25, 50], "color": "rgba(234,179,8,0.15)"},
                {"range": [50, 75], "color": "rgba(249,115,22,0.15)"},
                {"range": [75,100], "color": "rgba(239,68,68,0.15)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.8,
                "value": probability * 100,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Predict
# ────────────────────────────────────────────────────────────────────────────
def page_predict(predictor):
    st.header("🎯 Single Customer Prediction")
    st.caption("Score one customer and get instant churn risk, revenue impact, retention strategy and SHAP explanation.")

    if predictor is None:
        st.error("Model not loaded. Run `python src/run_training.py` first.")
        return

    # ── input form ───────────────────────────────────────────────────────────
    with st.form("page_predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Account Info")
            customer_id     = st.text_input("Customer ID", value="CUST-00001")
            tenure          = st.number_input("Tenure (months)", min_value=0, max_value=720, value=6)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=1.0, max_value=10000.0, value=950.0, step=10.0)
            total_charges   = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges))

        with c2:
            st.subheader("Contract & Service")
            contract_type    = st.selectbox("Contract Type",    ["Month-to-Month", "One Year", "Two Year"])
            internet_service = st.selectbox("Internet Service", ["Fiber Optic", "DSL", "No"])
            payment_method   = st.selectbox("Payment Method",   ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"])

        with c3:
            st.subheader("Usage & Support")
            support_calls    = st.number_input("Support Calls",   min_value=0, max_value=100,  value=7)
            number_of_logins = st.number_input("Logins (period)", min_value=0, max_value=1000, value=3)
            usage_hours      = st.number_input("Usage Hours",     min_value=0.0, max_value=8760.0, value=12.5)

        submitted = st.form_submit_button("⚡ Predict Churn Risk", width="stretch", type="primary")

    if not submitted:
        if st.session_state.get("last_result") is not None:
            st.info("Showing last prediction. Submit the form to score a new customer.")
            _render_predict_results(st.session_state["last_result"], st.session_state["last_input"])
        else:
            st.info("👆 Fill in the form above and click **Predict Churn Risk**")
        return

    with st.spinner("Scoring customer …"):
        input_dict = {
            "customer_id":      customer_id,
            "tenure":           tenure,
            "monthly_charges":  monthly_charges,
            "total_charges":    total_charges,
            "support_calls":    support_calls,
            "contract_type":    contract_type,
            "internet_service": internet_service,
            "payment_method":   payment_method,
            "number_of_logins": number_of_logins,
            "usage_hours":      usage_hours,
        }
        result = predictor.predict_single(input_dict)

    st.session_state["last_result"] = result
    st.session_state["last_input"]  = input_dict
    _render_predict_results(result, input_dict)


def _render_predict_results(result, inp):
    """Render the full prediction result panel."""
    st.divider()
    st.subheader("Prediction Results")

    col_gauge, col_metrics, col_actions = st.columns([1.2, 1, 1.4])

    # ── gauge ─────────────────────────────────────────────────────────────────
    with col_gauge:
        st.plotly_chart(gauge_fig(result.churn_probability),
                        width="stretch", config={"displayModeBar": False})
        verdict = "⚠️ WILL CHURN" if result.churn_label else "✅ WILL RETAIN"
        st.markdown(f"<h3 style='text-align:center'>{verdict}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center'>{risk_label(result.risk_segment)}</p>",
                    unsafe_allow_html=True)

    # ── metrics ───────────────────────────────────────────────────────────────
    with col_metrics:
        st.subheader("Key Metrics")
        st.metric("Churn Probability",         f"{result.churn_probability:.1%}")
        st.metric("Risk Segment",              risk_label(result.risk_segment))
        st.metric("Expected Revenue Loss",     f"${result.expected_revenue_loss:,.2f}",
                  help="churn_probability × monthly_charges × 6 months")
        st.metric("Monthly Charges",           f"${inp.get('monthly_charges', 0):,.2f}")
        st.metric("Tenure",                    f"{inp.get('tenure', 0)} months")

    # ── retention + SHAP ─────────────────────────────────────────────────────
    with col_actions:
        st.subheader("🛡️ Retention Strategy")
        for action in result.retention_strategy:
            st.info(f"💡 {action}")

        st.subheader("🔍 SHAP — Top Churn Drivers")
        for i, reason in enumerate(result.top_reasons, 1):
            st.warning(f"**#{i}** {reason}")

# ────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Model Insights
# ────────────────────────────────────────────────────────────────────────────
def page_model_insights(metadata):
    st.header("📊 Model Insights")
    st.caption("Training metrics, model comparison, and feature importance.")

    if not metadata:
        st.error("model_metadata.json not found. Run training first.")
        return

    best     = metadata.get("best_model", {})
    all_runs = metadata.get("all_runs", [])

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st.subheader("Best Model Performance")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Model",     best.get("run_id", "–").replace("_", " ").title())
    k2.metric("ROC-AUC",   f"{best.get('test_roc_auc', 0):.4f}")
    k3.metric("Precision", f"{best.get('test_precision', 0):.4f}")
    k4.metric("Recall",    f"{best.get('test_recall', 0):.4f}")
    k5.metric("F1",        f"{best.get('test_f1', 0):.4f}")
    st.caption(
        f"SMOTE: {'✅ Yes' if best.get('smote') else '❌ No'}  ·  "
        f"CV AUC: {best.get('cv_roc_auc', 0):.4f} ± {best.get('cv_roc_auc_std', 0):.4f}  ·  "
        f"Trained: {metadata.get('trained_at','–')[:19].replace('T',' ')} UTC"
    )
    st.divider()

    if all_runs:
        left, right = st.columns(2)
        runs_df = pd.DataFrame(all_runs)

        with left:
            st.subheader("ROC-AUC Comparison")
            fig = px.bar(
                runs_df.sort_values("test_roc_auc", ascending=True),
                x="test_roc_auc", y="run_id", orientation="h",
                color="smote",
                color_discrete_map={True: "#3b82f6", False: "#94a3b8"},
                labels={"test_roc_auc": "Test ROC-AUC", "run_id": ""},
                text_auto=".4f",
            )
            fig.update_layout(height=320, showlegend=True, legend_title="SMOTE",
                              xaxis=dict(range=[0.5, 1.0]),
                              paper_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=0, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")

        with right:
            st.subheader("Metric Radar")
            cats = ["ROC-AUC", "Precision", "Recall", "F1"]
            vals = [best.get("test_roc_auc", 0), best.get("test_precision", 0),
                    best.get("test_recall", 0),   best.get("test_f1", 0)]
            fig2 = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself",
                line=dict(color="#3b82f6", width=2),
                fillcolor="rgba(59,130,246,0.2)",
            ))
            fig2.update_layout(height=320,
                               polar=dict(radialaxis=dict(range=[0, 1])),
                               showlegend=False,
                               paper_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=40, r=40, t=20, b=20))
            st.plotly_chart(fig2, width="stretch")

        st.subheader("All Runs Summary")
        display = (
            runs_df[["run_id","smote","cv_roc_auc","test_roc_auc",
                     "test_precision","test_recall","test_f1"]]
            .rename(columns={"run_id":"Run","smote":"SMOTE","cv_roc_auc":"CV AUC",
                             "test_roc_auc":"Test AUC","test_precision":"Precision",
                             "test_recall":"Recall","test_f1":"F1"})
            .sort_values("Test AUC", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(display, width="stretch", hide_index=True)

    fi_df = load_feature_importance()
    if not fi_df.empty:
        st.divider()
        st.subheader("Feature Importance (Top 15)")
        top = fi_df.head(15).sort_values("importance")
        fig3 = px.bar(top, x="importance", y="feature", orientation="h",
                      color="importance",
                      color_continuous_scale=["#93c5fd", "#3b82f6", "#1d4ed8"],
                      labels={"importance": "Importance Score", "feature": ""})
        fig3.update_layout(height=420, coloraxis_showscale=False,
                           paper_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=0, r=10, t=10, b=10))
        st.plotly_chart(fig3, width="stretch")

# ────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Batch Prediction
# ────────────────────────────────────────────────────────────────────────────
def page_batch(predictor):
    # guard — this page must never show single-customer form content
    st.header("📂 Batch Prediction")
    st.caption("Upload a CSV file containing multiple customers and score them all at once.")

    if predictor is None:
        st.error("Model not loaded. Run `python src/run_training.py` first.")
        return

    st.info("ℹ️ This page scores **many customers at once** via CSV upload. To score a single customer manually, use **🎯 Predict**.")

    with st.expander("📋 Required CSV columns"):
        st.code(
            "customer_id, tenure, monthly_charges, support_calls,\n"
            "contract_type, internet_service, payment_method\n\n"
            "# Optional (auto-imputed if missing):\n"
            "total_charges, number_of_logins, usage_hours",
            language="text",
        )
        sample = load_dataset()
        if not sample.empty:
            st.caption("Sample from training data:")
            st.dataframe(sample.head(3), width="stretch", hide_index=True)

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"], key="batch_uploader")

    if uploaded is None:
        cached = st.session_state["batch_results_df"]
        # validate cached df has required columns before rendering
        required_cols = {"churn_label", "risk_segment", "churn_probability", "expected_revenue_loss"}
        if cached is not None and required_cols.issubset(set(cached.columns)):
            st.info(f"Showing last batch: **{st.session_state['batch_filename']}**")
            _render_batch_results(cached)
        else:
            # clear stale/incompatible cached data silently
            st.session_state["batch_results_df"] = None
            st.info("👆 Upload a CSV file to begin batch scoring. To score a single customer manually, use **🎯 Predict**.")
        return

    df = pd.read_csv(uploaded)
    st.success(f"✅ Loaded **{len(df):,}** customers × {df.shape[1]} columns")
    st.dataframe(df.head(5), width="stretch", hide_index=True)

    if st.button("⚡ Score All Customers", type="primary", width="stretch"):
        progress = st.progress(0, text="Scoring …")
        rows     = df.to_dict(orient="records")
        results, errors = [], []

        for i, row in enumerate(rows):
            try:
                r = predictor.predict_single(row)
                results.append({
                    "customer_id":           r.customer_id or str(i),
                    "churn_probability":     round(r.churn_probability, 4),
                    "churn_label":           r.churn_label,
                    "risk_segment":          r.risk_segment,
                    "expected_revenue_loss": r.expected_revenue_loss,
                    "retention_strategy":    " | ".join(r.retention_strategy),
                    "top_reasons":           " | ".join(r.top_reasons),
                })
            except Exception as e:
                errors.append({"row": i, "error": str(e)})
            progress.progress((i + 1) / len(rows), text=f"Scored {i+1} / {len(rows)}")

        progress.empty()
        res_df = pd.DataFrame(results)

        # save to session_state — separate key from predict page
        st.session_state["batch_results_df"] = res_df
        st.session_state["batch_filename"]   = uploaded.name

        # push stats to monitoring log
        high_risk_count = res_df["risk_segment"].isin(["High", "Critical"]).sum()
        st.session_state["monitoring_log"].append({
            "timestamp":     datetime.now().strftime("%H:%M:%S"),
            "total":         len(res_df),
            "avg_prob":      round(res_df["churn_probability"].mean() * 100, 2),
            "high_risk_pct": round(high_risk_count / len(res_df) * 100, 2),
            "total_loss":    round(res_df["expected_revenue_loss"].sum(), 2),
        })

        _render_batch_results(res_df)

        if errors:
            with st.expander(f"⚠️ {len(errors)} rows failed"):
                st.dataframe(pd.DataFrame(errors), width="stretch")


def _render_batch_results(res_df):
    """Render summary metrics, chart, table, and download for a batch result."""
    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Scored",         f"{len(res_df):,}")
    m2.metric("Predicted Churn",      f"{res_df['churn_label'].sum():,}")
    m3.metric("High / Critical Risk", f"{res_df['risk_segment'].isin(['High','Critical']).sum():,}")
    m4.metric("Total Expected Loss",  f"${res_df['expected_revenue_loss'].sum():,.0f}")

    seg = res_df["risk_segment"].value_counts().reset_index()
    seg.columns = ["Segment", "Count"]
    fig = px.bar(seg, x="Segment", y="Count", color="Segment",
                 color_discrete_map={"Critical":"#ef4444","High":"#f97316",
                                     "Medium":"#eab308","Low":"#22c55e"},
                 title="Risk Segment Distribution")
    fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Scored Results")
    st.dataframe(
        res_df.sort_values("churn_probability", ascending=False),
        width="stretch", hide_index=True,
    )

    st.download_button(
        "⬇️ Download Results CSV",
        data      = res_df.to_csv(index=False).encode(),
        file_name = "churn_predictions.csv",
        mime      = "text/csv",
        width="stretch",
    )

# ────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Analytics
# ────────────────────────────────────────────────────────────────────────────
def page_analytics():
    st.header("📈 Analytics")
    st.caption("Churn patterns and distributions from the training dataset.")

    df = load_dataset()
    if df.empty:
        st.error("churn_dataset.csv not found. Run `python data/generate_data.py` first.")
        return

    # ── overview ──────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers",    f"{len(df):,}")
    m2.metric("Overall Churn Rate", f"{df['churn'].mean():.1%}")
    m3.metric("Avg Monthly Charge", f"${df['monthly_charges'].mean():,.0f}")
    m4.metric("Avg Tenure",         f"{df['tenure'].mean():.0f} months")

    st.divider()

    # ── churn distribution ────────────────────────────────────────────────────
    st.subheader("Churn Distribution")
    left, right = st.columns(2)

    with left:
        churn_counts = df["churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        churn_counts["Churn"] = churn_counts["Churn"].map({0: "Retained", 1: "Churned"})
        fig_pie = px.pie(churn_counts, names="Churn", values="Count",
                         color="Churn",
                         color_discrete_map={"Retained": "#22c55e", "Churned": "#ef4444"},
                         hole=0.4)
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, width="stretch")

    with right:
        # monthly charges distribution by churn
        fig_hist = px.histogram(
            df, x="monthly_charges", color="churn",
            color_discrete_map={0: "#22c55e", 1: "#ef4444"},
            barmode="overlay", opacity=0.7, nbins=40,
            labels={"churn": "Churn", "monthly_charges": "Monthly Charges ($)"},
            title="Monthly Charges Distribution",
        )
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_hist, width="stretch")

    st.divider()

    # ── churn by contract ─────────────────────────────────────────────────────
    st.subheader("Churn by Contract Type")
    ct = (df.groupby("contract_type")["churn"].mean() * 100).round(1).reset_index()
    ct.columns = ["Contract Type", "Churn Rate (%)"]
    fig_ct = px.bar(ct, x="Contract Type", y="Churn Rate (%)",
                    color="Churn Rate (%)",
                    color_continuous_scale=["#22c55e", "#eab308", "#ef4444"],
                    text_auto=".1f")
    fig_ct.update_traces(texttemplate="%{text}%")
    fig_ct.update_layout(coloraxis_showscale=False,
                         paper_bgcolor="rgba(0,0,0,0)",
                         margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_ct, width="stretch")

    st.divider()

    # ── churn vs tenure ───────────────────────────────────────────────────────
    st.subheader("Churn Rate vs Tenure")
    left2, right2 = st.columns(2)

    with left2:
        df["tenure_bucket"] = pd.cut(
            df["tenure"], bins=[0, 12, 36, 60, 72],
            labels=["0–12 mo", "13–36 mo", "37–60 mo", "61–72 mo"],
            include_lowest=True,
        )
        tb = (df.groupby("tenure_bucket", observed=True)["churn"].mean() * 100).round(1).reset_index()
        tb.columns = ["Tenure Group", "Churn Rate (%)"]
        fig_tb = px.bar(tb, x="Tenure Group", y="Churn Rate (%)",
                        color="Churn Rate (%)",
                        color_continuous_scale=["#22c55e", "#eab308", "#ef4444"],
                        text_auto=".1f", title="Churn Rate by Tenure Group")
        fig_tb.update_traces(texttemplate="%{text}%")
        fig_tb.update_layout(coloraxis_showscale=False,
                             paper_bgcolor="rgba(0,0,0,0)",
                             margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_tb, width="stretch")

    with right2:
        # scatter: tenure vs monthly_charges coloured by churn
        sample = df.sample(min(1000, len(df)), random_state=42)
        fig_sc = px.scatter(
            sample, x="tenure", y="monthly_charges", color="churn",
            color_discrete_map={0: "#22c55e", 1: "#ef4444"},
            opacity=0.5,
            labels={"tenure": "Tenure (months)", "monthly_charges": "Monthly Charges ($)",
                    "churn": "Churn"},
            title="Tenure vs Monthly Charges (sampled 1k)",
        )
        fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                             margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_sc, width="stretch")

    st.divider()

    # ── support calls ─────────────────────────────────────────────────────────
    st.subheader("Churn Rate by Support Calls")
    sc = (df.groupby("support_calls")["churn"].mean() * 100).round(1).reset_index()
    sc.columns = ["Support Calls", "Churn Rate (%)"]
    fig_sc2 = px.line(sc, x="Support Calls", y="Churn Rate (%)",
                      markers=True, color_discrete_sequence=["#3b82f6"])
    fig_sc2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_sc2, width="stretch")

# ────────────────────────────────────────────────────────────────────────────
# PAGE 5 — Monitoring
# ────────────────────────────────────────────────────────────────────────────
def page_monitoring():
    st.header("🔍 Monitoring")
    st.caption("Live risk tracking across batch scoring runs. Alerts fire when high-risk % exceeds 70%.")

    log = st.session_state.get("monitoring_log", [])

    if not log:
        st.info(
            "No scoring runs logged yet. "
            "Go to **📂 Batch Prediction**, upload a CSV and score it — "
            "results will appear here automatically."
        )
        return

    log_df = pd.DataFrame(log)

    # ── alert banner ──────────────────────────────────────────────────────────
    latest = log_df.iloc[-1]
    if latest["high_risk_pct"] > HIGH_RISK_ALERT_THRESHOLD * 100:
        st.error(
            f"🚨 **Alert:** Latest batch has **{latest['high_risk_pct']:.1f}%** high-risk customers "
            f"— exceeds the {HIGH_RISK_ALERT_THRESHOLD*100:.0f}% threshold!"
        )
    else:
        st.success(
            f"✅ Latest batch: **{latest['high_risk_pct']:.1f}%** high-risk "
            f"— within acceptable threshold (<{HIGH_RISK_ALERT_THRESHOLD*100:.0f}%)"
        )

    st.divider()

    # ── latest run KPIs ───────────────────────────────────────────────────────
    st.subheader(f"Latest Run — {latest['timestamp']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers Scored",  f"{int(latest['total']):,}")
    m2.metric("Avg Churn Prob",    f"{latest['avg_prob']:.1f}%",
              delta=f"{'⚠️ HIGH' if latest['avg_prob'] > 50 else '✅ OK'}")
    m3.metric("High-Risk %",       f"{latest['high_risk_pct']:.1f}%",
              delta=f"{'🚨 ALERT' if latest['high_risk_pct'] > HIGH_RISK_ALERT_THRESHOLD*100 else '✅ OK'}")
    m4.metric("Total Expected Loss", f"${latest['total_loss']:,.0f}")

    st.divider()

    # ── trend charts (only if > 1 run) ────────────────────────────────────────
    if len(log_df) > 1:
        st.subheader("Trend Across Runs")
        left, right = st.columns(2)

        with left:
            fig1 = px.line(log_df, x="timestamp", y="avg_prob",
                           markers=True, title="Avg Churn Probability (%) Over Runs",
                           color_discrete_sequence=["#3b82f6"],
                           labels={"avg_prob": "Avg Prob (%)", "timestamp": "Run Time"})
            fig1.add_hline(y=50, line_dash="dash", line_color="#ef4444",
                           annotation_text="50% threshold")
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig1, width="stretch")

        with right:
            fig2 = px.line(log_df, x="timestamp", y="high_risk_pct",
                           markers=True, title="High-Risk % Over Runs",
                           color_discrete_sequence=["#f97316"],
                           labels={"high_risk_pct": "High-Risk (%)", "timestamp": "Run Time"})
            fig2.add_hline(y=HIGH_RISK_ALERT_THRESHOLD * 100,
                           line_dash="dash", line_color="#ef4444",
                           annotation_text=f"{HIGH_RISK_ALERT_THRESHOLD*100:.0f}% alert threshold")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig2, width="stretch")

    # ── full run history table ────────────────────────────────────────────────
    st.subheader("Run History")
    st.dataframe(
        log_df.rename(columns={
            "timestamp":     "Time",
            "total":         "Customers",
            "avg_prob":      "Avg Prob (%)",
            "high_risk_pct": "High-Risk (%)",
            "total_loss":    "Expected Loss ($)",
        }).sort_index(ascending=False),
        width="stretch",
        hide_index=True,
    )

    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state["monitoring_log"] = []
        st.rerun()

# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    init_session_state()
    predictor = load_predictor()
    metadata  = load_metadata()
    page      = render_sidebar(predictor, metadata)

    if   page == "🎯 Predict":          page_predict(predictor)
    elif page == "📊 Model Insights":   page_model_insights(metadata)
    elif page == "📂 Batch Prediction": page_batch(predictor)
    elif page == "📈 Analytics":        page_analytics()
    elif page == "🔍 Monitoring":       page_monitoring()

if __name__ == "__main__":
    main()