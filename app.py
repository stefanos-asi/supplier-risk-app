import json
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from supply_utils import (
    compute_supplier_stats,
    score_to_category,
    ALL_FEATURES,
    FEATURE_LABELS,
    FEATURE_GROUPS,
    RISK_CONFIG,
    INTERPRETATIONS,
    TARGET,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Supplier Risk Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f172a; color: #e2e8f0; }
section[data-testid="stSidebar"] { background: #0d1526; border-right: 1px solid #1e293b; }
.risk-banner { padding: 18px 24px; border-radius: 12px; margin: 12px 0 20px 0; display: flex; align-items: center; gap: 18px; border: 1px solid; }
.risk-label { font-size: 1.7rem; font-weight: 700; }
.risk-score { font-family: 'JetBrains Mono', monospace; font-size: 1rem; margin-top: 4px; opacity: 0.8; }
.kpi-card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px 16px; text-align: center; height: 100%; }
.kpi-val { font-size: 1.4rem; font-weight: 700; color: #f8fafc; font-family: 'JetBrains Mono', monospace; line-height: 1.2; }
.kpi-label { font-size: 0.68rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 5px; }
.kpi-sub { font-size: 0.66rem; color: #475569; margin-top: 2px; }
.section-label { font-size: 0.65rem; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 0.15em; margin: 20px 0 6px 0; padding-bottom: 4px; border-bottom: 1px solid #1e293b; }
.badge { display: inline-block; background: rgba(30,58,95,0.2); color: #7dd3fc; border: 1px solid rgba(30,64,175,0.27); border-radius: 5px; padding: 2px 9px; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; margin: 2px 2px 0 0; }
div[data-testid="stMetricValue"] { color: #f8fafc; }
div[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.75rem; }
.stTabs [data-baseweb="tab"] { color: #64748b; }
.stTabs [aria-selected="true"] { color: #e2e8f0 !important; }
.stButton > button { background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white; border: none; border-radius: 8px; font-weight: 600; letter-spacing: 0.03em; transition: all 0.2s; }
.stButton > button:hover { background: linear-gradient(135deg, #3b82f6, #2563eb); transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

# temporary debug — remove after fixing
import streamlit as st
try:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    st.write(f"URL length: {len(url)}")
    st.write(f"KEY length: {len(key)}")
    st.write(f"KEY starts with: {key[:10]}")
except Exception as e:
    st.error(f"Secrets error: {e}")

@st.cache_resource
def load_artifacts():
    model  = joblib.load("supplier_risk_model.pkl")
    scaler = joblib.load("supplier_risk_scaler.pkl")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, scaler, meta


@st.cache_data(ttl=300, show_spinner=False)
def load_supabase():
    try:
        from supabase import create_client, ClientOptions
        from config import SUPABASE_URL, SUPABASE_KEY
        sb = create_client(SUPABASE_URL, SUPABASE_KEY, options=ClientOptions(schema="public"))
        return pd.DataFrame(sb.table("fact_supply_chain").select("*").execute().data)
    except Exception as e:
        st.warning(f"Supabase connection failed: {e}")
        return None


try:
    model, scaler, meta = load_artifacts()
except Exception as e:
    st.error(f"Model not found: {e} — run train.py first.")
    st.stop()

fact_df   = load_supabase()
data_ok   = fact_df is not None
sup_stats = compute_supplier_stats(fact_df) if data_ok else None
ranges    = meta["feature_ranges"]


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    # plotly doesn't support 8-digit hex, so we convert manually
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def make_gauge(score: float, category: str) -> go.Figure:
    color = RISK_CONFIG[category]["color"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(score, 4),
        delta={"reference": 0.0, "valueformat": ".3f"},
        title={"text": "Volatility Risk Score (z)", "font": {"size": 14, "color": "#94a3b8"}},
        number={"font": {"color": color, "size": 40}},
        gauge={
            "axis":    {"range": [-3, 3], "tickwidth": 1, "tickcolor": "#475569"},
            "bar":     {"color": color, "thickness": 0.22},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [-3.0, -0.5], "color": hex_to_rgba("#10b981", 0.1)},
                {"range": [-0.5,  0.5], "color": hex_to_rgba("#f59e0b", 0.1)},
                {"range": [ 0.5,  1.5], "color": hex_to_rgba("#f97316", 0.1)},
                {"range": [ 1.5,  3.0], "color": hex_to_rgba("#ef4444", 0.1)},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.75, "value": score},
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
    return fig


def make_importance_bar(feat_imp: dict, imp_label: str) -> go.Figure:
    fi_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1]))
    vals      = list(fi_sorted.values())
    p66, p33  = np.percentile(vals, 66), np.percentile(vals, 33)
    colors    = ["#3b82f6" if v >= p66 else "#6366f1" if v >= p33 else "#475569" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=[FEATURE_LABELS.get(k, k) for k in fi_sorted],
        orientation="h", marker_color=colors,
        hovertemplate="%{y}: %{x:.5f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Feature Importance ({imp_label})", font=dict(size=13)),
        height=380, margin=dict(l=10, r=20, t=44, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def make_model_comparison_chart(comparison: list, best_name: str) -> go.Figure:
    df     = pd.DataFrame(comparison).sort_values("real_r2", ascending=True)
    colors = ["#10b981" if row["model"] == best_name else "#3b82f6" for _, row in df.iterrows()]
    fig    = go.Figure()
    fig.add_trace(go.Bar(
        name="Hold-out R2", x=df["real_r2"], y=df["model"],
        orientation="h", marker_color=colors,
        hovertemplate="<b>%{y}</b><br>R2: %{x:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="CV R2", x=df["cv_r2_mean"], y=df["model"],
        orientation="h", marker_color=colors, opacity=0.4,
        hovertemplate="<b>%{y}</b><br>CV R2: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        barmode="group",
        title=dict(text="Model Comparison — R2 (higher is better)", font=dict(size=13)),
        xaxis_title="R2", height=280, margin=dict(l=10, r=20, t=44, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#1e293b", range=[0, 1.05]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def make_supplier_bar(sup_stats: pd.DataFrame) -> go.Figure:
    df     = sup_stats.sort_values("volatility_risk_score", ascending=True)
    colors = [RISK_CONFIG.get(cat, {"color": "#94a3b8"})["color"] for cat in df["risk_category"]]
    fig = go.Figure(go.Bar(
        x=df["volatility_risk_score"], y=df["supplier_name"],
        orientation="h", marker_color=colors,
        text=df["risk_category"], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>",
    ))
    # threshold lines
    for x_val, color in [(-0.5, "#10b981"), (0.5, "#f97316"), (1.5, "#ef4444")]:
        fig.add_vline(x=x_val, line_dash="dot", line_color=color, line_width=1, opacity=0.3)
    fig.update_layout(
        title=dict(text="Supplier Volatility Risk Ranking", font=dict(size=13)),
        xaxis_title="Volatility Risk Score (z-normalized)",
        height=300, margin=dict(l=10, r=80, t=44, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", xaxis=dict(gridcolor="#1e293b"),
    )
    return fig


def make_scatter(fact_df: pd.DataFrame, sup_stats: pd.DataFrame) -> go.Figure:
    df = fact_df.merge(
        sup_stats[["supplier_name", "volatility_risk_score", "risk_category"]],
        on="supplier_name", how="left"
    ).dropna(subset=["volatility_risk_score", "defect_rates"])
    fig = px.scatter(
        df, x="volatility_risk_score", y="defect_rates",
        color="risk_category",
        color_discrete_map={k: v["color"] for k, v in RISK_CONFIG.items()},
        size="revenue_generated",
        hover_data=["supplier_name", "sku"],
        labels={"volatility_risk_score": "Risk Score", "defect_rates": "Defect Rate (%)"},
    )
    fig.update_layout(
        title=dict(text="Risk Score vs Defect Rate (bubble = revenue)", font=dict(size=13)),
        height=300, margin=dict(l=10, r=10, t=44, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
    )
    return fig


# sidebar
with st.sidebar:
    st.markdown("## ⚡ Risk Intelligence")
    st.markdown(
        f'<span class="badge">{meta["best_model_name"]}</span>'
        f'<span class="badge">R² {meta["real_r2"]:.3f}</span>'
        f'<span class="badge">RMSE {meta["real_rmse"]:.3f}</span>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    mode = st.radio("Input Mode", ["Manual Input", "Load Real Supplier"],
                    help="Load a real supplier to pre-fill inputs from Supabase")

    prefill = {}
    if mode == "Load Real Supplier" and data_ok:
        supplier = st.selectbox("Supplier", sup_stats["supplier_name"].dropna().unique())
        sr = sup_stats[sup_stats["supplier_name"] == supplier].iloc[0]
        fr = fact_df[fact_df["supplier_name"] == supplier].iloc[0]

        prefill = {
            "defect_rates":         float(fr.get("defect_rates", 2.0)),
            "supplier_lead_time":   float(fr.get("supplier_lead_time", 15)),
            "revenue_generated":    float(fr.get("revenue_generated", 50000)),
            "shipping_costs":       float(fr.get("shipping_costs", 500)),
            "revenue_volatility":   float(sr.get("revenue_volatility", 0)),
            "defect_volatility":    float(sr.get("defect_volatility", 0)),
            "lead_time_volatility": float(sr.get("lead_time_volatility", 0)),
            "velocity_volatility":  float(sr.get("velocity_volatility", 0)),
            "avg_revenue":          float(sr.get("avg_revenue", 50000)),
            "avg_defect_rate":      float(sr.get("avg_defect_rate", 2.0)),
            "avg_lead_time":        float(sr.get("avg_lead_time", 15)),
            "sku_count":            float(sr.get("sku_count", 5)),
        }

        real_score = float(sr["volatility_risk_score"])
        real_cat   = str(sr["risk_category"])
        rc         = RISK_CONFIG.get(real_cat, {"color": "#94a3b8", "bg": "rgba(148,163,184,0.08)", "icon": "❓"})
        st.markdown(
            f"<div style='background:{rc['bg']};border-radius:8px;padding:10px 14px;"
            f"margin:8px 0;border-left:3px solid {rc['color']};'>"
            f"<div style='font-size:0.68rem;color:#94a3b8;'>SQL ground truth</div>"
            f"<div style='font-size:1.1rem;font-weight:700;color:{rc['color']};font-family:monospace'>"
            f"{rc['icon']} {real_score:+.4f}</div>"
            f"<div style='font-size:0.72rem;color:{rc['color']};'>{real_cat}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        prefill = {feat: ranges[feat]["mean"] for feat in ALL_FEATURES}

    inputs = {}
    for group_name, feats in FEATURE_GROUPS.items():
        st.markdown(f'<div class="section-label">{group_name}</div>', unsafe_allow_html=True)
        for feat in feats:
            r   = ranges[feat]
            val = prefill.get(feat, r["mean"])
            lbl = FEATURE_LABELS[feat]
            if feat == "sku_count":
                inputs[feat] = float(st.number_input(lbl, min_value=0.0, value=float(round(val, 1)), step=1.0, key=feat))
            elif feat in ("revenue_generated", "avg_revenue"):
                inputs[feat] = float(st.number_input(lbl, min_value=0.0, value=float(round(val, 0)), step=1000.0, key=feat))
            else:
                f_min = max(0.0, r["min"] - r["std"])
                f_max = r["max"] + r["std"]
                step  = max(round((f_max - f_min) / 100, 3), 0.001)
                inputs[feat] = st.slider(lbl, min_value=round(f_min, 2), max_value=round(f_max, 2),
                                         value=round(float(val), 2), step=step, key=feat)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict Risk Score", use_container_width=True, type="primary")


st.markdown("# Supplier Risk Intelligence Platform")
st.caption(
    f"Model: **{meta['best_model_name']}** · "
    f"Hold-out R² **{meta['real_r2']:.3f}** · "
    f"RMSE **{meta['real_rmse']:.4f}** · "
    f"{meta['real_rows']} real rows + {meta['synthetic_rows']} synthetic"
)

tab1, tab2, tab3 = st.tabs(["🎯  Risk Prediction", "📊  Portfolio Overview", "🔬  Model Details"])


with tab1:
    if predict_btn:
        input_df   = pd.DataFrame([inputs])[ALL_FEATURES]
        pred_score = float(model.predict(scaler.transform(input_df))[0])
        pred_cat   = score_to_category(pred_score)
        cfg        = RISK_CONFIG[pred_cat]

        st.markdown(
            f'<div class="risk-banner" style="background:{cfg["bg"]};border-color:{cfg["color"]};">'
            f'<span style="font-size:2.2rem;">{cfg["icon"]}</span>'
            f'<div>'
            f'<div class="risk-label" style="color:{cfg["color"]};">{pred_cat}</div>'
            f'<div class="risk-score" style="color:{cfg["color"]};">Score: {pred_score:+.4f}</div>'
            f'</div></div>',
            unsafe_allow_html=True
        )

        kpi_data = [
            ("Predicted Score",    f"{pred_score:+.4f}", "z-composite"),
            ("Defect Rate",        f"{inputs['defect_rates']:.2f}%",       "SKU level"),
            ("Lead Time",          f"{inputs['supplier_lead_time']:.0f}d", "SKU level"),
            ("Revenue Volatility", f"{inputs['revenue_volatility']:.1f}",  "supplier std"),
            ("Defect Volatility",  f"{inputs['defect_volatility']:.2f}",   "supplier std"),
        ]
        for col, (lbl, val, sub) in zip(st.columns(5), kpi_data):
            col.markdown(
                f'<div class="kpi-card"><div class="kpi-val">{val}</div>'
                f'<div class="kpi-label">{lbl}</div>'
                f'<div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True
            )

        st.markdown("")
        col_g, col_i = st.columns(2)

        with col_g:
            st.plotly_chart(make_gauge(pred_score, pred_cat), use_container_width=True)
            st.markdown("""
| Score range | Category |
|---|---|
| > 1.5 | 🚨 Critical Volatility |
| 0.5 → 1.5 | 🔶 High Volatility |
| -0.5 → 0.5 | ⚠️ Moderate |
| < -0.5 | ✅ Stable |
""")

        with col_i:
            st.plotly_chart(
                make_importance_bar(meta["feature_importances"], meta["importance_label"]),
                use_container_width=True
            )

        # show delta vs SQL score when a real supplier is loaded
        if mode == "Load Real Supplier" and data_ok:
            delta = pred_score - real_score
            close = abs(delta) < 0.1
            st.markdown(
                f'<div style="background:#1e293b;border-radius:8px;padding:12px 18px;'
                f'border:1px solid #334155;margin-top:8px;">'
                f'<span style="color:#94a3b8;font-size:0.8rem;">Model vs SQL ground truth: </span>'
                f'<span style="font-family:monospace;color:{"#10b981" if close else "#f59e0b"};">Δ = {delta:+.4f}</span>'
                f'<span style="color:#64748b;font-size:0.75rem;margin-left:10px;">'
                f'SQL: {real_score:+.4f} → Model: {pred_score:+.4f}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.info(f"**Business Interpretation:** {INTERPRETATIONS[pred_cat]}")

    else:
        st.markdown(
            '<div style="background:#1e293b;border:1px dashed #334155;border-radius:12px;'
            'padding:48px;text-align:center;color:#475569;margin-top:24px;">'
            '<div style="font-size:2rem;margin-bottom:12px;">🔍</div>'
            '<div style="font-size:1rem;">Adjust parameters in the sidebar<br>'
            'and click <strong style="color:#94a3b8;">Predict Risk Score</strong></div>'
            '</div>',
            unsafe_allow_html=True
        )


with tab2:
    if not data_ok:
        st.warning("Supabase data unavailable. Check your connection.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(make_supplier_bar(sup_stats), use_container_width=True)
        with c2:
            st.plotly_chart(make_scatter(fact_df, sup_stats), use_container_width=True)

        st.markdown("#### Supplier Risk Summary")
        display = sup_stats[[
            "supplier_name", "sku_count", "avg_revenue", "avg_defect_rate", "avg_lead_time",
            "revenue_volatility", "defect_volatility", "lead_time_volatility",
            "volatility_risk_score", "risk_category"
        ]].copy().round(3)
        display.columns = [
            "Supplier", "SKUs", "Avg Revenue ($)", "Avg Defect %", "Avg Lead Time",
            "Rev Volatility", "Defect Volatility", "LT Volatility", "Risk Score", "Category"
        ]

        def style_category(val):
            return f"color: {RISK_CONFIG.get(val, {}).get('color', '#94a3b8')}; font-weight: 600"

        st.dataframe(display.style.applymap(style_category, subset=["Category"]),
                     use_container_width=True, hide_index=True)


with tab3:
    st.markdown("### Model Selection & Methodology")

    st.markdown(
        f'<div style="background:rgba(5,96,58,0.13);border:1px solid rgba(16,185,129,0.2);'
        f'border-radius:10px;padding:14px 20px;margin-bottom:20px;">'
        f'<span style="color:#6ee7b7;font-weight:700;font-size:1.05rem;">✅ Selected model: {meta["best_model_name"]}</span>'
        f'<span style="color:#94a3b8;font-size:0.85rem;margin-left:12px;">'
        f'Chosen by hold-out R² with Occam\'s razor tiebreak (simpler model preferred when performance difference ≤ 0.02)</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    for col, (lbl, val, sub) in zip(
        st.columns(4),
        [
            ("Hold-out R²",   f"{meta['real_r2']:.4f}",   "Real data only"),
            ("Hold-out RMSE", f"{meta['real_rmse']:.4f}",  "Real data only"),
            ("Hold-out MAE",  f"{meta['real_mae']:.4f}",   "Real data only"),
            ("CV R²",         f"{meta['cv_r2_mean']:.3f} ± {meta['cv_r2_std']:.3f}", "5-fold on augmented"),
        ]
    ):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-val">{val}</div>'
            f'<div class="kpi-label">{lbl}</div>'
            f'<div class="kpi-sub">{sub}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("")
    col_chart, col_table = st.columns([1.2, 1])

    with col_chart:
        st.plotly_chart(
            make_model_comparison_chart(meta["model_comparison"], meta["best_model_name"]),
            use_container_width=True
        )

    with col_table:
        st.markdown("#### All models evaluated")
        comp_df = pd.DataFrame(meta["model_comparison"]).rename(columns={
            "model":        "Model",
            "cv_r2_mean":   "CV R²",
            "cv_r2_std":    "CV R² std",
            "cv_rmse_mean": "CV RMSE",
            "real_r2":      "Hold-out R²",
            "real_rmse":    "Hold-out RMSE",
            "real_mae":     "Hold-out MAE",
        }).sort_values("Hold-out R²", ascending=False)

        def style_winner(row):
            if row["Model"] == meta["best_model_name"]:
                return ["color: #6ee7b7; font-weight: 700"] * len(row)
            return [""] * len(row)

        st.dataframe(comp_df.style.apply(style_winner, axis=1),
                     use_container_width=True, hide_index=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Why this model was chosen")
        st.markdown(f"""
The target (`volatility_risk_score`) is a weighted linear combination of z-normalized
inputs by construction:

```
score = z(revenue_std)    × 0.30
      + z(defect_std)     × 0.30
      + z(lead_time_std)  × 0.20
      + z(velocity_std)   × 0.20
```

Since the data-generating process is linear, a linear model is the right starting point.
We ran all four and selected **{meta["best_model_name"]}** based on hold-out performance.
When two models are within 0.02 R², the simpler one wins — it generalises better
and is a lot easier to explain.
        """)

    with col_r:
        st.markdown("#### Training setup")
        st.markdown(f"""
| | |
|---|---|
| **Real rows** | {meta['real_rows']} (SKU-level, from Supabase) |
| **Synthetic rows** | {meta['synthetic_rows']} (Gaussian-augmented) |
| **Noise factor** | {meta['noise_factor']*100:.0f}% |
| **Total training** | {meta['training_rows']} rows |
| **Test set** | Real data only — never seen in training |
| **CV strategy** | 5-fold KFold on augmented set |
| **Selection** | Hold-out R² + Occam's razor |
        """)

    st.markdown("---")
    st.info(
        "Real supply chain data was augmented with synthetic samples to make ML training viable. "
        "Synthetic rows are generated by adding calibrated Gaussian noise to real observations, "
        "then recomputing the risk score with the same formula as the SQL view — so there's no leakage. "
        "All metrics shown are evaluated on real data only."
    )