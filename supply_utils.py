import numpy as np
import pandas as pd


# SKU-level features — trimmed to what actually makes sense as predictors
SKU_FEATURES = [
    "defect_rates",
    "supplier_lead_time",
    "revenue_generated",
    "shipping_costs",
]

# Supplier-level features — these feed directly into the risk score formula
SUPPLIER_FEATURES = [
    "revenue_volatility",
    "defect_volatility",
    "lead_time_volatility",
    "velocity_volatility",
    "avg_revenue",
    "avg_defect_rate",
    "avg_lead_time",
    "sku_count",
]

ALL_FEATURES = SKU_FEATURES + SUPPLIER_FEATURES
TARGET = "volatility_risk_score"

FEATURE_LABELS = {
    "defect_rates":          "Defect Rate (%)",
    "supplier_lead_time":    "Lead Time (days)",
    "revenue_generated":     "Revenue ($)",
    "shipping_costs":        "Shipping Costs ($)",
    "revenue_volatility":    "Revenue Volatility (std)",
    "defect_volatility":     "Defect Rate Volatility (std)",
    "lead_time_volatility":  "Lead Time Volatility (std)",
    "velocity_volatility":   "Rev. Velocity Volatility (std)",
    "avg_revenue":           "Avg Revenue per SKU ($)",
    "avg_defect_rate":       "Avg Defect Rate (%)",
    "avg_lead_time":         "Avg Lead Time (days)",
    "sku_count":             "Number of SKUs",
}

FEATURE_GROUPS = {
    "📦 SKU-Level Inputs":       SKU_FEATURES,
    "🏭 Supplier-Level Context": SUPPLIER_FEATURES,
}

RISK_CONFIG = {
    "STABLE":              {"color": "#10b981", "bg": "rgba(16,185,129,0.08)",  "icon": "✅", "order": 0},
    "MODERATE":            {"color": "#f59e0b", "bg": "rgba(245,158,11,0.08)",  "icon": "⚠️",  "order": 1},
    "HIGH VOLATILITY":     {"color": "#f97316", "bg": "rgba(249,115,22,0.08)",  "icon": "🔶", "order": 2},
    "CRITICAL VOLATILITY": {"color": "#ef4444", "bg": "rgba(239,68,68,0.08)",   "icon": "🚨", "order": 3},
}

INTERPRETATIONS = {
    "STABLE": (
        "This supplier demonstrates consistent, predictable performance across all "
        "volatility dimensions. Low operational risk. Suitable for critical or sole-source components."
    ),
    "MODERATE": (
        "Some variability detected in one or more dimensions. Monitor defect trends and "
        "lead time consistency monthly. Consider light safety stock buffers."
    ),
    "HIGH VOLATILITY": (
        "Significant volatility detected. This supplier introduces meaningful operational "
        "uncertainty. Recommend dual-sourcing strategy and increased safety stock. "
        "Schedule a supplier performance review."
    ),
    "CRITICAL VOLATILITY": (
        "This supplier poses immediate operational risk. Volatility is significantly above "
        "baseline across multiple dimensions. Escalate to procurement leadership and activate "
        "contingency sourcing immediately."
    ),
}


def compute_supplier_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Python mirror of vw_supplier_volatility. Replicates the 5 SQL CTEs so that
    whatever the view returns, this function returns the same thing.
    Input is fact_supply_chain at SKU level.
    """
    df = df.copy()

    # revenue per day of lead time — same as base_metrics CTE
    df["revenue_velocity"] = df["revenue_generated"] / df["supplier_lead_time"].replace(0, np.nan)

    # aggregate to supplier level — mirrors supplier_aggregates CTE
    agg = df.groupby("supplier_name").agg(
        sku_count            = ("sku",               "count"),
        avg_revenue          = ("revenue_generated",  "mean"),
        revenue_volatility   = ("revenue_generated",  "std"),
        avg_defect_rate      = ("defect_rates",        "mean"),
        defect_volatility    = ("defect_rates",        "std"),
        avg_lead_time        = ("supplier_lead_time",  "mean"),
        lead_time_volatility = ("supplier_lead_time",  "std"),
        avg_velocity         = ("revenue_velocity",    "mean"),
        velocity_volatility  = ("revenue_velocity",    "std"),
    ).reset_index()

    vol_cols = ["revenue_volatility", "defect_volatility", "lead_time_volatility", "velocity_volatility"]

    # SQL STDDEV returns NULL for single-row groups, pandas does the same — fill with 0
    agg[vol_cols] = agg[vol_cols].fillna(0)

    # global averages and stds across all suppliers — mirrors global_baselines CTE
    g_mean = agg[vol_cols].mean()
    g_std  = agg[vol_cols].std().replace(0, np.nan)

    # z-score each volatility dimension — mirrors normalized_scores CTE
    agg["z_rev_vol"]  = (agg["revenue_volatility"]   - g_mean["revenue_volatility"])   / g_std["revenue_volatility"]
    agg["z_def_vol"]  = (agg["defect_volatility"]     - g_mean["defect_volatility"])     / g_std["defect_volatility"]
    agg["z_lead_vol"] = (agg["lead_time_volatility"]  - g_mean["lead_time_volatility"])  / g_std["lead_time_volatility"]
    agg["z_vel_vol"]  = (agg["velocity_volatility"]   - g_mean["velocity_volatility"])   / g_std["velocity_volatility"]

    z_cols = ["z_rev_vol", "z_def_vol", "z_lead_vol", "z_vel_vol"]
    agg[z_cols] = agg[z_cols].fillna(0)

    # weighted composite — same 30/30/20/20 weights as the SQL view
    agg["volatility_risk_score"] = (
        agg["z_rev_vol"]  * 0.30 +
        agg["z_def_vol"]  * 0.30 +
        agg["z_lead_vol"] * 0.20 +
        agg["z_vel_vol"]  * 0.20
    )

    # same CASE WHEN thresholds as SQL
    agg["risk_category"] = pd.cut(
        agg["volatility_risk_score"],
        bins=[-np.inf, -0.5, 0.5, 1.5, np.inf],
        labels=["STABLE", "MODERATE", "HIGH VOLATILITY", "CRITICAL VOLATILITY"]
    ).astype(str)

    return agg


def build_sku_feature_matrix(fact_df: pd.DataFrame, supplier_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Joins supplier-level stats back onto each SKU row.
    Same idea as SQL window functions — every SKU gets its supplier's
    volatility context as extra columns.
    """
    join_cols = ["supplier_name"] + SUPPLIER_FEATURES + [TARGET, "risk_category"]
    merged = fact_df.merge(supplier_stats[join_cols], on="supplier_name", how="left")
    return merged[ALL_FEATURES + [TARGET, "supplier_name", "risk_category"]].dropna()


def generate_synthetic_samples(
    real_df: pd.DataFrame,
    n_samples: int = 800,
    noise_factor: float = 0.08,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generates synthetic rows for training. We add calibrated Gaussian noise to real
    observations, then recompute the target from scratch using the same formula —
    so the target always stays consistent with the features (no leakage).
    """
    np.random.seed(random_state)

    vol_feat_weights = {
        "revenue_volatility":   0.30,
        "defect_volatility":    0.30,
        "lead_time_volatility": 0.20,
        "velocity_volatility":  0.20,
    }

    vol_cols  = list(vol_feat_weights.keys())
    g_mean    = real_df[vol_cols].mean()
    g_std     = real_df[vol_cols].std().replace(0, np.nan)
    feat_stds = {f: real_df[f].std() for f in ALL_FEATURES}
    real_array = real_df[ALL_FEATURES].values
    rows = []

    for _ in range(n_samples):
        base      = real_array[np.random.randint(0, len(real_array))].copy()
        feat_dict = dict(zip(ALL_FEATURES, base))

        for feat in ALL_FEATURES:
            std = feat_stds[feat]
            feat_dict[feat] = max(
                0.0,
                feat_dict[feat] + np.random.normal(0, noise_factor * std if std > 0 else 1e-6)
            )

        # recompute target from the noisy volatility inputs
        score = 0.0
        for vol_col, weight in vol_feat_weights.items():
            s = g_std[vol_col]
            z = (feat_dict[vol_col] - g_mean[vol_col]) / s if pd.notna(s) and s > 0 else 0.0
            score += z * weight

        feat_dict[TARGET] = score
        rows.append(feat_dict)

    return pd.DataFrame(rows)


def score_to_category(score: float) -> str:
    if score > 1.5:  return "CRITICAL VOLATILITY"
    if score > 0.5:  return "HIGH VOLATILITY"
    if score > -0.5: return "MODERATE"
    return "STABLE"