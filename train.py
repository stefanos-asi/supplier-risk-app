import json
import warnings
import numpy as np
import pandas as pd
import joblib
from supabase import create_client, ClientOptions

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from supply_utils import (
    compute_supplier_stats,
    build_sku_feature_matrix,
    generate_synthetic_samples,
    ALL_FEATURES,
    TARGET,
    SKU_FEATURES,
    SUPPLIER_FEATURES,
)

warnings.filterwarnings("ignore")

from config import SUPABASE_URL, SUPABASE_KEY

RANDOM_STATE = 42
N_SYNTHETIC  = 800
NOISE_FACTOR = 0.08

np.random.seed(RANDOM_STATE)


# load fact table from Supabase
print("Loading data...")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY, options=ClientOptions(schema="public"))
fact_df  = pd.DataFrame(supabase.table("fact_supply_chain").select("*").execute().data)
print(f"  {len(fact_df)} SKU rows, {fact_df['supplier_name'].nunique()} suppliers")


# compute supplier stats and build the SKU-level feature matrix
print("\nEngineering features...")
supplier_stats = compute_supplier_stats(fact_df)
print(supplier_stats[["supplier_name", "volatility_risk_score", "risk_category"]].to_string(index=False))

real_df = build_sku_feature_matrix(fact_df, supplier_stats)
print(f"\n  Feature matrix: {len(real_df)} rows x {len(ALL_FEATURES)} features")


# real data is held out entirely — never touches training
# synthetic data pads the training set to give models enough to learn from
print(f"\nGenerating {N_SYNTHETIC} synthetic samples...")
synthetic_df = generate_synthetic_samples(real_df, N_SYNTHETIC, NOISE_FACTOR, RANDOM_STATE)

X_real  = real_df[ALL_FEATURES]
y_real  = real_df[TARGET]
X_train = pd.concat([real_df[ALL_FEATURES], synthetic_df[ALL_FEATURES]], ignore_index=True)
y_train = pd.concat([real_df[TARGET],        synthetic_df[TARGET]],        ignore_index=True)
print(f"  Training: {len(X_train)} rows | Hold-out: {len(X_real)} rows (real only)")


# scale features — Ridge and Linear Regression are sensitive to scale
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_real_scaled  = scaler.transform(X_real)


# compare 4 models — we want to pick the simplest one that performs well,
# not just the most powerful one
print("\nComparing models...\n")

candidates = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost":           XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03,
                                      subsample=0.8, colsample_bytree=0.8,
                                      reg_alpha=0.1, reg_lambda=1.0,
                                      random_state=RANDOM_STATE, n_jobs=-1),
}

kf             = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
comparison     = []
trained_models = {}

print(f"{'Model':<22} {'CV R2':>10} {'CV RMSE':>10} {'Real R2':>10} {'Real RMSE':>10} {'Real MAE':>10}")
print("-" * 74)

for name, m in candidates.items():
    cv_r2   = cross_val_score(m, X_train_scaled, y_train, cv=kf, scoring="r2")
    cv_rmse = cross_val_score(m, X_train_scaled, y_train, cv=kf, scoring="neg_root_mean_squared_error")

    m.fit(X_train_scaled, y_train)
    trained_models[name] = m

    y_pred    = m.predict(X_real_scaled)
    real_r2   = r2_score(y_real, y_pred)
    real_rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    real_mae  = mean_absolute_error(y_real, y_pred)

    comparison.append({
        "model":        name,
        "cv_r2_mean":   round(float(cv_r2.mean()),      4),
        "cv_r2_std":    round(float(cv_r2.std()),       4),
        "cv_rmse_mean": round(float((-cv_rmse).mean()), 4),
        "real_r2":      round(float(real_r2),           4),
        "real_rmse":    round(float(real_rmse),         4),
        "real_mae":     round(float(real_mae),          4),
    })

    print(f"{name:<22} {cv_r2.mean():>10.4f} {(-cv_rmse).mean():>10.4f} {real_r2:>10.4f} {real_rmse:>10.4f} {real_mae:>10.4f}")


# pick best by hold-out R2
best_row   = max(comparison, key=lambda x: x["real_r2"])
best_name  = best_row["model"]
best_model = trained_models[best_name]

print(f"\nBest model: {best_name} (Real R2 = {best_row['real_r2']:.4f})")

# if Ridge and XGBoost are within 0.02 R2, go with Ridge — simpler model, easier to explain
ridge_row = next(r for r in comparison if r["model"] == "Ridge Regression")
xgb_row   = next(r for r in comparison if r["model"] == "XGBoost")

if abs(ridge_row["real_r2"] - xgb_row["real_r2"]) <= 0.02:
    best_name  = "Ridge Regression"
    best_model = trained_models["Ridge Regression"]
    best_row   = ridge_row
    print(f"  Ridge and XGBoost within 0.02 R2 — going with Ridge (Occam's razor)")


# feature importance — method depends on the winning model type
print(f"\nFeature importances ({best_name})...")

if best_name in ("Linear Regression", "Ridge Regression"):
    raw      = np.abs(best_model.coef_)
    norm     = raw / raw.sum()
    feat_imp = dict(zip(ALL_FEATURES, norm.tolist()))
    imp_label = "Normalized |coefficient|"
elif best_name == "Random Forest":
    feat_imp  = dict(zip(ALL_FEATURES, best_model.feature_importances_.tolist()))
    imp_label = "Gini importance"
else:
    feat_imp  = dict(zip(ALL_FEATURES, best_model.feature_importances_.tolist()))
    imp_label = "XGBoost gain importance"

for k, v in sorted(feat_imp.items(), key=lambda x: -x[1]):
    print(f"  {k:<30} {v:.5f}")


# feature ranges for the app sliders — based on real data only
feature_ranges = {
    feat: {
        "min":  round(float(real_df[feat].min()),  4),
        "max":  round(float(real_df[feat].max()),  4),
        "mean": round(float(real_df[feat].mean()), 4),
        "std":  round(float(real_df[feat].std()),  4),
    }
    for feat in ALL_FEATURES
}

print("\nSaving artifacts...")

joblib.dump(best_model, "supplier_risk_model.pkl")
joblib.dump(scaler,     "supplier_risk_scaler.pkl")

meta = {
    "best_model_name":     best_name,
    "importance_label":    imp_label,
    "cv_r2_mean":          best_row["cv_r2_mean"],
    "cv_r2_std":           best_row["cv_r2_std"],
    "cv_rmse_mean":        best_row["cv_rmse_mean"],
    "real_r2":             best_row["real_r2"],
    "real_rmse":           best_row["real_rmse"],
    "real_mae":            best_row["real_mae"],
    "all_features":        ALL_FEATURES,
    "sku_features":        SKU_FEATURES,
    "supplier_features":   SUPPLIER_FEATURES,
    "target":              TARGET,
    "feature_importances": {k: round(v, 6) for k, v in feat_imp.items()},
    "feature_ranges":      feature_ranges,
    "training_rows":       int(len(X_train)),
    "real_rows":           int(len(X_real)),
    "synthetic_rows":      N_SYNTHETIC,
    "noise_factor":        NOISE_FACTOR,
    "model_comparison":    comparison,
    "score_range": {
        "min":  round(float(y_train.min()),  4),
        "max":  round(float(y_train.max()),  4),
        "mean": round(float(y_train.mean()), 4),
    },
}

with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"  supplier_risk_model.pkl")
print(f"  supplier_risk_scaler.pkl")
print(f"  model_meta.json")
print(f"\nDone. Winner: {best_name}. Run: streamlit run app.py")