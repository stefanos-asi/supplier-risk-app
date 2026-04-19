import pandas as pd
import numpy as np
from supabase import create_client, ClientOptions
from config import SUPABASE_URL, SUPABASE_KEY
from supply_utils import compute_supplier_stats

# how much difference between SQL and Python is acceptable (rounding noise)
TOLERANCE = 0.01

sb = create_client(SUPABASE_URL, SUPABASE_KEY, options=ClientOptions(schema="public"))

fact_df = pd.DataFrame(sb.table("fact_supply_chain").select("*").execute().data)

# view lives in analytics schema — change ClientOptions if needed
sql_df = pd.DataFrame(
    sb.table("vw_supplier_volatility").select(
        "supplier_name, sku_count, avg_revenue, avg_defect_rate, "
        "avg_lead_time, volatility_risk_score, volatility_rank, risk_category"
    ).execute().data
)

print(f"Loaded {len(fact_df)} SKU rows from fact_supply_chain")
print(f"Loaded {len(sql_df)} rows from vw_supplier_volatility")

# supabase returns numeric columns as strings sometimes — cast everything explicitly
sql_df["sku_count"]             = sql_df["sku_count"].astype(int)
sql_df["avg_revenue"]           = sql_df["avg_revenue"].astype(float).round(2)
sql_df["avg_defect_rate"]       = sql_df["avg_defect_rate"].astype(float).round(3)
sql_df["avg_lead_time"]         = sql_df["avg_lead_time"].astype(float).round(2)
sql_df["volatility_risk_score"] = sql_df["volatility_risk_score"].astype(float).round(3)
sql_df["volatility_rank"]       = sql_df["volatility_rank"].astype(int)
sql_df["risk_category"]         = sql_df["risk_category"].astype(str).str.strip()

# compute Python scores and round to same precision as SQL view
py_df = compute_supplier_stats(fact_df)

py_df["volatility_rank"] = py_df["volatility_risk_score"].rank(ascending=False, method="min").astype(int)
py_df["avg_revenue"]           = py_df["avg_revenue"].round(2)
py_df["avg_defect_rate"]       = py_df["avg_defect_rate"].round(3)
py_df["avg_lead_time"]         = py_df["avg_lead_time"].round(2)
py_df["volatility_risk_score"] = py_df["volatility_risk_score"].round(3)

py_df = py_df[[
    "supplier_name", "sku_count", "avg_revenue", "avg_defect_rate",
    "avg_lead_time", "volatility_risk_score", "volatility_rank", "risk_category"
]]

merged = sql_df.merge(py_df, on="supplier_name", how="outer", suffixes=("_sql", "_py"))

FLOAT_COLS  = ["avg_revenue", "avg_defect_rate", "avg_lead_time", "volatility_risk_score"]
INT_COLS    = ["sku_count", "volatility_rank"]
STRING_COLS = ["risk_category"]

print("\nParity Check — SQL View vs Python compute_supplier_stats()")
print("=" * 80)

all_passed = True

for _, row in merged.iterrows():
    sup            = row["supplier_name"]
    supplier_passed = True
    failures       = []

    for col in FLOAT_COLS:
        sql_val = float(row[f"{col}_sql"])
        py_val  = float(row[f"{col}_py"])
        delta   = abs(sql_val - py_val)
        if delta > TOLERANCE:
            failures.append(f"  ❌ {col:<28} SQL={sql_val:.4f}  PY={py_val:.4f}  delta={delta:.4f}")
            supplier_passed = False

    for col in INT_COLS:
        sql_val = int(row[f"{col}_sql"])
        py_val  = int(row[f"{col}_py"])
        if sql_val != py_val:
            failures.append(f"  ❌ {col:<28} SQL={sql_val}  PY={py_val}")
            supplier_passed = False

    for col in STRING_COLS:
        sql_val = str(row[f"{col}_sql"]).strip()
        py_val  = str(row[f"{col}_py"]).strip()
        if sql_val != py_val:
            failures.append(f"  ❌ {col:<28} SQL='{sql_val}'  PY='{py_val}'")
            supplier_passed = False

    status = "PASS" if supplier_passed else "FAIL"
    print(f"\n{'✅' if supplier_passed else '❌'} {status}  {sup}")
    for f in failures:
        print(f)
        all_passed = False

# side by side table so you can eyeball all values at once
print("\n" + "=" * 80)
print("\nSide-by-side comparison:\n")

for col in FLOAT_COLS + INT_COLS + STRING_COLS:
    print(f"  {'Supplier':<14}{'SQL ' + col:<24}{'PY ' + col:<24}{'delta / match'}")
    print("  " + "-" * 65)
    for _, row in merged.iterrows():
        sup     = row["supplier_name"]
        sql_val = row[f"{col}_sql"]
        py_val  = row[f"{col}_py"]
        if col in FLOAT_COLS:
            delta = abs(float(sql_val) - float(py_val))
            flag  = "⚠️ " if delta > TOLERANCE else "✅ "
            print(f"  {sup:<14}{str(sql_val):<24}{str(py_val):<24}{flag}{delta:.4f}")
        else:
            match = str(sql_val).strip() == str(py_val).strip()
            print(f"  {sup:<14}{str(sql_val):<24}{str(py_val):<24}{'✅' if match else '❌'}")
    print()

print("=" * 80)
if all_passed:
    print("✅ ALL COLUMNS MATCH — Python perfectly mirrors your SQL view.")
else:
    print("❌ MISMATCHES FOUND — check the output above.")
    print("\n  Common causes:")
    print("  - SQL ROUND() on intermediate values vs Python rounding at the end")
    print("  - NULL handling: SQL STDDEV returns NULL for single-SKU suppliers, Python fills 0")
    print("  - View reads from a different table than fact_supply_chain")