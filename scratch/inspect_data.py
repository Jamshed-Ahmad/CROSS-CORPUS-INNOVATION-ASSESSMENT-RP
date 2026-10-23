import pandas as pd
import json

df_orig = pd.read_csv("reports/model_comparison_original_small.csv")
print("=== ORIGINAL SMALL / BEFORE FINE-TUNING ===")
print(df_orig.to_string())

df_comp = pd.read_csv("reports/model_comparison.csv")
print("\n=== EXPANDED / AFTER FINE-TUNING ===")
print(df_comp.to_string())

df_base = pd.read_csv("reports/baseline_results.csv")
print("\n=== BASELINE RESULTS ===")
print(df_base.to_string())
