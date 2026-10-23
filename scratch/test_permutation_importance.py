import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

project_root = Path(".")
backend_dir = project_root / "backend"
sys.path.insert(0, str(backend_dir))

from app.models.baseline_ann import FineTunedPaperPatentANN
from app.models.feature_fusion_ann import DeepFeatureFusionAttentionANN
from app.features.relationship_features import RELATIONSHIP_FEATURE_COLUMNS

test_df = pd.read_csv("data/training/test.csv")
scaler = joblib.load("models/relationship_model/feature_scaler.pkl")
X_test_raw = test_df[RELATIONSHIP_FEATURE_COLUMNS].values
y_test = test_df["label"].values.astype(int)

# Load Model 2 (Feature Fusion)
m2 = DeepFeatureFusionAttentionANN(num_features=10)
m2.load_state_dict(torch.load("models/feature_fusion_ann_model.pt", map_location="cpu"))
m2.eval()

X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)
prob_baseline = m2.predict_proba(X_test_scaled)
base_auc = roc_auc_score(y_test, prob_baseline)
base_f1 = f1_score(y_test, (prob_baseline >= 0.5).astype(int))

print(f"Base AUC: {base_auc:.4f}, Base F1: {base_f1:.4f}")

# Permutation Importance: Shuffle each feature in raw space, re-scale, and measure AUC drop
np.random.seed(42)
importance_results = []
n_repeats = 10

for col_idx, col_name in enumerate(RELATIONSHIP_FEATURE_COLUMNS):
    auc_drops = []
    f1_drops = []
    for _ in range(n_repeats):
        X_perm = X_test_raw.copy()
        perm_idx = np.random.permutation(len(X_perm))
        X_perm[:, col_idx] = X_perm[perm_idx, col_idx]
        X_perm_scaled = scaler.transform(X_perm).astype(np.float32)
        prob_perm = m2.predict_proba(X_perm_scaled)
        auc_perm = roc_auc_score(y_test, prob_perm)
        f1_perm = f1_score(y_test, (prob_perm >= 0.5).astype(int))
        auc_drops.append(base_auc - auc_perm)
        f1_drops.append(base_f1 - f1_perm)
    
    mean_drop = np.mean(auc_drops)
    std_drop = np.std(auc_drops)
    importance_results.append({
        "Feature": col_name,
        "Mean_AUC_Drop": round(float(mean_drop), 4),
        "Std_AUC_Drop": round(float(std_drop), 4),
        "Mean_F1_Drop": round(float(np.mean(f1_drops)), 4)
    })

df_imp = pd.DataFrame(importance_results).sort_values(by="Mean_AUC_Drop", ascending=False)
print("\n--- PERMUTATION FEATURE IMPORTANCE (MODEL 2 FEATURE FUSION) ---")
print(df_imp.to_string(index=False))
