"""
evaluate_baselines.py
=====================
Evaluates the EXISTING baseline models exactly as they currently operate
prior to any dataset expansion or fine-tuning.

Computes:
- Train Accuracy, Validation Accuracy, Test Accuracy
- Test Precision, Recall, F1-score, ROC-AUC, PR-AUC, Confusion Matrix
Saves results to reports/baseline_evaluation_report.json and reports/baseline_results.csv
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.models.baseline_ann import BaselinePaperPatentANN, FineTunedPaperPatentANN
from app.models.feature_fusion_ann import DeepFeatureFusionAttentionANN
from app.models.siamese_cross_attention_ann import SiameseCrossAttentionANN
from app.features.relationship_features import RELATIONSHIP_FEATURE_COLUMNS
from app.retrieval.embeddings import EmbeddingGenerator
from app.logging_config import logger

DATA_TRAINING = PROJECT_ROOT / "data" / "training"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_split(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float(np.mean(y_true))

    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "PR-AUC": round(pr_auc, 4),
        "ConfusionMatrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }


def run_baseline_evaluation():
    logger.info("Evaluating existing baseline models on original splits...")
    train_df = pd.read_csv(DATA_TRAINING / "train.csv")
    val_df = pd.read_csv(DATA_TRAINING / "validation.csv")
    test_df = pd.read_csv(DATA_TRAINING / "test.csv")

    X_train = train_df[RELATIONSHIP_FEATURE_COLUMNS].values
    y_train = train_df["label"].values.astype(int)

    X_val = val_df[RELATIONSHIP_FEATURE_COLUMNS].values
    y_val = val_df["label"].values.astype(int)

    X_test = test_df[RELATIONSHIP_FEATURE_COLUMNS].values
    y_test = test_df["label"].values.astype(int)

    # Load pre-trained embeddings for Siamese model
    encoder = EmbeddingGenerator()
    df_papers = pd.read_csv(DATA_PROCESSED / "papers_clean.csv")
    df_patents = pd.read_csv(DATA_PROCESSED / "patents_clean.csv")

    p_dict = {row["paper_id"]: f"{row['title']}. {str(row.get('abstract', ''))}" for _, row in df_papers.iterrows()}
    pt_dict = {row["patent_id"]: f"{row['title']}. {str(row.get('abstract', ''))} {str(row.get('claims', ''))}" for _, row in df_patents.iterrows()}

    p_all = list(p_dict.keys())
    pt_all = list(pt_dict.keys())

    p_embs = encoder.encode([p_dict[pid] for pid in p_all])
    pt_embs = encoder.encode([pt_dict[ptid] for ptid in pt_all])

    p_map = {pid: p_embs[i] for i, pid in enumerate(p_all)}
    pt_map = {ptid: pt_embs[i] for i, ptid in enumerate(pt_all)}

    p_train_embs = np.array([p_map.get(pid, np.zeros(384)) for pid in train_df["paper_id"]])
    pt_train_embs = np.array([pt_map.get(ptid, np.zeros(384)) for ptid in train_df["patent_id"]])

    p_val_embs = np.array([p_map.get(pid, np.zeros(384)) for pid in val_df["paper_id"]])
    pt_val_embs = np.array([pt_map.get(ptid, np.zeros(384)) for ptid in val_df["patent_id"]])

    p_test_embs = np.array([p_map.get(pid, np.zeros(384)) for pid in test_df["paper_id"]])
    pt_test_embs = np.array([pt_map.get(ptid, np.zeros(384)) for ptid in test_df["patent_id"]])

    # 1. Model 1 Baseline
    m1_base = BaselinePaperPatentANN(num_features=10)
    m1_base_path = MODELS_DIR / "baseline_ann_model.pt"
    if m1_base_path.exists():
        m1_base.load_state_dict(torch.load(m1_base_path, map_location="cpu"))
    m1_base.eval()

    # 2. Model 1 Fine-Tuned (Original)
    m1_ft = FineTunedPaperPatentANN(num_features=10)
    m1_ft_path = MODELS_DIR / "finetuned_ann_model.pt"
    if m1_ft_path.exists():
        m1_ft.load_state_dict(torch.load(m1_ft_path, map_location="cpu"))
    m1_ft.eval()

    # 3. Model 2 Feature Fusion
    m2 = DeepFeatureFusionAttentionANN(num_features=10)
    m2_path = MODELS_DIR / "feature_fusion_ann_model.pt"
    if m2_path.exists():
        m2.load_state_dict(torch.load(m2_path, map_location="cpu"))
    m2.eval()

    # 4. Model 3 Siamese Cross Attention
    m3 = SiameseCrossAttentionANN(embed_dim=384, proj_dim=256, num_rag_features=10)
    m3_path = MODELS_DIR / "relationship_model" / "relationship_model.pt"
    if m3_path.exists():
        m3.load_state_dict(torch.load(m3_path, map_location="cpu"))
    m3.eval()

    models = [
        ("Model 1 (Baseline ANN)", m1_base, False),
        ("Model 1 (Fine-Tuned ANN)", m1_ft, False),
        ("Model 2 (Feature-Fusion ANN)", m2, False),
        ("Model 3 (Siamese Cross-Attention)", m3, True)
    ]

    baseline_rows = []
    full_report = {}

    for name, model, is_multimodal in models:
        if is_multimodal:
            prob_train = model.predict_proba(p_train_embs, pt_train_embs, X_train)
            prob_val = model.predict_proba(p_val_embs, pt_val_embs, X_val)
            prob_test = model.predict_proba(p_test_embs, pt_test_embs, X_test)
        else:
            prob_train = model.predict_proba(X_train)
            prob_val = model.predict_proba(X_val)
            prob_test = model.predict_proba(X_test)

        train_acc = float(accuracy_score(y_train, (prob_train >= 0.5).astype(int)))
        val_acc = float(accuracy_score(y_val, (prob_val >= 0.5).astype(int)))
        test_metrics = evaluate_split(y_test, prob_test)

        baseline_rows.append({
            "Model": name,
            "Train_Acc": round(train_acc, 4),
            "Val_Acc": round(val_acc, 4),
            "Test_Acc": test_metrics["Accuracy"],
            "Precision": test_metrics["Precision"],
            "Recall": test_metrics["Recall"],
            "F1-Score": test_metrics["F1-Score"],
            "ROC-AUC": test_metrics["ROC-AUC"],
            "PR-AUC": test_metrics["PR-AUC"],
            "TN": test_metrics["ConfusionMatrix"]["TN"],
            "FP": test_metrics["ConfusionMatrix"]["FP"],
            "FN": test_metrics["ConfusionMatrix"]["FN"],
            "TP": test_metrics["ConfusionMatrix"]["TP"]
        })

        full_report[name] = {
            "Train_Accuracy": round(train_acc, 4),
            "Validation_Accuracy": round(val_acc, 4),
            **test_metrics
        }

    df_base = pd.DataFrame(baseline_rows)
    df_base.to_csv(REPORTS_DIR / "baseline_results.csv", index=False)
    with open(REPORTS_DIR / "baseline_evaluation_report.json", "w") as f:
        json.dump(full_report, f, indent=2)

    print("\n" + "=" * 80)
    print("BASELINE MODEL EVALUATION RESULTS (BEFORE FINE-TUNING)")
    print("=" * 80)
    print(df_base[["Model", "Train_Acc", "Val_Acc", "Test_Acc", "Precision", "Recall", "F1-Score", "ROC-AUC"]].to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_baseline_evaluation()
