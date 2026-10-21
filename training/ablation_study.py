"""
ablation_study.py
=================
Comprehensive Ablation Study on the Best-Performing ANN Architecture (Siamese Cross-Attention).
Systematically removes components to quantify their empirical contribution to test performance.

Ablation Configurations:
  1. Full Model (Proposed Architecture)
  2. Without Semantic Features (Zero out F1, F7, F8, F9)
  3. Without TF-IDF Similarity (Zero out F2)
  4. Without BM25 Score (Zero out F3)
  5. Without Metadata Features (Zero out F10: publication_year_diff)
  6. Without Embedding Interaction Terms (Omit |z_p - z_pt| and z_p * z_pt)
  7. Without Cross-Attention (Bypass Multi-Head Cross-Attention)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.models.siamese_cross_attention_ann import SiameseCrossAttentionANN
from app.features.relationship_features import RELATIONSHIP_FEATURE_COLUMNS
from app.retrieval.embeddings import EmbeddingGenerator
from app.logging_config import logger

DATA_TRAINING = PROJECT_ROOT / "data" / "training"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def run_ablation_study():
    logger.info("=" * 80)
    logger.info("RUNNING COMPREHENSIVE ABLATION STUDY ON SIAMESE CROSS-ATTENTION ANN")
    logger.info("=" * 80)

    test_df = pd.read_csv(DATA_TRAINING / "test.csv")
    scaler = joblib.load(MODELS_DIR / "relationship_model" / "feature_scaler.pkl")

    X_test_raw = test_df[RELATIONSHIP_FEATURE_COLUMNS].values
    X_test = scaler.transform(X_test_raw).astype(np.float32)
    y_test = test_df["label"].values.astype(int)

    encoder = EmbeddingGenerator()
    df_papers = pd.read_csv(DATA_PROCESSED / "papers_clean.csv")
    df_patents = pd.read_csv(DATA_PROCESSED / "patents_clean.csv")

    paper_emb_dict = {}
    for _, row in df_papers.iterrows():
        text = f"{row['title']}. {str(row.get('abstract', ''))}"
        paper_emb_dict[row["paper_id"]] = encoder.encode([text])[0]

    patent_emb_dict = {}
    for _, row in df_patents.iterrows():
        text = f"{row['title']}. {str(row.get('abstract', ''))} {str(row.get('claims', ''))}"
        patent_emb_dict[row["patent_id"]] = encoder.encode([text])[0]

    p_embs = np.array([paper_emb_dict.get(r["paper_id"], np.zeros(384)) for _, r in test_df.iterrows()], dtype=np.float32)
    pt_embs = np.array([patent_emb_dict.get(r["patent_id"], np.zeros(384)) for _, r in test_df.iterrows()], dtype=np.float32)

    # Load full trained model
    full_model = SiameseCrossAttentionANN(embed_dim=384, proj_dim=256, num_rag_features=10)
    full_model.load_state_dict(torch.load(MODELS_DIR / "relationship_model" / "relationship_model.pt", map_location="cpu"))
    full_model.eval()

    # Define feature index masks
    # [semantic_sim:0, tfidf_sim:1, bm25_score:2, jaccard_sim:3, word_overlap:4, title_sim:5, abstract_sim:6, emb_diff_norm:7, emb_product_mean:8, publication_year_diff:9]
    ablations = [
        ("Full Model (Proposed)", X_test, False, False),
        ("Without Semantic Features", X_test.copy(), False, False),
        ("Without TF-IDF", X_test.copy(), False, False),
        ("Without BM25", X_test.copy(), False, False),
        ("Without Metadata (Year Diff)", X_test.copy(), False, False),
        ("Without Embedding Interaction Terms", X_test.copy(), True, False),
        ("Without Cross-Attention", X_test.copy(), False, True),
    ]

    # Modify feature matrices for each ablation
    # Without Semantic Features: zero indices 0, 6, 7, 8
    ablations[1][1][:, [0, 6, 7, 8]] = 0.0
    # Without TF-IDF: zero index 1
    ablations[2][1][:, 1] = 0.0
    # Without BM25: zero index 2
    ablations[3][1][:, 2] = 0.0
    # Without Metadata: zero index 9
    ablations[4][1][:, 9] = 0.0

    ablation_results = []

    for name, feat_mat, no_interact, no_cross_attn in ablations:
        with torch.no_grad():
            t_p = torch.tensor(p_embs, dtype=torch.float32)
            t_pt = torch.tensor(pt_embs, dtype=torch.float32)
            t_rag = torch.tensor(feat_mat, dtype=torch.float32)

            z_p = full_model.paper_tower(t_p)
            z_pt = full_model.patent_tower(t_pt)

            if no_interact:
                diff_emb = torch.zeros_like(z_p)
                prod_emb = torch.zeros_like(z_p)
            else:
                diff_emb = torch.abs(z_p - z_pt)
                prod_emb = z_p * z_pt

            direct_fusion = torch.cat([z_p, z_pt, diff_emb, prod_emb], dim=-1)

            if no_cross_attn:
                cross_repr = torch.zeros(t_p.shape[0], full_model.cross_attention.output_dim)
            else:
                cross_repr, _ = full_model.cross_attention(z_p.unsqueeze(1), z_pt.unsqueeze(1))

            rag_repr = full_model.rag_projection(t_rag)
            fused = torch.cat([direct_fusion, cross_repr, rag_repr], dim=-1)
            y_prob = full_model.classifier(fused).numpy().flatten()

        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        row = {
            "Configuration": name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4),
            "PR-AUC": round(float(pr_auc), 4),
            "Delta_F1_vs_Full": round(float(f1 - ablation_results[0]["F1-Score"]), 4) if ablation_results else 0.0
        }
        ablation_results.append(row)
        logger.info(f"{name:38s} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Delta_F1: {row['Delta_F1_vs_Full']}")

    df_abl = pd.DataFrame(ablation_results)
    df_abl.to_csv(REPORTS_DIR / "ablation_study.csv", index=False)
    logger.info(f"\nSaved ablation study results to {REPORTS_DIR / 'ablation_study.csv'}")
    return df_abl


if __name__ == "__main__":
    run_ablation_study()
