"""
train_models_comparison.py
==========================
Comprehensive training and benchmarking pipeline for:
  1. Classical ML Baselines: Logistic Regression, SVM (RBF), Random Forest, Gradient Boosting
  2. Model 1 Baseline: Existing ANN (Original Architecture)
  3. Model 1 Fine-Tuned: Existing ANN (LayerNorm, LeakyReLU, Residual Attention, AdamW)
  4. Model 2 Proposed: Deep Feature-Fusion ANN with Attention
  5. Model 3 Proposed: Siamese / Dual-Tower ANN with Cross-Attention

Evaluates all models on identical splits, handles class weighting if imbalanced,
exports full metrics, confusion matrices, ROC/PR data, and generates a direct
Old Dataset vs. Expanded Dataset comparative report.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MultimodalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, X_tabular: np.ndarray, paper_embs: Dict[str, np.ndarray], patent_embs: Dict[str, np.ndarray]):
        self.df = df.reset_index(drop=True)
        self.X_tab = torch.tensor(X_tabular, dtype=torch.float32)
        self.paper_embs = paper_embs
        self.patent_embs = patent_embs
        self.y = torch.tensor(self.df["label"].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p_emb = torch.tensor(self.paper_embs.get(row["paper_id"], np.zeros(384)), dtype=torch.float32)
        pt_emb = torch.tensor(self.patent_embs.get(row["patent_id"], np.zeros(384)), dtype=torch.float32)
        rag_feat = self.X_tab[idx]
        return p_emb, pt_emb, rag_feat, self.y[idx]


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_tabular_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    epochs: int = 40,
    patience: int = 12,
    pos_weight: torch.Tensor = None
) -> Tuple[nn.Module, Dict[str, List[float]], float]:
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if hasattr(model, "forward_logits") else nn.BCELoss(weight=pos_weight)
    best_loss = float("inf")
    best_weights = None
    patience_counter = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_corr, tr_tot = 0.0, 0, 0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            prob = model(X_b)
            loss = criterion(prob, y_b)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * len(y_b)
            preds = (prob >= 0.5).float()
            tr_corr += (preds == y_b).sum().item()
            tr_tot += len(y_b)

        ep_tr_loss = tr_loss / tr_tot
        ep_tr_acc = tr_corr / tr_tot

        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                prob = model(X_b)
                loss = criterion(prob, y_b)
                v_loss += loss.item() * len(y_b)
                preds = (prob >= 0.5).float()
                v_corr += (preds == y_b).sum().item()
                v_tot += len(y_b)

        ep_v_loss = v_loss / v_tot
        ep_v_acc = v_corr / v_tot

        history["train_loss"].append(round(ep_tr_loss, 4))
        history["train_acc"].append(round(ep_tr_acc, 4))
        history["val_loss"].append(round(ep_v_loss, 4))
        history["val_acc"].append(round(ep_v_acc, 4))

        if scheduler:
            scheduler.step(ep_v_loss)

        if ep_v_loss < best_loss:
            best_loss = ep_v_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    train_time = time.time() - start_time
    if best_weights:
        model.load_state_dict(best_weights)
    return model, history, train_time


def run_full_training_and_comparison():
    logger.info("=" * 80)
    logger.info("BENCHMARKING 8 MODELS ON IDENTICAL DATASET")
    logger.info("=" * 80)

    train_df = pd.read_csv(DATA_TRAINING / "train.csv")
    val_df = pd.read_csv(DATA_TRAINING / "validation.csv")
    test_df = pd.read_csv(DATA_TRAINING / "test.csv")

    logger.info(f"Loaded splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Features
    X_train = train_df[RELATIONSHIP_FEATURE_COLUMNS].values
    y_train = train_df["label"].values.astype(int)

    X_val = val_df[RELATIONSHIP_FEATURE_COLUMNS].values
    y_val = val_df["label"].values.astype(int)

    X_test = test_df[RELATIONSHIP_FEATURE_COLUMNS].values
    y_test = test_df["label"].values.astype(int)

    # Class balance weights
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([float(n_neg) / float(n_pos)]) if n_pos > 0 else None
    logger.info(f"Training Class distribution: Pos={n_pos}, Neg={n_neg} (pos_weight={float(pos_weight.item()):.3f} if applicable)")

    # DataLoaders
    train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=32, shuffle=False)
    test_loader = DataLoader(TabularDataset(X_test, y_test), batch_size=32, shuffle=False)

    # Compute dense embeddings for Siamese Model 3
    logger.info("Loading / Computing dense embeddings for multimodal dual-tower model...")
    encoder = EmbeddingGenerator()
    df_papers_clean = pd.read_csv(DATA_PROCESSED / "papers_clean.csv")
    df_patents_clean = pd.read_csv(DATA_PROCESSED / "patents_clean.csv")

    paper_texts = {row["paper_id"]: f"{row['title']}. {str(row.get('abstract', ''))}" for _, row in df_papers_clean.iterrows()}
    patent_texts = {row["patent_id"]: f"{row['title']}. {str(row.get('abstract', ''))} {str(row.get('claims', ''))}" for _, row in df_patents_clean.iterrows()}

    all_paper_ids = list(paper_texts.keys())
    all_patent_ids = list(patent_texts.keys())

    p_raw_embs = encoder.encode([paper_texts[pid] for pid in all_paper_ids])
    pt_raw_embs = encoder.encode([patent_texts[ptid] for ptd in all_patent_ids for ptid in [ptd]])

    paper_emb_dict = {pid: p_raw_embs[i] for i, pid in enumerate(all_paper_ids)}
    patent_emb_dict = {ptid: pt_raw_embs[i] for i, ptid in enumerate(all_patent_ids)}

    p_test_embs = np.array([paper_emb_dict.get(pid, np.zeros(384)) for pid in test_df["paper_id"]])
    pt_test_embs = np.array([patent_emb_dict.get(ptid, np.zeros(384)) for ptid in test_df["patent_id"]])

    all_results = []
    all_histories = {}
    all_predictions = {}
    all_cms = {}

    # -------------------------------------------------------------------------
    # PART 1: TRADITIONAL MACHINE LEARNING BASELINES
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 1: EVALUATING TRADITIONAL MACHINE LEARNING BASELINES")
    logger.info("=" * 80)

    ml_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Support Vector Machine (RBF)": SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
        "Random Forest (100 trees)": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight="balanced"),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    }

    for name, clf in ml_models.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - t0

        t0 = time.time()
        y_prob = clf.predict_proba(X_test)[:, 1]
        inf_latency_ms = ((time.time() - t0) / len(X_test)) * 1000.0

        metrics = evaluate_predictions(y_test, y_prob)
        all_predictions[name] = {"y_true": y_test.tolist(), "y_prob": y_prob.tolist()}
        all_cms[name] = metrics["ConfusionMatrix"]

        all_results.append({
            "Category": "Traditional Baseline",
            "Model": name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1-Score": metrics["F1-Score"],
            "ROC-AUC": metrics["ROC-AUC"],
            "PR-AUC": metrics["PR-AUC"],
            "Parameters": getattr(clf, "n_features_in_", 10),
            "Train_Time_s": round(fit_time, 3),
            "Inf_Latency_ms": round(inf_latency_ms, 3)
        })
        logger.info(f"{name:32s} | Acc: {metrics['Accuracy']:.4f} | F1: {metrics['F1-Score']:.4f} | AUC: {metrics['ROC-AUC']:.4f}")

    # -------------------------------------------------------------------------
    # PART 2: MODEL 1 - BASELINE ANN (ORIGINAL ARCHITECTURE)
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2: TRAINING MODEL 1 (ORIGINAL BASELINE ANN)")
    logger.info("=" * 80)

    baseline_ann = BaselinePaperPatentANN(num_features=10, dropout_funnel=(0.35, 0.30, 0.20))
    opt1 = torch.optim.Adam(baseline_ann.parameters(), lr=1e-3, weight_decay=1e-4)
    baseline_ann, hist1, t1 = train_tabular_model(
        baseline_ann, train_loader, val_loader, opt1, epochs=40, patience=12
    )

    t0 = time.time()
    prob_m1_orig = baseline_ann.predict_proba(X_test)
    inf_lat_m1 = ((time.time() - t0) / len(X_test)) * 1000.0

    m1_orig_metrics = evaluate_predictions(y_test, prob_m1_orig)
    all_predictions["Existing ANN (Before Fine-Tuning)"] = {"y_true": y_test.tolist(), "y_prob": prob_m1_orig.tolist()}
    all_histories["Existing ANN (Before Fine-Tuning)"] = hist1
    all_cms["Existing ANN (Before Fine-Tuning)"] = m1_orig_metrics["ConfusionMatrix"]

    all_results.append({
        "Category": "Model 1 Baseline",
        "Model": "Existing ANN (Before Fine-Tuning)",
        "Accuracy": m1_orig_metrics["Accuracy"],
        "Precision": m1_orig_metrics["Precision"],
        "Recall": m1_orig_metrics["Recall"],
        "F1-Score": m1_orig_metrics["F1-Score"],
        "ROC-AUC": m1_orig_metrics["ROC-AUC"],
        "PR-AUC": m1_orig_metrics["PR-AUC"],
        "Parameters": count_parameters(baseline_ann),
        "Train_Time_s": round(t1, 3),
        "Inf_Latency_ms": round(inf_lat_m1, 3)
    })
    logger.info(f"Existing Baseline ANN | Acc: {m1_orig_metrics['Accuracy']:.4f} | F1: {m1_orig_metrics['F1-Score']:.4f} | AUC: {m1_orig_metrics['ROC-AUC']:.4f}")

    # -------------------------------------------------------------------------
    # PART 3: MODEL 1 - FINE-TUNED ANN (PaperPatentANN)
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 3: TRAINING MODEL 1 (FINE-TUNED ANN)")
    logger.info("=" * 80)

    finetuned_ann = FineTunedPaperPatentANN(num_features=10, dropout_funnel=(0.25, 0.20, 0.15))
    opt_ft = torch.optim.AdamW(finetuned_ann.parameters(), lr=5e-4, weight_decay=1e-4)
    sched_ft = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_ft, mode="min", factor=0.5, patience=4)

    finetuned_ann, hist_ft, t_ft = train_tabular_model(
        finetuned_ann, train_loader, val_loader, opt_ft, scheduler=sched_ft, epochs=50, patience=15
    )

    t0 = time.time()
    prob_m1_ft = finetuned_ann.predict_proba(X_test)
    inf_lat_ft = ((time.time() - t0) / len(X_test)) * 1000.0

    m1_ft_metrics = evaluate_predictions(y_test, prob_m1_ft)
    all_predictions["Existing ANN (Fine-Tuned)"] = {"y_true": y_test.tolist(), "y_prob": prob_m1_ft.tolist()}
    all_histories["Existing ANN (Fine-Tuned)"] = hist_ft
    all_cms["Existing ANN (Fine-Tuned)"] = m1_ft_metrics["ConfusionMatrix"]

    all_results.append({
        "Category": "Model 1 Fine-Tuned",
        "Model": "Existing ANN (Fine-Tuned)",
        "Accuracy": m1_ft_metrics["Accuracy"],
        "Precision": m1_ft_metrics["Precision"],
        "Recall": m1_ft_metrics["Recall"],
        "F1-Score": m1_ft_metrics["F1-Score"],
        "ROC-AUC": m1_ft_metrics["ROC-AUC"],
        "PR-AUC": m1_ft_metrics["PR-AUC"],
        "Parameters": count_parameters(finetuned_ann),
        "Train_Time_s": round(t_ft, 3),
        "Inf_Latency_ms": round(inf_lat_ft, 3)
    })
    logger.info(f"Existing Fine-Tuned ANN | Acc: {m1_ft_metrics['Accuracy']:.4f} | F1: {m1_ft_metrics['F1-Score']:.4f} | AUC: {m1_ft_metrics['ROC-AUC']:.4f}")

    # -------------------------------------------------------------------------
    # PART 4: MODEL 2 - DEEP FEATURE-FUSION ANN WITH ATTENTION
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 4: TRAINING MODEL 2 (FEATURE-FUSION ANN WITH ATTENTION)")
    logger.info("=" * 80)

    feature_fusion_ann = DeepFeatureFusionAttentionANN(num_features=10, group_proj_dim=64, hidden_dim=128, dropout=0.20)
    opt2 = torch.optim.AdamW(feature_fusion_ann.parameters(), lr=5e-4, weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, mode="min", factor=0.5, patience=4)

    feature_fusion_ann, hist2, t2 = train_tabular_model(
        feature_fusion_ann, train_loader, val_loader, opt2, scheduler=sched2, epochs=50, patience=15
    )

    t0 = time.time()
    prob_m2 = feature_fusion_ann.predict_proba(X_test)
    inf_lat_m2 = ((time.time() - t0) / len(X_test)) * 1000.0

    m2_metrics = evaluate_predictions(y_test, prob_m2)
    all_predictions["Feature-Fusion ANN with Attention"] = {"y_true": y_test.tolist(), "y_prob": prob_m2.tolist()}
    all_histories["Feature-Fusion ANN with Attention"] = hist2
    all_cms["Feature-Fusion ANN with Attention"] = m2_metrics["ConfusionMatrix"]

    all_results.append({
        "Category": "Model 2 Proposed",
        "Model": "Feature-Fusion ANN with Attention",
        "Accuracy": m2_metrics["Accuracy"],
        "Precision": m2_metrics["Precision"],
        "Recall": m2_metrics["Recall"],
        "F1-Score": m2_metrics["F1-Score"],
        "ROC-AUC": m2_metrics["ROC-AUC"],
        "PR-AUC": m2_metrics["PR-AUC"],
        "Parameters": count_parameters(feature_fusion_ann),
        "Train_Time_s": round(t2, 3),
        "Inf_Latency_ms": round(inf_lat_m2, 3)
    })
    logger.info(f"Feature-Fusion ANN | Acc: {m2_metrics['Accuracy']:.4f} | F1: {m2_metrics['F1-Score']:.4f} | AUC: {m2_metrics['ROC-AUC']:.4f}")

    # -------------------------------------------------------------------------
    # PART 5: MODEL 3 - SIAMESE DUAL-TOWER ANN WITH CROSS-ATTENTION
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 5: TRAINING MODEL 3 (SIAMESE CROSS-ATTENTION ANN)")
    logger.info("=" * 80)

    siamese_ann = SiameseCrossAttentionANN(embed_dim=384, proj_dim=256, num_rag_features=10, num_heads=4, hidden_dim=256, dropout=0.25)
    opt3 = torch.optim.AdamW(siamese_ann.parameters(), lr=5e-4, weight_decay=1e-4)
    sched3 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt3, mode="min", factor=0.5, patience=4)
    criterion = nn.BCELoss()

    train_multi_loader = DataLoader(
        MultimodalDataset(train_df, X_train, paper_emb_dict, patent_emb_dict),
        batch_size=32, shuffle=True
    )
    val_multi_loader = DataLoader(
        MultimodalDataset(val_df, X_val, paper_emb_dict, patent_emb_dict),
        batch_size=32, shuffle=False
    )

    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    hist3 = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    t0_m3 = time.time()
    for epoch in range(1, 51):
        siamese_ann.train()
        tr_loss, tr_corr, tr_tot = 0.0, 0, 0
        for p_b, pt_b, rag_b, y_b in train_multi_loader:
            opt3.zero_grad()
            prob, _ = siamese_ann(p_b, pt_b, rag_b)
            loss = criterion(prob, y_b)
            loss.backward()
            opt3.step()

            tr_loss += loss.item() * len(y_b)
            preds = (prob >= 0.5).float()
            tr_corr += (preds == y_b).sum().item()
            tr_tot += len(y_b)

        ep_tr_loss = tr_loss / tr_tot
        ep_tr_acc = tr_corr / tr_tot

        siamese_ann.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for p_b, pt_b, rag_b, y_b in val_multi_loader:
                prob, _ = siamese_ann(p_b, pt_b, rag_b)
                loss = criterion(prob, y_b)
                v_loss += loss.item() * len(y_b)
                preds = (prob >= 0.5).float()
                v_corr += (preds == y_b).sum().item()
                v_tot += len(y_b)

        ep_v_loss = v_loss / v_tot
        ep_v_acc = v_corr / v_tot

        hist3["train_loss"].append(round(ep_tr_loss, 4))
        hist3["train_acc"].append(round(ep_tr_acc, 4))
        hist3["val_loss"].append(round(ep_v_loss, 4))
        hist3["val_acc"].append(round(ep_v_acc, 4))

        sched3.step(ep_v_loss)

        if ep_v_loss < best_loss:
            best_loss = ep_v_loss
            best_state = {k: v.cpu().clone() for k, v in siamese_ann.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                logger.info(f"Model 3 Early stopping at epoch {epoch}")
                break

    t3 = time.time() - t0_m3
    if best_state:
        siamese_ann.load_state_dict(best_state)

    t0 = time.time()
    prob_m3 = siamese_ann.predict_proba(p_test_embs, pt_test_embs, X_test)
    inf_lat_m3 = ((time.time() - t0) / len(X_test)) * 1000.0

    m3_metrics = evaluate_predictions(y_test, prob_m3)
    all_predictions["Siamese Cross-Attention ANN"] = {"y_true": y_test.tolist(), "y_prob": prob_m3.tolist()}
    all_histories["Siamese Cross-Attention ANN"] = hist3
    all_cms["Siamese Cross-Attention ANN"] = m3_metrics["ConfusionMatrix"]

    all_results.append({
        "Category": "Model 3 Proposed",
        "Model": "Siamese Cross-Attention ANN",
        "Accuracy": m3_metrics["Accuracy"],
        "Precision": m3_metrics["Precision"],
        "Recall": m3_metrics["Recall"],
        "F1-Score": m3_metrics["F1-Score"],
        "ROC-AUC": m3_metrics["ROC-AUC"],
        "PR-AUC": m3_metrics["PR-AUC"],
        "Parameters": count_parameters(siamese_ann),
        "Train_Time_s": round(t3, 3),
        "Inf_Latency_ms": round(inf_lat_m3, 3)
    })
    logger.info(f"Siamese Cross-Attention ANN | Acc: {m3_metrics['Accuracy']:.4f} | F1: {m3_metrics['F1-Score']:.4f} | AUC: {m3_metrics['ROC-AUC']:.4f}")

    # -------------------------------------------------------------------------
    # PART 6: SAVE ARTIFACTS AND RESULTS
    # -------------------------------------------------------------------------
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    df_results.to_csv(REPORTS_DIR / "model_comparison_expanded.csv", index=False)
    logger.info(f"\nSaved benchmark comparison table to {REPORTS_DIR / 'model_comparison.csv'}")

    with open(REPORTS_DIR / "ann_training_histories.json", "w", encoding="utf-8") as f:
        json.dump(all_histories, f, indent=2)

    with open(REPORTS_DIR / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2)

    with open(REPORTS_DIR / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(all_cms, f, indent=2)

    # Save model checkpoints
    torch.save(baseline_ann.state_dict(), MODELS_DIR / "baseline_ann_model.pt")
    torch.save(finetuned_ann.state_dict(), MODELS_DIR / "finetuned_ann_model.pt")
    torch.save(feature_fusion_ann.state_dict(), MODELS_DIR / "feature_fusion_ann_model.pt")
    torch.save(siamese_ann.state_dict(), MODELS_DIR / "relationship_model" / "relationship_model.pt")

    # -------------------------------------------------------------------------
    # PART 7: OLD DATASET VS EXPANDED DATASET COMPARISON TABLE
    # -------------------------------------------------------------------------
    orig_path = REPORTS_DIR / "model_comparison_original_small.csv"
    if orig_path.exists():
        df_orig = pd.read_csv(orig_path)
        comparison_rows = []

        ann_names = [
            "Existing ANN (Before Fine-Tuning)",
            "Existing ANN (Fine-Tuned)",
            "Feature-Fusion ANN with Attention",
            "Siamese Cross-Attention ANN"
        ]

        for m_name in ann_names:
            orig_match = df_orig[df_orig["Model"] == m_name]
            new_match = df_results[df_results["Model"] == m_name]

            if not orig_match.empty and not new_match.empty:
                om = orig_match.iloc[0]
                nm = new_match.iloc[0]

                comparison_rows.append({
                    "Model": m_name,
                    "Old_Train_Size": 326,
                    "New_Train_Size": len(train_df),
                    "Old_Accuracy": om["Accuracy"],
                    "New_Accuracy": nm["Accuracy"],
                    "Delta_Accuracy": round(nm["Accuracy"] - om["Accuracy"], 4),
                    "Old_F1": om["F1-Score"],
                    "New_F1": nm["F1-Score"],
                    "Delta_F1": round(nm["F1-Score"] - om["F1-Score"], 4),
                    "Old_ROCAUC": om["ROC-AUC"],
                    "New_ROCAUC": nm["ROC-AUC"],
                    "Delta_ROCAUC": round(nm["ROC-AUC"] - om["ROC-AUC"], 4),
                    "Old_PRAUC": om["PR-AUC"],
                    "New_PRAUC": nm["PR-AUC"],
                    "Delta_PRAUC": round(nm["PR-AUC"] - om["PR-AUC"], 4)
                })

        df_dataset_comp = pd.DataFrame(comparison_rows)
        dataset_comp_path = REPORTS_DIR / "dataset_comparison.csv"
        df_dataset_comp.to_csv(dataset_comp_path, index=False)

        print("\n" + "=" * 80)
        print("OLD DATASET (~460 samples) VS EXPANDED DATASET BENCHMARK")
        print("=" * 80)
        print(df_dataset_comp[["Model", "Old_Train_Size", "New_Train_Size", "Old_F1", "New_F1", "Delta_F1", "Old_ROCAUC", "New_ROCAUC", "Delta_ROCAUC"]].to_string(index=False))
        print("=" * 80 + "\n")

    logger.info("Training and evaluation on expanded dataset completed successfully.")
    return df_results


if __name__ == "__main__":
    run_full_training_and_comparison()
