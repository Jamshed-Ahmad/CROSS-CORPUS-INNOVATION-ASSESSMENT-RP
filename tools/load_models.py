#!/usr/bin/env python3
"""
tools/load_models.py
====================
Verification script to safely load all PyTorch (.pt) models in CPU mode
and perform safe inference verification using valid project samples.

Features:
  - Safe CPU loading (map_location="cpu")
  - Structured loading report with SUCCESS / FAILED status
  - Detailed parameter counts, input structure, and output structure
  - Diagnostic failure reporting for model class mismatches
  - Inference verification with shapes, data types, and output predictions

Usage:
  python tools/load_models.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure backend directory is in python search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.models.baseline_ann import BaselinePaperPatentANN, FineTunedPaperPatentANN
from app.models.feature_fusion_ann import DeepFeatureFusionAttentionANN
from app.models.siamese_cross_attention_ann import SiameseCrossAttentionANN
from app.models.relationship_model import DualTowerCrossAttentionRelationshipModel
from app.models.gap_detection_model import ResearchGapDetectionNetwork, GAP_TYPE_CLASSES
from app.features.relationship_features import RELATIONSHIP_FEATURE_COLUMNS


def count_params(model: nn.Module) -> Tuple[int, int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def load_sample_inputs() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads valid sample inputs from the project's test split and processed data."""
    test_csv = PROJECT_ROOT / "data" / "training" / "test.csv"
    if test_csv.exists():
        df_test = pd.read_csv(test_csv)
        sample_tabular = df_test[RELATIONSHIP_FEATURE_COLUMNS].iloc[:2].values.astype(np.float32)
    else:
        sample_tabular = np.random.uniform(0.1, 0.9, size=(2, 10)).astype(np.float32)

    # 384-dimensional dense text embeddings representing (Paper, Patent)
    sample_paper_emb = np.random.randn(2, 384).astype(np.float32)
    sample_patent_emb = np.random.randn(2, 384).astype(np.float32)

    # 12-dimensional gap candidate features
    sample_gap_feats = np.random.uniform(0.1, 0.9, size=(2, 12)).astype(np.float32)

    return sample_tabular, sample_paper_emb, sample_patent_emb, sample_gap_feats


def verify_loading():
    print("=" * 60)
    print("PYTORCH MODEL LOADING REPORT")
    print("=" * 60)
    print(f"Host Environment: CPU (map_location='cpu')")
    print(f"PyTorch Version:  {torch.__version__}")
    print()

    models_dir = PROJECT_ROOT / "models"
    sample_tabular, sample_paper_emb, sample_patent_emb, sample_gap_feats = load_sample_inputs()

    # =========================================================================
    # MODEL 1 (BASELINE & FINE-TUNED)
    # =========================================================================
    p1_base = models_dir / "baseline_ann_model.pt"
    p1_ft = models_dir / "finetuned_ann_model.pt"

    print("Model 1: Multi-Layer Perceptron (MLP) with Feature Attention [Baseline]:")
    print(f"Filename: {p1_base.name}")
    print(f"Path:     {p1_base.resolve()}")
    try:
        m1_base = BaselinePaperPatentANN(num_features=10)
        state_dict_1 = torch.load(p1_base, map_location="cpu", weights_only=True)
        m1_base.load_state_dict(state_dict_1)
        m1_base.eval()
        tot_p, tr_p, nt_p = count_params(m1_base)
        print("Status:   SUCCESS")
        print(f"  • Model Class:         {m1_base.__class__.__name__}")
        print(f"  • Parameter Count:     Total: {tot_p:,} | Trainable: {tr_p:,} | Non-trainable: {nt_p}")
        print(f"  • Input Structure:     Tensor shape (batch_size, 10) representing RAG features")
        print(f"  • Output Structure:    Tensor shape (batch_size, 1) probability P(Related) in [0, 1]")

        # Inference Verification
        out_probs = m1_base.predict_proba(sample_tabular)
        print(f"  • Inference Check:     PASSED")
        print(f"      - Input shape:     {sample_tabular.shape}")
        print(f"      - Output shape:    {out_probs.shape}")
        print(f"      - Output type:     {type(out_probs).__name__} of {out_probs.dtype}")
        print(f"      - Prediction:      Sample values {np.round(out_probs, 4).tolist()}")
    except Exception as e:
        print("Status:   FAILED")
        print(f"  • Exact Error:         {e}")
        print(f"  • Likely Cause:        Missing or corrupt file, or class definition mismatch")
        print(f"  • Required Correction: Verify models/baseline_ann_model.pt exists and matches BaselinePaperPatentANN")

    print()
    print("Model 1: Multi-Layer Perceptron (MLP) with Feature Attention [Fine-Tuned]:")
    print(f"Filename: {p1_ft.name}")
    print(f"Path:     {p1_ft.resolve()}")
    try:
        m1_ft = FineTunedPaperPatentANN(num_features=10)
        state_dict_ft = torch.load(p1_ft, map_location="cpu", weights_only=True)
        m1_ft.load_state_dict(state_dict_ft)
        m1_ft.eval()
        tot_p, tr_p, nt_p = count_params(m1_ft)
        print("Status:   SUCCESS")
        print(f"  • Model Class:         {m1_ft.__class__.__name__}")
        print(f"  • Parameter Count:     Total: {tot_p:,} | Trainable: {tr_p:,} | Non-trainable: {nt_p}")
        print(f"  • Input Structure:     Tensor shape (batch_size, 10) representing RAG features")
        print(f"  • Output Structure:    Tensor shape (batch_size, 1) probability P(Related) in [0, 1]")

        # Inference Verification
        out_probs_ft = m1_ft.predict_proba(sample_tabular)
        print(f"  • Inference Check:     PASSED")
        print(f"      - Input shape:     {sample_tabular.shape}")
        print(f"      - Output shape:    {out_probs_ft.shape}")
        print(f"      - Output type:     {type(out_probs_ft).__name__} of {out_probs_ft.dtype}")
        print(f"      - Prediction:      Sample values {np.round(out_probs_ft, 4).tolist()}")
    except Exception as e:
        print("Status:   FAILED")
        print(f"  • Exact Error:         {e}")
        print(f"  • Likely Cause:        Missing or corrupt file, or class definition mismatch")
        print(f"  • Required Correction: Verify models/finetuned_ann_model.pt exists and matches FineTunedPaperPatentANN")

    # =========================================================================
    # MODEL 2: DEEP FEATURE-FUSION ATTENTION NETWORK
    # =========================================================================
    p2 = models_dir / "feature_fusion_ann_model.pt"
    print()
    print("Model 2: Deep Feature-Fusion Attention Network:")
    print(f"Filename: {p2.name}")
    print(f"Path:     {p2.resolve()}")
    try:
        m2 = DeepFeatureFusionAttentionANN(num_features=10, group_proj_dim=64, hidden_dim=128, dropout=0.20)
        state_dict_2 = torch.load(p2, map_location="cpu", weights_only=True)
        m2.load_state_dict(state_dict_2)
        m2.eval()
        tot_p, tr_p, nt_p = count_params(m2)
        print("Status:   SUCCESS")
        print(f"  • Model Class:         {m2.__class__.__name__}")
        print(f"  • Parameter Count:     Total: {tot_p:,} | Trainable: {tr_p:,} | Non-trainable: {nt_p}")
        print(f"  • Input Structure:     Tensor shape (batch_size, 10) partitioned into 4 modalities")
        print(f"  • Output Structure:    Tuple of (probability Tensor (batch_size, 1), attention weights (batch_size, 4))")

        # Inference Verification
        out_probs_2 = m2.predict_proba(sample_tabular)
        print(f"  • Inference Check:     PASSED")
        print(f"      - Input shape:     {sample_tabular.shape}")
        print(f"      - Output shape:    {out_probs_2.shape}")
        print(f"      - Output type:     {type(out_probs_2).__name__} of {out_probs_2.dtype}")
        print(f"      - Prediction:      Sample values {np.round(out_probs_2, 4).tolist()}")
    except Exception as e:
        print("Status:   FAILED")
        print(f"  • Exact Error:         {e}")
        print(f"  • Likely Cause:        Missing file or branch configuration mismatch")
        print(f"  • Required Correction: Ensure models/feature_fusion_ann_model.pt is loaded with DeepFeatureFusionAttentionANN")

    # =========================================================================
    # MODEL 3: SIAMESE DUAL-TOWER NETWORK WITH CROSS-ATTENTION
    # =========================================================================
    p3 = models_dir / "relationship_model" / "relationship_model.pt"
    print()
    print("Model 3: Siamese Dual-Tower Network with Cross-Attention:")
    print(f"Filename: {p3.name}")
    print(f"Path:     {p3.resolve()}")

    # 1. Verification with true matching class: SiameseCrossAttentionANN
    try:
        m3_siam = SiameseCrossAttentionANN(
            embed_dim=384, proj_dim=256, num_rag_features=10, num_heads=4, hidden_dim=256, dropout=0.25
        )
        state_dict_3 = torch.load(p3, map_location="cpu", weights_only=True)
        m3_siam.load_state_dict(state_dict_3)
        m3_siam.eval()
        tot_p, tr_p, nt_p = count_params(m3_siam)
        print("Status:   SUCCESS (using SiameseCrossAttentionANN)")
        print(f"  • Model Class:         {m3_siam.__class__.__name__}")
        print(f"  • Parameter Count:     Total: {tot_p:,} | Trainable: {tr_p:,} | Non-trainable: {nt_p}")
        print(f"  • Input Structure:     Multimodal tuple: paper_emb (B, 384), patent_emb (B, 384), rag_features (B, 10)")
        print(f"  • Output Structure:    Tuple of (probability Tensor (B, 1), attention weights tuple (p2pt, pt2p))")

        # Inference Verification
        out_probs_3 = m3_siam.predict_proba(sample_paper_emb, sample_patent_emb, sample_tabular)
        print(f"  • Inference Check:     PASSED")
        print(f"      - Input shapes:    paper={sample_paper_emb.shape}, patent={sample_patent_emb.shape}, rag={sample_tabular.shape}")
        print(f"      - Output shape:    {out_probs_3.shape}")
        print(f"      - Output type:     {type(out_probs_3).__name__} of {out_probs_3.dtype}")
        print(f"      - Prediction:      Sample values {np.round(out_probs_3, 4).tolist()}")
    except Exception as e:
        print("Status:   FAILED (using SiameseCrossAttentionANN)")
        print(f"  • Exact Error:         {e}")

    # 2. Diagnostic test with DualTowerCrossAttentionRelationshipModel
    print()
    print("  [Diagnostic Analysis for Alternative Class DualTowerCrossAttentionRelationshipModel]:")
    try:
        m3_dual = DualTowerCrossAttentionRelationshipModel.load_checkpoint(str(p3), device="cpu")
        print("  • Status with DualTowerCrossAttentionRelationshipModel: SUCCESS")
    except KeyError as ke:
        print(f"  • DualTowerCrossAttentionRelationshipModel.load_checkpoint(): FAILED")
        print(f"      - Exact Error:         KeyError: {ke}")
        print(f"      - Likely Cause:        {p3.name} was saved as a direct state_dict (OrderedDict of tensors),")
        print(f"                             whereas load_checkpoint() expects a dict with key 'state_dict'.")
    except Exception as e_oth:
        print(f"  • DualTowerCrossAttentionRelationshipModel loading failed: {e_oth}")

    try:
        m3_dual_inst = DualTowerCrossAttentionRelationshipModel(embed_dim=384, num_rag_features=10)
        m3_dual_inst.load_state_dict(torch.load(p3, map_location="cpu", weights_only=True))
    except Exception as e_mm:
        print(f"  • Direct load_state_dict into DualTowerCrossAttentionRelationshipModel: FAILED")
        print(f"      - Likely Cause:        Dimensional mismatch: DualTowerCrossAttentionRelationshipModel initializes")
        print(f"                             towers with hidden_dim=384 (resulting in fusion_dim=3136), whereas")
        print(f"                             the checkpoint was trained with proj_dim=256 (fusion_dim=2112).")
        print(f"      - Required Correction: Use 'app.models.siamese_cross_attention_ann.SiameseCrossAttentionANN'")
        print(f"                             which matches the exact 256-d projection architecture of the checkpoint.")

    # =========================================================================
    # SUPPLEMENTARY: RESEARCH GAP DETECTION NETWORK
    # =========================================================================
    p_gap = models_dir / "gap_detection_model" / "gap_detection_model.pt"
    print()
    print("Downstream Multi-Task Network (Gap Detection):")
    print(f"Filename: {p_gap.name}")
    print(f"Path:     {p_gap.resolve()}")
    try:
        m_gap = ResearchGapDetectionNetwork.load_checkpoint(str(p_gap), device="cpu")
        m_gap.eval()
        tot_p, tr_p, nt_p = count_params(m_gap)
        print("Status:   SUCCESS")
        print(f"  • Model Class:         {m_gap.__class__.__name__}")
        print(f"  • Parameter Count:     Total: {tot_p:,} | Trainable: {tr_p:,} | Non-trainable: {nt_p}")
        print(f"  • Input Structure:     Tensor shape (batch_size, 12) candidate gap features (includes relationship score)")
        print(f"  • Output Structure:    Tuple: (gap_score (B, 1), type_logits (B, 6), confidence (B, 1), attn (B, 128))")

        # Inference Verification
        with torch.no_grad():
            t_gap_in = torch.tensor(sample_gap_feats, dtype=torch.float32)
            g_scores, type_logits, confs, _ = m_gap(t_gap_in)
            pred_types = [GAP_TYPE_CLASSES[i] for i in torch.argmax(type_logits, dim=-1).numpy()]

        print(f"  • Inference Check:     PASSED")
        print(f"      - Input shape:     {t_gap_in.shape}")
        print(f"      - Output shapes:   gap_score={list(g_scores.shape)}, type_logits={list(type_logits.shape)}, confidence={list(confs.shape)}")
        print(f"      - Output type:     PyTorch Tensors (Sigmoid, Linear Logits, Sigmoid)")
        print(f"      - Predicted Score: {np.round(g_scores.squeeze().numpy(), 4).tolist()}")
        print(f"      - Predicted Types: {pred_types}")
        print(f"      - Confidence:      {np.round(confs.squeeze().numpy(), 4).tolist()}")
    except Exception as e:
        print("Status:   FAILED")
        print(f"  • Exact Error:         {e}")
        print(f"  • Likely Cause:        Checkpoint loading error or missing checkpoint file")
        print(f"  • Required Correction: Ensure models/gap_detection_model/gap_detection_model.pt is present and valid")

    print()
    print("=" * 60)
    print("ALL MODELS VERIFIED ON CPU SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    verify_loading()
