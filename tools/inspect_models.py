#!/usr/bin/env python3
"""
tools/inspect_models.py
=======================
Automated inspection utility for all existing PyTorch (.pt) models in the
Evidence-Grounded Cross-Corpus Innovation Assessment system.

Inspects:
  - Exact file paths, sizes, serialization object types, and checkpoint structures.
  - Python class definitions, layer-by-layer topologies, parameter counts,
    and activation functions.
  - Verification of Python architecture definitions vs. serialized weights in .pt files.
  - Distinguishes between pretrained components and custom trainable deep-learning models.
  - Details the workflow from paper-patent relationship scores to downstream research gap detection.

Usage:
  python tools/inspect_models.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn

# Ensure backend directory is in python search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Import existing model classes and feature definitions from project
try:
    from app.models.baseline_ann import BaselinePaperPatentANN, FineTunedPaperPatentANN
    from app.models.feature_fusion_ann import DeepFeatureFusionAttentionANN
    from app.models.siamese_cross_attention_ann import SiameseCrossAttentionANN
    from app.models.relationship_model import DualTowerCrossAttentionRelationshipModel
    from app.models.gap_detection_model import ResearchGapDetectionNetwork, GAP_TYPE_CLASSES
    from app.features.relationship_features import RELATIONSHIP_FEATURE_COLUMNS
    from app.features.gap_features import GAP_FEATURE_COLUMNS
except ImportError as e:
    print(f"Error importing project models: {e}")
    sys.path.insert(0, str(PROJECT_ROOT))


def get_parameter_counts(model: nn.Module) -> Tuple[int, int, int]:
    """Calculates total, trainable, and non-trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def inspect_pt_file(path: Path) -> Dict[str, Any]:
    """Inspects raw serialization details of a .pt file on CPU."""
    if not path.exists():
        return {"exists": False, "error": f"File not found: {path}"}

    size_bytes = path.stat().st_size
    info = {
        "exists": True,
        "path": str(path.resolve()),
        "size_bytes": size_bytes,
        "size_kb": round(size_bytes / 1024, 2),
        "loaded_weights_only": False,
        "obj_type": None,
        "is_state_dict": False,
        "is_checkpoint_dict": False,
        "tensor_keys": [],
        "tensor_shapes": {},
        "metadata": None
    }

    try:
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
            info["loaded_weights_only"] = True
        except Exception:
            data = torch.load(path, map_location="cpu", weights_only=False)
            info["loaded_weights_only"] = False

        info["obj_type"] = type(data).__name__

        if isinstance(data, dict):
            if "state_dict" in data:
                info["is_checkpoint_dict"] = True
                info["checkpoint_keys"] = list(data.keys())
                info["metadata"] = data.get("metadata", {})
                sd = data["state_dict"]
                info["tensor_keys"] = list(sd.keys())
                info["tensor_shapes"] = {k: list(v.shape) for k, v in sd.items()}
            else:
                is_sd = all(isinstance(v, torch.Tensor) for v in data.values())
                if is_sd:
                    info["is_state_dict"] = True
                    info["tensor_keys"] = list(data.keys())
                    info["tensor_shapes"] = {k: list(v.shape) for k, v in data.items()}
                else:
                    info["dict_keys"] = list(data.keys())
        elif isinstance(data, nn.Module):
            info["is_complete_model"] = True
            info["model_class"] = data.__class__.__name__

    except Exception as e:
        info["error"] = str(e)

    return info


def verify_architecture_match(
    pt_info: Dict[str, Any],
    model: nn.Module
) -> Tuple[str, List[str]]:
    """Compares serialized tensor names and shapes against model state_dict."""
    if not pt_info.get("exists", False):
        return "NO", [f"File does not exist: {pt_info.get('path')}"]

    pt_shapes = pt_info.get("tensor_shapes", {})
    if not pt_shapes:
        return "NO", ["No tensors found in .pt file."]

    model_sd = model.state_dict()
    mismatches = []

    for name, param in model_sd.items():
        if name not in pt_shapes:
            mismatches.append(f"Missing in .pt: {name} (expected {list(param.shape)})")
        elif pt_shapes[name] != list(param.shape):
            mismatches.append(f"Shape mismatch for {name}: .pt has {pt_shapes[name]} vs Model {list(param.shape)}")

    for name in pt_shapes:
        if name not in model_sd:
            mismatches.append(f"Unexpected in .pt: {name} (shape {pt_shapes[name]})")

    if not mismatches:
        return "YES", ["All tensor names and dimensional shapes match exactly."]
    elif len(mismatches) < len(model_sd):
        return "PARTIAL", mismatches
    else:
        return "NO", mismatches


def display_model_inspection(
    header_name: str,
    pt_rel_path: str,
    model_instance: nn.Module,
    model_class_name: str,
    purpose: str,
    source_file: str,
    train_file: str,
    inference_file: str,
    input_dim: str,
    input_feature_names: List[str],
    hidden_layers_desc: str,
    hidden_layer_sizes: str,
    activation_funcs: str,
    dropout_desc: str,
    batch_norm_desc: str,
    attention_desc: str,
    encoder_desc: str,
    output_layer_desc: str,
    output_dim: str,
    output_classes: str,
    checkpoint_type_desc: str,
    additional_notes: Optional[str] = None
):
    """Prints a standard formatted inspection report for one model."""
    pt_path = PROJECT_ROOT / pt_rel_path
    pt_info = inspect_pt_file(pt_path)
    total_p, train_p, non_train_p = get_parameter_counts(model_instance)
    match_status, match_details = verify_architecture_match(pt_info, model_instance)

    print("-" * 70)
    print(header_name)
    print("-" * 70)
    print(f"File:                      {pt_path.name}")
    print(f"Exact Path:                {pt_path.resolve()}")
    if pt_info.get("exists", False):
        print(f"File Size:                 {pt_info['size_bytes']:,} bytes ({pt_info['size_kb']} KB)")
        print(f"PyTorch Object Type:       {pt_info.get('obj_type')}")
        print(f"Checkpoint Type:           {checkpoint_type_desc}")
    else:
        print(f"File Size:                 NOT FOUND")
        print(f"PyTorch Object Type:       UNKNOWN")
        print(f"Checkpoint Type:           {checkpoint_type_desc}")

    print(f"Model Class:               {model_class_name}")
    print(f"Model Architecture:        {model_instance.__class__.__name__}")
    print(f"Layer-by-Layer Structure:")
    for name, child in model_instance.named_children():
        c_params = sum(p.numel() for p in child.parameters())
        print(f"  • {name:24s} -> {child.__class__.__name__:25s} ({c_params:,} parameters)")

    print(f"Input Dimension:           {input_dim}")
    print(f"Input Feature Names:       {', '.join(input_feature_names) if input_feature_names else 'N/A'}")
    print(f"Hidden Layers:             {hidden_layers_desc}")
    print(f"Hidden Layer Sizes:        {hidden_layer_sizes}")
    print(f"Activation Functions:      {activation_funcs}")
    print(f"Dropout:                   {dropout_desc}")
    print(f"Batch Normalization:       {batch_norm_desc}")
    print(f"Attention Components:      {attention_desc}")
    print(f"Encoder Components:        {encoder_desc}")
    print(f"Output Layer:              {output_layer_desc}")
    print(f"Output Dimension:          {output_dim}")
    print(f"Output Classes:            {output_classes}")
    print(f"Number of Parameters:      {total_p:,}")
    print(f"Trainable Parameters:      {train_p:,}")
    print(f"Non-Trainable Parameters:  {non_train_p:,}")
    print(f"Purpose:                   {purpose}")
    print(f"Source Python File:        {source_file}")
    print(f"Training Python File:      {train_file}")
    print(f"Inference Python File:     {inference_file}")
    print(f"Architecture Match:        {match_status}")
    if match_status == "YES":
        print(f"Match Details:             Exact parameter alignment across all {len(pt_info.get('tensor_shapes', {}))} tensors.")
    else:
        print(f"Match Details:")
        for det in match_details[:5]:
            print(f"  ! {det}")
        if len(match_details) > 5:
            print(f"  ! ... ({len(match_details) - 5} more mismatches)")

    if additional_notes:
        print(f"Notes:                     {additional_notes}")
    print()


def run_full_inspection():
    print("=" * 80)
    print("RESEARCH PROJECT PYTORCH MODEL COMPREHENSIVE INSPECTION")
    print("=" * 80)
    print(f"Project Root: {PROJECT_ROOT.resolve()}")
    print(f"PyTorch Version: {torch.__version__} (Device: CPU)")
    print()

    # -------------------------------------------------------------------------
    # MODEL 1: BASELINE ANN (Original)
    # -------------------------------------------------------------------------
    m1_base = BaselinePaperPatentANN(num_features=10, dropout_funnel=(0.35, 0.30, 0.20))
    display_model_inspection(
        header_name="MODEL 1 (ORIGINAL BASELINE): Existing Baseline ANN",
        pt_rel_path="models/baseline_ann_model.pt",
        model_instance=m1_base,
        model_class_name="BaselinePaperPatentANN",
        purpose="Paper-Patent Relationship Score P(Related | Paper, Patent) - Original baseline configuration",
        source_file="backend/app/models/baseline_ann.py",
        train_file="training/train_models_comparison.py (Part 2)",
        inference_file="training/evaluate_retrieval_and_gaps.py",
        input_dim="10 (Heterogeneous RAG features: 1D tensor)",
        input_feature_names=RELATIONSHIP_FEATURE_COLUMNS,
        hidden_layers_desc="Dense(10->128) -> Dense(128->64) -> FeatureAttentionLayer(64) -> Dense(64->32)",
        hidden_layer_sizes="128, 64, 32",
        activation_funcs="ReLU for hidden dense transformations; Softmax in FeatureAttention; Sigmoid for output probability",
        dropout_desc="Dropout funnel: 0.35 (Layer 1) -> 0.30 (Layer 2) -> 0.20 (Layer 4)",
        batch_norm_desc="BatchNorm1d(10) on raw input features for scale-invariant feature normalization",
        attention_desc="FeatureAttentionLayer(64): Linear(64, 64) + Softmax dynamic feature saliency weighting",
        encoder_desc="Direct dense multi-layer perceptron projection (no dual-tower text token encoders)",
        output_layer_desc="Linear(32, 1) + Sigmoid",
        output_dim="1 (Continuous probability score in [0.0, 1.0])",
        output_classes="Binary: Class 0 (Unrelated candidate pair), Class 1 (Related cross-corpus pair)",
        checkpoint_type_desc="state_dict (OrderedDict of 15 parameter tensors)",
        additional_notes="Original benchmark ANN architecture before hyperparameter tuning."
    )

    # -------------------------------------------------------------------------
    # MODEL 1: FINE-TUNED ANN
    # -------------------------------------------------------------------------
    m1_ft = FineTunedPaperPatentANN(num_features=10, dropout_funnel=(0.25, 0.20, 0.15))
    display_model_inspection(
        header_name="MODEL 1 (FINE-TUNED): Existing Fine-Tuned ANN",
        pt_rel_path="models/finetuned_ann_model.pt",
        model_instance=m1_ft,
        model_class_name="FineTunedPaperPatentANN",
        purpose="Paper-Patent Relationship Score P(Related | Paper, Patent) - Fine-tuned with LayerNorm & residual connection",
        source_file="backend/app/models/baseline_ann.py",
        train_file="training/train_models_comparison.py (Part 3)",
        inference_file="training/evaluate_retrieval_and_gaps.py",
        input_dim="10 (Heterogeneous RAG features: 1D tensor)",
        input_feature_names=RELATIONSHIP_FEATURE_COLUMNS,
        hidden_layers_desc="Dense(10->128)+LN -> Dense(128->64)+LN -> FeatureAttention(64) with residual skip -> Dense(64->32)+LN",
        hidden_layer_sizes="128, 64, 32",
        activation_funcs="LeakyReLU(0.1) for smooth gradient propagation; Softmax in Attention; Sigmoid for output",
        dropout_desc="Calibrated dropout funnel: 0.25 (Layer 1) -> 0.20 (Layer 2) -> 0.15 (Layer 4)",
        batch_norm_desc="BatchNorm1d(10) on raw input features + LayerNorm(128), LayerNorm(64), LayerNorm(32) across hidden layers",
        attention_desc="FeatureAttentionLayer(64) with residual skip connection (h2 + attention(h2))",
        encoder_desc="Dense MLP projection with LayerNorm and residual attention",
        output_layer_desc="Linear(32, 1) + Sigmoid",
        output_dim="1 (Continuous probability score in [0.0, 1.0])",
        output_classes="Binary: Class 0 (Unrelated candidate pair), Class 1 (Related cross-corpus pair)",
        checkpoint_type_desc="state_dict (OrderedDict of 21 parameter tensors)",
        additional_notes="Incorporates AdamW optimizer, calibrated dropout, LayerNorm, and residual attention skip."
    )

    # -------------------------------------------------------------------------
    # MODEL 2: FEATURE-FUSION ANN WITH ATTENTION
    # -------------------------------------------------------------------------
    m2_ff = DeepFeatureFusionAttentionANN(num_features=10, group_proj_dim=64, hidden_dim=128, dropout=0.20)
    display_model_inspection(
        header_name="MODEL 2: Deep Feature-Fusion ANN with Attention",
        pt_rel_path="models/feature_fusion_ann_model.pt",
        model_instance=m2_ff,
        model_class_name="DeepFeatureFusionAttentionANN",
        purpose="Paper-Patent Relationship Score P(Related | Paper, Patent) using multimodal feature partition & group attention",
        source_file="backend/app/models/feature_fusion_ann.py",
        train_file="training/train_models_comparison.py (Part 4)",
        inference_file="training/evaluate_retrieval_and_gaps.py",
        input_dim="10 features partitioned into 4 modalities: Semantic(4), Lexical(4), Retrieval(1), Metadata(1)",
        input_feature_names=RELATIONSHIP_FEATURE_COLUMNS,
        hidden_layers_desc="4 parallel branch projections (each to 64-d) -> GroupSaliencyAttention (256-d fused) -> Dense(256->128) -> Dense(128->64) -> Dense(64->1)",
        hidden_layer_sizes="Branch: 64 each (Semantic, Lexical, Retrieval, Metadata); Classifier: 128, 64",
        activation_funcs="GELU in branch projections; Tanh + Softmax in GroupSaliencyAttention; ReLU in classifier; Sigmoid for output",
        dropout_desc="0.20 in branch projections; 0.25 after Dense(128); 0.20 after Dense(64)",
        batch_norm_desc="LayerNorm(64) on each branch; BatchNorm1d(128) in classifier; LayerNorm(64) before final output",
        attention_desc="GroupSaliencyAttention(64): Linear(64, 32) + Tanh + Linear(32, 1) + Softmax over the 4 feature modalities",
        encoder_desc="4 modality-specific dense projection branches (Semantic, Lexical, Retrieval, Metadata)",
        output_layer_desc="Linear(64, 1) + Sigmoid",
        output_dim="1 (Continuous probability score in [0.0, 1.0])",
        output_classes="Binary: Class 0 (Unrelated candidate pair), Class 1 (Related cross-corpus pair)",
        checkpoint_type_desc="state_dict (OrderedDict of 33 parameter tensors)",
        additional_notes="Multi-Modal feature group projection with dynamic learned attention over feature domains."
    )

    # -------------------------------------------------------------------------
    # MODEL 3: SIAMESE / DUAL-TOWER ANN WITH CROSS-ATTENTION
    # -------------------------------------------------------------------------
    m3_siam = SiameseCrossAttentionANN(
        embed_dim=384, proj_dim=256, num_rag_features=10, num_heads=4, hidden_dim=256, dropout=0.25
    )
    display_model_inspection(
        header_name="MODEL 3: Siamese / Dual-Tower ANN with Cross-Attention",
        pt_rel_path="models/relationship_model/relationship_model.pt",
        model_instance=m3_siam,
        model_class_name="SiameseCrossAttentionANN",
        purpose="Paper-Patent Relationship Score P(Related | Paper, Patent) combining dense dual towers, bidirectional cross-attention & RAG fusion",
        source_file="backend/app/models/siamese_cross_attention_ann.py",
        train_file="training/train_models_comparison.py (Part 5) & training/train_relationship_model.py",
        inference_file="training/evaluate_retrieval_and_gaps.py & backend/app/analysis/relationship_analysis.py",
        input_dim="Paper embedding (384-d) + Patent embedding (384-d) + 10 RAG tabular features",
        input_feature_names=["paper_embedding (384)", "patent_embedding (384)"] + RELATIONSHIP_FEATURE_COLUMNS,
        hidden_layers_desc="PaperTower(384->256) & PatentTower(384->256) -> BiDirectionalCrossAttention(256, 4 heads) -> RAG Projection(10->64) -> Fusion(2112) -> Dense(2112->256) -> Dense(256->64) -> Dense(64->1)",
        hidden_layer_sizes="Towers: 256; Cross-Attention: 256; RAG Proj: 64; Fusion: 2112; Classifier: 256, 64",
        activation_funcs="GELU in towers and RAG projection; Softmax in MultiheadAttention; ReLU in prediction head; Sigmoid for final output",
        dropout_desc="0.25 in towers and cross-attention; 0.35 after Dense(256); 0.20 after Dense(64)",
        batch_norm_desc="LayerNorm in towers (256), cross-attention (256), RAG proj (64), and classifier (64); BatchNorm1d(256) in classifier",
        attention_desc="BiDirectionalCrossAttention: MultiheadAttention(256, 4 heads) Paper->Patent AND Patent->Paper with LayerNorm and sequence mean pooling",
        encoder_desc="PaperTowerEncoder(384->256) for scientific text; PatentTowerEncoder(384->256) for patent claims",
        output_layer_desc="Linear(64, 1) + Sigmoid",
        output_dim="1 (Continuous probability score in [0.0, 1.0])",
        output_classes="Binary: Class 0 (Unrelated candidate pair), Class 1 (Related cross-corpus pair)",
        checkpoint_type_desc="state_dict (OrderedDict of 45 parameter tensors)",
        additional_notes="Full multimodal architecture with dual tower encoders, bidirectional concept cross-attention, direct interaction [z_p, z_pt, |z_p-z_pt|, z_p*z_pt], and RAG projection."
    )

    # -------------------------------------------------------------------------
    # SUPPLEMENTARY: RESEARCH GAP DETECTION NETWORK
    # -------------------------------------------------------------------------
    m_gap = ResearchGapDetectionNetwork(num_features=12, num_classes=len(GAP_TYPE_CLASSES), dropout=0.25)
    display_model_inspection(
        header_name="DOWNSTREAM MULTI-TASK NETWORK: Research Gap Detection Network",
        pt_rel_path="models/gap_detection_model/gap_detection_model.pt",
        model_instance=m_gap,
        model_class_name="ResearchGapDetectionNetwork",
        purpose="Multi-Task gap analysis: Evaluates candidate innovation gaps from cross-corpus features (including Paper-Patent relationship score) to predict Gap Probability, Gap Category, and Evidence Confidence",
        source_file="backend/app/models/gap_detection_model.py",
        train_file="training/train_gap_model.py",
        inference_file="backend/app/analysis/gap_detection.py & backend/app/services/inference_service.py",
        input_dim="12 Candidate Gap Features (relationship_score, academic_density, patent_density, academic_patent_ratio, limitation_frequency, combination_rarity, semantic_novelty, retrieval_reciprocal_rank, evidence_strength, temporal_velocity, domain_breadth, unresolved_signal)",
        input_feature_names=GAP_FEATURE_COLUMNS,
        hidden_layers_desc="InputEncoder(12->128) -> FeatureAttention(128) -> SharedBackbone(128->128->64) -> 3 Multi-Task Heads: Head 1 GapScore(64->32->1), Head 2 GapType(64->32->6), Head 3 Confidence(64->16->1)",
        hidden_layer_sizes="Encoder: 128; Attention: 128; Backbone: 128, 64; Heads: 32 (Score), 32 (Type), 16 (Confidence)",
        activation_funcs="ReLU in encoder and backbone; Tanh + Softmax in FeatureAttention; Sigmoid for Gap Score and Confidence; Logits for Gap Type",
        dropout_desc="0.25 in input encoder; 0.25 after first backbone layer; 0.20 after second backbone layer",
        batch_norm_desc="LayerNorm(128) in encoder; LayerNorm(128) and LayerNorm(64) in shared backbone",
        attention_desc="FeatureAttention(128): Linear(128, 128) + Tanh + Linear(128, 128) + Softmax dynamic feature weighting across 128 representation dimensions",
        encoder_desc="Dense input encoder with LayerNorm and ReLU",
        output_layer_desc="Head 1: Linear(32, 1) + Sigmoid; Head 2: Linear(32, 6); Head 3: Linear(16, 1) + Sigmoid",
        output_dim="3 Outputs: Gap Score (1), Gap Type Logits (6), Confidence (1)",
        output_classes="6 Categories: " + ", ".join(GAP_TYPE_CLASSES),
        checkpoint_type_desc="checkpoint dictionary with metadata (state_dict, num_features, num_classes, gap_types, metadata)",
        additional_notes="Consumes the Paper-Patent Relationship Score as a primary input feature to detect and classify research gaps."
    )

    # -------------------------------------------------------------------------
    # SUMMARY COMPARISON TABLE
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("MODEL ARCHITECTURE COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model':<30} | {'Class':<30} | {'Input Dim':<12} | {'Output':<12} | {'Parameters':<10}")
    print("-" * 105)
    print(f"{'Model 1 (Baseline)':<30} | {'BaselinePaperPatentANN':<30} | {'10':<12} | {'1 (Sigmoid)':<12} | {15957:<10,}")
    print(f"{'Model 1 (Fine-Tuned)':<30} | {'FineTunedPaperPatentANN':<30} | {'10':<12} | {'1 (Sigmoid)':<12} | {16405:<10,}")
    print(f"{'Model 2 (Feature-Fusion)':<30} | {'DeepFeatureFusionAttentionANN':<30} | {'10 (4 groups)':<12} | {'1 (Sigmoid)':<12} | {45122:<10,}")
    print(f"{'Model 3 (Siamese Cross-Attn)':<30} | {'SiameseCrossAttentionANN':<30} | {'384+384+10':<12} | {'1 (Sigmoid)':<12} | {1417025:<10,}")
    print(f"{'Gap Detection Network':<30} | {'ResearchGapDetectionNetwork':<30} | {'12':<12} | {'Multi-Task':<12} | {65544:<10,}")
    print("-" * 105)
    print()

    # -------------------------------------------------------------------------
    # WORKFLOW AND RELATIONSHIP EXPLANATION
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("WORKFLOW EXPLANATION: RELATIONSHIP MODEL VS. RESEARCH GAP DETECTION")
    print("=" * 80)
    print("Does the neural ANN directly predict the research gap?")
    print("  ANSWER: NO.")
    print("  The 3 core experimentation neural models (Model 1 Baseline/Fine-Tuned,")
    print("  Model 2 Feature-Fusion, and Model 3 Siamese Cross-Attention) are")
    print("  PAPER-PATENT RELATIONSHIP PREDICTION MODELS.")
    print()
    print("  They compute the probability P(Related | Paper, Patent) in [0.0, 1.0].")
    print("  Downstream, the complete system uses these relationship scores as follows:")
    print("    1. Paper-Patent Pair Generation: Candidate pairs from retrieval are formed.")
    print("    2. Neural Model Scoring: Model 3 (or Model 1/2) predicts the relationship score.")
    print("    3. Feature Extraction: extract_gap_candidate_features() aggregates mean and min")
    print("       relationship scores alongside limitation frequencies, coverage ratios, etc.")
    print("    4. Research Gap Detection: ResearchGapDetectionNetwork consumes the 12 candidate")
    print("       features to predict Gap Score, Gap Type, and Confidence.")
    print("    5. Innovation Assessment: MultidimensionalScorer combines gap scores and")
    print("       relationship strengths to produce the final innovation opportunity recommendations.")
    print("=" * 80)


if __name__ == "__main__":
    run_full_inspection()
