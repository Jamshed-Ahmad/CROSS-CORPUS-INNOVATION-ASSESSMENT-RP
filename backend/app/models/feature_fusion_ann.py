"""
feature_fusion_ann.py
=====================
MODEL 2: Deep Feature-Fusion Attention Network.

ARCHITECTURE SPECIFICATION:
  Features are partitioned into four distinct technical modalities:
    1. Semantic Features:
       [semantic_similarity, abstract_similarity, emb_diff_norm, emb_product_mean]
    2. Lexical Features:
       [tfidf_similarity, jaccard_similarity, word_overlap_count, title_similarity]
    3. Retrieval Features:
       [bm25_score]
    4. Metadata Features:
       [publication_year_diff]

  Each group passes through a dedicated dense projection branch with LayerNorm and GELU:
    Semantic (4)  -> Dense(64) + LayerNorm + GELU + Dropout(0.20)
    Lexical (4)   -> Dense(64) + LayerNorm + GELU + Dropout(0.20)
    Retrieval (1) -> Dense(32) + LayerNorm + GELU + Dropout(0.20)
    Metadata (1)  -> Dense(32) + LayerNorm + GELU + Dropout(0.20)

  Feature Fusion & Group Attention:
    All 4 projected representations are aligned and weighted by a learned
    Group Saliency Attention mechanism that dynamically scores which feature
    domain (semantic, lexical, retrieval, metadata) provides the strongest evidence.

  Classification Backbone:
    Fused Vector -> Dense(128) + BatchNorm1d + ReLU + Dropout(0.25)
                 -> Dense(64) + LayerNorm + ReLU + Dropout(0.20)
                 -> Dense(1) + Sigmoid -> P(Related | Paper, Patent) in [0, 1]
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List


class GroupSaliencyAttention(nn.Module):
    """
    Learns dynamic attention weights across the heterogeneous feature groups.
    Computes normalized attention coefficients alpha_k over group representations.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, group_tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Stack group vectors: (B, num_groups, hidden_dim)
        stacked = torch.stack(group_tensors, dim=1)
        scores = self.attn_net(stacked).squeeze(-1)  # (B, num_groups)
        weights = torch.softmax(scores, dim=-1)      # (B, num_groups)

        # Apply weights: (B, num_groups, 1) * (B, num_groups, hidden_dim)
        weighted = stacked * weights.unsqueeze(-1)
        # Flatten fused representation: (B, num_groups * hidden_dim)
        fused = weighted.reshape(stacked.shape[0], -1)
        return fused, weights


class DeepFeatureFusionAttentionANN(nn.Module):
    """
    MODEL 2: Deep Feature-Fusion ANN with Multi-Modal Group Attention.
    """
    # Feature indices in standard 10-feature vector:
    # 0: semantic_similarity, 1: tfidf_similarity, 2: bm25_score, 3: jaccard_similarity,
    # 4: word_overlap_count, 5: title_similarity, 6: abstract_similarity,
    # 7: emb_diff_norm, 8: emb_product_mean, 9: publication_year_diff
    SEMANTIC_INDICES = [0, 6, 7, 8]  # 4 features
    LEXICAL_INDICES = [1, 3, 4, 5]   # 4 features
    RETRIEVAL_INDICES = [2]          # 1 feature
    METADATA_INDICES = [9]           # 1 feature

    def __init__(
        self,
        num_features: int = 10,
        group_proj_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.20
    ):
        super().__init__()
        self.group_proj_dim = group_proj_dim

        # 1. Semantic Branch
        self.semantic_branch = nn.Sequential(
            nn.Linear(len(self.SEMANTIC_INDICES), group_proj_dim),
            nn.LayerNorm(group_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 2. Lexical Branch
        self.lexical_branch = nn.Sequential(
            nn.Linear(len(self.LEXICAL_INDICES), group_proj_dim),
            nn.LayerNorm(group_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 3. Retrieval Branch (projected to group_proj_dim for uniform attention)
        self.retrieval_branch = nn.Sequential(
            nn.Linear(len(self.RETRIEVAL_INDICES), group_proj_dim),
            nn.LayerNorm(group_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 4. Metadata Branch (projected to group_proj_dim for uniform attention)
        self.metadata_branch = nn.Sequential(
            nn.Linear(len(self.METADATA_INDICES), group_proj_dim),
            nn.LayerNorm(group_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 5. Group Saliency Attention Layer
        self.group_attention = GroupSaliencyAttention(hidden_dim=group_proj_dim)

        # 6. Classification Backbone
        fused_dim = group_proj_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout + 0.05),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sem_in = x[:, self.SEMANTIC_INDICES]
        lex_in = x[:, self.LEXICAL_INDICES]
        ret_in = x[:, self.RETRIEVAL_INDICES]
        met_in = x[:, self.METADATA_INDICES]

        h_sem = self.semantic_branch(sem_in)
        h_lex = self.lexical_branch(lex_in)
        h_ret = self.retrieval_branch(ret_in)
        h_met = self.metadata_branch(met_in)

        fused, group_weights = self.group_attention([h_sem, h_lex, h_ret, h_met])
        prob = self.classifier(fused)
        return prob, group_weights

    def predict_proba(self, X: np.ndarray, device: str = "cpu") -> np.ndarray:
        self.eval()
        self.to(device)
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32).to(device)
            prob, _ = self.forward(tensor_x)
            return prob.cpu().numpy().flatten()
