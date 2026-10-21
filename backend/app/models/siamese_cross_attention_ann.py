"""
siamese_cross_attention_ann.py
==============================
MODEL 3: Siamese / Dual-Tower Neural Network with Bidirectional Cross-Attention.

ARCHITECTURE SPECIFICATION:
  Input:
    - Research Paper Representation (MiniLM-L6-v2, 384-d)
    - Patent Representation (MiniLM-L6-v2, 384-d)
    - 10 Heterogeneous RAG Features

  Dual Towers:
    Paper Tower  -> Dense(384 -> 256) + LayerNorm + GELU + Dropout(0.20) -> z_p
    Patent Tower -> Dense(384 -> 256) + LayerNorm + GELU + Dropout(0.20) -> z_pt

  Concept-Level Cross-Attention:
    Bidirectional Multi-Head Attention (4 heads, d_k=64):
      Paper attends to Patent:  A_{paper -> patent} = Softmax(Q_p K_pt^T / sqrt(d_k)) V_pt
      Patent attends to Paper:  A_{patent -> paper} = Softmax(Q_pt K_p^T / sqrt(d_k)) V_p

  Interaction & RAG Feature Fusion:
    Direct Interaction: [z_p, z_pt, |z_p - z_pt|, z_p * z_pt] (dim = 256 * 4 = 1024)
    Cross-Attention Output: [A_{p->pt}, A_{pt->p}, diff, prod] (dim = 256 * 4 = 1024)
    RAG Features Projection: Dense(10 -> 64) + LayerNorm + GELU
    Total Fusion Dimension: 1024 + 1024 + 64 = 2112

  Prediction Head:
    Fused Vector -> Dense(256) + BatchNorm1d + ReLU + Dropout(0.35)
                 -> Dense(64) + LayerNorm + ReLU + Dropout(0.20)
                 -> Dense(1) + Sigmoid -> P(Related | Paper, Patent) in [0, 1]
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from app.models.paper_encoder import PaperTowerEncoder
from app.models.patent_encoder import PatentTowerEncoder
from app.models.cross_attention import BiDirectionalCrossAttention


class SiameseCrossAttentionANN(nn.Module):
    """
    MODEL 3: Siamese / Dual-Tower Neural Network with Bidirectional Cross-Attention.
    """
    def __init__(
        self,
        embed_dim: int = 384,
        proj_dim: int = 256,
        num_rag_features: int = 10,
        num_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.25
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.num_rag_features = num_rag_features

        # 1. Dual Towers
        self.paper_tower = PaperTowerEncoder(embed_dim=embed_dim, hidden_dim=proj_dim, dropout=dropout)
        self.patent_tower = PatentTowerEncoder(embed_dim=embed_dim, hidden_dim=proj_dim, dropout=dropout)

        # 2. Multi-Head Cross-Attention
        self.cross_attention = BiDirectionalCrossAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 3. RAG Feature Projection
        self.rag_projection = nn.Sequential(
            nn.Linear(num_rag_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 4. Fusion Dimension
        fusion_dim = (proj_dim * 4) + self.cross_attention.output_dim + 64

        # 5. Prediction Head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout + 0.10),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        paper_emb: torch.Tensor,
        patent_emb: torch.Tensor,
        rag_features: torch.Tensor,
        paper_seq: Optional[torch.Tensor] = None,
        patent_seq: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # 1. Project through dedicated towers
        z_p = self.paper_tower(paper_emb)      # (B, proj_dim)
        z_pt = self.patent_tower(patent_emb)   # (B, proj_dim)

        diff_emb = torch.abs(z_p - z_pt)
        prod_emb = z_p * z_pt
        direct_fusion = torch.cat([z_p, z_pt, diff_emb, prod_emb], dim=-1)  # (B, 4*proj_dim)

        # 2. Cross-Attention
        if paper_seq is None:
            paper_seq = z_p.unsqueeze(1)
        else:
            paper_seq = self.paper_tower(paper_seq)

        if patent_seq is None:
            patent_seq = z_pt.unsqueeze(1)
        else:
            patent_seq = self.patent_tower(patent_seq)

        cross_repr, attn_weights = self.cross_attention(paper_seq, patent_seq)

        # 3. Project RAG Features
        rag_repr = self.rag_projection(rag_features)

        # 4. Full Multimodal Fusion
        fused = torch.cat([direct_fusion, cross_repr, rag_repr], dim=-1)

        # 5. Relationship Prediction
        prob = self.classifier(fused)
        return prob, attn_weights

    def predict_proba(
        self,
        paper_embs: np.ndarray,
        patent_embs: np.ndarray,
        rag_features: np.ndarray,
        device: str = "cpu"
    ) -> np.ndarray:
        self.eval()
        self.to(device)
        with torch.no_grad():
            t_p = torch.tensor(paper_embs, dtype=torch.float32).to(device)
            t_pt = torch.tensor(patent_embs, dtype=torch.float32).to(device)
            t_rag = torch.tensor(rag_features, dtype=torch.float32).to(device)
            prob, _ = self.forward(t_p, t_pt, t_rag)
            return prob.cpu().numpy().flatten()
