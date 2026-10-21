"""
__init__.py
===========
Neural models package for Evidence-Grounded Cross-Corpus Innovation Assessment.
Contains:
  - DualTowerCrossAttentionRelationshipModel (Siamese / Dual-Tower + Cross-Attention + RAG Fusion)
  - ResearchGapDetectionNetwork (Multi-Task Evidence-Grounded Gap Detection Network)
  - PaperTowerEncoder & PatentTowerEncoder
  - BiDirectionalCrossAttention
"""

from app.models.paper_encoder import PaperTowerEncoder
from app.models.patent_encoder import PatentTowerEncoder
from app.models.cross_attention import BiDirectionalCrossAttention
from app.models.relationship_model import DualTowerCrossAttentionRelationshipModel
from app.models.baseline_ann import BaselinePaperPatentANN, FineTunedPaperPatentANN
from app.models.feature_fusion_ann import DeepFeatureFusionAttentionANN
from app.models.siamese_cross_attention_ann import SiameseCrossAttentionANN
from app.models.gap_detection_model import ResearchGapDetectionNetwork, GAP_TYPE_CLASSES

__all__ = [
    "PaperTowerEncoder",
    "PatentTowerEncoder",
    "BiDirectionalCrossAttention",
    "DualTowerCrossAttentionRelationshipModel",
    "BaselinePaperPatentANN",
    "FineTunedPaperPatentANN",
    "DeepFeatureFusionAttentionANN",
    "SiameseCrossAttentionANN",
    "ResearchGapDetectionNetwork",
    "GAP_TYPE_CLASSES"
]
