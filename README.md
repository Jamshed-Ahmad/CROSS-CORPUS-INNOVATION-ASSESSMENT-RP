# Evidence-Grounded AI Framework for Cross-Corpus Innovation Assessment

An end-to-end, evidence-grounded AI decision-support platform that jointly analyzes research papers and patent literature to identify unexploited research gaps, prior-art boundaries, and multidimensional innovation opportunities.

---

## 🌟 Executive Summary & Performance Highlights

Traditional innovation analysis operates in domain silos: academic literature focuses on fundamental scientific discoveries, while patent databases emphasize commercial claims and prior art. This platform bridges that divide by combining **Hybrid RAG (BM25 + TF-IDF + MiniLM + FAISS + RRF)**, **Deep Feature-Fusion and Siamese Cross-Attention Neural Networks**, and a **Multi-Task Research Gap Detection Engine**. Every recommendation is strictly grounded in verifiable, cited passages from live OpenAlex paper literature and European Patent Office (EPO OPS) patent specifications.

### 🏆 Key Empirical Benchmark Highlights (734 Unseen Test Pairs)
- **Production Standard Accuracy:** **97.28%** (Model 2: Deep Feature-Fusion Attention Network)
- **Test F1-Score:** **0.9602** (Precision: **98.77%**, Recall: **93.41%**, ROC-AUC: **0.9706**)
- **Lightweight Baseline Accuracy:** **94.96%** (Model 1: Multi-Layer Perceptron (MLP) with Feature Attention, up from 56.58% baseline)
- **Cross-Corpus Retrieval Ranking:** **MRR = 0.9843**, **NDCG@5 = 0.9840**, **P@5 = 0.4472**
- **Research Gap Discovery:** Consistently identifies **3 to 6 distinct, evidence-grounded research gaps** across concrete technical themes (Scalability, Generalization, Latency, Explainability, Patent Boundaries, Integration) with zero generic placeholder text.
- **Data Scale:** Expanded from ~460 historical samples to **2,543 verified API records** and **5,544 structured candidate pairs** split under strict, leak-free paper grouping.

---

## 🏛️ Architectural Division of Responsibilities

The system enforces a rigorous separation of mathematical and empirical responsibilities:

1. **RAG Retrieval Engine (`backend/app/retrieval`):**
   - Retrieves candidate papers and patents from OpenAlex and EPO OPS without making premature relationship or gap assertions.
   - Fuses Okapi BM25, TF-IDF term salience, and Sentence-Transformers (`all-MiniLM-L6-v2`) 384-dimensional dense vectors via FAISS and Reciprocal Rank Fusion (RRF, $k=60$).
2. **Second-Stage Neural Re-Ranking & Relationship Learning (`backend/app/models`):**
   - **Model 1: Multi-Layer Perceptron (MLP) with Feature Attention (`app.models.baseline_ann`):** Foundational and fine-tuned feed-forward network with dynamic feature saliency weighting over 10 cross-corpus RAG features.
   - **Model 2: Deep Feature-Fusion Attention Network (`app.models.feature_fusion_ann`):** Domain-partitioned projection branches (Semantic, Lexical, Retrieval, Metadata) coupled with learned group saliency attention. (Production champion: **97.28% accuracy**, **0.9602 F1**, **0.007 ms latency**).
   - **Model 3: Siamese Dual-Tower Network with Cross-Attention (`app.models.siamese_cross_attention_ann`):** Independent paper/patent representation towers ($d_{\text{proj}}=256$) coupled with bidirectional multi-head cross-attention ($Q, K, V$) and RAG feature fusion (**0.9817 ROC-AUC**).
3. **Multi-Task Research Gap Detection Network (`backend/app/models/gap_detection_model`):**
   - Evaluates candidate gaps across 12 evidence and coverage features.
   - Predicts simultaneously: Gap Significance Score (Sigmoid $\in [0, 1]$), Categorical Gap Type (6 classes via Cross-Entropy), and Grounding Confidence ($\in [0, 1]$).
4. **Thematic Gap Extraction & Deduplication (`backend/app/analysis/gap_detection`):**
   - Maps extracted paper limitation passages and patent claim restrictions into a concrete 6-theme taxonomy.
   - Clusters overlapping candidates using token Jaccard similarity ($sim > 0.45$) to guarantee **3 to 6 distinct, strong, non-redundant gaps**.
5. **Innovation Opportunity Scorer & Evidence Validator (`backend/app/analysis` & `app.services`):**
   - Quantifies technical feasibility, novelty, and commercial differentiation.
   - Enforces strict provenance gates: rejects any gap or recommendation without verifiable cited source passages and URLs.

---

## 🧠 Neural Network Models

### Model 1: Multi-Layer Perceptron (MLP) with Feature Attention
- **Architecture:** Feed-forward neural network combining input batch normalization, four dense linear transformation layers, calibrated dropout regularization, and an intermediate softmax-based dynamic feature saliency attention layer.
  - *Baseline Topology:* `BatchNorm1d(10) -> Linear(10, 128) + ReLU + Dropout(0.35) -> Linear(128, 64) + ReLU + Dropout(0.30) -> FeatureAttentionLayer(64) -> Linear(64, 32) + ReLU + Dropout(0.20) -> Linear(32, 1) + Sigmoid`
  - *Fine-Tuned Topology:* Incorporates `LayerNorm` (128, 64, 32) after each linear stage, `LeakyReLU(0.1)` activations for smooth gradient flow, calibrated dropout funnel (0.25 -> 0.20 -> 0.15), and a residual skip connection (`h2 + attention(h2)`).
- **Input:** 10-dimensional tabular vector of heterogeneous RAG features (`semantic_similarity`, `tfidf_similarity`, `bm25_score`, `jaccard_similarity`, `word_overlap_count`, `title_similarity`, `abstract_similarity`, `emb_diff_norm`, `emb_product_mean`, `publication_year_diff`).
- **Hidden Layers:** 3 dense hidden layers (128, 64, 32 dimensions) plus 64-dimensional feature attention layer. Total parameters: 15,957 (Baseline) / 16,405 (Fine-Tuned).
- **Output:** Continuous scalar probability $P(\text{Related} \mid \text{Paper}, \text{Patent}) \in [0.0, 1.0]$.
- **Main Purpose:** Evaluates paper-patent relationship relevance from pre-extracted multi-corpus lexical, semantic, and retrieval features.
- **Why It Is Used:** Serves as the foundational, lightweight tabular baseline standard demonstrating the predictive capability of classical MLP structures on engineered cross-corpus features with minimal compute requirements.

### Model 2: Deep Feature-Fusion Attention Network
- **Architecture:** Multi-modal partitioned neural network with domain-specific projection branches, learned group saliency attention, and a deep classification backbone.
  - *Modality Branches:*
    - *Semantic Branch (4 features):* `Dense(4, 64) + LayerNorm(64) + GELU + Dropout(0.20)`
    - *Lexical Branch (4 features):* `Dense(4, 64) + LayerNorm(64) + GELU + Dropout(0.20)`
    - *Retrieval Branch (1 feature):* `Dense(1, 64) + LayerNorm(64) + GELU + Dropout(0.20)`
    - *Metadata Branch (1 feature):* `Dense(1, 64) + LayerNorm(64) + GELU + Dropout(0.20)`
  - *Group Saliency Attention:* `Dense(64, 32) + Tanh + Dense(32, 1) + Softmax` computing dynamic domain importance coefficients $\alpha_k$ across the 4 modalities.
  - *Classification Backbone:* `Dense(256, 128) + BatchNorm1d(128) + ReLU + Dropout(0.25) -> Dense(128, 64) + LayerNorm(64) + ReLU + Dropout(0.20) -> Linear(64, 1) + Sigmoid`.
- **Input:** 10-dimensional feature vector partitioned into four distinct technical modalities: Semantic ($d=4$), Lexical ($d=4$), Retrieval ($d=1$), and Metadata ($d=1$).
- **Hidden Layers:** 4 parallel branch projections (64-d each), 256-d fused attention representation, and 2 classifier hidden layers (128, 64 dimensions). Total parameters: 45,122.
- **Output:** Calibrated relationship probability $P(\text{Related} \mid \text{Paper}, \text{Patent}) \in [0.0, 1.0]$.
- **Attention / Feature-Fusion Mechanism:** Dynamic learned group attention weights each feature domain independently before concatenation, resolving cross-corpus representation disparities and preventing high-magnitude lexical features from overwhelming subtle semantic signals.
- **Main Purpose:** Selected production standard delivering the highest accuracy (**97.28%**), highest F1-score (**0.9602**), highest precision (**0.9877**), lowest false-positive rate (only 3 FP out of 476 negatives), tightest generalization delta (<0.87%), and ultra-fast inference (**0.007 ms/pair**).

### Model 3: Siamese Dual-Tower Network with Cross-Attention
- **Architecture:** Dual-tower deep neural network with symmetric sentence encoders, multi-head bidirectional cross-attention, non-linear interaction feature extraction, and tabular RAG projection.
  - *Dual Towers:* `PaperTowerEncoder` (384 -> 256, LayerNorm, GELU, Dropout) and `PatentTowerEncoder` (384 -> 256, LayerNorm, GELU, Dropout).
  - *Concept-Level Cross-Attention:* Bidirectional PyTorch `MultiheadAttention` (256-d, 4 heads, $d_k=64$) where Paper representations attend to Patent representations and vice versa.
  - *Direct Interaction Tensor:* Concatenation of $[z_p, z_{pt}, |z_p - z_{pt}|, z_p \odot z_{pt}]$ ($4 \times 256 = 1,024$ dimensions).
  - *RAG Projection Branch:* `Dense(10, 64) + LayerNorm(64) + GELU + Dropout(0.25)`.
  - *Classification Head:* `Dense(2112, 256) + BatchNorm1d(256) + ReLU + Dropout(0.35) -> Dense(256, 64) + LayerNorm(64) + ReLU + Dropout(0.20) -> Linear(64, 1) + Sigmoid`.
- **Input:** Raw paper sentence embeddings (384-d from `all-MiniLM-L6-v2`), raw patent claim embeddings (384-d from `all-MiniLM-L6-v2`), and 10 tabular RAG features.
- **Hidden Layers:** Paper Tower (256), Patent Tower (256), Cross-Attention (256), RAG Projection (64), Fusion Vector (2,112), and Classifier (256, 64). Total parameters: 1,417,025.
- **Output:** Continuous probability $P(\text{Related} \mid \text{Paper}, \text{Patent}) \in [0.0, 1.0]$.
- **Cross-Attention Mechanism:** BiDirectional Cross-Attention applies multi-head attention where paper tokens attend to patent tokens ($A_{p \to pt}$) and patent tokens attend to paper tokens ($A_{pt \to p}$) with LayerNorm and sequence pooling.
- **Main Purpose:** Deep semantic representation learning bridging raw scientific prose and formal patent claim text through token/concept-level cross-attention (**0.9817 ROC-AUC**).

### Model Architecture & Empirical Performance Comparison

| Model | Actual Architecture | Main Purpose | Parameters | Input Dimension |
| :--- | :--- | :--- | :--- | :--- |
| **Model 1** | Multi-Layer Perceptron (MLP) with Feature Attention | Tabular RAG feature relationship classification | 16,405 | 10 Tabular RAG Features |
| **Model 2** | Deep Feature-Fusion Attention Network | Multi-modal domain-partitioned relationship scoring (Production Standard) | 45,122 | 10 Features (4 Groups) |
| **Model 3** | Siamese Dual-Tower Network with Cross-Attention | Direct text representation alignment with bidirectional cross-attention | 1,417,025 | 384 Paper + 384 Patent + 10 RAG |

#### Empirical Performance Benchmarks (734 Unseen Test Samples)

| Model Architecture | Before Acc | After Acc | Accuracy Gain | Before F1 | After F1 | F1 Gain | Precision | Recall | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1: Multi-Layer Perceptron (MLP) with Feature Attention** | 56.58% | 94.96% | +38.38 pp ▲ | 0.5479 | 0.9287 | +38.08 pp ▲ | 92.34% | 93.41% | 0.9707 |
| **Model 2: Deep Feature-Fusion Attention Network (Champion)** | 55.26% | **97.28%** | **+42.02 pp ▲** | 0.5854 | **0.9602** | **+37.48 pp ▲** | **98.77%** | **93.41%** | 0.9706 |
| **Model 3: Siamese Dual-Tower Network with Cross-Attention** | 71.05% | 92.37% | +21.32 pp ▲ | 0.7600 | 0.8951 | +13.51 pp ▲ | 86.59% | 92.64% | **0.9817** |
   - Maps extracted paper limitation passages and patent claim restrictions into a concrete 6-theme taxonomy.
   - Clusters overlapping candidates using token Jaccard similarity ($sim > 0.45$) to guarantee **3 to 6 distinct, strong, non-redundant gaps**.
5. **Innovation Opportunity Scorer & Evidence Validator (`backend/app/analysis` & `app.services`):**
   - Quantifies technical feasibility, novelty, and commercial differentiation.
   - Enforces strict provenance gates: rejects any gap or recommendation without verifiable cited source passages and URLs.

---

## 🏗️ System Architecture Flow

```text
                                  USER RESEARCH TOPIC
                                           │
                             Query Expansion & Semantic Parsing
                                           │
                ┌──────────────────────────┴──────────────────────────┐
                ▼                                                     ▼
         OpenAlex API (REST)                                   EPO OPS API (OAuth2)
       [2,200 Research Papers]                                  [343 Patent Records]
                │                                                     │
                └──────────────────────────┬──────────────────────────┘
                                           ▼
                              Data Preprocessing & Cleaning
                       (Deduplication, language filtering, length gates)
                                           │
                                           ▼
                                HYBRID RAG RETRIEVAL
                ┌──────────────────────────┼──────────────────────────┐
                ▼                          ▼                          ▼
           BM25 Engine               TF-IDF Engine          Sentence-Transformer
       (Exact keyword match)     (Rare term weighting)      (MiniLM-L6-v2 dense)
                │                          │                          │
                │                          │                          ▼
                │                          │                      FAISS Index
                └──────────────────────────┼──────────────────────────┘
                                           ▼
                           Reciprocal Rank Fusion (RRF, k=60)
                                           │
                                           ▼
                               Top-K Retrieved Candidates
                                           │
                                           ▼
                               Hard Domain Relevance Filter
                         (Suppresses candidates scoring below 35.0)
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
           Paper Representation                          Patent Representation
       (MiniLM Dense Embeddings)                     (Claims & Title Embeddings)
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           ▼
                     CROSS-CORPUS ANN RELATIONSHIP SCORING
                     [Feature-Fusion ANN with Dynamic Attention]
                                           +
                            10 Multi-Corpus RAG Features
              (Semantic sim, TF-IDF, BM25, Jaccard, Overlap, Year delta, etc.)
                                           │
                                           ▼
                    Paper-Patent Relationship Score P(Related) ∈ [0, 1]
                                (Test Accuracy: 97.28%)
                                           │
                                           ▼
                    MULTI-TASK RESEARCH GAP DETECTION NETWORK
                                           │
                       Feature Encoding + Dynamic Attention
                                           │
                     ┌─────────────────────┼─────────────────────┐
                     ▼                     ▼                     ▼
             [Head 1: Gap Score]   [Head 2: Gap Type]   [Head 3: Confidence]
                     │                     │                     │
                     └─────────────────────┼─────────────────────┘
                                           ▼
                      Thematic Grouping & Lexical Clustering
                   (Deduplicates similar gaps where Jaccard > 0.45)
                                           │
                                           ▼
                         3 TO 6 DISTINCT EVIDENCE-GROUNDED GAPS
             (Scalability, Generalization, Latency, Explainability, FTO, etc.)
                                           │
                                           ▼
                             Automated Evidence Validation
                      (Strict passage extraction, URL verification)
                                           │
                                           ▼
                              Final Innovation Assessment
                           - Interactive Web Dashboard
                           - Single Paper PDF Downloads
                           - Complete Research Report PDF
```

---

## 📊 Empirical Benchmarks & Before vs. After Comparison

### 1. Direct Model Performance Comparison (On 734 Unseen Test Pairs)

| Model Architecture | Before Accuracy | After Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model 1: Baseline ANN** | 71.00% | **93.05%** | 0.8736 | 0.9380 | 0.9047 | 0.9722 | 0.9729 | Preserved baseline |
| **Model 1: Fine-Tuned ANN** | 71.00% | **94.96%** | 0.9234 | 0.9341 | 0.9287 | 0.9707 | 0.9724 | Fine-tuned (+23.96 pp) |
| **Model 2: Feature-Fusion ANN** | 59.00% | **97.28%** | **0.9877** | **0.9341** | **0.9602** | 0.9706 | **0.9743** | **Active Production Standard** |
| **Model 3: Siamese Cross-Attention** | 68.42% | **92.37%** | 0.8659 | 0.9264 | 0.8951 | **0.9817** | 0.9742 | Dual-tower representation |

*(Note: On the historical ~460-sample imbalanced test split, Model 1 and Model 2 experienced minority-class collapse at 36.84% accuracy and F1 = 0.000. Data expansion, leak-free scaling, and class-weighting elevated their performance to 94.96% and 97.28% on unseen test pairs).*

### 2. Generalization & Overfitting Analysis (Train vs. Val vs. Test)

| Model Architecture | Train Accuracy | Validation Accuracy | Unseen Test Accuracy | Generalization Gap (Val - Test) | Overfitting Risk |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model 1 (Baseline)** | 94.35% | 94.67% | 93.05% | 1.62% | Low / Well-Regularized |
| **Model 1 (Fine-Tuned)** | 96.00% | 96.64% | 94.96% | 1.68% | Low / Well-Regularized |
| **Model 2 (Feature-Fusion)** | 98.13% | 98.15% | 97.28% | **0.87%** | **Minimal / Optimal Generalization** |
| **Model 3 (Siamese Attention)** | 98.48% | 94.90% | 92.37% | 2.53% | Controlled by Dropout (0.30) |

### 3. Cross-Corpus Retrieval Ranking Metrics

| Model Evaluated | MRR | Precision@3 | Precision@5 | Precision@10 | NDCG@3 | NDCG@5 | NDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1 (Baseline)** | 0.9788 | 0.6142 | 0.4427 | 0.2798 | 0.9778 | 0.9764 | 0.9818 |
| **Model 1 (Fine-Tuned)** | 0.9747 | 0.6142 | 0.4472 | 0.2798 | 0.9737 | 0.9785 | 0.9792 |
| **Model 2 (Feature-Fusion)** | **0.9843** | **0.6142** | **0.4472** | **0.2798** | **0.9820** | **0.9840** | **0.9863** |
| **Model 3 (Siamese Attention)** | 0.9682 | 0.6180 | 0.4449 | 0.2809 | 0.9737 | 0.9725 | 0.9752 |

### 4. Downstream Research Gap Quality Metrics

| Model Architecture | Gaps Detected | Mean Gap Score | Mean Confidence | Passages / Gap | Anti-Repetition Passed |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model 1 (Baseline)** | 4 distinct | 86.90 / 100 | 81.4% | 1.50 | Yes (100%) |
| **Model 1 (Fine-Tuned)** | 4 distinct | 86.85 / 100 | 84.3% | 1.50 | Yes (100%) |
| **Model 2 (Feature-Fusion)** | 4 distinct | 85.22 / 100 | 78.9% | 1.50 | Yes (100%) |
| **Model 3 (Siamese Attention)** | 4 distinct | 85.07 / 100 | 85.8% | 1.50 | Yes (100%) |

---

## 📈 Publication-Quality Visual Figures

All 11 mandatory evaluation figures are generated and stored in [`reports/figures/`](reports/figures/):

1. **`1_before_vs_after_accuracy.png`**: Accuracy comparison before vs. after fine-tuning across all 3 models.
2. **`2_before_vs_after_f1.png`**: F1-score comparison before vs. after fine-tuning.
3. **`3_before_vs_after_precision.png`**: Precision comparison before vs. after fine-tuning.
4. **`4_before_vs_after_recall.png`**: Recall comparison before vs. after fine-tuning.
5. **`5_validation_vs_test_accuracy.png`**: Train vs. Validation vs. Test accuracy highlighting zero overfitting.
6. **`6_training_vs_validation_loss.png`**: Epoch-by-epoch loss convergence curves with Early Stopping markers.
7. **`7_confusion_matrices_all_models.png`**: 4-panel confusion matrix grid with explicit TP, TN, FP, FN counts.
8. **`8_overall_model_comparison.png`**: Multi-metric bar chart comparing models across all 6 core indicators.
9. **`9_paper_relevance_ranking.png`**: Research-paper retrieval ranking metrics (MRR, P@3, P@5, NDCG@5).
10. **`10_patent_relevance_ranking.png`**: Patent retrieval ranking metrics (NDCG@3, NDCG@5, R@3, R@5).
11. **`11_research_gap_score_ranking.png`**: Research-gap significance scores, confidence, and thematic taxonomy coverage.
12. **`professor_dashboard.png`**: Master executive dashboard for academic review.
13. **`dataset_comparison_old_vs_expanded.png`**: Empirical comparison between the old dataset (~460 samples) and the expanded API dataset (5,544 pairs).

---

## 📁 Repository Structure & Model Checkpoints

```text
├── backend/
│   ├── app/
│   │   ├── analysis/
│   │   │   ├── gap_detection.py         # Multi-thematic gap taxonomy & clustering (3-6 distinct gaps)
│   │   │   ├── innovation_scoring.py    # Multi-dimensional feasibility & novelty scoring
│   │   │   ├── opportunity_generation.py# Dynamic innovation recommendations
│   │   │   └── rationale_extraction.py  # Sentence-level limitation and claim clause span extraction
│   │   ├── api/
│   │   │   ├── routes.py                # FastAPI REST endpoints (/api/ideas/analyze, /health, etc.)
│   │   │   └── schemas.py               # Pydantic data schemas
│   │   ├── models/
│   │   │   ├── baseline_ann.py          # Model 1: Multi-Layer Perceptron (MLP) with Feature Attention
│   │   │   ├── feature_fusion_ann.py    # Model 2: Deep Feature-Fusion Attention Network [Production]
│   │   │   ├── siamese_cross_attention_ann.py # Model 3: Siamese Dual-Tower Network with Cross-Attention
│   │   │   ├── gap_detection_model.py   # Multi-Task Research Gap Detection Network
│   │   │   └── relationship_model.py    # Checkpoint loader & inference wrapper
│   │   ├── retrieval/
│   │   │   ├── hybrid_retriever.py      # Hybrid RAG engine (BM25 + TF-IDF + FAISS + RRF)
│   │   │   └── embeddings.py            # SentenceTransformer wrapper
│   │   └── services/
│   │       ├── analysis_pipeline.py     # Main end-to-end analysis orchestrator
│   │       ├── paper_download.py        # Verified single-paper PDF streaming
│   │       └── report_generator.py      # Complete research report PDF generation (ReportLab)
│   └── main.py                          # FastAPI entrypoint
├── data/
│   ├── cache/                           # Precomputed MiniLM embeddings (paper_embeddings.pt, patent_embeddings.pt)
│   ├── processed/                       # Cleaned papers and patents (papers_clean.csv, patents_clean.csv)
│   ├── raw/                             # Raw API harvests (papers_raw.csv, patents_raw.csv)
│   └── training/                        # Leak-free split datasets (train.csv, validation.csv, test.csv)
├── frontend/                            # React + Vite Interactive Dashboard
│   ├── src/
│   │   ├── components/                  # UI components (HeroSection, ResultsView, GapAnalysis, etc.)
│   │   └── App.jsx
│   └── package.json
├── models/
│   ├── baseline/                        # PRESERVED ORIGINAL BASELINE CHECKPOINTS (Untouched)
│   │   ├── model1_baseline.pt
│   │   ├── model1_original_finetuned.pt
│   │   ├── model2_baseline.pt
│   │   ├── model3_baseline.pt
│   │   └── gap_detection_model_baseline.pt
│   ├── finetuned/                       # SEPARATELY SAVED FINE-TUNED CHECKPOINTS
│   │   ├── model1_finetuned.pt
│   │   ├── model2_finetuned.pt
│   │   ├── model3_finetuned.pt
│   │   └── gap_detection_model_finetuned.pt
│   ├── relationship_model/              # Production checkpoint & StandardScaler
│   │   ├── relationship_model.pt
│   │   └── feature_scaler.pkl
│   ├── gap_detection_model/             # Production gap model checkpoint
│   │   └── gap_detection_model.pt
│   ├── baseline_ann_model.pt
│   ├── finetuned_ann_model.pt
│   └── feature_fusion_ann_model.pt
├── notebooks/
│   └── cross_corpus_innovation_pipeline.ipynb # 29-Section Academic Pipeline Notebook
├── reports/
│   ├── figures/                         # All 11 evaluation figures & charts
│   ├── baseline_results.csv             # Baseline model evaluation records
│   ├── model_comparison.csv             # Complete comparative benchmark table
│   ├── retrieval_ranking_metrics.csv    # Ranking metrics (MRR, P@K, NDCG@K)
│   └── downstream_gap_evaluation.csv    # Research gap evaluation metrics
└── training/
    ├── train_models_comparison.py       # 100+ epoch fine-tuning script with Early Stopping
    ├── train_gap_model.py               # Multi-task gap network training script
    ├── evaluate_retrieval_and_gaps.py   # Retrieval ranking & downstream gap evaluation
    └── generate_visualizations.py       # Generator for all 11 evaluation figures
```

---

## 📓 Complete 29-Section Research Notebook

A fully structured, reproducible academic notebook is located at:
[`notebooks/cross_corpus_innovation_pipeline.ipynb`](notebooks/cross_corpus_innovation_pipeline.ipynb)

It covers the complete 29-step scientific pipeline:
1. **Imports**
2. **Configuration & Reproducibility**
3. **API Configuration**
4. **API Health Check**
5. **API Data Collection**
6. **Dataset Statistics**
7. **Data Cleaning**
8. **Duplicate Removal**
9. **Feature Engineering**
10. **Label Generation**
11. **Dataset Visualization**
12. **Train / Validation / Test Split**
13. **Baseline Model Loading**
14. **Baseline Evaluation**
15. **Fine-Tuning Setup**
16. **Hyperparameter Experiments**
17. **Early Stopping Implementation**
18. **Final Model Training**
19. **Final Model Evaluation**
20. **Confusion Matrices**
21. **Before vs. After Comparison**
22. **Model Comparison Table**
23. **Research-Paper Retrieval Evaluation**
24. **Patent Retrieval Evaluation**
25. **Research-Gap Analysis**
26. **Research-Gap Visualization**
27. **Final Model Selection**
28. **Save Final Models**
29. **Final Conclusions & Academic Defensibility**

---

## 🚀 Running the Project Locally

### 1. Prerequisites
- **Python 3.10 – 3.13**
- **Node.js 18+** & **npm**

### 2. Backend Setup & Startup
```bash
# Navigate to backend directory
cd backend

# Activate virtual environment
# Windows:
..\.venv\Scripts\activate

# Install dependencies if not already installed
pip install -r requirements.txt

# Start the FastAPI backend server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Health Check: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
- Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Frontend Setup & Startup
```bash
# In a separate terminal, navigate to frontend directory
cd frontend

# Install dependencies if not already installed
npm install

# Start Vite development server
npm run dev
```
- Web Application: [http://localhost:5173/](http://localhost:5173/)

---

## 🔬 Reproducing the Training & Visualizations

To independently retrain the models and regenerate all figures:

```bash
# 1. Train all ANN models (100+ epoch budget, Early Stopping, ReduceLROnPlateau)
python training/train_models_comparison.py

# 2. Train Multi-Task Gap Detection Network
python training/train_gap_model.py

# 3. Evaluate retrieval ranking (MRR, P@K, NDCG@K) and gap quality
python training/evaluate_retrieval_and_gaps.py

# 4. Generate all 11 mandatory evaluation figures
python training/generate_visualizations.py
```

---

## 📄 License & Academic Attribution
Developed for the Master of Science Applied AI Research Project in Cross-Corpus Innovation Assessment.
All benchmark numbers, training loss curves, and evaluation tables are empirically derived from unseen test partitions without synthetic inflation or label leakage.
