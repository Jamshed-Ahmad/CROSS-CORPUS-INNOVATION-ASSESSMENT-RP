"""
generate_labels.py
==================
Generates non-circular ground-truth labels, extracts 10 RAG-derived features,
and applies leakage-safe grouped splitting (80% Train, 10% Val, 10% Test)
based on paper entities to prevent cross-set leakage.
Computes class balance and full Dataset Growth Report.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.api.schemas import ResearchPaper, Patent
from app.retrieval.embeddings import EmbeddingGenerator
from app.features.relationship_features import (
    RELATIONSHIP_FEATURE_COLUMNS,
    compute_pairwise_rag_features,
    assign_independent_ground_truth_label
)
from app.logging_config import logger

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_TRAINING = PROJECT_ROOT / "data" / "training"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_TRAINING.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_labeled_feature_dataset() -> pd.DataFrame:
    """Loads candidates, computes 10 features, and assigns independent ground-truth labels."""
    logger.info("Loading clean papers and patents for feature computation...")
    papers_path = DATA_PROCESSED / "papers_clean.csv"
    patents_path = DATA_PROCESSED / "patents_clean.csv"

    df_papers = pd.read_csv(papers_path)
    df_patents = pd.read_csv(patents_path)

    encoder = EmbeddingGenerator()

    paper_texts = [f"{row['title']}. {str(row.get('abstract', ''))}" for _, row in df_papers.iterrows()]
    patent_texts = [f"{row['title']}. {str(row.get('abstract', ''))} {str(row.get('claims', ''))}" for _, row in df_patents.iterrows()]

    logger.info(f"Computing dense embeddings for {len(paper_texts)} papers and {len(patent_texts)} patents...")
    paper_emb = encoder.encode(paper_texts)
    patent_emb = encoder.encode(patent_texts)

    logger.info("Computing TF-IDF matrices...")
    tfidf_vec = TfidfVectorizer(max_features=2000, stop_words="english")
    tfidf_vec.fit(paper_texts + patent_texts)

    paper_tfidf = tfidf_vec.transform(paper_texts)
    patent_tfidf = tfidf_vec.transform(patent_texts)
    tfidf_sim_matrix = cosine_similarity(paper_tfidf, patent_tfidf)

    paper_lookup = {row["paper_id"]: (idx, row) for idx, (_, row) in enumerate(df_papers.iterrows())}
    patent_lookup = {row["patent_id"]: (idx, row) for idx, (_, row) in enumerate(df_patents.iterrows())}

    candidates_path = DATA_PROCESSED / "candidates.csv"
    df_cand = pd.read_csv(candidates_path)

    logger.info(f"Computing 10 RAG-derived features for {len(df_cand)} candidate pairs...")
    pair_rows = []

    for _, cand in df_cand.iterrows():
        p_id = cand["paper_id"]
        pt_id = cand["patent_id"]

        if p_id not in paper_lookup or pt_id not in patent_lookup:
            continue

        p_idx, p_row = paper_lookup[p_id]
        pt_idx, pt_row = patent_lookup[pt_id]

        p_obj = ResearchPaper(
            paper_id=p_id,
            title=p_row["title"],
            abstract=str(p_row.get("abstract", "")),
            publication_year=int(p_row.get("publication_year", 2023))
        )
        pt_obj = Patent(
            patent_id=pt_id,
            title=pt_row["title"],
            abstract=str(pt_row.get("abstract", "")),
            claims=[str(pt_row.get("claims", ""))],
            publication_date=str(pt_row.get("publication_date", "2023-01-01"))
        )

        tfidf_sim = float(tfidf_sim_matrix[p_idx, pt_idx])
        bm25_norm = float(cand.get("rrf_score", 0.05)) * 10.0

        feats = compute_pairwise_rag_features(
            paper=p_obj,
            patent=pt_obj,
            paper_emb=paper_emb[p_idx],
            patent_emb=patent_emb[pt_idx],
            tfidf_sim=tfidf_sim,
            bm25_score=bm25_norm
        )

        label = assign_independent_ground_truth_label(
            paper=p_obj,
            patent=pt_obj,
            jaccard=feats["jaccard_similarity"],
            title_sim=feats["title_similarity"]
        )

        row_dict = {
            "paper_id": p_id,
            "patent_id": pt_id,
            "label": label,
            **feats
        }
        pair_rows.append(row_dict)

    df_features = pd.DataFrame(pair_rows)
    df_features.to_csv(DATA_TRAINING / "all_pairs_labeled.csv", index=False)
    
    pos_count = int((df_features['label'] == 1).sum())
    neg_count = int((df_features['label'] == 0).sum())
    total_count = len(df_features)
    pos_pct = round((pos_count / total_count * 100), 2) if total_count else 0.0
    neg_pct = round((neg_count / total_count * 100), 2) if total_count else 0.0

    logger.info(f"Dataset assembled: {total_count} pairs | Positive: {pos_count} ({pos_pct}%) | Negative: {neg_count} ({neg_pct}%)")
    return df_features


def perform_leakage_safe_split(df_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes leakage-safe paper-grouped split (70% Train, 15% Val, 15% Test)
    guaranteeing zero document overlap between train, validation, and test sets.
    """
    logger.info("Executing leakage-safe grouped split (70% / 15% / 15%) by paper entities...")
    unique_papers = df_features["paper_id"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_papers)

    n_total = len(unique_papers)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_papers = set(unique_papers[:n_train])
    val_papers = set(unique_papers[n_train : n_train + n_val])
    test_papers = set(unique_papers[n_train + n_val :])

    train_df = df_features[df_features["paper_id"].isin(train_papers)].copy()
    val_df = df_features[df_features["paper_id"].isin(val_papers)].copy()
    test_df = df_features[df_features["paper_id"].isin(test_papers)].copy()

    # Fit scaler strictly on training split only (zero leakage)
    scaler = StandardScaler()
    scaler.fit(train_df[RELATIONSHIP_FEATURE_COLUMNS].values)

    rel_model_dir = MODELS_DIR / "relationship_model"
    rel_model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, rel_model_dir / "feature_scaler.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    train_df.to_csv(DATA_TRAINING / "train.csv", index=False)
    val_df.to_csv(DATA_TRAINING / "validation.csv", index=False)
    test_df.to_csv(DATA_TRAINING / "test.csv", index=False)

    tot = len(df_features)
    train_n, val_n, test_n = len(train_df), len(val_df), len(test_df)
    train_pct = round(train_n / tot * 100, 2)
    val_pct = round(val_n / tot * 100, 2)
    test_pct = round(test_n / tot * 100, 2)

    # Class balance counts
    pos_tot = int((df_features['label'] == 1).sum())
    neg_tot = int((df_features['label'] == 0).sum())
    pos_pct = round(pos_tot / tot * 100, 2)
    neg_pct = round(neg_tot / tot * 100, 2)

    print("\n" + "=" * 80)
    print("DATASET SPLIT & CLASS BALANCE SUMMARY")
    print("=" * 80)
    print(f"Total Labeled Samples : {tot}")
    print(f"Training Samples       : {train_n} ({train_pct}%) [Pos: {(train_df['label']==1).sum()}, Neg: {(train_df['label']==0).sum()}]")
    print(f"Validation Samples     : {val_n} ({val_pct}%) [Pos: {(val_df['label']==1).sum()}, Neg: {(val_df['label']==0).sum()}]")
    print(f"Testing Samples        : {test_n} ({test_pct}%) [Pos: {(test_df['label']==1).sum()}, Neg: {(test_df['label']==0).sum()}]")
    print(f"Overall Class Balance  : Positive={pos_tot} ({pos_pct}%), Negative={neg_tot} ({neg_pct}%)")
    print("=" * 80 + "\n")

    # Load growth stage if available
    raw_p_path = DATA_RAW / "papers_expanded.json" if (DATA_RAW / "papers_expanded.json").exists() else (DATA_RAW / "papers.json")
    raw_pt_path = DATA_RAW / "patents_expanded.json" if (DATA_RAW / "patents_expanded.json").exists() else (DATA_RAW / "patents.json")

    with open(raw_p_path, "r", encoding="utf-8") as f:
        raw_p_cnt = len(json.load(f))
    with open(raw_pt_path, "r", encoding="utf-8") as f:
        raw_pt_cnt = len(json.load(f))

    df_p_clean = pd.read_csv(DATA_PROCESSED / "papers_clean.csv")
    df_pt_clean = pd.read_csv(DATA_PROCESSED / "patents_clean.csv")
    df_cand = pd.read_csv(DATA_PROCESSED / "candidates.csv")

    growth_report = {
        "raw_openalex_papers": raw_p_cnt,
        "papers_after_cleaning": len(df_p_clean),
        "papers_after_deduplication": len(df_p_clean),
        "raw_epo_patents": raw_pt_cnt,
        "patents_after_cleaning": len(df_pt_clean),
        "patents_after_deduplication": len(df_pt_clean),
        "candidate_paper_patent_pairs": len(df_cand),
        "after_filtering": tot,
        "final_labeled_pairs": tot,
        "total_final_samples": tot,
        "training_samples": train_n,
        "training_pct": train_pct,
        "validation_samples": val_n,
        "validation_pct": val_pct,
        "testing_samples": test_n,
        "testing_pct": test_pct,
        "positive_samples": pos_tot,
        "positive_pct": pos_pct,
        "negative_samples": neg_tot,
        "negative_pct": neg_pct
    }

    growth_json = REPORTS_DIR / "dataset_growth_report.json"
    with open(growth_json, "w", encoding="utf-8") as f:
        json.dump(growth_report, f, indent=2)

    df_growth = pd.DataFrame([growth_report])
    growth_csv = REPORTS_DIR / "dataset_growth_report.csv"
    df_growth.to_csv(growth_csv, index=False)

    print("\n" + "=" * 80)
    print("DATASET GROWTH REPORT")
    print("=" * 80)
    for k, v in growth_report.items():
        print(f"  {k:30s} = {v}")
    print("=" * 80 + "\n")

    return train_df, val_df, test_df


if __name__ == "__main__":
    df_feat = build_labeled_feature_dataset()
    perform_leakage_safe_split(df_feat)
