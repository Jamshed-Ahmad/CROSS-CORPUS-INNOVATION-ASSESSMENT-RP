"""
build_pairs.py
==============
Converts raw API records (or expanded harvest) into clean processed datasets and
generates candidate paper-patent pairs using hybrid RAG retrieval
(BM25 + TF-IDF + FAISS + RRF) with configurable limits.
Computes and records the full Dataset Growth Report.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.api.schemas import ResearchPaper, Patent
from app.ingestion.preprocessing import preprocess_papers, preprocess_patents
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.tfidf import TFIDFRetriever
from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.vector_index import VectorIndex
from app.retrieval.fusion import reciprocal_rank_fusion
from app.logging_config import logger

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Configurable candidate generation limits
TOP_K_PER_PAPER = int(os.getenv("TOP_K_PER_PAPER", "12"))
MAX_PAIRS = int(os.getenv("MAX_PAIRS", "6000"))


def clean_and_normalize_data(raw_papers_path: Path, raw_patents_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[ResearchPaper], List[Patent], Dict[str, Any]]:
    """Cleans, validates, and deduplicates raw paper and patent records."""
    logger.info(f"Loading raw datasets from {raw_papers_path} and {raw_patents_path}...")

    with open(raw_papers_path, "r", encoding="utf-8") as f:
        raw_papers_data = json.load(f)
        raw_papers = [ResearchPaper(**item) for item in raw_papers_data]

    with open(raw_patents_path, "r", encoding="utf-8") as f:
        patent_items = json.load(f)
        raw_patents = []
        for item in patent_items:
            if isinstance(item.get("claims"), str):
                item["claims"] = [item["claims"]]
            if isinstance(item.get("classifications"), str):
                item["classifications"] = [c.strip() for c in item["classifications"].split(",") if c.strip()]
            raw_patents.append(Patent(**item))

    raw_papers_count = len(raw_papers)
    raw_patents_count = len(raw_patents)

    clean_papers, paper_profile = preprocess_papers(raw_papers)
    clean_patents, patent_profile = preprocess_patents(raw_patents)

    logger.info(f"Papers cleaned: {len(clean_papers)} / {raw_papers_count} | Profile: {paper_profile}")
    logger.info(f"Patents cleaned: {len(clean_patents)} / {raw_patents_count} | Profile: {patent_profile}")

    df_papers = pd.DataFrame([p.model_dump() for p in clean_papers])
    df_patents = pd.DataFrame([pt.model_dump() for pt in clean_patents])

    # Save processed datasets
    df_papers.to_csv(DATA_PROCESSED / "papers_clean.csv", index=False)
    df_patents.to_csv(DATA_PROCESSED / "patents_clean.csv", index=False)

    growth_stage = {
        "raw_openalex_papers": raw_papers_count,
        "papers_after_cleaning": paper_profile.get("total_input", raw_papers_count),
        "papers_after_deduplication": len(clean_papers),
        "raw_epo_patents": raw_patents_count,
        "patents_after_cleaning": patent_profile.get("total_input", raw_patents_count),
        "patents_after_deduplication": len(clean_patents)
    }

    return df_papers, df_patents, clean_papers, clean_patents, growth_stage


def generate_candidate_pairs(
    clean_papers: List[ResearchPaper],
    clean_patents: List[Patent],
    top_k_per_paper: int = TOP_K_PER_PAPER,
    max_pairs: int = MAX_PAIRS
) -> pd.DataFrame:
    """
    Generates candidate pairs using hybrid RAG retrieval.
    Avoids full O(N x M) Cartesian explosion by selecting top-K candidates per paper.
    """
    logger.info(f"Building hybrid retrieval indices for candidate pair generation ({len(clean_papers)} papers, {len(clean_patents)} patents)...")
    encoder = EmbeddingGenerator()
    bm25 = BM25Retriever()
    tfidf = TFIDFRetriever()
    vector_index = VectorIndex(encoder)

    bm25.index_documents(clean_papers, clean_patents)
    tfidf.index_documents(clean_papers, clean_patents)
    vector_index.index_documents(clean_papers, clean_patents)

    patent_dict = {pt.patent_id: pt for pt in clean_patents}
    candidate_rows = []

    logger.info(f"Executing hybrid RAG retrieval across {len(clean_papers)} papers (top_k={top_k_per_paper})...")
    for p_idx, paper in enumerate(clean_papers):
        paper_query = f"{paper.title}. {paper.abstract or ''}"

        bm25_hits = [h for h in bm25.search(paper_query, top_k=top_k_per_paper * 4) if len(h) >= 3 and h[2] == "patent"][:top_k_per_paper * 2]
        tfidf_hits = [h for h in tfidf.search(paper_query, top_k=top_k_per_paper * 4) if len(h) >= 3 and h[2] == "patent"][:top_k_per_paper * 2]
        vec_hits = [h for h in vector_index.search(paper_query, top_k=top_k_per_paper * 4) if len(h) >= 3 and h[2] == "patent"][:top_k_per_paper * 2]

        fused = reciprocal_rank_fusion([bm25_hits, tfidf_hits, vec_hits], top_n=top_k_per_paper)

        for pt_id, rrf_score, _ in fused:
            if pt_id in patent_dict:
                pt = patent_dict[pt_id]
                candidate_rows.append({
                    "paper_id": paper.paper_id,
                    "patent_id": pt.patent_id,
                    "paper_title": paper.title,
                    "patent_title": pt.title,
                    "paper_abstract": paper.abstract or "",
                    "patent_abstract": pt.abstract or "",
                    "patent_claims": " ".join(pt.claims or []),
                    "rrf_score": round(float(rrf_score), 5),
                    "paper_year": paper.publication_year or 2023,
                    "patent_year": int(pt.publication_date[:4]) if pt.publication_date else 2023
                })

        if len(candidate_rows) >= max_pairs * 1.2:
            break

    raw_candidates_count = len(candidate_rows)
    df_candidates = pd.DataFrame(candidate_rows).drop_duplicates(subset=["paper_id", "patent_id"])
    if len(df_candidates) > max_pairs:
        df_candidates = df_candidates.iloc[:max_pairs]

    logger.info(f"Candidate pairs generated: {raw_candidates_count} -> {len(df_candidates)} after deduplication/filtering.")
    df_candidates.to_csv(DATA_PROCESSED / "candidates.csv", index=False)

    return df_candidates


if __name__ == "__main__":
    raw_p = DATA_RAW / "papers_expanded.json" if (DATA_RAW / "papers_expanded.json").exists() else (DATA_RAW / "papers.json")
    raw_pt = DATA_RAW / "patents_expanded.json" if (DATA_RAW / "patents_expanded.json").exists() else (DATA_RAW / "patents.json")

    df_papers, df_patents, clean_p, clean_pt, growth_stage = clean_and_normalize_data(raw_p, raw_pt)
    df_cand = generate_candidate_pairs(clean_p, clean_pt, top_k_per_paper=TOP_K_PER_PAPER, max_pairs=MAX_PAIRS)
