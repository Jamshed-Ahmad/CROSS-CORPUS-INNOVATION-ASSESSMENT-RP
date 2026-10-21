import json
import os
import requests
from typing import List, Optional, Dict, Any
from app.ingestion.base import BasePaperAdapter
from app.api.schemas import ResearchPaper
from app.config import settings
from app.logging_config import logger

def reconstruct_openalex_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """Reconstructs plain text abstract from OpenAlex abstract_inverted_index format."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return None
    try:
        max_idx = -1
        for positions in inverted_index.values():
            if positions:
                max_idx = max(max_idx, max(positions))
        if max_idx < 0:
            return None
        
        words = [""] * (max_idx + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                if 0 <= pos <= max_idx:
                    words[pos] = word
        abstract_str = " ".join(words).strip()
        return abstract_str if abstract_str else None
    except Exception as e:
        logger.warning(f"Failed to reconstruct OpenAlex abstract: {e}")
        return None

class MockPaperAdapter(BasePaperAdapter):
    """Adapter for loading research papers from mock datasets or local JSON fixtures."""
    def __init__(self, fixture_path: str = None):
        self.fixture_path = fixture_path
        self._papers: List[ResearchPaper] = []
        self._load_fixtures()

    def _load_fixtures(self):
        if self.fixture_path and os.path.exists(self.fixture_path):
            try:
                with open(self.fixture_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._papers = []
                    for idx, item in enumerate(data):
                        if not item.get("journal"):
                            item["journal"] = "Journal of Machine Learning Research" if idx % 2 == 0 else "IEEE Transactions on Neural Networks"
                        if not item.get("citation_count"):
                            item["citation_count"] = 45 + (idx * 17) % 80
                        if not item.get("open_access_status"):
                            item["open_access_status"] = "gold" if idx % 3 != 0 else "hybrid"
                        if item.get("is_open_access") is None:
                            item["is_open_access"] = True
                        if not item.get("pdf_url") and idx % 4 != 3:
                            item["pdf_url"] = "https://arxiv.org/pdf/2303.08774.pdf"
                        self._papers.append(ResearchPaper(**item))
                    logger.info(f"Loaded {len(self._papers)} papers from {self.fixture_path}")
            except Exception as e:
                logger.error(f"Failed to load paper fixture: {e}")

    def fetch_papers(self, query: str, limit: int = 10) -> List[ResearchPaper]:
        query_words = set(query.lower().split())
        scored_papers = []
        for paper in self._papers:
            text = f"{paper.title} {paper.abstract or ''} {' '.join(paper.keywords)}".lower()
            overlap = sum(1 for word in query_words if word in text)
            scored_papers.append((overlap, paper))
        scored_papers.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored_papers[:limit]] if scored_papers else self._papers[:limit]


class OpenAlexPaperAdapter(BasePaperAdapter):
    """Production live adapter fetching research papers from OpenAlex Works API with pagination & dual auth."""

    BASE_URL = "https://api.openalex.org/works"

    def __init__(self, api_key: str = None, timeout: int = None, max_results: int = None):
        self.api_key = api_key if api_key is not None else settings.clean_openalex_api_key
        self.timeout = timeout or settings.OPENALEX_TIMEOUT_SECONDS
        self.max_results = max_results or settings.OPENALEX_MAX_RESULTS

    def fetch_papers(self, query: str, limit: int = 10) -> List[ResearchPaper]:
        req_limit = min(limit, self.max_results)
        papers: List[ResearchPaper] = []
        page = 1
        per_page = min(50, req_limit)

        logger.info(f"Initiating OpenAlex multi-page retrieval for '{query[:40]}' (target={req_limit})...")

        while len(papers) < req_limit:
            batch_limit = min(per_page, req_limit - len(papers))
            params = {
                "search": query,
                "per_page": batch_limit,
                "page": page,
                "sort": "relevance_score:desc"
            }

            headers = {
                "User-Agent": "CrossCorpusInnovationFramework/1.0 (mailto:research@innovation.org)"
            }

            # Method 1: Bearer Token Header (if key present)
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            try:
                response = requests.get(self.BASE_URL, params=params, headers=headers, timeout=self.timeout)

                # Fallback to query param if Bearer returned 401
                if response.status_code == 401 and self.api_key:
                    logger.warning("OpenAlex Bearer auth returned 401. Testing query-parameter auth fallback...")
                    del headers["Authorization"]
                    params["api_key"] = self.api_key
                    response = requests.get(self.BASE_URL, params=params, headers=headers, timeout=self.timeout)

                # Fallback to polite pool if key still fails
                if response.status_code == 401:
                    logger.warning("OpenAlex API key rejected. Falling back to polite public pool...")
                    params.pop("api_key", None)
                    headers.pop("Authorization", None)
                    response = requests.get(self.BASE_URL, params=params, headers=headers, timeout=self.timeout)

                response.raise_for_status()
                data = response.json()

                total_available = data.get("meta", {}).get("count", 0)
                results = data.get("results", [])

                if not results:
                    break

                for item in results:
                    paper = self._map_to_schema(item)
                    if paper:
                        # Avoid duplicates
                        if not any(p.paper_id == paper.paper_id for p in papers):
                            papers.append(paper)
                    if len(papers) >= req_limit:
                        break

                page += 1
                # If page exceeds available results or no more results
                if len(results) < batch_limit:
                    break

            except Exception as e:
                logger.error(f"OpenAlex query error on page {page} ({e}).")
                break

        logger.info(f"OpenAlex retrieval complete: {len(papers)} papers successfully parsed.")
        return papers

    def _map_to_schema(self, item: Dict[str, Any]) -> Optional[ResearchPaper]:
        try:
            work_id = item.get("id") or f"openalex-{item.get('doi', 'unknown')}"
            title = item.get("title") or "Untitled Publication"
            
            # Reconstruct abstract from inverted index
            abstract_inv = item.get("abstract_inverted_index")
            abstract_text = reconstruct_openalex_abstract(abstract_inv)

            # Authors
            authorships = item.get("authorships", [])
            authors = []
            for a in authorships:
                name = a.get("author", {}).get("display_name")
                if name:
                    authors.append(name)

            # Publication year & DOI
            pub_year = item.get("publication_year")
            doi = item.get("doi")

            # Source URL and Open Access PDF Detection
            primary_loc = item.get("primary_location") or {}
            best_oa = item.get("best_oa_location") or {}
            
            pdf_url = best_oa.get("pdf_url") or primary_loc.get("pdf_url")
            if not pdf_url:
                for loc in (item.get("locations") or []):
                    if isinstance(loc, dict) and loc.get("pdf_url"):
                        pdf_url = loc.get("pdf_url")
                        break
            
            # Check for arXiv pattern
            if not pdf_url and doi and "arxiv" in str(doi).lower():
                clean_arxiv = str(doi).lower().split("arxiv.")[-1]
                pdf_url = f"https://arxiv.org/pdf/{clean_arxiv}.pdf"

            oa_info = item.get("open_access") or {}
            oa_status = oa_info.get("oa_status") or ("gold" if pdf_url else "closed")
            is_oa = bool(oa_info.get("is_oa", False) or pdf_url)
            cited_by = int(item.get("cited_by_count") or 0)

            # Journal / Source display name
            journal_name = (
                (primary_loc.get("source") or {}).get("display_name") or
                (best_oa.get("source") or {}).get("display_name") or
                (item.get("host_venue") or {}).get("display_name") or
                "Scholarly Publication"
            )

            source_url = item.get("doi") or primary_loc.get("landing_page_url") or pdf_url or work_id

            # Keywords & Concepts
            keywords = []
            for concept in item.get("concepts", []):
                disp_name = concept.get("display_name")
                if disp_name:
                    keywords.append(disp_name)
            for kw in item.get("keywords", []):
                disp_name = kw.get("display_name")
                if disp_name and disp_name not in keywords:
                    keywords.append(disp_name)

            # Referenced Works
            referenced_works = item.get("referenced_works", [])

            return ResearchPaper(
                paper_id=work_id,
                title=title,
                abstract=abstract_text,
                authors=authors,
                publication_year=pub_year,
                doi=doi,
                keywords=keywords[:10],
                source_url=source_url,
                references=referenced_works[:10],
                source_provider="openalex",
                dataset_version="openalex-v1",
                journal=journal_name,
                citation_count=cited_by,
                open_access_status=oa_status,
                is_open_access=is_oa,
                pdf_url=pdf_url
            )
        except Exception as e:
            logger.warning(f"Error mapping OpenAlex item: {e}")
            return None
