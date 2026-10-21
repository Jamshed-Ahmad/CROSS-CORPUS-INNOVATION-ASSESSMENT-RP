from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional, Union, Dict, Any
from app.api.schemas import AnalysisRequest, AnalysisResponse, ResearchPaper, Patent
from app.services.analysis_pipeline import AnalysisPipelineService
from app.services.report_generator import generate_research_report_pdf
from app.services.paper_download import stream_verified_paper_pdf, sanitize_filename
from app.database.session import get_db
from app.database.repository import AnalysisRepository
from app.logging_config import logger

from app.services.health_check import check_openalex_health, check_epo_health

router = APIRouter()
pipeline_service = AnalysisPipelineService()
repo = AnalysisRepository()

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Cross-Corpus Innovation Assessment API",
        "embedding_model": pipeline_service.encoder.model_name
    }

@router.get("/health/diagnostics")
def api_diagnostics():
    openalex_res = check_openalex_health()
    epo_res = check_epo_health()
    return {
        "status": "success",
        "openalex": openalex_res,
        "epo_ops": epo_res
    }

@router.post("/ideas/analyze", response_model=AnalysisResponse)
def analyze_idea(request: AnalysisRequest, db: Session = Depends(get_db)):
    try:
        response = pipeline_service.run_analysis(request)
        repo.save_analysis(db, response)
        return response
    except Exception as e:
        logger.error(f"Error executing analysis pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analyses/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    res = repo.get_analysis(db, analysis_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Analysis ID '{analysis_id}' not found.")
    return res

@router.get("/analyses/{analysis_id}/evidence")
def get_analysis_evidence(analysis_id: str, db: Session = Depends(get_db)):
    res = repo.get_analysis(db, analysis_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Analysis ID '{analysis_id}' not found.")
    return {"analysis_id": analysis_id, "evidence": res.evidence}

@router.post("/search/papers")
def search_papers(query: str = Query(..., min_length=2), top_k: int = Query(5, ge=1)):
    papers = pipeline_service.paper_adapter.fetch_papers(query, limit=top_k)
    return {"query": query, "count": len(papers), "results": papers}

@router.post("/search/patents")
def search_patents(query: str = Query(..., min_length=2), top_k: int = Query(5, ge=1)):
    patents = pipeline_service.patent_adapter.fetch_patents(query, limit=top_k)
    return {"query": query, "count": len(patents), "results": patents}

# ---------------------------------------------------------------------------
# Complete Research Analysis Report PDF & Paper Download Endpoints
# ---------------------------------------------------------------------------

@router.get("/reports/{analysis_id}/pdf")
def download_analysis_report_pdf(analysis_id: str, db: Session = Depends(get_db)):
    """
    Generate and download the complete 16-section research analysis report PDF
    from an existing, stored analysis run without re-executing retrieval or neural models.
    """
    analysis = repo.get_analysis(db, analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis ID '{analysis_id}' not found. Cannot generate report."
        )

    try:
        pdf_buffer = generate_research_report_pdf(analysis)
        pdf_bytes = pdf_buffer.getvalue()
        safe_topic = sanitize_filename(analysis.parsed_idea.original_text or analysis.analysis_id, default="Research_Report")
        filename = f"Research_Report_{safe_topic[:40]}_{analysis_id}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(pdf_bytes)),
                "Cache-Control": "no-cache"
            }
        )
    except Exception as e:
        logger.error(f"Failed to generate research report PDF for {analysis_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate research report PDF: {str(e)}")


@router.post("/reports/generate")
def generate_report_from_payload(
    analysis_id: Optional[str] = Query(None),
    payload: Optional[Dict[str, Any]] = Body(None),
    db: Session = Depends(get_db)
):
    """
    Generate and stream research report PDF either from an analysis_id query parameter
    or from the current frontend analysis payload.
    """
    analysis = None
    target_id = analysis_id or (payload.get("analysis_id") if payload else None)

    if target_id:
        analysis = repo.get_analysis(db, target_id)

    if not analysis and payload:
        try:
            analysis = AnalysisResponse(**payload)
        except Exception as e:
            logger.warning(f"Could not parse analysis payload: {e}")

    if not analysis:
        raise HTTPException(
            status_code=400,
            detail="Valid analysis_id or complete analysis JSON payload is required to generate the report."
        )

    try:
        pdf_buffer = generate_research_report_pdf(analysis)
        pdf_bytes = pdf_buffer.getvalue()
        safe_topic = sanitize_filename(analysis.parsed_idea.original_text or analysis.analysis_id, default="Research_Report")
        filename = f"Research_Report_{safe_topic[:40]}_{analysis.analysis_id}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(pdf_bytes)),
                "Cache-Control": "no-cache"
            }
        )
    except Exception as e:
        logger.error(f"Failed to generate research report PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate research report PDF: {str(e)}")


@router.get("/papers/download")
def download_individual_paper(
    url: str = Query(..., description="Legitimate, accessible external full-text PDF URL"),
    title: str = Query("research_paper", description="Paper title for safe download naming")
):
    """
    Directly and securely stream a verified research paper PDF to the browser.
    Validates URL against SSRF, internal networks, paywalls, and ensures PDF content integrity.
    """
    return stream_verified_paper_pdf(url=url, paper_title=title)

