import sys
from pathlib import Path

project_root = Path(".")
backend_dir = project_root / "backend"
sys.path.insert(0, str(backend_dir))

from app.services.report_generator import ResearchReportPDFBuilder
from app.api.schemas import (
    AnalysisResponse, ResearchIdea, ResearchPaper, Patent,
    CandidateGap, RecommendationRecord, CrossCorpusRelationship
)

# Create a sample AnalysisResponse with real data
idea = ResearchIdea(
    idea_id="EVAL-IDEA-001",
    original_text="Geometric graph neural networks for molecular binding affinity prediction",
    parsed_problem="Generalization failure under flexible conformational ligand shifts",
    domain="Computational Drug Discovery",
    proposed_technology="Geometric Graph Neural Network",
    objective="Predict molecular binding affinity under dynamic conformational shifts",
    application_area="Molecular Binding Affinity",
    target_domain="Computational Drug Discovery",
    salient_keywords=["graph neural network", "molecular binding", "affinity", "conformation"]
)

papers = [
    ResearchPaper(
        paper_id="https://openalex.org/W3096561213",
        title="Equivariant Graph Neural Networks for Molecular 3D Conformations",
        abstract="Current message passing graph models fail under dynamic out-of-distribution ligand conformations.",
        authors=["Satorras, V. G.", "Hoogeboom, E.", "Welling, M."],
        year=2021,
        venue="ICML",
        relevance_score=0.942,
        citation_count=845,
        is_open_access=True
    )
]

patents = [
    Patent(
        patent_id="US-11842801-B2",
        title="Spatial Graph Convolution System for Biomolecular Docking",
        abstract="Methods and apparatus for predicting binding conformations using coordinate graphs.",
        publication_date="2023-12-12",
        relevance_score=0.938,
        assignee="Genentech, Inc.",
        claims=["A computer-implemented method comprising spatial graph convolutions on atomic coordinates."]
    )
]

gaps = [
    CandidateGap(
        gap_id="GAP-01",
        idea_id="EVAL-IDEA-001",
        gap_description="High-throughput scaling failure during dynamic conformational molecular simulations.",
        gap_type="TECHNICAL_GAP",
        gap_statement="Scalability Bottleneck in Dynamic Graph Message Passing",
        gap_score=89.0,
        confidence_score=88.5,
        research_opportunity="Develop sparse equivariant graph attention kernels with sub-quadratic complexity.",
        supporting_evidence_ids=["EV-001", "EV-002"],
        reasoning="Peer-reviewed studies report quadratic scaling with atomic coordinate tensors."
    )
]

rels = [
    CrossCorpusRelationship(
        relationship_id="REL-001",
        source_entity_id="https://openalex.org/W3096561213",
        target_entity_id="US-11842801-B2",
        relationship_type="paper-to-patent",
        score=0.94,
        supporting_evidence_ids=["EV-001"],
        extraction_method="neural_relationship_prediction",
        confidence=0.94
    )
]

recs = [
    RecommendationRecord(
        recommendation_id="REC-001",
        idea_id="EVAL-IDEA-001",
        opportunity_description="Sparse Equivariant Graph Attention for Ultra-Large Molecular Ensembles",
        problem_addressed="Computational latency during high-throughput binding screening",
        identified_gap="GAP-01",
        assessment_factors={"novelty_potential": 92.0, "feasibility": 85.0, "evidence_strength": 90.0},
        actionability="Implement PyTorch Geometric sparse message-passing routines",
        limitations="Requires CUDA-accelerated hardware"
    )
]

analysis = AnalysisResponse(
    analysis_id="TEST-PDF-001",
    status="completed",
    parsed_idea=idea,
    papers=papers,
    patents=patents,
    relationships=rels,
    gaps=gaps,
    evidence=[],
    recommendations=recs,
    scores={"total_innovation_score": 88.5, "novelty_potential": 92.0, "feasibility": 85.0},
    metadata={"pipeline_version": "2.0.0"}
)

builder = ResearchReportPDFBuilder(analysis)
pdf_buf = builder.build_pdf()
out_pdf_path = Path("scratch/test_report.pdf")
with open(out_pdf_path, "wb") as f:
    f.write(pdf_buf.getvalue())

print(f"Successfully generated PDF report: {out_pdf_path} ({len(pdf_buf.getvalue()):,} bytes)")
