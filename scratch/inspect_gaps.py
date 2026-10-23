import sys
import json
from pathlib import Path

project_root = Path(".")
backend_dir = project_root / "backend"
sys.path.insert(0, str(backend_dir))

from app.analysis.gap_detection import GapDetector
from app.api.schemas import ResearchIdea, EvidenceRecord, CrossCorpusRelationship

gap_detector = GapDetector(
    model_path=str(project_root / "models" / "gap_detection_model" / "gap_detection_model.pt")
)

test_idea = ResearchIdea(
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

evidence_pool = [
    EvidenceRecord(
        evidence_id="EV-001",
        document_id="https://openalex.org/W3096561213",
        document_type="paper",
        passage_text="Current message passing graph models exhibit severe generalization drops when tested on unseen dynamic ligand conformations.",
        source_url="https://doi.org/10.1038/s41598-020-74902-8",
        evidence_type="limitation",
        confidence=0.92
    ),
    EvidenceRecord(
        evidence_id="EV-002",
        document_id="US-11842801-B2",
        document_type="patent",
        passage_text="Spatial convolution architectures are limited to static rigid-body docking and lack rotational covariance guarantees.",
        source_url="https://patents.google.com/patent/US11842801B2/en",
        evidence_type="limitation",
        confidence=0.89
    ),
    EvidenceRecord(
        evidence_id="EV-003",
        document_id="https://openalex.org/W3032123378",
        document_type="paper",
        passage_text="Scalability bottlenecks occur when evaluating multi-body atomic interactions in high-throughput chemical screening.",
        source_url="https://doi.org/10.1021/acs.jcim.1c00000",
        evidence_type="limitation",
        confidence=0.86
    ),
    EvidenceRecord(
        evidence_id="EV-004",
        document_id="EP-3891720-A1",
        document_type="patent",
        passage_text="Latency constraints prevent real-time inference during ultra-large virtual screening campaigns on commercial clouds.",
        source_url="https://worldwide.espacenet.com/patent/search/family/072481928/publication/EP3891720A1",
        evidence_type="limitation",
        confidence=0.84
    ),
    EvidenceRecord(
        evidence_id="EV-005",
        document_id="https://openalex.org/W2963412093",
        document_type="paper",
        passage_text="Opaque feature representations hinder biophysical explainability and regulatory auditing of binding predictions.",
        source_url="https://doi.org/10.1093/bioinformatics/btz555",
        evidence_type="limitation",
        confidence=0.87
    )
]

rels = [
    CrossCorpusRelationship(
        relationship_id="REL-001",
        source_entity_id="https://openalex.org/W3096561213",
        target_entity_id="US-11842801-B2",
        relationship_type="paper-to-patent",
        score=0.91,
        supporting_evidence_ids=["EV-001", "EV-002"],
        extraction_method="neural_relationship_prediction",
        confidence=0.91
    ),
    CrossCorpusRelationship(
        relationship_id="REL-002",
        source_entity_id="https://openalex.org/W3032123378",
        target_entity_id="EP-3891720-A1",
        relationship_type="paper-to-patent",
        score=0.86,
        supporting_evidence_ids=["EV-003", "EV-004"],
        extraction_method="neural_relationship_prediction",
        confidence=0.86
    )
]

gaps = gap_detector.detect_gaps(
    idea=test_idea,
    paper_rationales={},
    patent_rationales={},
    evidence_pool=evidence_pool,
    relationships=rels,
    papers_count=5,
    patents_count=2
)

print(f"Total gaps detected: {len(gaps)}")
for g in gaps:
    print(f"ID: {g.gap_id} | Type: {g.gap_type} | Score: {g.gap_score:.1f} | Conf: {g.confidence_score:.1f} | Evidence: {len(g.supporting_evidence_ids)}")
    print(f"  Statement: {g.gap_statement}")
    print(f"  Direction: {g.potential_research_direction}\n")
