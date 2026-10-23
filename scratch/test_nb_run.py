from pathlib import Path

fig_dir = Path("results/figures")
print("Testing that all required images exist in results/figures/:")
req_images = [
    "accuracy_before_after.png",
    "precision_comparison.png",
    "recall_comparison.png",
    "f1_comparison.png",
    "roc_curves.png",
    "roc_auc_comparison.png",
    "pr_curves.png",
    "confusion_matrix_model1.png",
    "confusion_matrix_model2.png",
    "confusion_matrix_model3.png",
    "confusion_matrices_before_after.png",
    "training_validation_accuracy_model1.png",
    "training_validation_accuracy_model2.png",
    "training_validation_accuracy_model3.png",
    "training_validation_loss_model1.png",
    "training_validation_loss_model2.png",
    "training_validation_loss_model3.png",
    "training_validation_loss_all_models.png",
    "learning_rate_curves.png",
    "model_radar.png",
    "model_improvement.png",
    "model_ranking.png",
    "retrieval_performance.png",
    "paper_relevance.png",
    "patent_relevance.png",
    "paper_patent_relationship.png",
    "research_gap_scores.png",
    "research_gap_types.png",
    "gap_evidence_grounding.png",
    "dataset_distribution.png",
    "train_val_test_split.png",
    "class_distribution.png",
    "feature_importance.png",
    "model_input_features.png",
    "professor_dashboard.png",
    "before_after_master_comparison.png"
]

all_ok = True
for img_name in req_images:
    p = fig_dir / img_name
    if not p.exists() or p.stat().st_size == 0:
        print(f"MISSING: {img_name}")
        all_ok = False
    else:
        print(f"  [OK] {img_name:40s} ({p.stat().st_size:,} bytes)")

if all_ok:
    print(f"\nSUCCESS: All {len(req_images)} required figures exist and have valid non-zero byte size!")
