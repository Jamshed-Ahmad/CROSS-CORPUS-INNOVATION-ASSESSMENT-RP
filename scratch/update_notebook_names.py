import json
from pathlib import Path

nb_path = Path("notebooks/cross_corpus_innovation_pipeline.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

replacements = [
    # Section 13
    (
        "Loading the pre-existing ANN models preserved in `models/baseline/` without modification.\n",
        "Loading the baseline model checkpoints (Model 1: Multi-Layer Perceptron (MLP) with Feature Attention, Model 2: Deep Feature-Fusion Attention Network, and Model 3: Siamese Dual-Tower Network with Cross-Attention) preserved in `models/baseline/` without modification.\n"
    ),
    # Section 16
    (
        '"Model 1 (Fine-Tuned)": {"lr": 1e-3, "batch_size": 32, "dropout": 0.25, "weight_decay": 1e-4}',
        '"Model 1: Multi-Layer Perceptron (MLP) with Feature Attention": {"lr": 1e-3, "batch_size": 32, "dropout": 0.25, "weight_decay": 1e-4}'
    ),
    (
        '"Model 2 (Feature-Fusion)": {"lr": 1e-3, "batch_size": 32, "dropout": 0.25, "weight_decay": 1e-4}',
        '"Model 2: Deep Feature-Fusion Attention Network": {"lr": 1e-3, "batch_size": 32, "dropout": 0.25, "weight_decay": 1e-4}'
    ),
    (
        '"Model 3 (Siamese Cross-Attention)": {"lr": 1e-4, "batch_size": 32, "dropout": 0.30, "weight_decay": 1e-4}',
        '"Model 3: Siamese Dual-Tower Network with Cross-Attention": {"lr": 1e-4, "batch_size": 32, "dropout": 0.30, "weight_decay": 1e-4}'
    ),
    # Section 18
    (
        "Training Model 1, Model 2, and Model 3 with a 100-epoch budget and ReduceLROnPlateau scheduling.\n",
        "Training Model 1 (Multi-Layer Perceptron (MLP) with Feature Attention), Model 2 (Deep Feature-Fusion Attention Network), and Model 3 (Siamese Dual-Tower Network with Cross-Attention) with a 100-epoch budget and ReduceLROnPlateau scheduling.\n"
    ),
    # Section 21
    (
        '"Model": "Model 1: Existing ANN"',
        '"Model": "Model 1: Multi-Layer Perceptron (MLP) with Feature Attention"'
    ),
    (
        '"Model": "Model 2: Feature-Fusion ANN"',
        '"Model": "Model 2: Deep Feature-Fusion Attention Network"'
    ),
    (
        '"Model": "Model 3: Siamese Attention"',
        '"Model": "Model 3: Siamese Dual-Tower Network with Cross-Attention"'
    ),
    # Section 27
    (
        "- **Model 2 (Deep Feature-Fusion ANN with Attention)** achieves",
        "- **Model 2: Deep Feature-Fusion Attention Network** achieves"
    ),
    (
        "- **Model 1 (Fine-Tuned ANN)** is selected as the lightweight tabular standard",
        "- **Model 1: Multi-Layer Perceptron (MLP) with Feature Attention** is selected as the lightweight tabular standard"
    ),
    (
        "- **Model 3 (Siamese Cross-Attention)** is selected when raw text",
        "- **Model 3: Siamese Dual-Tower Network with Cross-Attention** is selected when raw text"
    ),
    (
        'best_model_name = "Feature-Fusion ANN with Attention"',
        'best_model_name = "Model 2: Deep Feature-Fusion Attention Network"'
    ),
    # Section 31
    (
        "  Model 1 (Existing Baseline ANN):\n",
        "  Model 1 (Multi-Layer Perceptron (MLP) with Feature Attention):\n"
    ),
    (
        "  Model 2 (Deep Feature-Fusion ANN with Attention):\n",
        "  Model 2 (Deep Feature-Fusion Attention Network):\n"
    ),
    (
        "  Model 3 (Siamese Cross-Attention Dual-Tower ANN):\n",
        "  Model 3 (Siamese Dual-Tower Network with Cross-Attention):\n"
    ),
    (
        "  Model Architecture   : Model 2 (Deep Feature-Fusion ANN with Attention)\n",
        "  Model Architecture   : Model 2 (Deep Feature-Fusion Attention Network)\n"
    ),
    # Section 32
    (
        "**Model 2 (Deep Feature-Fusion ANN with Attention)** achieved the largest empirical improvement",
        "**Model 2 (Deep Feature-Fusion Attention Network)** achieved the largest empirical improvement"
    ),
    (
        "outperforming Model 1 (**94.96%**) and Model 3 (**92.37%**).",
        "outperforming Model 1: Multi-Layer Perceptron (MLP) with Feature Attention (**94.96%**) and Model 3: Siamese Dual-Tower Network with Cross-Attention (**92.37%**)."
    ),
    (
        "Model 2 was selected because it achieved the **highest composite rank score (96.58 / 100)**",
        "**Model 2: Deep Feature-Fusion Attention Network** was selected because it achieved the **highest composite rank score (96.58 / 100)**"
    )
]

modified_count = 0
for cell in nb["cells"]:
    if "source" in cell:
        new_source = []
        for line in cell["source"]:
            updated_line = line
            for old_text, new_text in replacements:
                if old_text in updated_line:
                    updated_line = updated_line.replace(old_text, new_text)
                    modified_count += 1
            new_source.append(updated_line)
        cell["source"] = new_source

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Updated notebook {nb_path} with {modified_count} replacements.")
