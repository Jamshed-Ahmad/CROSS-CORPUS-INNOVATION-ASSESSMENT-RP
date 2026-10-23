import json
from pathlib import Path

nb_path = Path("notebooks/cross_corpus_innovation_pipeline.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    ctype = cell.get("cell_type")
    src = cell.get("source", [])
    first_line = src[0].strip() if src else "EMPTY"
    print(f"Cell {i:2d} [{ctype:8s}]: {first_line[:80]}")
