import json

nb = json.load(open("notebooks/cross_corpus_innovation_pipeline.ipynb", encoding="utf-8"))
print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "markdown":
        text = "".join(c["source"]).strip().split("\n")[0]
        print(f"Cell {i:2d} [MD]: {text}")
    else:
        text = "".join(c["source"]).strip().split("\n")[0]
        print(f"Cell {i:2d} [CODE]: {text[:60]}")
