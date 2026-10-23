import json

nb = json.load(open("notebooks/cross_corpus_innovation_pipeline.ipynb", encoding="utf-8"))
for i in range(36, 58):
    c = nb["cells"][i]
    print(f"=== Cell {i} ({c['cell_type']}) ===")
    src = "".join(c["source"]).strip()
    print(src[:250] + ("..." if len(src) > 250 else ""))
    print()
