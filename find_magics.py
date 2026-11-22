import json

with open('polypddpm-master.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    lines = [l for l in src.splitlines() if l.strip().startswith('%%') or l.strip().startswith('%')]
    if lines:
        print(f"Cell {idx} has magics:")
        for l in lines:
            print("  ", l)

