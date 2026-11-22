import json

with open('polypddpm-master.ipynb', 'r', encoding='utf-8') as f:
    nb1 = json.load(f)

for idx in [1, 3, 5]:
    c = nb1['cells'][idx]
    src = ''.join(c.get('source', []))
    print(f"*** CELL {idx} (lines: {len(src.splitlines())}) ***")
    print(src)
    print("=" * 60)

