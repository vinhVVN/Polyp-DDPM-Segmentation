import json

with open('polypddpm-master.ipynb', 'r', encoding='utf-8') as f:
    nb1 = json.load(f)

print("=== POLYPDDPM-MASTER FIRST 15 CELLS ===")
for i in range(15):
    c = nb1['cells'][i]
    src = ''.join(c.get('source', []))
    print(f"Cell {i} ({c['cell_type']}):")
    for line in src.splitlines()[:20]:
        print("  ", line[:100])
    if len(src.splitlines()) > 20:
        print(f"   ... ({len(src.splitlines()) - 20} more lines)")

print("\n=== POLYP_DDPM_EVALUATION SOURCES KEYS ===")
with open('Polyp_DDPM_Evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

for i in range(len(nb2['cells'])):
    c = nb2['cells'][i]
    src = ''.join(c.get('source', []))
    if 'SOURCES' in src:
        print(f"Eval Cell {i} contains SOURCES!")
        # Let's extract keys of SOURCES
        import ast
        try:
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == 'SOURCES':
                            if isinstance(node.value, ast.Dict):
                                keys = [k.value for k in node.value.keys if hasattr(k, 'value')]
                                print(f"  SOURCES keys: {keys}")
        except Exception as e:
            print("  ast parse error:", e)

