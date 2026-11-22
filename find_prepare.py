import json

for name in ['polypddpm-master.ipynb', 'Polyp_DDPM_Evaluation.ipynb']:
    with open(name, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"=== Searching in {name} ===")
    for idx, c in enumerate(nb['cells']):
        src = ''.join(c.get('source', []))
        if 'prepare_report' in src:
            print(f"Cell {idx} contains 'prepare_report':")
            for line in src.splitlines():
                if 'prepare_report' in line:
                    print("  ", line)
        if 'RUNTIME_SOURCES' in src:
            print(f"Cell {idx} contains 'RUNTIME_SOURCES':")
            for line in src.splitlines():
                if 'RUNTIME_SOURCES' in line:
                    print("  ", line)
        if 'def prepare_' in src:
            print(f"Cell {idx} contains 'def prepare_':")
            for line in src.splitlines():
                if 'def prepare_' in line:
                    print("  ", line)

