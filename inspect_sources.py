import json
import ast

with open('Polyp_DDPM_Evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

for idx, c in enumerate(nb2['cells']):
    src = ''.join(c.get('source', []))
    if 'SOURCES' in src:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'SOURCES':
                        for k, v in zip(node.value.keys, node.value.values):
                            filename = k.value
                            content = v.value
                            print(f"File in SOURCES: {filename} ({len(content)} chars, {len(content.splitlines())} lines)")
                            for search_term in ['prepare_report', 'orchestrator', 'report', 'run_extension', 'implementation']:
                                if search_term in content:
                                    print(f"   --> contains {search_term}!")

