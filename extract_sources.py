import json
import ast
from pathlib import Path

with open('Polyp_DDPM_Evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

extracted_dir = Path('_extracted_sources')
extracted_dir.mkdir(exist_ok=True)

for idx, c in enumerate(nb2['cells']):
    src = ''.join(c.get('source', []))
    if 'SOURCES' in src:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'SOURCES':
                        for k, v in zip(node.value.keys, node.value.values):
                            fname = k.value
                            content = v.value
                            out_p = extracted_dir / fname
                            out_p.parent.mkdir(parents=True, exist_ok=True)
                            out_p.write_text(content, encoding='utf-8')
                            print(f"Extracted: {fname}")

print("Extraction done.")
