import json

with open('polypddpm-master.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

files_to_find = [
    'color_base.py', 'color_stage.py', 'common.py', 'models.py',
    'notebook_support.py', 'orchestrator.py', 'presentation.py',
    'project_worker.py', 'refiner_core.py', 'report_ui.py',
    'texture_worker.py', 'workflow.py'
]

for idx, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    for fname in files_to_find:
        if fname in src:
            print(f"Cell {idx} ({c['cell_type']}) mentions {fname}")

