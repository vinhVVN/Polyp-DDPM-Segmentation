import os
from pathlib import Path

for root, dirs, files in os.walk('_extracted_sources'):
    for f in files:
        p = Path(root) / f
        try:
            content = p.read_text(encoding='utf-8')
            for term in ['register_cell_magic', 'magic', 'prepare_report', 'orchestrator', 'report', 'RUNTIME_SOURCES', 'register_line_cell_magic']:
                if term in content:
                    print(f"{p} contains '{term}'")
        except Exception:
            pass

