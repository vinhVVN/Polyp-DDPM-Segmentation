import json
import sys

def inspect_nb(nb_path):
    print(f"=== INSPECTING: {nb_path} ===")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"nbformat: {nb.get('nbformat')}.{nb.get('nbformat_minor')}")
    print(f"Total cells: {len(nb['cells'])}")
    md_count = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
    code_count = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
    print(f"Markdown cells: {md_count}, Code cells: {code_count}")
    
    for idx, c in enumerate(nb['cells']):
        cell_type = c['cell_type']
        source = "".join(c.get('source', []))
        first_line = source.strip().split('\n')[0] if source.strip() else ""
        outputs = c.get('outputs', [])
        out_types = [o.get('output_type', '') for o in outputs]
        has_display = any('data' in o for o in outputs)
        
        # Look for headings or function/class definitions or key execution lines
        if cell_type == 'markdown':
            lines = [l.strip() for l in source.strip().split('\n') if l.strip().startswith('#')]
            if lines:
                print(f"[Cell {idx:02d} MD] {lines[0]} (total headings: {len(lines)})")
        else:
            defs = [l.strip() for l in source.strip().split('\n') if l.strip().startswith('def ') or l.strip().startswith('class ')]
            def_str = f" | defs: {defs[:3]}" if defs else ""
            out_str = f" | outs: {len(outputs)} ({','.join(out_types[:3])})" if outputs else ""
            print(f"[Cell {idx:02d} CODE] len={len(source.splitlines())} lines | first: {first_line[:60]}{def_str}{out_str}")

inspect_nb("polypddpm-master.ipynb")
inspect_nb("Polyp_DDPM_Evaluation.ipynb")
