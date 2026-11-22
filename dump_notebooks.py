import json

def dump_notebook(nb_path, out_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    with open(out_path, 'w', encoding='utf-8') as out:
        out.write(f"NOTEBOOK: {nb_path}\n")
        out.write(f"Total cells: {len(nb['cells'])}\n\n")
        for idx, c in enumerate(nb['cells']):
            out.write(f"============================== CELL {idx} ({c['cell_type'].upper()}) ==============================\n")
            src = ''.join(c.get('source', []))
            out.write(src + '\n')
            outs = c.get('outputs', [])
            if outs:
                out.write(f"\n--- OUTPUTS ({len(outs)}) ---\n")
                for o_idx, o in enumerate(outs):
                    otype = o.get('output_type')
                    if otype == 'stream':
                        text = ''.join(o.get('text', []))[:300].replace('\n', ' ')
                        out.write(f"  [out {o_idx}: stream] {text}\n")
                    elif otype in ('display_data', 'execute_result'):
                        keys = list(o.get('data', {}).keys())
                        out.write(f"  [out {o_idx}: {otype}] keys: {keys}\n")
                        if 'text/plain' in o.get('data', {}):
                            tplain = ''.join(o['data']['text/plain'])[:200].replace('\n', ' ')
                            out.write(f"     text/plain: {tplain}\n")
                    elif otype == 'error':
                        out.write(f"  [out {o_idx}: ERROR] {o.get('ename')}: {o.get('evalue')}\n")
            out.write('\n\n')

dump_notebook('polypddpm-master.ipynb', 'dump_master.txt')
dump_notebook('Polyp_DDPM_Evaluation.ipynb', 'dump_eval.txt')
print("Dump completed.")
