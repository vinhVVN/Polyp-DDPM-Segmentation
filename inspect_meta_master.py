import json

with open('polypddpm-master.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

meta = nb['metadata']['polyp_ddpm_master']
for k, v in meta.items():
    if isinstance(v, dict):
        print(f"Key: {k} (dict with {len(v)} items):")
        for subk, subv in v.items():
            print(f"   {subk}: {str(subv)[:80]}")
    else:
        print(f"Key: {k}: {v}")

