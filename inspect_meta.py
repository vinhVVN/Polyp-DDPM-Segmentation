import json

with open('polypddpm-master.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Notebook metadata keys:", list(nb.get('metadata', {}).keys()))
for k, v in nb.get('metadata', {}).items():
    if isinstance(v, dict):
        print(f"Metadata key {k}: dict with keys {list(v.keys())}")
    elif isinstance(v, str):
        print(f"Metadata key {k}: str len={len(v)}")
    else:
        print(f"Metadata key {k}: {type(v)}")

# Check if metadata has sources or hashes
if 'sources' in nb.get('metadata', {}):
    print("Found 'sources' in metadata!")
if 'runtime' in nb.get('metadata', {}):
    print("Found 'runtime' in metadata!")

# Let's check all metadata recursively
def search_dict(d, path=""):
    for k, v in d.items():
        curr = f"{path}.{k}" if path else k
        if 'orchestrator' in str(k) or 'orchestrator' in str(v)[:100]:
            print(f"Found orchestrator at metadata path: {curr}")
        if isinstance(v, dict):
            search_dict(v, curr)

search_dict(nb.get('metadata', {}))

