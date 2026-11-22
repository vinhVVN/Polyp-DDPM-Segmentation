with open('polypddpm-master.ipynb', 'r', encoding='utf-8') as f:
    raw = f.read()

pos = 0
while True:
    pos = raw.find('orchestrator.py', pos)
    if pos == -1:
        break
    print(f"=== FOUND AT {pos} ===")
    start = max(0, pos - 200)
    end = min(len(raw), pos + 300)
    print(raw[start:end])
    print("-" * 50)
    pos += len('orchestrator.py')

