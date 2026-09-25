import json, sys
sys.path.insert(0, 'c:/Prompshield')
with open('c:/Prompshield/evaluation/promptshield_benchmark/ground_truth.json', encoding='utf-8') as f:
    data = json.load(f)
errors = []
total_ents = 0
for item in data:
    text = item['text']
    for e in item.get('entities', []):
        total_ents += 1
        actual = text[e['start']:e['end']]
        if actual != e['text']:
            errors.append(f"[{item['id']}] MISMATCH: expected '{e['text']}' got '{actual}' at [{e['start']}:{e['end']}]")
if errors:
    for err in errors:
        print(err)
    print(f'TOTAL ERRORS: {len(errors)}')
else:
    print(f'ALL {total_ents} entity offsets VALID across {len(data)} prompts. PASS.')
