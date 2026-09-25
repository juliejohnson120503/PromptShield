"""
Fix all character offset errors in ground_truth.json.
This script recomputes each entity's start/end by searching for the
entity text inside the prompt, then validates every offset.
It writes a corrected file and reports any entities it could not resolve.
"""
import json, re
from pathlib import Path

GT_PATH = Path("c:/Prompshield/evaluation/promptshield_benchmark/ground_truth.json")

with open(GT_PATH, encoding="utf-8") as f:
    data = json.load(f)

all_errors = []
fixed_count = 0
conflict_count = 0

for item in data:
    text = item["text"]
    entities = item.get("entities", [])
    new_entities = []

    for e in entities:
        entity_text = e["text"]
        old_start   = e["start"]
        old_end     = e["end"]
        label       = e["label"]

        # Check if existing offset is already correct
        if text[old_start:old_end] == entity_text:
            new_entities.append(e)
            continue

        # Search for the entity text in the prompt
        matches = [m.start() for m in re.finditer(re.escape(entity_text), text)]

        if len(matches) == 0:
            all_errors.append(
                f"[{item['id']}] CANNOT FIND '{entity_text}' in text"
            )
            conflict_count += 1
            new_entities.append(e)  # keep original, flag it
            continue

        if len(matches) == 1:
            new_start = matches[0]
            new_end   = new_start + len(entity_text)
        else:
            # Multiple matches — pick the closest to the old start
            best = min(matches, key=lambda s: abs(s - old_start))
            new_start = best
            new_end   = best + len(entity_text)

        # Verify
        assert text[new_start:new_end] == entity_text, \
            f"Verification failed: {text[new_start:new_end]!r} != {entity_text!r}"

        if new_start != old_start or new_end != old_end:
            print(f"  FIXED [{item['id']}] '{entity_text}' ({label}): "
                  f"[{old_start}:{old_end}] -> [{new_start}:{new_end}]")
            fixed_count += 1

        new_entities.append({
            "text": entity_text,
            "start": new_start,
            "end": new_end,
            "label": label,
        })

    item["entities"] = new_entities

# Write corrected file
with open(GT_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n--- Summary ---")
print(f"Fixed: {fixed_count} offsets")
print(f"Unresolvable: {conflict_count}")
if all_errors:
    for e in all_errors:
        print(f"  ERROR: {e}")

# Final validation pass
print("\n--- Final validation pass ---")
errors = []
total = 0
for item in data:
    text = item["text"]
    for e in item.get("entities", []):
        total += 1
        actual = text[e["start"]:e["end"]]
        if actual != e["text"]:
            errors.append(f"[{item['id']}] '{e['text']}' != '{actual}' at [{e['start']}:{e['end']}]")

if errors:
    print(f"REMAINING ERRORS ({len(errors)}):")
    for err in errors:
        print(f"  {err}")
else:
    print(f"ALL {total} entity offsets VALID across {len(data)} prompts. PASS.")
