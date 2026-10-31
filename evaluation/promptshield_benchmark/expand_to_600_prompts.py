#!/usr/bin/env python3
"""
PromptShield Benchmark Dataset Expander (500 -> 600 Prompts)
=============================================================
Adds 100 challenging, real-world prompts reflecting all core problems solved:
1. Casing invariance for PERSON names (lowercase, alternating mock case, uppercase)
2. Typo and shorthand context cues for credentials (passwrd, p@ssword, pswd, pwd, PassWord)
3. Typo and shorthand context cues for bank accounts (acc no, act no, a/c no, ac no)
4. Mathematical / algorithmic continuous number queries as negative controls (zero PII)
5. Indian Identity Documents (Aadhaar with adhr/adhar, PAN with pan no, Driving License, Voter ID)
6. Foreign phone numbers (US NANP, UK national/mobile, UAE, Australia, Europe)
7. Lifecycle-anchored dates (birthday / anniversary on today/yesterday) vs generic relative dates
8. Gazetteer public figures vs private individuals

Ensures:
- Exact character-level offset calculation: text[start:end] == entity['text']
- Zero offset mismatches
- Backs up ground_truth.json to ground_truth_500_backup.json
"""

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

BENCHMARK_DIR = Path(__file__).resolve().parent
GT_FILE = BENCHMARK_DIR / "ground_truth.json"
BACKUP_FILE = BENCHMARK_DIR / "ground_truth_500_backup.json"


def build_prompt_with_entities(template: str, entities_to_place: List[Tuple[str, str, str]]) -> Dict:
    """
    Constructs a prompt and computes exact [start, end) offsets for each entity.
    entities_to_place: List of (placeholder_key, entity_text, label)
    template: Text containing {placeholder_key} markers.
    """
    lookup = {k: (val, label) for k, val, label in entities_to_place}
    
    parts = []
    clean_entities = []
    last_idx = 0
    current_pos = 0

    matches = list(re.finditer(r"\{([a-zA-Z0-9_]+)\}", template))
    matched_keys = {m.group(1) for m in matches}
    for k in lookup:
        if k not in matched_keys:
            raise ValueError(f"Placeholder '{{{k}}}' declared in entities_to_place but NOT found in template: '{template}'")

    for match in matches:
        start, end = match.span()
        key = match.group(1)
        if key not in lookup:
            continue

        prefix = template[last_idx:start]
        parts.append(prefix)
        current_pos += len(prefix)

        entity_val, entity_label = lookup[key]
        ent_start = current_pos
        ent_end = ent_start + len(entity_val)

        parts.append(entity_val)
        current_pos += len(entity_val)

        clean_entities.append({
            "text": entity_val,
            "start": ent_start,
            "end": ent_end,
            "label": entity_label
        })

        last_idx = end

    suffix = template[last_idx:]
    parts.append(suffix)
    final_text = "".join(parts)

    return {
        "text": final_text,
        "entities": clean_entities
    }


def get_100_new_prompt_specs() -> List[Tuple[str, List[Tuple[str, str, str]]]]:
    specs: List[Tuple[str, List[Tuple[str, str, str]]]] = [
        # =====================================================================
        # 1. CASING INVARIANCE FOR PERSON NAMES (20 prompts)
        # =====================================================================
        # 1.1 Lowercase self-introductions
        ("my name is {name}.", [("name", "anand", "PERSON")]),
        ("i am {name} and I work here.", [("name", "deepak", "PERSON")]),
        ("call me {name} if you have any questions.", [("name", "priya", "PERSON")]),
        ("this is {name} from engineering.", [("name", "rohit", "PERSON")]),
        ("Hello, {name} here.", [("name", "sunil", "PERSON")]),

        # 1.2 Lowercase names in sentence contexts
        ("Please send the quarterly file to {name} today.", [("name", "anand", "PERSON")]),
        ("I had an architectural review meeting with {name} yesterday.", [("name", "deepak", "PERSON")]),
        ("The deployment was approved by {name} this morning.", [("name", "priya", "PERSON")]),
        ("{name} will be presenting the quarterly revenue report.", [("name", "rohit", "PERSON")]),
        ("Check with {name} before pushing the database changes.", [("name", "sunil", "PERSON")]),

        # 1.3 Alternating / mixed case (mock case)
        ("my name is {name}.", [("name", "AnAnD", "PERSON")]),
        ("Please email {name} regarding the project roadmap.", [("name", "SuNiL", "PERSON")]),
        ("I discussed the incident with {name} earlier.", [("name", "RoHiT", "PERSON")]),
        ("Call me {name} on my extension.", [("name", "PrIyA", "PERSON")]),
        ("{name} confirmed that the server migration is complete.", [("name", "AnAnD", "PERSON")]),

        # 1.4 All uppercase names
        ("my name is {name}.", [("name", "ANAND", "PERSON")]),
        ("Tell {name} to review the pull request.", [("name", "ROHIT", "PERSON")]),
        ("The release notes were authored by {name}.", [("name", "DEEPAK", "PERSON")]),
        ("Forward the legal contract to {name} for signing.", [("name", "PRIYA", "PERSON")]),
        ("{name} signed off on the security compliance checklist.", [("name", "SUNIL", "PERSON")]),

        # =====================================================================
        # 2. TYPO & SHORTHAND CREDENTIALS (Problem 1) (15 prompts)
        # =====================================================================
        ("passwrd: {p1}", [("p1", "AlphaSecret#99", "PASSWORD")]),
        ("my passwrd is {p1}", [("p1", "MyTopSecret99!", "PASSWORD")]),
        ("enter your passwrd: {p1} to authenticate.", [("p1", "WinterPass#2024", "PASSWORD")]),
        ("p@ssword: {p1}", [("p1", "K9#mP2$vL8", "PASSWORD")]),
        ("my p@ssword is {p1} for the staging cluster.", [("p1", "CloudAdmin#772", "PASSWORD")]),
        ("pswd: {p1}", [("p1", "SuperSecr3t!", "PASSWORD")]),
        ("my pswd is {p1} on the portal.", [("p1", "VaultKey#9182", "PASSWORD")]),
        ("pwd: {p1}", [("p1", "TestPass1234#", "PASSWORD")]),
        ("my pwd is {p1}", [("p1", "SecureDev#5544", "PASSWORD")]),
        ("temp pwd: {p1}", [("p1", "TempCode#881", "PASSWORD")]),
        ("PassWord: {p1}", [("p1", "RiverStone#9472", "PASSWORD")]),
        ("pAsSwOrD: {p1}", [("p1", "MixedCase#9911", "PASSWORD")]),
        ("PASSWORD: {p1}", [("p1", "ALLCAPSPASS#12", "PASSWORD")]),
        ("credentials: {p1}", [("p1", "admin/SecretRoot#88", "PASSWORD")]),
        ("passcode: {p1} for building access.", [("p1", "948271", "PASSWORD")]),

        # =====================================================================
        # 3. TYPO & SHORTHAND BANK ACCOUNTS (Problem 1) (15 prompts)
        # =====================================================================
        ("acc no: {acc}", [("acc", "123456789012", "BANK_ACCOUNT")]),
        ("my acc no is {acc}", [("acc", "987654321098", "BANK_ACCOUNT")]),
        ("savings acc no: {acc}", [("acc", "554433221100", "BANK_ACCOUNT")]),
        ("act no: {acc}", [("acc", "987654321012", "BANK_ACCOUNT")]),
        ("my act no is {acc}", [("acc", "112233445566", "BANK_ACCOUNT")]),
        ("bank act: {acc}", [("acc", "445566778899", "BANK_ACCOUNT")]),
        ("a/c no: {acc}", [("acc", "998877665544", "BANK_ACCOUNT")]),
        ("my a/c no is {acc}", [("acc", "123123123123", "BANK_ACCOUNT")]),
        ("bank a/c: {acc}", [("acc", "321321321321", "BANK_ACCOUNT")]),
        ("ac no: {acc}", [("acc", "665544332211", "BANK_ACCOUNT")]),
        ("accnt no: {acc}", [("acc", "778899001122", "BANK_ACCOUNT")]),
        ("customer acc no: {acc}", [("acc", "889900112233", "BANK_ACCOUNT")]),
        ("transfer funds to acc no {acc} immediately.", [("acc", "456789012345", "BANK_ACCOUNT")]),
        # Negative controls: Normal English words / legal acts
        ("Act 1986 of Parliament applies to this legal case.", []),
        ("We need to act 4 times to complete the workflow cycle.", []),

        # =====================================================================
        # 4. MATHEMATICAL & ALGORITHMIC NUMBER QUERIES (12 prompts)
        # Zero-PII Negative Controls — abstract numbers must NOT be masked as phone/id
        # =====================================================================
        ("find the sum of digits of a number 9845124523", []),
        ("What is the last 4 digits of 456245627895", []),
        ("Calculate the square root of 9845124523", []),
        ("Find the prime factors of 9876543210", []),
        ("Reverse the digits of 9123456780", []),
        ("Count the number of digits in 452345624562", []),
        ("Write a Python regex for matching a 10-digit number like 9876543210", []),
        ("Sort the digits of 9845124523 in ascending order", []),
        ("Compute 9845124523 modulo 97", []),
        ("Show the binary representation of 9876543210", []),
        ("Calculate the sum of the first and last digit of 8765432109", []),
        ("Check whether 9845124523 is divisible by 11", []),

        # =====================================================================
        # 5. INDIAN IDENTITY DOCUMENTS (15 prompts)
        # =====================================================================
        ("my adhr no is {id}", [("id", "456245627895", "NATIONAL_ID")]),
        ("my adhar number is {id}", [("id", "987654321098", "NATIONAL_ID")]),
        ("aadhaar no: {id}", [("id", "345678901234", "NATIONAL_ID")]),
        ("uidai id: {id}", [("id", "567890123456", "NATIONAL_ID")]),
        ("my aadhaar number is {id}, please verify it.", [("id", "4562 4562 7895", "NATIONAL_ID")]),
        ("identity card number {id} was submitted for KYC.", [("id", "9876 5432 1098", "NATIONAL_ID")]),
        ("my pan no is {pan}", [("pan", "ABCDE1234F", "TAX_ID")]),
        ("pan card no: {pan}", [("pan", "BKRPD5432M", "TAX_ID")]),
        ("Employee PAN {pan} was recorded in the payroll database.", [("pan", "XYZPA9876K", "TAX_ID")]),
        ("permanent account number: {pan}", [("pan", "CRKPS1234L", "TAX_ID")]),
        ("my dl no is {dl}", [("dl", "DL-1420110012345", "DRIVER_LICENSE")]),
        ("driving license: {dl}", [("dl", "MH-0220190054321", "DRIVER_LICENSE")]),
        ("voter id: {vid}", [("vid", "WBD1234567", "NATIONAL_ID")]),
        ("epic card no: {vid}", [("vid", "ABC9876543", "NATIONAL_ID")]),
        # Negative control: Coding / regex prompt with zero ownership
        ("Write regex to validate PAN like ABCDE1234F", []),

        # =====================================================================
        # 6. FOREIGN PHONE FORMATS (10 prompts)
        # =====================================================================
        ("Call me at {phone} for the US client briefing.", [("phone", "+1 (202) 555-0199", "PHONE")]),
        ("Contact {loc} office on {phone}.", [("loc", "Chicago", "LOCATION"), ("phone", "312-555-0143", "PHONE")]),
        ("Support hotline is {phone}.", [("phone", "1-800-555-0199", "PHONE")]),
        ("{loc} support is {phone} if you need assistance.", [("loc", "UK", "LOCATION"), ("phone", "020 7946 0958", "PHONE")]),
        ("Reach {loc} dispatch at {phone}.", [("loc", "London", "LOCATION"), ("phone", "01632 960001", "PHONE")]),
        ("{loc} mobile number is {phone}.", [("loc", "UK", "LOCATION"), ("phone", "07911 123456", "PHONE")]),
        ("Contact {name} on {phone} regarding the {loc} shipment.", [("name", "Tariq", "PERSON"), ("phone", "+971 50 123 4567", "PHONE"), ("loc", "Dubai", "LOCATION")]),
        ("Reach {loc} office on {phone}.", [("loc", "UAE", "LOCATION"), ("phone", "050 123 4567", "PHONE")]),
        ("Call {loc} headquarters on {phone}.", [("loc", "Sydney", "LOCATION"), ("phone", "02 9876 5432", "PHONE")]),
        ("Contact {loc} operations at {phone}.", [("loc", "Paris", "LOCATION"), ("phone", "01 42 68 55 55", "PHONE")]),

        # =====================================================================
        # 7. LIFECYCLE-ANCHORED DATES VS GENERIC RELATIVE DATES (8 prompts)
        # =====================================================================
        ("{date} is my birthday.", [("date", "Today", "DATE")]),
        ("My birthday is {date}.", [("date", "today", "DATE")]),
        ("{date} is his birthday.", [("date", "Tomorrow", "DATE")]),
        ("{date} was my anniversary.", [("date", "Yesterday", "DATE")]),
        # Negative controls: Generic temporal words without biographical context
        ("What is the weather today in {loc}?", [("loc", "New York", "LOCATION")]),
        ("Can you summarize today's technology news?", []),
        ("I need to finish this report today before evening.", []),
        ("Tomorrow will be a busy day for our team.", []),

        # =====================================================================
        # 8. GAZETTEER: PUBLIC FIGURES VS PRIVATE INDIVIDUALS (5 prompts)
        # =====================================================================
        ("Who is {name} and what are his major policies?", [("name", "Narendra Modi", "PERSON")]),
        ("What are the latest AI products announced by {org}?", [("org", "Microsoft", "ORGANIZATION")]),
        ("HR coordinator {name} attended the onboarding session.", [("name", "Priya Nair", "PERSON")]),
        ("Software developer {name} submitted the pull request.", [("name", "Rahul Verma", "PERSON")]),
        ("Delivery contract was awarded to {org}.", [("org", "Acme Corp", "ORGANIZATION")]),
    ]

    return specs


def main():
    print("=" * 70)
    print("  PromptShield Benchmark Dataset Expander (500 -> 600 Prompts)")
    print("=" * 70)

    if not GT_FILE.exists():
        print(f"[FATAL] Ground truth file not found at: {GT_FILE}")
        sys.exit(1)

    # Use the 500-prompt baseline if backup exists, otherwise create it
    if BACKUP_FILE.exists():
        with open(BACKUP_FILE, encoding="utf-8") as f:
            existing_data = json.load(f)[:500]
    else:
        shutil.copyfile(GT_FILE, BACKUP_FILE)
        print(f"[INFO] Backed up current 500 prompts to: {BACKUP_FILE}")
        with open(GT_FILE, encoding="utf-8") as f:
            existing_data = json.load(f)[:500]

    print(f"[INFO] Loaded existing baseline: {len(existing_data)} prompts.")

    specs = get_100_new_prompt_specs()
    print(f"[INFO] Defined {len(specs)} new prompt specifications.")

    start_idx = len(existing_data) + 1
    new_prompts = []

    for i, (template, entities_to_place) in enumerate(specs):
        prompt_id = f"test_{start_idx + i:03d}"
        
        try:
            rendered = build_prompt_with_entities(template, entities_to_place)
        except Exception as e:
            print(f"[ERROR] Failed building prompt {prompt_id}: {e}")
            sys.exit(1)

        new_entry = {
            "id": prompt_id,
            "text": rendered["text"],
            "entities": rendered["entities"]
        }
        new_prompts.append(new_entry)

    combined_data = existing_data + new_prompts
    print(f"[INFO] Combined total prompts: {len(combined_data)}")

    # Strict full validation of every entity offset in all prompts
    print("[INFO] Verifying character offsets across all prompts...")
    errors = []
    total_entities = 0
    for item in combined_data:
        text = item["text"]
        for e in item.get("entities", []):
            total_entities += 1
            actual = text[e["start"]:e["end"]]
            if actual != e["text"]:
                errors.append(f"[{item['id']}] Mismatch: expected '{e['text']}', got '{actual}' at [{e['start']}:{e['end']}] in '{text}'")

    if errors:
        print(f"[FATAL] Found {len(errors)} offset mismatches!")
        for err in errors[:10]:
            print(f"  {err}")
        sys.exit(1)

    print(f"[INFO] All {total_entities} entity offsets strictly verified! 100% VALID.")

    # Save to ground_truth.json
    with open(GT_FILE, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Updated {GT_FILE} with exactly {len(combined_data)} prompts.")


if __name__ == "__main__":
    main()
