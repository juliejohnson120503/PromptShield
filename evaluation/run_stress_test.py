"""
PromptShield AI — Synthetic Stress Test Runner (Phase 1–3 Verification)
Runs the full 20-entity onboarding prompt through the production pipeline:
GLiNER2PIIDetector -> SensitiveDataDetector -> EntityFusionEngine -> ContextAnalyzer -> PolicyEngine.
"""

import os
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.core import PromptShieldCore

prompt = (
    "I am preparing a confidential onboarding record for employee Elena Rodriguez. "
    "Her passport number is X12345678, driver's license number is D123-4567-8901, "
    "and national identification number is 784-1987-1234567-1. Her date of birth "
    "is 17 September 1994 and her mother's maiden name is Fernandez. Her IPv4 "
    "address is 192.168.14.27, IPv6 address is "
    "2001:db8:85a3::8a2e:370:7334, and device MAC address is "
    "3C:52:82:4A:91:BF. The server can be accessed using username erodriguez, "
    "password RiverStone#9472, API key sk_test_8Kx29Qa7Lm4Np2Vz, access token "
    "ghp_X7a9Bc2De4Fg6Hi8Jk0Lm2No4Pq6Rs8Tu0Vw, and database connection string "
    "postgresql://erodriguez:RiverStone9472@db.internal.example:5432/payroll. "
    "Her bank account number is 001234567890, IBAN is "
    "GB82WEST12345698765432, SWIFT/BIC is DEUTDEFF500, and tax identification "
    "number is 123-45-6789. Her medical record number is MRN-84729163 and health "
    "insurance member ID is HIN-72819456. She will attend a conference in "
    "Singapore next month and works for Orion Analytics. Write a professional "
    "summary of her onboarding requirements without exposing information that is "
    "unnecessary for the task."
)

def run():
    print("=" * 80)
    print("INITIALIZING PROMPTSHIELD CORE...")
    print("=" * 80)
    shield = PromptShieldCore()

    print("\n" + "=" * 80)
    print("PHASE 1: DETECTION & FUSION OUTPUT")
    print("=" * 80)
    entities = shield.detector.detect(prompt)
    for i, e in enumerate(entities, 1):
        print(f"[{i:2d}] {e.text:<45} | {e.entity_type.value:<20} | {e.source:<12} | conf={e.confidence:.2f} | span=[{e.start}:{e.end}]")

    print("\n" + "=" * 80)
    print("PHASE 2: CONTEXT ANALYSIS OUTPUT")
    print("=" * 80)
    context_res = shield.analyze_context(prompt)
    print(f"Classified Task: {context_res.task_type.value} (conf={context_res.task_confidence:.2f})")
    for i, c in enumerate(context_res.entity_contexts, 1):
        print(f"[{i:2d}] {c.entity_text:<45} | {c.entity_type.value:<20} | {c.role_category.value:<22} | 1stParty={str(c.is_first_party):<5} | Pub={str(c.is_public_knowledge):<5} | Cue: {c.context_cue}")

    print("\n" + "=" * 80)
    print("PHASE 3: POLICY ENGINE REPORT")
    print("=" * 80)
    report = shield.evaluate_policy(prompt)
    print(f"Overall Risk Level: {report.overall_risk_level.value}")
    print(f"Summary: Total={len(report.entity_policies)} | MASK={len(report.entities_to_mask)} | RETAIN={len(report.entities_to_retain)} | USER_APPROVAL={len(report.entities_needing_approval)}")
    print("-" * 80)
    for i, p in enumerate(report.entity_policies, 1):
        print(f"[{i:2d}] {p.entity_text:<45} | {p.entity_type.value:<20} | {p.decision.value:<8} | {p.risk_level.value:<8} (score={p.risk_score:.2f})")
        print(f"     Reason: {p.reason}")

    print("\n" + "=" * 80)
    print("VERIFICATION OF SPECIFIC EXPECTED ENTITIES")
    print("=" * 80)
    expected_to_mask = [
        "Elena Rodriguez",
        "X12345678",
        "D123-4567-8901",
        "784-1987-1234567-1",
        "17 September 1994",
        "Fernandez",
        "192.168.14.27",
        "2001:db8:85a3::8a2e:370:7334",
        "3C:52:82:4A:91:BF",
        "erodriguez",
        "RiverStone#9472",
        "sk_test_8Kx29Qa7Lm4Np2Vz",
        "ghp_X7a9Bc2De4Fg6Hi8Jk0Lm2No4Pq6Rs8Tu0Vw",
        "postgresql://erodriguez:RiverStone9472@db.internal.example:5432/payroll",
        "001234567890",
        "GB82WEST12345698765432",
        "DEUTDEFF500",
        "123-45-6789",
        "MRN-84729163",
        "HIN-72819456",
    ]

    expected_to_retain = [
        "Singapore",
        "next month",
        "Orion Analytics",
    ]

    detected_texts = {p.entity_text: p for p in report.entity_policies}

    print("\n--- MASK AUDIT ---")
    for exp in expected_to_mask:
        found = detected_texts.get(exp)
        if found:
            status = "PASS" if found.decision.value == "MASK" else f"FAIL (decision={found.decision.value})"
            print(f"  [{status}] {exp} -> {found.entity_type.value} -> {found.decision.value}")
        else:
            matched = [k for k in detected_texts if exp in k or k in exp]
            print(f"  [MISSING/PARTIAL] {exp} -> candidates: {matched}")

    print("\n--- RETAIN AUDIT ---")
    for exp in expected_to_retain:
        found = detected_texts.get(exp)
        if found:
            status = "PASS" if found.decision.value == "RETAIN" else f"FAIL (decision={found.decision.value})"
            print(f"  [{status}] {exp} -> {found.entity_type.value} -> {found.decision.value}")
        else:
            matched = [k for k in detected_texts if exp in k or k in exp]
            print(f"  [NOT DETECTED AS ENTITY (Retained by default)] {exp}")

    print("\n--- DESCRIPTOR LABELS AUDIT ---")
    for label in ["MAC", "IBAN", "BIC", "SWIFT"]:
        if label in detected_texts:
            print(f"  [FAIL] Label '{label}' detected as sensitive entity: {detected_texts[label]}")
        else:
            print(f"  [PASS] Label '{label}' NOT present in sensitive entities")

if __name__ == "__main__":
    run()
