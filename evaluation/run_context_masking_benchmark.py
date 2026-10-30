#!/usr/bin/env python3
"""
PromptShield AI — Context-Aware Masking Benchmark (Phase 4 & End-to-End Evaluation)
===================================================================================
Evaluates the full end-to-end context-aware sanitization pipeline:
Prompt -> SensitiveDataDetector -> ContextAnalyzer -> PolicyEngine -> SemanticMasker

Measures:
1. Context-Aware Decision Accuracy (MASK vs. RETAIN vs. USER_APPROVAL)
2. Personal PII Protection (Sensitivity / Safety)
3. Public Knowledge & Casual Context Preservation (Utility / Non-Over-Masking)
4. Value-Required Target of Analysis Handling
5. End-to-End Sanitization Latency (ms)
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core import PromptShieldCore
from backend.models import EntityType, RoleCategory
from backend.policy_engine import PolicyDecision


@dataclass
class ContextTestCase:
    test_id: str
    category: str
    prompt: str
    target_entity: str
    expected_decision: PolicyDecision
    should_be_in_sanitized: bool  # True if target_entity literal should remain in sanitized prompt
    description: str


CONTEXT_BENCHMARK_CASES: List[ContextTestCase] = [
    # -------------------------------------------------------------------------
    # Category 1: First-Party Personal PII (Should MASK)
    # -------------------------------------------------------------------------
    ContextTestCase(
        test_id="ctx_001",
        category="Personal PII",
        prompt="My birthday is today",
        target_entity="today",
        expected_decision=PolicyDecision.MASK,
        should_be_in_sanitized=False,
        description="First-party birthday reference must be masked as DOB",
    ),
    ContextTestCase(
        test_id="ctx_002",
        category="Personal PII",
        prompt="My name is Julie Johnson and my email is julie.johnson@example.com",
        target_entity="julie.johnson@example.com",
        expected_decision=PolicyDecision.MASK,
        should_be_in_sanitized=False,
        description="First-party personal email must be masked",
    ),
    ContextTestCase(
        test_id="ctx_003",
        category="Personal PII",
        prompt="Call me on my personal phone +1 555-123-4567 regarding the account",
        target_entity="+1 555-123-4567",
        expected_decision=PolicyDecision.MASK,
        should_be_in_sanitized=False,
        description="First-party personal phone number must be masked",
    ),
    ContextTestCase(
        test_id="ctx_004",
        category="Personal PII",
        prompt="My password is SuperSecret#2024 for logging into the portal",
        target_entity="SuperSecret#2024",
        expected_decision=PolicyDecision.MASK,
        should_be_in_sanitized=False,
        description="Authentication credential must always be masked",
    ),
    ContextTestCase(
        test_id="ctx_005",
        category="Personal PII",
        prompt="Please wire the refund to my bank account 001234567890",
        target_entity="001234567890",
        expected_decision=PolicyDecision.MASK,
        should_be_in_sanitized=False,
        description="First-party bank account number must be masked",
    ),

    # -------------------------------------------------------------------------
    # Category 2: Casual Conversation & Ambiguous Words (Should RETAIN)
    # -------------------------------------------------------------------------
    ContextTestCase(
        test_id="ctx_006",
        category="Casual Context",
        prompt="Today I need to play football with my college friends",
        target_entity="Today",
        expected_decision=PolicyDecision.RETAIN,
        should_be_in_sanitized=True,
        description="Casual time expression in recreational context must be retained",
    ),
    ContextTestCase(
        test_id="ctx_007",
        category="Casual Context",
        prompt="Yesterday was a great conference on distributed systems architecture",
        target_entity="Yesterday",
        expected_decision=PolicyDecision.RETAIN,
        should_be_in_sanitized=True,
        description="Casual past reference must be retained unchanged",
    ),
    ContextTestCase(
        test_id="ctx_008",
        category="Casual Context",
        prompt="Deploy the application on Ubuntu 22.04 LTS using Docker containers",
        target_entity="Ubuntu 22.04 LTS",
        expected_decision=PolicyDecision.RETAIN,
        should_be_in_sanitized=True,
        description="Operating system / environment software name must be retained",
    ),
    ContextTestCase(
        test_id="ctx_009",
        category="Casual Context",
        prompt="Configure nginx web server for IPv6 protocol routing",
        target_entity="IPv6",
        expected_decision=PolicyDecision.RETAIN,
        should_be_in_sanitized=True,
        description="Network protocol standard acronym must be retained",
    ),

    # -------------------------------------------------------------------------
    # Category 3: Public Knowledge & Famous Figures (Should RETAIN)
    # -------------------------------------------------------------------------
    ContextTestCase(
        test_id="ctx_010",
        category="Public Knowledge",
        prompt="Tell me about Elon Musk and the founding of SpaceX",
        target_entity="Elon Musk",
        expected_decision=PolicyDecision.RETAIN,
        should_be_in_sanitized=True,
        description="Public celebrity / CEO inquiry must be retained for accuracy",
    ),
    ContextTestCase(
        test_id="ctx_011",
        category="Public Knowledge",
        prompt="Sundar Pichai leads Google as chief executive officer",
        target_entity="Sundar Pichai",
        expected_decision=PolicyDecision.RETAIN,
        should_be_in_sanitized=True,
        description="Public corporate leadership inquiry must be retained",
    ),
    ContextTestCase(
        test_id="ctx_012",
        category="Public Knowledge",
        prompt="What is the capital of France? Is it Paris?",
        target_entity="Paris",
        expected_decision=PolicyDecision.RETAIN,
        should_be_in_sanitized=True,
        description="General geographic knowledge question must retain location",
    ),
    ContextTestCase(
        test_id="ctx_013",
        category="Public Knowledge",
        prompt="Amazon and Microsoft dominate the global enterprise cloud market",
        target_entity="Microsoft",
        expected_decision=PolicyDecision.RETAIN,
        should_be_in_sanitized=True,
        description="Public corporate references in industry context must be retained",
    ),

    # -------------------------------------------------------------------------
    # Category 4: Target of Analysis / Value Required (USER_APPROVAL)
    # -------------------------------------------------------------------------
    ContextTestCase(
        test_id="ctx_014",
        category="Target of Analysis",
        prompt="How strong is my password Apple#123?",
        target_entity="Apple#123",
        expected_decision=PolicyDecision.USER_APPROVAL,
        should_be_in_sanitized=False,  # Unapproved by default, so placeholder inserted
        description="Password strength analysis requires user approval",
    ),
    ContextTestCase(
        test_id="ctx_015",
        category="Target of Analysis",
        prompt="How many digits are in my password Secret9942?",
        target_entity="Secret9942",
        expected_decision=PolicyDecision.USER_APPROVAL,
        should_be_in_sanitized=False,
        description="Character count on password requires user approval",
    ),
    ContextTestCase(
        test_id="ctx_016",
        category="Target of Analysis",
        prompt="Calculate my age if my birth date is 15/08/1990",
        target_entity="15/08/1990",
        expected_decision=PolicyDecision.USER_APPROVAL,
        should_be_in_sanitized=False,
        description="Mathematical calculation on birth date requires user approval",
    ),

    # -------------------------------------------------------------------------
    # Category 5: Multi-Entity Enterprise Scenarios
    # -------------------------------------------------------------------------
    ContextTestCase(
        test_id="ctx_017",
        category="Enterprise Context",
        prompt="Email Robert Taylor at robert.taylor@example.com regarding project Alpha",
        target_entity="Robert Taylor",
        expected_decision=PolicyDecision.MASK,
        should_be_in_sanitized=False,
        description="Recipient person in enterprise outreach email must be masked",
    ),
    ContextTestCase(
        test_id="ctx_018",
        category="Enterprise Context",
        prompt="Dear Kavya Reddy, your ticket SR-1091 has been resolved.",
        target_entity="Kavya Reddy",
        expected_decision=PolicyDecision.MASK,
        should_be_in_sanitized=False,
        description="Addressed customer name in support notification must be masked",
    ),
    ContextTestCase(
        test_id="ctx_019",
        category="Enterprise Context",
        prompt="The server is at IP 10.0.0.1 and connection string postgresql://user:pass@db:5432/main",
        target_entity="postgresql://user:pass@db:5432/main",
        expected_decision=PolicyDecision.MASK,
        should_be_in_sanitized=False,
        description="Credential-bearing connection string must be masked",
    ),
    ContextTestCase(
        test_id="ctx_020",
        category="Enterprise Context",
        prompt="Must mention Infosys in the draft partnership proposal",
        target_entity="Infosys",
        expected_decision=PolicyDecision.RETAIN,
        should_be_in_sanitized=True,
        description="Explicit user directive to mention an organization must be retained",
    ),
]


def run_benchmark():
    print("=" * 75)
    print("  PROMPTSHIELD CONTEXT-AWARE MASKING BENCHMARK (Phase 4)")
    print("=" * 75)

    print("\n[INFO] Initializing PromptShieldCore...")
    t0 = time.perf_counter()
    core = PromptShieldCore()
    t1 = time.perf_counter()
    print(f"[INFO] Initialized in {(t1 - t0) * 1000.0:.1f} ms.\n")

    total_cases = len(CONTEXT_BENCHMARK_CASES)
    passed_cases = 0
    decision_matches = 0
    sanitization_matches = 0
    latencies: List[float] = []

    print(f"{'ID':<8} | {'Category':<18} | {'Decision':<14} | {'Policy Match':<12} | {'Sanitization':<12} | {'Latency':<9}")
    print("-" * 80)

    category_stats: Dict[str, Dict[str, int]] = {}

    for case in CONTEXT_BENCHMARK_CASES:
        cat = case.category
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "decision_ok": 0, "sanitize_ok": 0}
        category_stats[cat]["total"] += 1

        t_start = time.perf_counter()
        policy_report = core.evaluate_policy(case.prompt)
        sanitized_res = core.sanitize(case.prompt)
        t_end = time.perf_counter()

        latency_ms = (t_end - t_start) * 1000.0
        latencies.append(latency_ms)

        # 1. Check policy decision for target entity
        target_policy = None
        for p in policy_report.entity_policies:
            if case.target_entity.lower() in p.entity_text.lower() or p.entity_text.lower() in case.target_entity.lower():
                target_policy = p
                break

        actual_decision = target_policy.decision if target_policy else PolicyDecision.RETAIN
        policy_ok = (actual_decision == case.expected_decision)
        if policy_ok:
            decision_matches += 1
            category_stats[cat]["decision_ok"] += 1

        # 2. Check sanitization result
        sanitized_text = sanitized_res.sanitized_prompt
        if case.should_be_in_sanitized:
            sanitize_ok = (case.target_entity.lower() in sanitized_text.lower())
        else:
            # Should be replaced by a placeholder
            sanitize_ok = (case.target_entity.lower() not in sanitized_text.lower())

        if sanitize_ok:
            sanitization_matches += 1
            category_stats[cat]["sanitize_ok"] += 1

        case_passed = policy_ok and sanitize_ok
        if case_passed:
            passed_cases += 1

        status_policy = "PASS" if policy_ok else f"FAIL ({actual_decision.value})"
        status_sanitize = "PASS" if sanitize_ok else "FAIL"

        print(f"{case.test_id:<8} | {case.category:<18} | {actual_decision.value:<14} | {status_policy:<12} | {status_sanitize:<12} | {latency_ms:6.1f} ms")

    # Summary Statistics
    print("=" * 80)
    print("  CONTEXT-AWARE BENCHMARK SUMMARY")
    print("=" * 80)

    accuracy_policy = (decision_matches / total_cases) * 100.0
    accuracy_sanitize = (sanitization_matches / total_cases) * 100.0
    overall_pass_rate = (passed_cases / total_cases) * 100.0
    mean_latency = sum(latencies) / len(latencies)
    median_latency = sorted(latencies)[len(latencies) // 2]
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

    print(f"Total Test Cases:            {total_cases}")
    print(f"Policy Decision Accuracy:    {accuracy_policy:.1f}% ({decision_matches}/{total_cases})")
    print(f"Sanitization Correctness:    {accuracy_sanitize:.1f}% ({sanitization_matches}/{total_cases})")
    print(f"Overall End-to-End Pass:     {overall_pass_rate:.1f}% ({passed_cases}/{total_cases})")
    print(f"Mean Pipeline Latency:       {mean_latency:.1f} ms")
    print(f"Median Pipeline Latency:     {median_latency:.1f} ms")
    print(f"P95 Pipeline Latency:        {p95_latency:.1f} ms")

    print("\nPER-CATEGORY ACCURACY:")
    print("-" * 55)
    for cat, stats in category_stats.items():
        tot = stats["total"]
        dec = stats["decision_ok"]
        san = stats["sanitize_ok"]
        print(f"  {cat:<22}: Policy={dec}/{tot} ({dec/tot*100:.0f}%) | Masking={san}/{tot} ({san/tot*100:.0f}%)")

    print("\nCONCLUSION:")
    if overall_pass_rate >= 90.0:
        print("  [SUCCESS] Context-aware security layer operates with high precision and safe selective masking.")
    else:
        print("  [WARN] Context-aware layer has edge cases requiring policy tuning.")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
