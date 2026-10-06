"""
Interactive Demonstration CLI for PromptShield AI (Phases 1–6).
Showcases:
- Phase 1: Structured & NER entity detection, normalization, baseline mapping.
- Phase 2: Task classification and Contextual Role analysis (e.g. Julie vs Elon Musk).
- Phase 3: Risk assessment and Privacy Policy decisions (MASK / RETAIN / USER_APPROVAL).
- Phase 4: Context-aware semantic masking.
- Phase 5: Controlled response restoration & LLM integration.
- Phase 6: Available as a REST API Service via `python run_server.py`.
"""


import sys
import os

# Ensure UTF-8 output on Windows consoles
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core import PromptShieldCore
from backend.policy_engine import PolicyDecision, RiskLevel
from backend.models import SemanticMaskingResult


SAMPLE_PROMPTS = [
    (
        "Personal Identifier vs Public Figure (Julie vs Elon Musk)",
        "My name is Julie Johnson and my email is julie@example.com. Tell me about Elon Musk and SpaceX."
    ),
    (
        "Task-Relevant Business Outreach",
        "My name is John Mathew and my email is john@gmail.com. "
        "Write a professional email to ABC Technologies regarding my application for the London office."
    ),
    (
        "Credentials & API Keys",
        "Deploy the backend using AWS key AKIAIOSFODNN7EXAMPLE, "
        "OpenAI key sk-live-abc1234567890def12345678, and database password='DbMasterPass!99'."
    ),
    (
        "Financial & Contact Information",
        "Please charge card 4111-1111-1111-1111 for renewal. "
        "Send receipt to billing@innovate.org or phone +91 98765 43210 on 2025-04-10."
    ),
    (
        "Pure Public Knowledge Q&A",
        "What is the capital of France and who founded Microsoft?"
    ),
    (
        "Informal name with contraction (i am / i'm)",
        "i'm amina and my email is amina@gmail.com, write a formal letter to elon musk."
    ),
    (
        "Abbreviations & Shortforms (i m julie, psswrd)",
        "i m julie and my psswrd is 98765432"
    ),
    (
        "Customer Support Multi-Entity (Anjali Nair, Infosys, Kochi, ORD-78291)",
        "Yesterday, Rahul Menon from Infosys contacted me about customer Anjali Nair, whose order ORD-78291 was shipped to Kochi. Her email is anjali.nair@gmail.com and she can be reached at +91 9876543210. The support team recorded her customer ID as CUST-10458 and the request came from IP address 192.168.1.45."
    ),
]


# Colour / style helpers (ANSI — work on most modern terminals)
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"

DECISION_COLOURS = {
    PolicyDecision.MASK:          RED,
    PolicyDecision.RETAIN:        GREEN,
    PolicyDecision.USER_APPROVAL: YELLOW,
    PolicyDecision.REVIEW:        CYAN,
}

RISK_COLOURS = {
    RiskLevel.CRITICAL: RED,
    RiskLevel.HIGH:     "\033[31m",     # dark red
    RiskLevel.MEDIUM:   YELLOW,
    RiskLevel.LOW:      GREEN,
    RiskLevel.NONE:     DIM,
}


def _colour(text: str, col: str) -> str:
    return f"{col}{text}{RESET}"


def print_banner():
    w = 90
    print("=" * w)
    print("  PROMPTSHIELD AI — Complete Security Pipeline (Phases 1–6)".center(w))
    print("=" * w)
    print("  A Context-Aware Security Layer for Safe, Responsible & Enterprise-Ready AI Systems".center(w))
    print("-" * w)
    print(("  Pipeline: Detect (P1) -> Context (P2) -> Policy (P3) -> Mask (P4) -> Restore (P5) -> REST API (P6)").center(w))
    print("  Upcoming: Browser Extension (P7) | Benchmark Dashboard (P8)".center(w))
    print("=" * w)


def display_results(shield: PromptShieldCore, prompt: str):
    # Run all phases
    policy_report  = shield.evaluate_policy(prompt)   # P1+P2+P3
    baseline_res   = shield.sanitize_baseline(prompt)  # P1 baseline for reference
    p4_result      = shield.sanitize(prompt)           # Phase 4 context-aware masking

    W = 90
    print("\n" + "-" * W)

    # -- [1] Original Prompt -------------------------------------------------
    print(f"\n{BOLD}[1] ORIGINAL PROMPT:{RESET}")
    print(f'    "{prompt}"')

    # -- [2] Phase 1: Detected Entities --------------------------------------
    print(f"\n{BOLD}[2] PHASE 1 — DETECTED ENTITIES:{RESET}")
    if not baseline_res.entities:
        print("    (No sensitive or named entities detected)")
    else:
        hdr = f"    {'TYPE':<15} | {'TEXT':<22} | {'NORMALIZED':<20} | {'CONF':<6} | DETECTOR"
        print(hdr)
        print("    " + "─" * 78)
        for e in baseline_res.entities:
            print(
                f"    {e.entity_type.value:<15} | {e.text[:21]:<22} | "
                f"{e.normalized_value[:19]:<20} | {e.confidence:<6.2f} | {e.detector}"
            )

    # -- [3] Phase 2: Task & Context -----------------------------------------
    print(f"\n{BOLD}[3] PHASE 2 — TASK & CONTEXT UNDERSTANDING:{RESET}")
    print(f"    Task: {CYAN}{policy_report.task_type}{RESET}  "
          f"(Confidence: {policy_report.task_confidence:.2f})")

    ctx_result = shield.analyze_context(prompt)
    if not ctx_result.entity_contexts:
        print("    (No entity contexts evaluated)")
    else:
        hdr = (f"    {'ENTITY':<18} | {'ROLE':<22} | "
               f"{'1ST':<5} | {'PUB':<5} | {'TASK':<5} | CUE")
        print(hdr)
        print("    " + "─" * 80)
        for cr in ctx_result.entity_contexts:
            print(
                f"    {cr.entity_text[:17]:<18} | {cr.role_category.value:<22} | "
                f"{'Y' if cr.is_first_party else 'N':<5} | "
                f"{'Y' if cr.is_public_knowledge else 'N':<5} | "
                f"{'Y' if cr.is_task_relevant else 'N':<5} | {cr.context_cue[:28]}"
            )

    # -- [4] Phase 3: Risk & Policy ------------------------------------------
    print(f"\n{BOLD}[4] PHASE 3 — RISK ASSESSMENT & POLICY DECISIONS:{RESET}")
    overall_col = RISK_COLOURS.get(policy_report.overall_risk_level, "")
    print(f"    Overall Risk: {_colour(policy_report.overall_risk_level.value, overall_col)}")

    if not policy_report.entity_policies:
        print("    (No entities evaluated)")
    else:
        hdr = (f"    {'ENTITY':<18} | {'DECISION':<14} | {'RISK':<8} | "
               f"{'SCORE':<6} | REASON")
        print(hdr)
        print("    " + "─" * 84)
        for p in policy_report.entity_policies:
            dec_col  = DECISION_COLOURS.get(p.decision, "")
            risk_col = RISK_COLOURS.get(p.risk_level, "")
            print(
                f"    {p.entity_text[:17]:<18} | "
                f"{_colour(p.decision.value, dec_col):<23} | "
                f"{_colour(p.risk_level.value, risk_col):<17} | "
                f"{p.risk_score:<6.2f} | {p.reason[:38]}"
            )

    # -- [5] Phase 1 Baseline Masking (reference) ----------------------------
    print(f"\n{BOLD}[5] PHASE 1 BASELINE MASKING (blind — reference only):{RESET}")
    print(f'    "{baseline_res.sanitized_prompt}"')
    print(f"\n{BOLD}[6] LOCAL MAPPING (phase 1 store, never sent to LLM):{RESET}")
    if not baseline_res.mapping:
        print("    (Empty mapping store)")
    else:
        for placeholder, secret in baseline_res.mapping.items():
            print(f"    {placeholder:<18} ===>  \"{secret}\"")

    # -- [7] Phase 4: Context-Aware Semantic Masking -------------------------
    print(f"\n{BOLD}[7] PHASE 4 — CONTEXT-AWARE SEMANTIC MASKING:{RESET}")
    print(f'    SANITIZED PROMPT: "{p4_result.sanitized_prompt}"')
    print(f"    Overall Risk:     {_colour(p4_result.overall_risk_level, overall_col)}")
    print()
    decisions = [
        ("  MASKED",   p4_result.masked_entities,   RED,    lambda e: f"{_colour(e.entity_text, RED)} -> {e.entity_type.value}"),
        ("  RETAINED", p4_result.retained_entities, GREEN,  lambda e: f"{_colour(e.entity_text, GREEN)}"),
        ("  APPROVAL_REQUIRED", p4_result.approval_required, YELLOW, lambda a: f"{_colour(a.entity_text, YELLOW)} ({a.placeholder})"),
        ("  APPROVED", p4_result.approved_entities,  GREEN,  lambda a: f"{_colour(a.entity_text, GREEN)} [approved]"),
        ("  REVIEW",   p4_result.review_required,    CYAN,   lambda a: f"{_colour(a.entity_text, CYAN)} (review)"),
    ]
    for label, items, col, fmt in decisions:
        if items:
            formatted = ", ".join(fmt(x) for x in items)
            print(f"    [{label.strip():<18}]  {formatted}")

    if p4_result.mapping:
        print(f"\n{BOLD}    LOCAL MAPPING (phase 4, never sent to LLM):{RESET}")
        for ph, raw in p4_result.mapping.items():
            print(f"      {ph:<22} ===>  \"{raw}\"")

    # -- [8] Phase 5: Downstream LLM & Controlled Restoration ----------------
    print(f"\n{BOLD}[8] PHASE 5 — DOWNSTREAM LLM GENERATION & CONTROLLED RESTORATION:{RESET}")
    exchange = shield.execute_and_restore(prompt, session_id=p4_result.session_id)
    print(f"    Provider:         {CYAN}{exchange.provider_name}{RESET}")
    print(f"    Raw LLM Output:   \"{exchange.raw_llm_response}\"")
    rest_status_col = GREEN if exchange.restoration_result.status.value in ("SUCCESS", "UNMODIFIED") else YELLOW
    print(f"    Restoration Status: {_colour(exchange.restoration_result.status.value, rest_status_col)}")
    print(f'    RESTORED OUTPUT:  "{exchange.restored_response}"')
    if exchange.restoration_result.restorations:
        restored_names = [
            f"{r.placeholder} -> {r.original_value}"
            for r in exchange.restoration_result.restorations
            if r.restored
        ]
        if restored_names:
            print(f"    Restored Secrets: {', '.join(restored_names)}")
        blocked = [
            f"{r.placeholder} ({r.reason})"
            for r in exchange.restoration_result.restorations
            if not r.restored
        ]
        if blocked:
            print(f"    Quarantined:      {', '.join(blocked)}")

    # -- [9] Phase 6: REST API Service Response -----------------------------
    print(f"\n{BOLD}[9] PHASE 6 — REST API SERVICE (FastAPI JSON Output):{RESET}")
    from backend.api.service import PromptShieldService
    api_service = PromptShieldService()
    api_response = api_service.process_end_to_end(prompt, session_id=p4_result.session_id)
    import json
    api_json = {
        "endpoint": "POST /process",
        "http_status": 200,
        "session_id": api_response.session_id,
        "sanitized_prompt": api_response.sanitized_prompt,
        "restored_response": api_response.restored_response,
        "metrics": {
            "entities_detected": api_response.detected_entities_count,
            "masked_count": api_response.masked_count,
            "restored_count": api_response.restored_count,
            "blocked_count": api_response.blocked_count,
            "leakage_detected": api_response.leakage_detected,
            "latency_ms": api_response.total_latency_ms,
        }
    }
    print(f"    {CYAN}HTTP JSON Payload delivered to Browser Extension / Dashboard:{RESET}")
    for line in json.dumps(api_json, indent=4).splitlines():
        print(f"    {line}")

    print("-" * W)



def run_demo():
    print_banner()

    # Support direct prompt via command-line argument:
    # Example: python -m cli.demo "i m julie and my psswrd is 98765432"
    if len(sys.argv) > 1:
        custom_input = " ".join(sys.argv[1:]).strip()
        print(f"\n  [>] Running direct CLI analysis for custom prompt...")
        shield = PromptShieldCore(use_spacy=True)
        display_results(shield, custom_input)
        return

    print("  [>] Initializing PromptShield AI defense pipeline (Regex + Presidio + GLiNER2-PII + SpaCy NER)...")
    shield = PromptShieldCore(use_spacy=True)
    print("  [✓] All defense layers active (GLiNER2 production pipeline ready)!\n")

    while True:
        print("\nChoose a sample prompt or enter your own:")
        for idx, (title, _) in enumerate(SAMPLE_PROMPTS, 1):
            print(f"  [{idx}] {title}")
        print(f"  [{len(SAMPLE_PROMPTS) + 1}] Enter custom prompt")
        print("  [0] Exit")

        choice = input(f"\nEnter selection (0-{len(SAMPLE_PROMPTS) + 1}): ").strip()
        if choice == "0":
            print("\nExiting PromptShield Demo. Goodbye!")
            break

        selected_prompt = None
        if choice.isdigit() and 1 <= int(choice) <= len(SAMPLE_PROMPTS):
            _, selected_prompt = SAMPLE_PROMPTS[int(choice) - 1]
        elif choice == str(len(SAMPLE_PROMPTS) + 1):
            selected_prompt = input("\nEnter your custom prompt: ").strip()
            if not selected_prompt:
                print("[!] Prompt cannot be empty.")
                continue
        else:
            print("[!] Invalid option. Please try again.")
            continue

        display_results(shield, selected_prompt)


if __name__ == "__main__":
    run_demo()
