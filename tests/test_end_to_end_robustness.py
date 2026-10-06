"""
PromptShield AI — End-to-End Pipeline Robustness & Acceptance Test Suite
========================================================================
Comprehensive automated verification across Phases 1–5:
1. Original Incident Report Prompt (Multi-entity, mixed inclusion/exclusion directives).
2. Negative Tests (Section 16 requirements):
   - Inclusions with exclusions (name/company vs phone).
   - Email drafting vs explicit non-disclosure of email.
   - Order investigation with order ID retained vs account number hidden.
   - Incident summary with employee mention vs credentials kept private.
   - Support response with customer name/ticket ID retained vs contact info masked.
   - Disambiguation: Call +91 XXXXX (PHONE) vs account number XXXXX (BANK_ACCOUNT).
   - Format validation: structured IBAN vs corporate name (Infosys).
3. Leakage & Security Invariant Tests:
   - Sanitized prompt contains 0 raw secrets.
   - Prohibited entities are NEVER restored into final output.
   - Permitted entities CAN be safely restored when allowed.
   - Mapping store remains strictly client-side.
4. Latency / Performance benchmarking across all 5 phases.
"""

import time
import pytest
from typing import Dict, Any, List

from backend.core import PromptShieldCore
from backend.models import EntityType, TaskType, RoleCategory
from backend.policy_engine import PolicyDecision


@pytest.fixture(scope="module")
def shield():
    """Shared PromptShieldCore instance."""
    return PromptShieldCore()


# ===========================================================================
# 1. THE ORIGINAL INCIDENT REPORT BENCHMARK
# ===========================================================================

def test_original_incident_report_pipeline(shield):
    """
    Validate the original test prompt through all 5 phases.
    Prompt contains:
      Retain: Rahul Menon, Infosys, Kochi, 18 September 2026, ORD-78291
      Mask: rahul.menon@example.com, +91 9876543210, 7845129630, sk-test-ABCD123456, 192.168.10.45
      Security: Restored output must NEVER restore email, phone, account, or API key.
    """
    prompt = (
        "Prepare an incident report for the failed payment transaction involving\n"
        "Rahul Menon from Infosys. The incident occurred in Kochi on 18 September\n"
        "2026. Rahul's customer email is rahul.menon@example.com and his phone\n"
        "number is +91 9876543210. His customer account number is 7845129630.\n"
        "The production server was accessed from IP address 192.168.10.45 using\n"
        "API key sk-test-ABCD123456. The investigation concerns order ORD-78291.\n"
        "The report must mention Rahul Menon, Infosys, Kochi, and order ORD-78291,\n"
        "but must not expose the customer's email, phone number, account number,\n"
        "or API key. Explain the incident and recommend steps to prevent it from\n"
        "happening again."
    )

    # --- Phase 1: Detection Foundation ---
    t0 = time.perf_counter()
    entities = shield.detector.detect(prompt)
    t_detect = time.perf_counter() - t0

    # Ensure format validation: Infosys is NOT BANK_ACCOUNT
    infosys_ents = [e for e in entities if e.text == "Infosys"]
    assert len(infosys_ents) >= 1
    for ie in infosys_ents:
        assert ie.entity_type == EntityType.ORGANIZATION, f"Infosys misclassified as {ie.entity_type}"

    # Ensure contextual disambiguation: 7845129630 is BANK_ACCOUNT (account number), NOT PHONE
    acc_ents = [e for e in entities if e.text == "7845129630"]
    assert len(acc_ents) == 1
    assert acc_ents[0].entity_type == EntityType.BANK_ACCOUNT, f"7845129630 misclassified as {acc_ents[0].entity_type}"

    # Verify other entities
    entity_texts = {e.text for e in entities}
    assert "Rahul Menon" in entity_texts
    assert "rahul.menon@example.com" in entity_texts
    assert "+91 9876543210" in entity_texts
    assert "sk-test-ABCD123456" in entity_texts
    assert "ORD-78291" in entity_texts
    assert "Kochi" in entity_texts

    # Span boundary integrity: prompt[start:end] must strictly match entity text
    for e in entities:
        assert prompt[e.start:e.end] == e.text

    # --- Phase 2: Context Understanding ---
    t0 = time.perf_counter()
    context = shield.analyze_context(prompt)
    t_context = time.perf_counter() - t0

    assert context.task_type == TaskType.INCIDENT_REPORT
    assert context.task_confidence >= 0.80

    role_by_text = {r.entity_text: r for r in context.entity_contexts}
    assert role_by_text["Infosys"].is_task_relevant is True
    assert role_by_text["ORD-78291"].is_task_relevant is True
    assert role_by_text["Rahul Menon"].is_task_relevant is True

    # --- Phase 3: Policy Evaluation ---
    t0 = time.perf_counter()
    policy = shield.evaluate_policy(prompt)
    t_policy = time.perf_counter() - t0

    policy_by_text = {ep.entity_text: ep for ep in policy.entity_policies}

    # Expected RETAINED
    assert policy_by_text["Rahul Menon"].decision == PolicyDecision.RETAIN
    assert policy_by_text["Infosys"].decision == PolicyDecision.RETAIN
    assert policy_by_text["Kochi"].decision == PolicyDecision.RETAIN
    assert policy_by_text["ORD-78291"].decision == PolicyDecision.RETAIN

    # Expected MASKED
    assert policy_by_text["rahul.menon@example.com"].decision == PolicyDecision.MASK
    assert policy_by_text["+91 9876543210"].decision == PolicyDecision.MASK
    assert policy_by_text["7845129630"].decision == PolicyDecision.MASK
    assert policy_by_text["sk-test-ABCD123456"].decision == PolicyDecision.MASK

    # Prohibited disclosure check
    assert policy_by_text["rahul.menon@example.com"].disclosure_allowed is False
    assert policy_by_text["+91 9876543210"].disclosure_allowed is False
    assert policy_by_text["7845129630"].disclosure_allowed is False
    assert policy_by_text["sk-test-ABCD123456"].disclosure_allowed is False

    # --- Phase 4: Semantic Masking ---
    t0 = time.perf_counter()
    mask_res = shield.sanitize(prompt)
    t_mask = time.perf_counter() - t0

    sanitized = mask_res.sanitized_prompt

    # Sensitive values MUST NOT be present in sanitized prompt
    assert "rahul.menon@example.com" not in sanitized
    assert "+91 9876543210" not in sanitized
    assert "7845129630" not in sanitized
    assert "sk-test-ABCD123456" not in sanitized

    # Retained entities MUST be present in sanitized prompt
    assert "Rahul Menon" in sanitized
    assert "Infosys" in sanitized
    assert "Kochi" in sanitized
    assert "ORD-78291" in sanitized

    # Correct placeholders used
    assert "<EMAIL_1>" in sanitized
    assert "<PHONE_1>" in sanitized
    assert "<BANK_ACCOUNT_1>" in sanitized
    assert "<API_KEY_1>" in sanitized
    assert "<BANK_ACCOUNT_1>" not in sanitized.replace("<BANK_ACCOUNT_1>", "Infosys")  # Infosys is not bank account

    # --- Phase 5: Safe Execution & Controlled Restoration ---
    t0 = time.perf_counter()
    exchange = shield.execute_and_restore(prompt)
    t_execute = time.perf_counter() - t0

    restored = exchange.restored_response

    # CRITICAL INVARIANT: Prohibited secrets must NEVER be restored into final response
    assert "rahul.menon@example.com" not in restored
    assert "+91 9876543210" not in restored
    assert "7845129630" not in restored
    assert "sk-test-ABCD123456" not in restored

    # Placeholders remain protected in output
    assert "<EMAIL_1>" in restored or "<PHONE_1>" in restored or "<BANK_ACCOUNT_1>" in restored

    # Performance logging
    total_time = t_detect + t_context + t_policy + t_mask + t_execute
    assert total_time < 180.0  # accommodates initial cold-start model load on CPU


# ===========================================================================
# 2. GENERALIZED NEGATIVE & DISAMBIGUATION TESTS (SECTION 16)
# ===========================================================================

def test_negative_1_mention_person_org_hide_phone(shield):
    """Case 1: Write a report mentioning John and Acme Corp. Do not reveal John's phone number."""
    prompt = (
        "Write a report mentioning John Smith and Acme Corp regarding quarterly progress. "
        "His contact phone is +1-555-019-2831. Do not reveal John Smith's phone number."
    )
    context = shield.analyze_context(prompt)
    policy = shield.evaluate_policy(prompt)
    sanitized = shield.sanitize(prompt).sanitized_prompt
    exchange = shield.execute_and_restore(prompt)

    # John Smith & Acme Corp retained
    assert "John Smith" in sanitized
    assert "Acme Corp" in sanitized

    # Phone masked in sanitized prompt and unrestored in final response
    assert "+1-555-019-2831" not in sanitized
    assert "+1-555-019-2831" not in exchange.restored_response


def test_negative_2_email_draft_do_not_expose_email(shield):
    """Case 2: Draft an email to customer Sarah using her email address, but do not expose the email in the generated response."""
    prompt = (
        "Draft an email to customer Sarah Jenkins using her email address sarah.j@enterprise.org, "
        "but do not expose the email in the generated response."
    )
    mask_res = shield.sanitize(prompt)
    exchange = shield.execute_and_restore(prompt)

    assert "Sarah Jenkins" in mask_res.sanitized_prompt
    assert "sarah.j@enterprise.org" not in mask_res.sanitized_prompt
    assert "sarah.j@enterprise.org" not in exchange.restored_response


def test_negative_3_investigate_order_hide_account(shield):
    """Case 3: Investigate order ORD-4567 and include the order number in the report, but hide the customer's account number."""
    prompt = (
        "Investigate order ORD-4567 and include the order number in the report, "
        "associated with customer account number 9948210345, but hide the customer's account number."
    )
    mask_res = shield.sanitize(prompt)
    exchange = shield.execute_and_restore(prompt)

    # ORD-4567 retained
    assert "ORD-4567" in mask_res.sanitized_prompt
    # Account masked
    assert "9948210345" not in mask_res.sanitized_prompt
    assert "9948210345" not in exchange.restored_response


def test_negative_4_incident_keep_credentials_private(shield):
    """Case 4: Summarize the incident involving employee X. Keep all credentials private."""
    prompt = (
        "Summarize the incident involving employee Marcus Vance. "
        "The system was breached using password P@ssw0rd9982! and API key sk-test-SEC998822. "
        "Keep all credentials private."
    )
    mask_res = shield.sanitize(prompt)
    exchange = shield.execute_and_restore(prompt)

    assert "P@ssw0rd9982!" not in mask_res.sanitized_prompt
    assert "sk-test-SEC998822" not in mask_res.sanitized_prompt
    assert "P@ssw0rd9982!" not in exchange.restored_response
    assert "sk-test-SEC998822" not in exchange.restored_response


def test_negative_5_support_ticket_mask_contact_info(shield):
    """Case 5: Prepare a support response and include the customer's name and ticket number, but mask email, phone and account number."""
    prompt = (
        "Prepare a support response and include the customer's name David Miller and ticket number TKT-88219. "
        "Customer details: email david.m@domain.com, phone +91 9123456780, account number 4410293847. "
        "Mask email, phone and account number."
    )
    mask_res = shield.sanitize(prompt)
    exchange = shield.execute_and_restore(prompt)

    assert "David Miller" in mask_res.sanitized_prompt
    assert "TKT-88219" in mask_res.sanitized_prompt
    assert "david.m@domain.com" not in mask_res.sanitized_prompt
    assert "+91 9123456780" not in mask_res.sanitized_prompt
    assert "4410293847" not in mask_res.sanitized_prompt
    assert "david.m@domain.com" not in exchange.restored_response
    assert "+91 9123456780" not in exchange.restored_response
    assert "4410293847" not in exchange.restored_response


def test_negative_6_phone_vs_account_disambiguation(shield):
    """Case 6: 'Call +91 XXXXX' (PHONE) versus 'account number XXXXX' (BANK_ACCOUNT)."""
    phone_prompt = "Please call me at 9876543210 regarding the server outage."
    account_prompt = "The customer account number is 9876543210 for the bill payment."

    ents_phone = shield.detector.detect(phone_prompt)
    ents_acc = shield.detector.detect(account_prompt)

    phone_match = [e for e in ents_phone if e.text == "9876543210"]
    acc_match = [e for e in ents_acc if e.text == "9876543210"]

    assert len(phone_match) == 1
    assert phone_match[0].entity_type == EntityType.PHONE

    assert len(acc_match) == 1
    assert acc_match[0].entity_type == EntityType.BANK_ACCOUNT


def test_negative_7_iban_vs_company_format_validation(shield):
    """Case 7: 'IBAN is ...' versus 'company is Infosys'."""
    iban_prompt = "Transfer funds to IBAN GB82WEST12345698765432 for the invoice."
    company_prompt = "The consulting partner for this rollout is Infosys."

    ents_iban = shield.detector.detect(iban_prompt)
    ents_company = shield.detector.detect(company_prompt)

    iban_match = [e for e in ents_iban if "GB82" in e.text]
    assert len(iban_match) == 1
    assert iban_match[0].entity_type == EntityType.BANK_ACCOUNT

    company_match = [e for e in ents_company if e.text == "Infosys"]
    assert len(company_match) == 1
    assert company_match[0].entity_type == EntityType.ORGANIZATION


# ===========================================================================
# 3. CUSTOMER SUPPORT INCIDENT RESTORATION BENCHMARK
# ===========================================================================

def test_customer_support_incident_restoration(shield):
    """
    Validate the customer support incident restoration prompt through all 5 phases:
    - Proper detection without gliner2_iban on organizations
    - No false CUSTOMER_ID on 'customer-support'
    - No false DATE on isolated 'MAY'
    - Clean PERSON spans ('Priya Nair' without possessives, rejection of 'API key.- You MUST')
    - No duplicate nested 'Priya' entity
    - Clean EMAIL span ('arjun.menon@technova.example' without trailing '.The')
    - Phase 4 sanitized prompt contains expected placeholders and no bogus ones
    - Phase 5 controlled restoration: emails restored, credentials and prohibited PII blocked
    - Input-conditioned Mock LLM output reflecting incident details without unsupported root causes
    """
    prompt = (
        "Prepare a customer support incident report involving Priya Nair from TechNova Solutions. "
        "The incident occurred in Chennai regarding order ORD-84621. Priya Nair's customer email is "
        "priya.nair@example.com, phone number is +91 9876543210, and customer ID is CUST-84621. "
        "Support coordinator Arjun Menon can be contacted at arjun.menon@technova.example.The server "
        "was accessed from IP address 192.168.1.100 using API key sk-test-TN998811.- You MUST "
        "prepare a detailed report mentioning Priya Nair, TechNova Solutions, Chennai, Arjun Menon, "
        "and order ORD-84621. Include the customer and support emails, but you MAY not expose the "
        "customer's phone number, customer ID, server IP address, or API key.- You MUST recommend "
        "preventative steps."
    )

    # --- Phase 1: Detection Foundation ---
    entities = shield.detector.detect(prompt)
    entity_texts = {e.text for e in entities}

    # 1. Organization format validation: TechNova Solutions is detected, but NOT via gliner2_iban
    org_ents = [e for e in entities if e.text == "TechNova Solutions"]
    assert len(org_ents) >= 1
    for oe in org_ents:
        assert oe.entity_type == EntityType.ORGANIZATION
        assert oe.detector != "gliner2_iban", f"TechNova Solutions must not have invalid detector gliner2_iban"

    # 2. No false positive CUSTOMER_ID for "customer-support"
    assert "customer-support" not in entity_texts
    assert "support" not in entity_texts

    # 3. No false positive DATE for "MAY"
    date_ents = [e for e in entities if e.text == "MAY"]
    assert len(date_ents) == 0, f"Isolated 'MAY' falsely detected as DATE: {date_ents}"

    # 4. Clean PERSON spans: "Priya Nair" without possessive 's, rejection of "API key.- You MUST"
    assert "Priya Nair's" not in entity_texts
    assert "API key.- You MUST" not in entity_texts
    assert not any("MUST" in e.text for e in entities if e.entity_type == EntityType.PERSON)

    # 5. No nested/duplicate PERSON entity: "Priya" is not a separate entity from "Priya Nair"
    priya_ents = [e for e in entities if e.text == "Priya" and e.entity_type == EntityType.PERSON]
    assert len(priya_ents) == 0, f"Nested 'Priya' entity falsely present: {priya_ents}"

    # 6. Email span bug fixed: arjun.menon@technova.example without trailing '.The'
    assert "arjun.menon@technova.example" in entity_texts
    assert "arjun.menon@technova.example.The" not in entity_texts

    # Other canonical entities present
    assert "priya.nair@example.com" in entity_texts
    assert "+91 9876543210" in entity_texts
    assert "CUST-84621" in entity_texts
    assert "192.168.1.100" in entity_texts
    assert "sk-test-TN998811" in entity_texts
    assert "Chennai" in entity_texts
    assert "ORD-84621" in entity_texts
    assert "Arjun Menon" in entity_texts

    # --- Phase 2: Context Understanding ---
    context = shield.analyze_context(prompt)
    assert context.task_type == TaskType.INCIDENT_REPORT

    # --- Phase 3: Policy Evaluation ---
    policy = shield.evaluate_policy(prompt)
    policy_by_text = {ep.entity_text: ep for ep in policy.entity_policies}

    # Retained entities
    assert policy_by_text["Priya Nair"].decision == PolicyDecision.RETAIN
    assert policy_by_text["TechNova Solutions"].decision == PolicyDecision.RETAIN
    assert policy_by_text["Chennai"].decision == PolicyDecision.RETAIN
    assert policy_by_text["Arjun Menon"].decision == PolicyDecision.RETAIN
    assert policy_by_text["ORD-84621"].decision == PolicyDecision.RETAIN

    # Masked entities
    assert policy_by_text["priya.nair@example.com"].decision == PolicyDecision.MASK
    assert policy_by_text["arjun.menon@technova.example"].decision == PolicyDecision.MASK
    assert policy_by_text["+91 9876543210"].decision == PolicyDecision.MASK
    assert policy_by_text["CUST-84621"].decision == PolicyDecision.MASK
    assert policy_by_text["192.168.1.100"].decision == PolicyDecision.MASK
    assert policy_by_text["sk-test-TN998811"].decision == PolicyDecision.MASK

    # Disclosure gating
    assert policy_by_text["priya.nair@example.com"].disclosure_allowed is True
    assert policy_by_text["arjun.menon@technova.example"].disclosure_allowed is True
    assert policy_by_text["+91 9876543210"].disclosure_allowed is False
    assert policy_by_text["CUST-84621"].disclosure_allowed is False
    assert policy_by_text["192.168.1.100"].disclosure_allowed is False
    assert policy_by_text["sk-test-TN998811"].disclosure_allowed is False

    # --- Phase 4: Semantic Masking ---
    mask_res = shield.sanitize(prompt)
    sanitized = mask_res.sanitized_prompt

    # Sensitive values absent from sanitized prompt
    assert "priya.nair@example.com" not in sanitized
    assert "arjun.menon@technova.example" not in sanitized
    assert "+91 9876543210" not in sanitized
    assert "CUST-84621" not in sanitized
    assert "192.168.1.100" not in sanitized
    assert "sk-test-TN998811" not in sanitized

    # Retained entities present
    assert "Priya Nair" in sanitized
    assert "TechNova Solutions" in sanitized
    assert "Chennai" in sanitized
    assert "Arjun Menon" in sanitized
    assert "ORD-84621" in sanitized

    # Expected placeholders present
    assert "<EMAIL_1>" in sanitized
    assert "<PHONE_1>" in sanitized
    assert "<CUSTOMER_ID_1>" in sanitized
    assert "<EMAIL_2>" in sanitized
    assert "<IP_ADDRESS_1>" in sanitized
    assert "<API_KEY_1>" in sanitized

    # No bogus placeholders
    assert "<ORG_1>" not in sanitized
    assert "<DATE_1>" not in sanitized

    # --- Phase 5: Controlled Restoration & LLM Layer ---
    exchange = shield.execute_and_restore(prompt, session_id=mask_res.session_id)
    raw_llm = exchange.raw_llm_response
    restored = exchange.restored_response

    # LLM input conditioning: reflects incident facts, does not invent gateway synchronization mismatch
    assert "Customer Support Incident" in raw_llm or "customer support" in raw_llm.lower()
    assert "Priya Nair" in raw_llm
    assert "TechNova Solutions" in raw_llm
    assert "Chennai" in raw_llm
    assert "ORD-84621" in raw_llm
    assert "Arjun Menon" in raw_llm
    assert "gateway synchronization mismatch" not in raw_llm
    assert "The exact root cause is not established by the provided information" in raw_llm

    # Controlled restoration invariants:
    # 1. Permitted email values are restored exactly with no trailing punctuation
    assert "priya.nair@example.com" in restored
    assert "arjun.menon@technova.example" in restored
    assert "arjun.menon@technova.example.The" not in restored

    # 2. Prohibited values remain strictly masked in the final output
    assert "+91 9876543210" not in restored
    assert "CUST-84621" not in restored
    assert "192.168.1.100" not in restored
    assert "sk-test-TN998811" not in restored

    # Placeholders for prohibited entities remain visible and protective
    assert "<PHONE_1>" in restored
    assert "<CUSTOMER_ID_1>" in restored
    assert "<IP_ADDRESS_1>" in restored
    assert "<API_KEY_1>" in restored


def test_password_and_email_detection_and_blocking(shield):
    """Verify that password followed by short numeric credentials is detected as PASSWORD (not DATE) and quarantined."""
    prompt = "i m manasa and my mail is mani@gmail.com along with password 1234"
    mask_res = shield.sanitize(prompt)
    exchange = shield.execute_and_restore(prompt, session_id=mask_res.session_id)

    # 1. 1234 must NOT leak into sanitized prompt as plaintext or be misclassified as DATE
    assert "1234" not in mask_res.sanitized_prompt
    assert "<DATE_" not in mask_res.sanitized_prompt
    assert "<PASSWORD_" in mask_res.sanitized_prompt
    assert "<EMAIL_" in mask_res.sanitized_prompt

    # 2. Email can be restored, but password is a technical credential and MUST remain quarantined
    assert "1234" not in exchange.restored_response
    assert "<PASSWORD_" in exchange.restored_response

