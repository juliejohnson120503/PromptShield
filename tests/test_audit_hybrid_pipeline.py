"""
Unit and regression tests for the audited hybrid entity/PII detection pipeline.
Validates all required test cases (Tests 5, 10, 11, 14, 17, 18, 19, 20) as well as
unseen generalization cases without hard-coding.
"""

import pytest
from backend.core import PromptShieldCore
from backend.detector import SensitiveDataDetector
from backend.models import EntityType


@pytest.fixture(scope="module")
def core():
    return PromptShieldCore(use_spacy=True)


@pytest.fixture(scope="module")
def detector():
    return SensitiveDataDetector(use_spacy=True)


def test_case_5_address_and_person(core):
    """
    TEST 5:
    'Please deliver the package to John Mathew at 24 MG Road, Kochi, Kerala 682016.'
    Expected:
    <PERSON_1> -> John Mathew
    <ADDRESS_1> -> 24 MG Road, Kochi, Kerala 682016
    """
    prompt = "Please deliver the package to John Mathew at 24 MG Road, Kochi, Kerala 682016."
    res = core.sanitize_baseline(prompt)
    mapping = core.get_local_mappings(res.session_id)

    assert "John Mathew" in mapping.values()
    assert "24 MG Road, Kochi, Kerala 682016" in mapping.values()
    assert "<PERSON_1>" in res.sanitized_prompt
    assert "<ADDRESS_1>" in res.sanitized_prompt
    # No duplicate internal location masks like <LOCATION_1>, <LOCATION_2>
    assert "<LOCATION" not in res.sanitized_prompt


def test_case_10_ip_word_not_org(core):
    """
    TEST 10:
    'The network team discussed the IP address and server configuration during the meeting.'
    Expected: NO PII (the word 'IP' must NOT be classified as ORG).
    """
    prompt = "The network team discussed the IP address and server configuration during the meeting."
    res = core.sanitize_baseline(prompt)
    mapping = core.get_local_mappings(res.session_id)

    assert len(res.entities) == 0
    assert len(mapping) == 0
    assert res.sanitized_prompt == prompt


def test_case_11_generic_words_not_org(core):
    """
    TEST 11:
    'The project meeting will be held tomorrow after lunch. The team will discuss the database architecture.'
    Expected: No generic false positives like project -> ORG, team -> ORG, database -> ORG.
    """
    prompt = "The project meeting will be held tomorrow after lunch. The team will discuss the database architecture."
    res = core.sanitize_baseline(prompt)

    org_entities = [e for e in res.entities if e.entity_type == EntityType.ORGANIZATION]
    assert len(org_entities) == 0


def test_case_14_contextual_apple_vs_apple_inc(core):
    """
    TEST 14:
    'I ate an apple while reading a report about Apple Inc. in California.'
    Expected:
    <ORG_1> -> Apple Inc.
    <LOCATION_1> -> California
    The first 'apple' must NOT be classified as an organization.
    The pronoun 'I' must NOT be classified as a person.
    """
    prompt = "I ate an apple while reading a report about Apple Inc. in California."
    res = core.sanitize_baseline(prompt)
    mapping = core.get_local_mappings(res.session_id)

    assert "Apple Inc." in mapping.values()
    assert "California" in mapping.values()
    assert "apple" not in mapping.values()
    assert "I" not in mapping.values()

    entity_map = {e.text: e.entity_type for e in res.entities}
    assert entity_map.get("Apple Inc.") == EntityType.ORGANIZATION
    assert entity_map.get("California") == EntityType.LOCATION
    assert "apple" not in entity_map


def test_case_17_custom_id_recognizers(core):
    """
    TEST 17:
    'The support request was submitted by customer CUST-90812 for order ORD-55123 and ticket TKT-99182.'
    Expected:
    <CUSTOMER_ID_1> -> CUST-90812
    <ORDER_ID_1> -> ORD-55123
    <TICKET_ID_1> -> TKT-99182
    """
    prompt = "The support request was submitted by customer CUST-90812 for order ORD-55123 and ticket TKT-99182."
    res = core.sanitize_baseline(prompt)
    mapping = core.get_local_mappings(res.session_id)

    assert mapping.get("<CUSTOMER_ID_1>") == "CUST-90812"
    assert mapping.get("<ORDER_ID_1>") == "ORD-55123"
    assert mapping.get("<TICKET_ID_1>") == "TKT-99182"


def test_case_18_repeated_entities_and_alias_coreference(core):
    """
    TEST 18:
    'John Mathew placed order ORD-12345. Later, John asked the support team to check whether order ORD-12345 had been shipped.'
    Expected:
    <PERSON_1> -> John Mathew (and alias John reuses <PERSON_1>)
    <ORDER_ID_1> -> ORD-12345 (repeated order maps to same placeholder)
    """
    prompt = "John Mathew placed order ORD-12345. Later, John asked the support team to check whether order ORD-12345 had been shipped."
    res = core.sanitize_baseline(prompt)

    assert res.sanitized_prompt.count("<ORDER_ID_1>") == 2
    assert "<ORDER_ID_2>" not in res.sanitized_prompt
    assert res.sanitized_prompt.count("<PERSON_1>") == 2
    assert "<PERSON_2>" not in res.sanitized_prompt


def test_case_19_false_positive_cleanliness(core):
    """
    TEST 19:
    'Machine learning models can identify patterns in large datasets and help organizations automate repetitive tasks.'
    Expected: NO PII / NO SENSITIVE ENTITIES.
    """
    prompt = "Machine learning models can identify patterns in large datasets and help organizations automate repetitive tasks."
    res = core.sanitize_baseline(prompt)

    assert len(res.entities) == 0
    assert res.sanitized_prompt == prompt


def test_case_20_complex_multi_entity(core):
    """
    TEST 20:
    Complex multi-entity prompt covering DATE, PERSON, ORG, ORDER_ID, EMAIL, PHONE, CUSTOMER_ID, ADDRESS, IP.
    """
    prompt = (
        "Yesterday, Rahul Menon from Infosys contacted customer Anjali Nair regarding order ORD-78291. "
        "Anjali's email is anjali.nair@gmail.com and her phone number is +91 9876543210. "
        "Her customer ID is CUST-10458, and she asked the support team to deliver the replacement to "
        "24 MG Road, Kochi, Kerala 682016. The request was submitted from IP address 192.168.1.45."
    )
    res = core.sanitize_baseline(prompt)
    mapping = core.get_local_mappings(res.session_id)

    assert "Yesterday" in mapping.values()
    assert "Rahul Menon" in mapping.values()
    assert "Infosys" in mapping.values()
    assert "Anjali Nair" in mapping.values()
    assert "ORD-78291" in mapping.values()
    assert "anjali.nair@gmail.com" in mapping.values()
    assert "+91 9876543210" in mapping.values()
    assert "CUST-10458" in mapping.values()
    assert "24 MG Road, Kochi, Kerala 682016" in mapping.values()
    assert "192.168.1.45" in mapping.values()

    # Verify no raw sensitive data leaks into sanitized prompt
    assert "anjali.nair@gmail.com" not in res.sanitized_prompt
    assert "192.168.1.45" not in res.sanitized_prompt
    assert "CUST-10458" not in res.sanitized_prompt
    assert "ORD-78291" not in res.sanitized_prompt
    assert "9876543210" not in res.sanitized_prompt


def test_unseen_generalization(core):
    """
    Test completely unseen data to prove pipeline generality:
    Different customer code (CUST-44129), different order (ORD-99014),
    different ticket (TKT-11234), different address (120 Nehru Marg, Pune, Maharashtra 411001),
    different person and company.
    """
    prompt = (
        "Agent Priya Sharma at TechCorp created ticket TKT-11234 for customer CUST-44129 "
        "regarding shipment ORD-99014 to 120 Nehru Marg, Pune, Maharashtra 411001."
    )
    res = core.sanitize_baseline(prompt)
    mapping = core.get_local_mappings(res.session_id)

    assert mapping.get("<TICKET_ID_1>") == "TKT-11234"
    assert mapping.get("<CUSTOMER_ID_1>") == "CUST-44129"
    assert mapping.get("<ORDER_ID_1>") == "ORD-99014"
    assert mapping.get("<ADDRESS_1>") == "120 Nehru Marg, Pune, Maharashtra 411001"


def test_explainable_reporting(detector):
    """
    Verify explainable report formatting capability:
    ENTITY | TYPE | SOURCE | CONFIDENCE
    """
    prompt = "Contact Rahul Menon at Infosys regarding order ORD-78291."
    entities = detector.detect(prompt)
    report = detector.format_explainable_report(entities)

    assert "ENTITY" in report
    assert "TYPE" in report
    assert "SOURCE" in report
    assert "CONFIDENCE" in report
    assert "Rahul Menon" in report
    assert "Infosys" in report
    assert "ORD-78291" in report
