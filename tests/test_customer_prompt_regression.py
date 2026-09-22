"""
PromptShield AI — Regression test for complex multi-entity prompt.
Verifies that customer IDs, people, organizations, locations, emails,
phones, order IDs, and IP addresses are correctly detected without
cross-entity interference or false positive technical abbreviations.
"""

import pytest
from backend.core import PromptShieldCore
from backend.models import EntityType


def test_complex_customer_support_prompt():
    prompt = (
        "Yesterday, Rahul Menon from Infosys contacted me about customer "
        "Anjali Nair, whose order ORD-78291 was shipped to Kochi. "
        "Her email is anjali.nair@gmail.com and she can be reached at "
        "+91 9876543210. The support team recorded her customer ID as "
        "CUST-10458 and the request came from IP address 192.168.1.45."
    )

    shield = PromptShieldCore(use_spacy=True)
    res = shield.sanitize_baseline(prompt)

    mapping = shield.get_local_mappings(res.session_id)
    mapping_values = set(mapping.values())

    # 1. Check all key entities are properly extracted
    assert "Yesterday" in mapping_values
    assert "Rahul Menon" in mapping_values
    assert "Infosys" in mapping_values
    assert "Anjali Nair" in mapping_values
    assert "ORD-78291" in mapping_values
    assert "Kochi" in mapping_values
    assert "anjali.nair@gmail.com" in mapping_values
    assert "+91 9876543210" in mapping_values
    assert "CUST-10458" in mapping_values
    assert "192.168.1.45" in mapping_values

    # 2. Check no false positive "IP" or broken names
    assert "IP" not in mapping_values
    assert "Anjali" not in mapping.values() or "Anjali Nair" in mapping.values()

    # 3. Check entity types
    entity_map = {e.text: e.entity_type for e in res.entities}
    assert entity_map.get("Yesterday") == EntityType.DATE
    assert entity_map.get("Rahul Menon") == EntityType.PERSON
    assert entity_map.get("Infosys") == EntityType.ORGANIZATION
    assert entity_map.get("Anjali Nair") == EntityType.PERSON
    assert entity_map.get("ORD-78291") == EntityType.ORDER_ID
    assert entity_map.get("Kochi") == EntityType.LOCATION
    assert entity_map.get("anjali.nair@gmail.com") == EntityType.EMAIL
    assert entity_map.get("+91 9876543210") == EntityType.PHONE
    assert entity_map.get("CUST-10458") == EntityType.CUSTOMER_ID
    assert entity_map.get("192.168.1.45") == EntityType.IP_ADDRESS
