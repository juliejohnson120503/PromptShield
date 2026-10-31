"""
Test Suite: Indian Identity Documents, Math/Abstract Context, and Foreign Phones
================================================================================
Validates:
1. Abstract mathematical and algorithmic queries with 10/12-digit numbers are NOT masked.
2. Indian identity documents (Aadhaar, PAN, DL, Voter ID) are detected when ownership/context is present.
3. Indian chat abbreviations (adhr no, pan no, ph no) trigger correct identification.
4. Foreign phone numbers (US, UK, UAE, Australia, Europe) are detected and masked.
5. Technical/programming examples (e.g. 'write regex for PAN ABCDE1234F') are retained in plaintext.
"""

import pytest
from backend.core import PromptShieldCore


@pytest.fixture(scope="module")
def core():
    return PromptShieldCore()


def test_math_10_digit_number_not_phone(core):
    """Abstract 10-digit number in a mathematical query must remain in plaintext."""
    prompt = "find the sum of digits of a number 9845124523"
    result = core.sanitize(prompt)
    assert result.sanitized_prompt == prompt, f"Expected '{prompt}', got '{result.sanitized_prompt}'"
    assert "9845124523" in result.sanitized_prompt
    assert "<PHONE_" not in result.sanitized_prompt


def test_math_12_digit_number_not_national_id(core):
    """Abstract 12-digit number in a string/digit query must remain in plaintext."""
    prompt = "What is the last 4 digits of 456245627895"
    result = core.sanitize(prompt)
    assert result.sanitized_prompt == prompt
    assert "456245627895" in result.sanitized_prompt
    assert "<NATIONAL_ID_" not in result.sanitized_prompt


def test_aadhaar_with_adhr_shorthand_masked(core):
    """Aadhaar number with chat shorthand ('adhr no') and personal ownership must be masked."""
    prompt = "My adhr no is 456245627895"
    result = core.sanitize(prompt)
    assert "<NATIONAL_ID_1>" in result.sanitized_prompt
    assert "456245627895" not in result.sanitized_prompt


def test_aadhaar_spaced_format_masked(core):
    """Spaced 12-digit Aadhaar format with ownership context must be masked."""
    prompt = "My aadhaar number is 4562 4562 7895, please verify it."
    result = core.sanitize(prompt)
    assert "<NATIONAL_ID_1>" in result.sanitized_prompt
    assert "4562 4562 7895" not in result.sanitized_prompt


def test_pan_card_with_ownership_masked(core):
    """Indian PAN card with personal ownership must be masked as TAX_ID."""
    prompt = "My PAN is ABCDE1234F"
    result = core.sanitize(prompt)
    assert "<TAX_ID_1>" in result.sanitized_prompt
    assert "ABCDE1234F" not in result.sanitized_prompt
    # Document label 'PAN' must not be masked as ORG
    assert "<ORG_" not in result.sanitized_prompt


def test_pan_card_programming_regex_retained(core):
    """PAN in a coding/regex prompt has zero personal ownership and should be retained."""
    prompt = "Write regex to validate PAN like ABCDE1234F"
    result = core.sanitize(prompt)
    assert "ABCDE1234F" in result.sanitized_prompt
    assert "<PERSON_" not in result.sanitized_prompt


def test_indian_mobile_with_communication_cue_masked(core):
    """Raw 10-digit number with communication verb must be detected as PHONE."""
    prompt = "call me at 9845124523"
    result = core.sanitize(prompt)
    assert "<PHONE_1>" in result.sanitized_prompt
    assert "9845124523" not in result.sanitized_prompt


def test_foreign_phone_us_nanp(core):
    """US/North American phone number must be masked."""
    prompt = "Call me at +1 (202) 555-0199 for the details."
    result = core.sanitize(prompt)
    assert "<PHONE_1>" in result.sanitized_prompt
    assert "(202) 555-0199" not in result.sanitized_prompt


def test_foreign_phone_uk_national(core):
    """UK national phone number with leading 0 must be masked."""
    prompt = "UK support is 020 7946 0958 if you need assistance."
    result = core.sanitize(prompt)
    assert "<PHONE_1>" in result.sanitized_prompt
    assert "020 7946 0958" not in result.sanitized_prompt


def test_foreign_phone_uae_international(core):
    """UAE international phone number must be masked."""
    prompt = "Contact Tariq on +971 50 123 4567 regarding the delivery."
    result = core.sanitize(prompt)
    assert "<PHONE_1>" in result.sanitized_prompt
    assert "+971 50 123 4567" not in result.sanitized_prompt


def test_indian_driving_license_masked(core):
    """Indian driving license with DL shorthand must be masked."""
    prompt = "My dl no is DL-1420110012345, please update the records."
    result = core.sanitize(prompt)
    assert "<DRIVER_LICENSE_1>" in result.sanitized_prompt
    assert "DL-1420110012345" not in result.sanitized_prompt
