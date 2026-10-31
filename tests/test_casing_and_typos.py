"""
Test Suite: Casing Invariance and Typo / Abbreviation Handling
==============================================================
Validates:
1. Casing invariance for PERSON names (lowercase, alternating case, uppercase, titlecase).
2. Casing invariance for credentials and sensitive entity labels (PassWord, pAsSwOrD, PASSWORD).
3. Typos and informal abbreviations for sensitive cues:
   - Passwords: passwrd, p@ssword, pswd, pwd
   - Bank accounts: acc no, act no, a/c no, ac no, accnt no
   - Identity documents: adhr no, pan no
   - Communication: mob no, ph no
"""

import pytest
from backend.core import PromptShieldCore


@pytest.fixture(scope="module")
def core():
    return PromptShieldCore()


# ============================================================
# 1. CASING INVARIANCE FOR PERSON NAMES
# ============================================================

def test_person_lowercase(core):
    """Full lowercase name in conversational self-intro must be masked."""
    prompt = "my name is anand"
    result = core.sanitize(prompt)
    assert "<PERSON_1>" in result.sanitized_prompt
    assert "anand" not in result.sanitized_prompt


def test_person_alternating_case(core):
    """Mixed/alternating case name (AnAnD) must be masked."""
    prompt = "my name is AnAnD"
    result = core.sanitize(prompt)
    assert "<PERSON_1>" in result.sanitized_prompt
    assert "AnAnD" not in result.sanitized_prompt


def test_person_all_caps(core):
    """All-caps name (ANAND) must be masked."""
    prompt = "my name is ANAND"
    result = core.sanitize(prompt)
    assert "<PERSON_1>" in result.sanitized_prompt
    assert "ANAND" not in result.sanitized_prompt


def test_person_conversational_variations(core):
    """Various conversational introducers with casing variations."""
    cases = [
        ("i am anand", "anand"),
        ("call me AnAnD", "AnAnD"),
        ("this is anand", "anand"),
        ("Hello, anand here", "anand"),
        ("meet with anand tomorrow at 5pm", "anand"),
        ("meet with AnAnD tomorrow at 5pm", "AnAnD"),
        ("Tell ANAND to finish the report", "ANAND"),
    ]
    for prompt, raw_name in cases:
        result = core.sanitize(prompt)
        assert "<PERSON_" in result.sanitized_prompt, f"Failed to mask person in '{prompt}'"
        assert raw_name not in result.sanitized_prompt, f"Raw name '{raw_name}' leaked in '{result.sanitized_prompt}'"


# ============================================================
# 2. CASING INVARIANCE FOR CREDENTIAL LABELS
# ============================================================

def test_password_mixed_case_label(core):
    """Mixed case 'PassWord' or 'pAsSwOrD' label must be masked."""
    for label in ["PassWord", "pAsSwOrD", "PASSWORD", "password"]:
        prompt = f"{label}: Secret123#"
        result = core.sanitize(prompt)
        assert "<PASSWORD_1>" in result.sanitized_prompt
        assert "Secret123#" not in result.sanitized_prompt


def test_password_natural_language_mixed_case(core):
    """Natural language password assignment with mixed case."""
    prompt = "my PassWord is Secret999"
    result = core.sanitize(prompt)
    assert "<PASSWORD_1>" in result.sanitized_prompt
    assert "Secret999" not in result.sanitized_prompt


# ============================================================
# 3. TYPOS AND INFORMAL ABBREVIATIONS (PROBLEM 1)
# ============================================================

def test_password_typo_passwrd(core):
    """Typo 'passwrd' (missing 'o') must be recognized."""
    prompt = "passwrd: secret999"
    result = core.sanitize(prompt)
    assert "<PASSWORD_1>" in result.sanitized_prompt
    assert "secret999" not in result.sanitized_prompt


def test_password_typo_natural_language(core):
    """Typo 'passwrd is ...' in sentence must be recognized."""
    prompt = "my passwrd is secret999"
    result = core.sanitize(prompt)
    assert "<PASSWORD_1>" in result.sanitized_prompt
    assert "secret999" not in result.sanitized_prompt


def test_password_shorthand_pswd_pwd(core):
    """Shorthand 'pswd' and 'pwd' must be recognized."""
    cases = [
        ("my pswd is supersecret", "supersecret"),
        ("my pwd is testpass123", "testpass123"),
    ]
    for prompt, secret in cases:
        result = core.sanitize(prompt)
        assert "<PASSWORD_1>" in result.sanitized_prompt, f"Failed on '{prompt}'"
        assert secret not in result.sanitized_prompt


def test_bank_account_shorthand_acc_no(core):
    """'acc no' abbreviation must be recognized as BANK_ACCOUNT."""
    prompt = "my acc no is 123456789012"
    result = core.sanitize(prompt)
    assert "<BANK_ACCOUNT_1>" in result.sanitized_prompt
    assert "123456789012" not in result.sanitized_prompt


def test_bank_account_shorthand_act_no(core):
    """'act no' abbreviation must be recognized as BANK_ACCOUNT."""
    prompt = "my act no is 987654321098"
    result = core.sanitize(prompt)
    assert "<BANK_ACCOUNT_1>" in result.sanitized_prompt
    assert "987654321098" not in result.sanitized_prompt


def test_bank_account_shorthand_slash_format(core):
    """'a/c no' format must be recognized as BANK_ACCOUNT."""
    prompt = "my a/c no is 112233445566"
    result = core.sanitize(prompt)
    assert "<BANK_ACCOUNT_1>" in result.sanitized_prompt
    assert "112233445566" not in result.sanitized_prompt


def test_bank_account_colon_formats(core):
    """Abbreviated colon triggers like 'acc no: ...' and 'act no: ...'."""
    cases = [
        ("acc no: 123456789012", "123456789012"),
        ("act no: 987654321098", "987654321098"),
        ("a/c no: 112233445566", "112233445566"),
        ("ac no: 123456789012", "123456789012"),
        ("accnt no: 123456789012", "123456789012"),
    ]
    for prompt, raw_num in cases:
        result = core.sanitize(prompt)
        assert "<BANK_ACCOUNT_1>" in result.sanitized_prompt, f"Failed on '{prompt}'"
        assert raw_num not in result.sanitized_prompt


def test_aadhaar_shorthand_adhr_no(core):
    """'adhr no' abbreviation must be recognized as NATIONAL_ID."""
    prompt = "my adhr no is 456245627895"
    result = core.sanitize(prompt)
    assert "<NATIONAL_ID_1>" in result.sanitized_prompt
    assert "456245627895" not in result.sanitized_prompt


def test_pan_shorthand_pan_no(core):
    """'pan no' abbreviation must be recognized as TAX_ID."""
    prompt = "my pan no is ABCDE1234F"
    result = core.sanitize(prompt)
    assert "<TAX_ID_1>" in result.sanitized_prompt
    assert "ABCDE1234F" not in result.sanitized_prompt


def test_mobile_shorthand_mob_and_ph_no(core):
    """'mob no' and 'ph no' abbreviations must be recognized as PHONE."""
    cases = [
        ("my mob no is 9876543210", "9876543210"),
        ("contact me at ph no 9876543210", "9876543210"),
    ]
    for prompt, raw_phone in cases:
        result = core.sanitize(prompt)
        assert "<PHONE_1>" in result.sanitized_prompt, f"Failed on '{prompt}'"
        assert raw_phone not in result.sanitized_prompt


def test_user_birthday_lifecycle_and_person_unification(core):
    """
    Validates user prompt:
    'My name is anand. Anand Likes football. Today is my birthday. What is the first 2 digits of 452345624562. My facebook pwd is abcsghs.'
    - 'anand' and 'Anand' unify to <PERSON_1>.
    - 'Today' in birthday context is protected as <DOB_1>.
    - 452345624562 in math context is retained.
    - 'pwd is abcsghs' is masked as <PASSWORD_1>.
    """
    prompt = "My name is anand. Anand Likes football. Today is my birthday. What is the first 2 digits of 452345624562. My facebook pwd is abcsghs."
    result = core.sanitize(prompt)
    assert "<PERSON_1>" in result.sanitized_prompt
    assert "<PERSON_2>" not in result.sanitized_prompt
    assert "<DOB_1>" in result.sanitized_prompt or "<DATE_1>" in result.sanitized_prompt
    assert "Today is my birthday" not in result.sanitized_prompt
    assert "452345624562" in result.sanitized_prompt
    assert "<PASSWORD_1>" in result.sanitized_prompt
    assert "abcsghs" not in result.sanitized_prompt

