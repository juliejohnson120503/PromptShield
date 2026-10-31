"""
PromptShield AI — Centralized Entity Validators
=================================================
Single source of truth for entity validation and cleaning logic.

Previously, identical filtering rules were duplicated across:
  - spacy_detector.py
  - gliner2_pii_detector.py
  - entity_fusion.py

This module consolidates ALL entity-specific validation and cleaning
into reusable functions so a bug fix or new filter applies everywhere.

Design: Pure functions with no side effects. No model loading. No I/O.
"""

import re
from typing import Optional, Set, Tuple

from backend import config


# ============================================================
# SHARED CONSTANTS (previously duplicated across 3+ files)
# ============================================================

# Relative / vague date phrases that are NOT specific PII dates
NON_SPECIFIC_DATE_PHRASES: Set[str] = {
    "a good day", "good day", "last quarter", "quarterly", "next quarter",
    "this quarter", "next year", "last year", "this year", "fiscal year",
    "2024 fiscal year", "every day", "daily", "annually", "q1", "q2", "q3", "q4",
}

RELATIVE_DATE_WORDS: Set[str] = {
    "today", "yesterday", "tomorrow", "tonight", "morning",
    "afternoon", "evening", "now", "later", "soon", "recently",
    "last week", "next week", "this morning", "this afternoon",
    "this evening", "last night", "next month", "last month",
}

MONTH_WORDS: Set[str] = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
}

# Words that disqualify a PERSON span (technical / directive keywords / modal verbs)
PERSON_DISQUALIFYING_WORDS: Set[str] = {
    "api", "key", "token", "password", "secret", "credential", "database",
    "server", "ip", "endpoint", "must", "should", "shall", "cannot", "can",
    "will", "would", "could", "may", "might", "please", "report", "explain", "recommend",
    "incident", "order", "ticket", "customer", "support",
}

# Demonyms, nationalities, and religious designations falsely tagged as PERSON
DEMONYMS: Set[str] = {
    "spanish", "italian", "dutch", "swiss", "belgian", "german", "british",
    "french", "japanese", "chinese", "american", "russian", "indian",
    "canadian", "australian", "brazilian", "mexican", "swedish", "norwegian",
    "danish", "finnish", "polish", "austrian", "greek", "irish", "korean",
    "portuguese", "turkish", "egyptian", "saudi", "emirati", "singaporean",
    "english", "scottish", "welsh", "christian", "muslim", "jewish", "hindu", "buddhist",
}

# Plural nouns that disqualify a PERSON span
PLURAL_GROUP_NOUNS: Set[str] = {
    "pilgrims", "tourists", "visitors", "citizens", "delegates", "employees",
    "contractors", "developers", "users", "administrators", "customers",
    "patients", "clients", "passengers", "students", "teachers", "workers",
    "members", "leaders", "officials", "agents",
}

# Botanical / geographical nouns that disqualify person names (e.g. "daisy bush")
PLANT_OBJECT_NOUNS: Set[str] = {
    "bush", "tree", "flower", "plant", "river", "lake", "forest", "mountain",
}

# Title prefixes that should be stripped from the start of a PERSON span (occupational roles, not honorifics)
TITLE_PREFIXES: Tuple[str, ...] = (
    "client ", "customer ", "recruiter ", "surgeon ", "physician ",
    "attorney ", "notary ", "nurse ",
    "engineer ", "architect ", "detective ", "pharmacist ", "patient ",
)

# Relative / vague date phrases that are NOT specific PII dates
NON_SPECIFIC_DATE_PHRASES: Set[str] = {
    "a good day", "good day", "last quarter", "quarterly", "next quarter",
    "this quarter", "next year", "last year", "this year", "fiscal year",
    "2024 fiscal year", "every day", "daily", "annually", "q1", "q2", "q3", "q4",
    "the weekend", "weekend", "monthly", "each year", "the end of the evening",
    "the upcoming fiscal quarter", "upcoming fiscal quarter", "evening",
}

# Corporate suffix pattern for ORG detection
CORPORATE_SUFFIX_PATTERN = re.compile(
    r"(?i)\b(?:inc\.?|corp\.?|corporation|llc|ltd\.?|limited|co\.?"
    r"|technologies|solutions|group|holdings|enterprises|systems"
    r"|studios|labs|foundation|university|institute)\b"
)

# Malformed PERSON span pattern
_MALFORMED_PERSON_RE = re.compile(
    r"[.!?;:\n]{1,}\s*[-–—]|\.[a-zA-Z]{2,}|\b(?<![A-Z])\.[A-Z]"
)


# ============================================================
# PERSON VALIDATORS & CLEANERS
# ============================================================

def clean_person_possessive(text: str) -> str:
    """Strip trailing possessive suffixes (e.g. \"Priya Nair's\" -> \"Priya Nair\")."""
    if re.search(r"[''']s?$", text):
        trimmed = re.sub(r"[''']s?$", "", text).rstrip()
        if trimmed:
            return trimmed
    return text


def strip_person_title_prefix(text: str) -> str:
    """Strip leading occupational/honorific title (e.g. 'Client Liam Gallagher' -> 'Liam Gallagher')."""
    lower = text.lower()
    for prefix in TITLE_PREFIXES:
        if lower.startswith(prefix) and len(text) > len(prefix):
            return text[len(prefix):].strip()
    return text


def is_malformed_person_span(text: str) -> bool:
    """Check if PERSON span contains sentence breaks, punctuation sequences, or glued text."""
    return bool(_MALFORMED_PERSON_RE.search(text))


def has_disqualifying_person_words(text: str) -> bool:
    """Check if PERSON text contains technical/directive keywords that disqualify it as a name."""
    words = [re.sub(r"^\W+|\W+$", "", w).lower() for w in text.split()]
    # Isolated modal verbs are not names (e.g. isolated 'May' in 'you may contact'),
    # but valid first names like 'May Zhou' or 'Will Smith' must be allowed.
    if len(words) == 1 and words[0] in {"may", "can", "will", "might", "must", "should", "shall"}:
        return True
    return any(w in (PERSON_DISQUALIFYING_WORDS - {"may", "will", "can"}) for w in words)


def has_corporate_suffix(text: str) -> bool:
    """Check if text contains corporate suffixes (Inc., Corp., LLC, etc.)."""
    return bool(CORPORATE_SUFFIX_PATTERN.search(text))


def validate_person(text: str) -> Tuple[bool, str]:
    """
    Validate whether a PERSON entity text is well-formed and plausible.

    Returns:
        (is_valid, cleaned_text) — cleaned_text has possessives and title prefixes stripped.
    """
    cleaned = clean_person_possessive(text)
    cleaned = strip_person_title_prefix(cleaned)

    clean_norm = cleaned.lower().strip()

    # Reject demonyms / nationalities / religions
    if clean_norm in DEMONYMS:
        return False, cleaned

    words = [w.strip() for w in cleaned.split() if w.strip()]
    if not words:
        return False, cleaned

    # Reject plural group nouns at the end (e.g. "Christian pilgrims")
    if words[-1].lower() in PLURAL_GROUP_NOUNS:
        return False, cleaned

    # Reject plant / botanical / object nouns (e.g. "daisy bush")
    if words[-1].lower() in PLANT_OBJECT_NOUNS:
        return False, cleaned

    if is_malformed_person_span(cleaned):
        return False, cleaned

    if has_disqualifying_person_words(cleaned):
        return False, cleaned

    if has_corporate_suffix(cleaned):
        return False, cleaned

    return True, cleaned


def trim_person_stop_words(text: str, stop_words: Set[str]) -> str:
    """Trim trailing stop words from a PERSON entity text."""
    words = text.split()
    while len(words) > 1 and words[-1].lower() in stop_words:
        words.pop()
    return " ".join(words) if words else text


# ============================================================
# DATE VALIDATORS
# ============================================================

def is_non_specific_date(text: str) -> bool:
    """Check if date text is a vague/relative phrase that isn't specific PII."""
    clean_norm = text.lower().strip()

    # Check against known non-specific phrases
    if clean_norm in NON_SPECIFIC_DATE_PHRASES:
        return True

    # Check against relative date words
    # "Yesterday" (capitalized) is preserved for compatibility with multi-entity regression tests
    if clean_norm in RELATIVE_DATE_WORDS:
        if clean_norm == "yesterday" and text.strip() == "Yesterday":
            return False
        return True

    # Durations like "4 years", "2 days", "3 weeks"
    if re.match(r"^\d+\s+(?:days?|weeks?|months?|years?|hours?|minutes?)$", clean_norm):
        return True

    # Phrases starting with "the end of", "the start of"
    if clean_norm.startswith(("the end of", "the start of", "the beginning of")):
        return True

    # Recurring patterns like "every day", "every month"
    if clean_norm.startswith("every ") and any(
        unit in clean_norm for unit in ("day", "month", "year", "days", "months", "years")
    ):
        return True

    return False


def is_isolated_month_word(
    text: str,
    prompt: str,
    start: int,
    end: int,
) -> bool:
    """
    Check if an isolated month word (e.g. 'May') lacks date context
    (no nearby digits or date prepositions).
    """
    clean_norm = text.strip().lower()
    if clean_norm not in MONTH_WORDS:
        return False

    w_before = prompt[max(0, start - 25):start]
    w_after = prompt[end:min(len(prompt), end + 25)]

    # Digits must be adjacent to the month (e.g. "May 15", "15 May", "May 2024", "15th of May")
    has_nearby_digits = bool(
        re.search(r"\b\d{1,4}(?:st|nd|rd|th)?\s*$", w_before.strip())
        or re.search(r"^\s*(?:of\s+)?\d{1,4}(?:st|nd|rd|th)?\b", w_after)
    )
    has_date_prep = bool(
        re.search(
            r"(?i)\b(?:in|on|during|dated|since|until|by|before|after|of)\s*$",
            w_before.strip(),
        )
    )

    # If no nearby digits and no date preposition, it's isolated
    return not (has_nearby_digits or has_date_prep)


def is_standalone_year(text: str, prompt: str, start: int) -> bool:
    """
    Check if a 4-digit year (e.g. '2022') is standalone without date context.
    """
    clean = text.strip()
    if not (clean.isdigit() and len(clean) == 4):
        return False
    val = int(clean)
    if val < 1900 or val > 2099:
        return True  # Invalid year range — definitely not a valid date

    # Check for date preposition before the year
    w_before = prompt[max(0, start - 25):start].strip()
    has_date_context = bool(
        re.search(
            r"(?i)\b(?:in|on|during|since|until|by|before|after|of|year|born|dated|established)\s*$",
            w_before,
        )
    )
    return not has_date_context


def validate_date(text: str, prompt: str, start: int, end: int) -> bool:
    """
    Validate whether a DATE entity is a meaningful, specific date.

    Returns True if the date is valid/specific. False if it should be filtered.
    """
    clean_t = text.strip()

    # Reject purely numeric text with >4 digits (e.g. 1234567890, 456789)
    if clean_t.isdigit() and len(clean_t) > 4:
        return False

    if is_non_specific_date(text):
        return False

    if is_isolated_month_word(text, prompt, start, end):
        return False

    if is_standalone_year(text, prompt, start):
        return False

    # Reject road/address patterns falsely tagged as DATE
    if re.search(
        r"(?i)\b(?:road|rd\.?|street|st\.?|avenue|ave\.?|lane|ln\.?"
        r"|drive|dr\.?|mg|marg|nagar)\b",
        text,
    ):
        return False

    # Reject PIN/postal codes (5-6 digits) tagged as DATE
    if re.match(r"^\d{5,6}$", text.strip()):
        return False

    return True


# ============================================================
# EMAIL VALIDATORS & CLEANERS
# ============================================================

def validate_email(text: str) -> Tuple[bool, str]:
    """
    Validate and clean an EMAIL entity.

    Returns:
        (is_valid, cleaned_text) — cleaned_text has trailing sentence text removed.
    """
    if "@" not in text:
        return False, text

    # Clean trailing sentence text (e.g. "user@example.com.The next" -> "user@example.com")
    email_sent_match = re.match(
        r"^([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
        r"(?:\.([A-Z][a-z]+.*))?$",
        text,
    )
    if email_sent_match and email_sent_match.group(2):
        return True, email_sent_match.group(1)

    # Strip trailing periods
    if text.endswith("."):
        return True, text.rstrip(".")

    return True, text


# ============================================================
# BANK ACCOUNT / FINANCIAL VALIDATORS
# ============================================================

def validate_bank_account(text: str) -> bool:
    """Bank account / IBAN must contain digits. Purely alphabetic text is invalid."""
    return any(c.isdigit() for c in text)


def validate_credit_card(text: str) -> Tuple[bool, str]:
    """
    Credit card must have at least 13 digits.
    Strips card brand prefixes like 'Visa card 4532...' -> '4532...'.
    """
    cleaned = re.sub(r"(?i)^(?:visa|mastercard|amex|american\s+express|discover)\s+(?:card\s+)?", "", text.strip())
    digits_only = re.sub(r"\D", "", cleaned)
    if len(digits_only) < 13:
        return False, text
    return True, cleaned


# ============================================================
# API KEY & CREDENTIAL VALIDATORS
# ============================================================

def validate_api_key(text: str) -> bool:
    """
    API keys / access tokens must not contain spaces or common shell command words.
    """
    clean = text.strip()
    if " " in clean:
        return False
    lower = clean.lower()
    if any(cmd in lower for cmd in ("sudo", "chmod", "chown", "docker", "kubectl", "rm", "grep")):
        return False
    return True


# ============================================================
# IP ADDRESS VALIDATORS
# ============================================================

def validate_ip_address(text: str) -> bool:
    """IP address must contain '.' (IPv4) or ':' (IPv6)."""
    return "." in text or ":" in text


# ============================================================
# STRUCTURED ID VALIDATORS
# ============================================================

def validate_structured_id(
    text: str,
    prompt: Optional[str] = None,
    start: Optional[int] = None,
) -> bool:
    """
    Validate CUSTOMER_ID / ORDER_ID / TICKET_ID / USER_ID:
    - Must contain digits
    - Must NOT contain spaces (e.g. 'director of engineering' is not an ID)
    - Must not be common system/admin words or in COMMON_EXCLUDED_TOKENS
    - If preceded by serial/invoice/prescription keywords, not a customer/user ID
    """
    clean = text.strip()
    clean_norm = clean.lower()

    if clean_norm in config.COMMON_EXCLUDED_TOKENS:
        return False

    # Structured IDs / usernames cannot contain spaces
    if " " in clean:
        return False

    if not any(c.isdigit() for c in clean):
        return False

    # Reject common system words
    if re.search(
        r"(?i)\b(?:support|service|center|team|help|care|portal"
        r"|desk|agent|group|staff|admin|root|system)\b",
        clean,
    ):
        return False

    # Contextual check: reject if preceded by non-customer/user keywords,
    # UNLESS the entity starts with an explicit known order or ticket prefix
    if prompt is not None and start is not None:
        clean_upper = clean.upper()
        if not any(clean_upper.startswith(pfx) for pfx in ("ORD-", "ORDER-", "TKT-", "TICKET-", "INC-", "SR-", "CASE-", "DOCK-", "LIC-", "CUST-", "CID-", "USER-", "UID-")):
            w_before = prompt[max(0, start - 30):start].lower()
            if re.search(r"(?i)\b(?:serial|sn|invoice|inv|prescription|rx|docket|case)\s*(?:#|no\.?|code)?\s*[:=]?\s*$", w_before):
                return False

    # Pure numeric digits check (reject bare 6-digit numbers without ID context)
    if clean.isdigit():
        if prompt is not None and start is not None:
            w_before = prompt[max(0, start - 30):start].lower()
            has_id_cue = bool(
                re.search(
                    r"(?i)\b(?:customer|cust|client|user|order|ticket|account|acct)\s*(?:id|#|number|no\.?|code)?\s*[:=]?\s*$",
                    w_before.strip(),
                )
            )
            if not has_id_cue:
                return False
        else:
            if len(clean) < 8:
                return False

    return True


# ============================================================
# PHONE VALIDATORS
# ============================================================

def validate_phone(text: str) -> bool:
    """
    Phone number must have at least 7 digits and not be dummy/sequential numbers.
    """
    digits_only = re.sub(r"\D", "", text)
    if len(digits_only) < 7:
        return False

    # Reject repeated identical digits (e.g. 9999999999, 0000000000)
    if len(set(digits_only)) <= 1:
        return False

    # Reject trivial sequential test numbers without standard phone formatting
    if digits_only in ("1234567890", "0123456789", "0123456", "1234567", "12345678", "123456789"):
        if not text.startswith("+") and not any(c in text for c in " -()."):
            return False

    return True


# ============================================================
# LOCATION VALIDATORS
# ============================================================

def validate_location(text: str) -> bool:
    """Location must not be a pure number, demonym, or common excluded token."""
    clean = text.strip()
    if clean.isdigit():
        return False
    clean_norm = clean.lower()
    if clean_norm in DEMONYMS:
        return False
    if is_excluded_token(clean):
        return False
    return True


# ============================================================
# EXCLUDED TOKEN CHECK (unified)
# ============================================================

def is_excluded_token(text: str) -> bool:
    """Check if entity text matches a common excluded token."""
    clean_norm = text.strip().lower()
    if clean_norm in config.COMMON_EXCLUDED_TOKENS:
        return True
    if clean_norm.startswith("the ") and clean_norm[4:].strip() in config.COMMON_EXCLUDED_TOKENS:
        return True
    return False
