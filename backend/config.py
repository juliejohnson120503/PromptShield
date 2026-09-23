"""
PromptShield AI — Central Detection Configuration (Phase 1).

This module is the single source of truth for:
  - All compiled regex patterns for structured sensitive data.
  - Presidio entity type → PromptShield EntityType normalization map.
  - spaCy label → PromptShield EntityType normalization map.
  - Entity type conflict-resolution priority order.
  - Confidence baselines for each detector source.

Design rationale
----------------
Keeping every pattern, mapping, and threshold here (instead of scattered
through individual detector modules) makes them:
  • Readable at a glance
  • Testable in isolation
  • Easy to extend without touching detector logic
  • Auditable for review / academic presentation
"""

import re
from typing import Dict, List, Tuple, Set


# ============================================================
# COMMON EXCLUDED TOKENS (Keywords falsely tagged as entities)
# ============================================================
COMMON_EXCLUDED_TOKENS: Set[str] = {
    # Credentials, technical protocols & field labels
    "password", "psswrd", "pswrd", "pswd", "pwd", "pass", "pin", "passcode",
    "email", "mail", "phone", "username", "user", "name", "id", "card",
    "letter", "note", "message", "text", "prompt", "code", "token", "key",
    "ip", "ip address", "url", "uri", "http", "https", "ftp", "dns", "tcp", "udp",
    "vpn", "api", "ai", "ml", "os", "ssh", "ssl", "tls", "customer", "order", "ticket",
    # Pronouns & function words falsely tagged as PERSON
    "i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    # Generic collective nouns, roles, and meeting concepts (not named organizations)
    "team", "the team", "network team", "the network team", "support team", "the support team",
    "project", "database", "server", "meeting", "architecture", "report",
    "organization", "organizations", "company", "companies", "department", "departments",
    "model", "models", "dataset", "datasets", "pattern", "patterns", "task", "tasks",
    "package", "replacement",
}


# ============================================================
# EMAIL
# ============================================================
EMAIL_PATTERN: re.Pattern = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# ============================================================
# PHONE
# ============================================================
# Each pattern is (compiled_regex, detector_label)
PHONE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # International with leading + code: e.g. +91 98765 43210, +1 (555) 234-5678, +44 20 7946 0958
    (
        re.compile(
            r"\+\d{1,3}[\s-]?(?:\(\d{1,4}\)[\s.-]?)?\d{2,5}[\s.-]?\d{2,5}(?:[\s.-]?\d{2,5})?\b"
        ),
        "phone_international",
    ),
    # Separated local/national phone numbers (must contain parens, spaces, or hyphens):
    # e.g. (555) 234-5678, 020-7946-0958, 98765-43210, 98765 43210
    (
        re.compile(
            r"(?:\(\d{2,4}\)[\s.-]?\d{3,4}[\s.-]?\d{3,5}|\b[6-9]\d{4}[\s.-]\d{5}\b|\b\d{3}[.-]\d{3}[.-]\d{4}\b|\b\d{4,5}[\s.-]\d{5,6}\b)\b"
        ),
        "phone_separated",
    ),
    # Standard 10-digit Indian / US without punctuation: 9876543210, 8005551234
    (
        re.compile(r"\b(?:[6-9]\d{9}|[2-9]\d{2}[2-9]\d{6})\b"),
        "phone_10digit",
    ),
]

# ============================================================
# CREDIT CARD  (candidates – must be Luhn-validated downstream)
# ============================================================
CREDIT_CARD_PATTERN: re.Pattern = re.compile(
    r"\b(?:\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,4}"
    r"|\d{4}[ -]?\d{6}[ -]?\d{4,5}"
    r"|\d{13,19})\b"
)

# ============================================================
# API KEY / ACCESS TOKEN
# ============================================================
# Each entry: (compiled_regex, detector_label, confidence)
API_KEY_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
    # OpenAI API keys: sk-..., sk-proj-..., sk-live-..., sk-test-...
    (
        re.compile(r"\b(?:sk-(?:proj-|live-|test-)?[A-Za-z0-9_\-]{8,})\b"),
        "openai_api_key",
        0.99,
    ),
    # Google API Key: AIza... (typically 39 characters total)
    (
        re.compile(r"\bAIza[0-9A-Za-z_\-]{30,45}\b"),
        "google_api_key",
        0.99,
    ),
    # GitHub Tokens: ghp_..., github_pat_...
    (
        re.compile(r"\b(?:ghp_[0-9a-zA-Z]{36}|github_pat_[0-9a-zA-Z_]{22,})\b"),
        "github_pat",
        0.99,
    ),
    # AWS Access Key ID: AKIA...
    (
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "aws_access_key",
        0.98,
    ),
    # Generic Key/Secret assignment: api_key = "...", token: '...'
    (
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret[_-]?token|auth[_-]?token|access[_-]?token)"
            r"\s*[:=]\s*['\"]?([A-Za-z0-9_\-\.]{16,})['\"]?"
        ),
        "generic_key_assignment",
        0.95,
    ),
    # Bearer tokens
    (
        re.compile(r"(?i)\bbearer\s+([A-Za-z0-9_\-\.]{20,})\b"),
        "bearer_token",
        0.93,
    ),
]

# JWT token (three base64url segments separated by dots)
ACCESS_TOKEN_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
    (
        re.compile(
            r"\bey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
        ),
        "jwt_token",
        0.97,
    ),
]

# ============================================================
# PASSWORD / CREDENTIAL
# ============================================================
PASSWORD_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # password = "...", pwd: "...", psswrd: '...', pass: '...'
    (
        re.compile(
            r"(?i)\b(?:password|passwd|psswrd|pswrd|pswd|pass|pwd|passphrase|userpass|passcode|pword)"
            r"\s*[:=]\s*['\"]?([^'\"\s\r\n,.;]{3,})['\"]?"
        ),
        "password_assignment",
    ),
    # "my password is ...", "psswrd is 98765432", "my psswrd is ...", "pwd is ..."
    (
        re.compile(
            r"(?i)\b(?:(?:my|the|user|acc|account|login)\s+)?(?:login\s+)?(?:password|passwd|psswrd|pswrd|pswd|pass|pwd|passphrase|passcode|pword)"
            r"\s+(?:is|was|=|:)\s*['\"]?([^'\"\s\r\n,.;]{3,})['\"]?"
        ),
        "password_natural_lang",
    ),
    # "pin is 1234", "pin: 1234", "my pin code is 4321"
    (
        re.compile(
            r"(?i)\b(?:(?:my|the|user|login|card|atm)\s+)?(?:pin(?:\s*code)?|passcode)"
            r"\s*(?:is|was|=|:)\s*['\"]?([0-9]{4,8})['\"]?"
        ),
        "pin_natural_lang",
    ),
    # "credentials: username/password"
    (
        re.compile(
            r"(?i)\b(?:credentials?\s*[:=]\s*['\"]?([^'\"\s\r\n]+)['\"]?)"
        ),
        "credential_assignment",
    ),
]

# ============================================================
# BANK ACCOUNT / IBAN
# ============================================================
BANK_ACCOUNT_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
    # IBAN: e.g. GB82 WEST 1234 5698 7654 32  (up to 34 alphanumeric + spaces)
    (
        re.compile(
            r"\b[A-Z]{2}\d{2}[\s-]?(?:[A-Z0-9]{4}[\s-]?){1,7}[A-Z0-9]{1,4}\b"
        ),
        "iban",
        0.90,
    ),
    # Generic account number keyword trigger: account number 1234567890
    (
        re.compile(
            r"(?i)\b(?:account\s*(?:number|no\.?|#)\s*[:\-]?\s*)(\d{6,18})\b"
        ),
        "account_number",
        0.85,
    ),
]

# ============================================================
# IP ADDRESS
# ============================================================
IP_ADDRESS_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
    # IPv4
    (
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        "ipv4",
        0.97,
    ),
    # IPv6 (simplified canonical form)
    (
        re.compile(
            r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
        ),
        "ipv6",
        0.97,
    ),
]

# ============================================================
# CONFIGURABLE ID PATTERNS (project-specific)
# ============================================================
# ============================================================
# PROJECT-SPECIFIC / STRUCTURED IDs
# ============================================================
ID_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
    # ORDER_ID: explicit prefix format (e.g. ORD-12345, ORD-78291, ORDER-55123)
    (
        re.compile(r"\b(?:ORD|ORDER)[-_][A-Za-z0-9]{4,16}\b", re.IGNORECASE),
        "order_id",
        0.95,
    ),
    # ORDER_ID: keyword context trigger (e.g. "order #ORD-123456", "order ORD-78291")
    (
        re.compile(r"(?i)\border(?:\s+id|\s+#|no\.?|number)?\s*(?:is|as|[:\-#=])?\s*(ORD[-\s]?[A-Za-z0-9]{4,12})\b"),
        "order_id",
        0.92,
    ),
    # CUSTOMER_ID: explicit prefix format (e.g. CUST-10458, CUST-90812, CUSTOMER-1234)
    (
        re.compile(r"\b(?:CUST|CUSTOMER)[-_][A-Za-z0-9]{4,16}\b", re.IGNORECASE),
        "customer_id",
        0.95,
    ),
    # CUSTOMER_ID: keyword context trigger (e.g. "customer id: CUST-10458", "customer ID as 10458")
    (
        re.compile(r"(?i)\bcustomer(?:\s+id|_id|\s*#|\s*no\.?|\s+number|\s+code)\s*(?:is|as|[:\-#=])?\s*([A-Za-z0-9_\-]*\d[A-Za-z0-9_\-]*)\b"),
        "customer_id",
        0.90,
    ),
    # TICKET_ID: explicit prefix format (e.g. TKT-99182, TICKET-1042, SR-9912, INC-8821)
    (
        re.compile(r"\b(?:TKT|TICKET|SR|INC)[-_][A-Za-z0-9]{4,16}\b", re.IGNORECASE),
        "ticket_id",
        0.95,
    ),
    # TICKET_ID: keyword context trigger (e.g. "ticket #99182", "ticket id: TKT-99182")
    (
        re.compile(r"(?i)\bticket(?:\s+id|_id|\s*#|\s*no\.?|\s+number)?\s*(?:is|as|[:\-#=])?\s*([A-Za-z0-9_\-]*\d[A-Za-z0-9_\-]*)\b"),
        "ticket_id",
        0.90,
    ),
    # USER_ID: keyword context trigger e.g. "user id: U12345", "user id as usr_998"
    (
        re.compile(r"(?i)\buser(?:\s+id|_id|\s*#|\s*no\.?|\s+number|\s+code)\s*(?:is|as|[:\-#=])?\s*([A-Za-z0-9_\-]*\d[A-Za-z0-9_\-]*)\b"),
        "user_id",
        0.85,
    ),
]

# ============================================================
# PHYSICAL / POSTAL ADDRESS
# ============================================================
ADDRESS_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
    # Full postal address with street number, street designator, optional localities, and 5-6 digit PIN/postal code
    # e.g., "24 MG Road, Kochi, Kerala 682016" or "120 Nehru Marg, Pune, Maharashtra 411001"
    (
        re.compile(
            r"(?<![-A-Za-z0-9/])\b\d{1,5}(?:[/-]\d{1,5})?[A-Za-z]?\s+"
            r"(?:(?:(?![a-z])[A-Za-z0-9\'-]+|\d+(?:st|nd|rd|th)?)\s+){1,4}"
            r"(?:Road|Rd\.?|Street|St\.?|Avenue|Ave\.?|Lane|Ln\.?|Drive|Dr\.?|Boulevard|Blvd\.?|Marg|Nagar|Colony|Sector|Bazaar|Cross|Main|Way|Place)\b"
            r"(?:,\s*[A-Za-z0-9\s\'-]+){0,4}(?:,\s*|\s+)\b\d{5,6}\b",
        ),
        "address_with_postal_code",
        0.96,
    ),
    # Street address without postal code (e.g., "24 MG Road", "221B Baker Street")
    (
        re.compile(
            r"(?<![-A-Za-z0-9/])\b\d{1,5}(?:[/-]\d{1,5})?[A-Za-z]?\s+"
            r"(?:(?:(?![a-z])[A-Za-z0-9\'-]+|\d+(?:st|nd|rd|th)?)\s+){1,4}"
            r"(?:Road|Rd\.?|Street|St\.?|Avenue|Ave\.?|Lane|Ln\.?|Drive|Dr\.?|Boulevard|Blvd\.?|Marg|Nagar|Colony|Sector|Bazaar|Cross|Main|Way|Place)\b",
        ),
        "street_address",
        0.90,
    ),
]

# ============================================================
# DATE PATTERNS  (kept for regex_detector fallback)
# ============================================================
DATE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # ISO-like YYYY-MM-DD or YYYY/MM/DD
    (
        re.compile(r"\b(?:19|20)\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])\b"),
        "date_iso",
    ),
    # DD-MM-YYYY or DD/MM/YYYY
    (
        re.compile(r"\b(?:0[1-9]|[12]\d|3[01])[-/.](?:0[1-9]|1[0-2])[-/.](?:19|20)\d{2}\b"),
        "date_dmy",
    ),
    # Month DD, YYYY  or  DD Month YYYY
    (
        re.compile(
            r"\b(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+(?:19|20)\d{2}\b",
            re.IGNORECASE,
        ),
        "date_month_dd_yyyy",
    ),
    (
        re.compile(
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|"
            r"July|August|September|October|November|December|"
            r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(?:,)?\s+(?:19|20)\d{2}\b",
            re.IGNORECASE,
        ),
        "date_dd_month_yyyy",
    ),
]


# ============================================================
# PRESIDIO → PROMPTSHIELD ENTITY TYPE MAPPING
# ============================================================
# Maps Presidio's canonical entity type strings to PromptShield EntityType values.
# Any Presidio type NOT in this dict is silently ignored (not treated as an error).
PRESIDIO_ENTITY_MAPPING: Dict[str, str] = {
    "EMAIL_ADDRESS":    "EMAIL",
    "PHONE_NUMBER":     "PHONE",
    "CREDIT_CARD":      "CREDIT_CARD",
    "CRYPTO":           "API_KEY",           # Crypto wallet addresses → API_KEY category
    "IP_ADDRESS":       "IP_ADDRESS",
    "IBAN_CODE":        "BANK_ACCOUNT",
    "PERSON":           "PERSON",
    "LOCATION":         "LOCATION",
    "DATE_TIME":        "DATE",
    "NRP":              "PERSON",            # Nationality, religion, political group → broad PERSON
    "MEDICAL_LICENSE":  "API_KEY",           # Professional IDs → credential-like
    "US_SSN":           "USER_ID",
    "US_DRIVER_LICENSE": "USER_ID",
    "US_PASSPORT":      "USER_ID",
    "US_BANK_NUMBER":   "BANK_ACCOUNT",
    "ORGANIZATION":     "ORGANIZATION",
    "CUSTOMER_ID":      "CUSTOMER_ID",
    "ORDER_ID":         "ORDER_ID",
    "TICKET_ID":        "TICKET_ID",
}

# Presidio entity strings to request from the analyzer engine
# Note: ORGANIZATION is excluded — Presidio 2.x does not ship an 'en' ORGANIZATION
# recognizer by default and emits a noisy warning; spaCy covers ORG via NER.
PRESIDIO_REQUESTED_ENTITIES: List[str] = [
    e for e in PRESIDIO_ENTITY_MAPPING.keys() if e != "ORGANIZATION"
]


# ============================================================
# SPACY LABEL → PROMPTSHIELD ENTITY TYPE MAPPING
# ============================================================
SPACY_ENTITY_MAPPING: Dict[str, str] = {
    "PERSON":   "PERSON",
    "ORG":      "ORGANIZATION",
    "GPE":      "LOCATION",        # Geo-political entity (country, city, state)
    "LOC":      "LOCATION",        # Non-GPE locations (mountain, water, etc.)
    "DATE":     "DATE",
    "NORP":     "PERSON",          # Nationalities, religious, political groups
}


# ============================================================
# GLINER CONFIGURATION & ENTITY MAPPING (Deep Neural NER)
# ============================================================
GLINER_MODEL_NAME: str = "urchade/gliner_small-v2.1"
GLINER_DEFAULT_LABELS: List[str] = [
    "person",
    "organization",
    "location",
    "password",
    "username",
]
GLINER_ENTITY_MAPPING: Dict[str, str] = {
    "person":           "PERSON",
    "organization":     "ORGANIZATION",
    "company":          "ORGANIZATION",
    "location":         "LOCATION",
    "city":             "LOCATION",
    "country":          "LOCATION",
    "password":         "PASSWORD",
    "passcode":         "PASSWORD",
    "pin":              "PASSWORD",
    "username":         "USER_ID",
    "email":            "EMAIL",
    "phone number":     "PHONE",
    "credit card":      "CREDIT_CARD",
    "bank account":     "BANK_ACCOUNT",
    "date":             "DATE",
}


# ============================================================
# CONFLICT RESOLUTION — ENTITY TYPE PRIORITY
# ============================================================
# Higher number = higher priority when resolving overlapping spans.
# Structured / credential detections beat generic NER.
ENTITY_TYPE_PRIORITY: Dict[str, int] = {
    "API_KEY":       10,
    "ACCESS_TOKEN":  10,
    "CREDIT_CARD":    9,
    "PASSWORD":       8,
    "EMAIL":          7,
    "PHONE":          6,
    "BANK_ACCOUNT":   6,
    "IP_ADDRESS":     6,
    "ORDER_ID":       6,
    "CUSTOMER_ID":    6,
    "TICKET_ID":      6,
    "USER_ID":        5,
    "ADDRESS":        5,  # Higher than individual LOCATION so full address subsumes nested city/state
    "DATE":           4,
    "PERSON":         3,
    "ORGANIZATION":   3,
    "LOCATION":       2,
}

# SOURCE priority when entity types are the same but sources differ.
# Higher = preferred on exact-span merging.
SOURCE_PRIORITY: Dict[str, int] = {
    "regex":            5,    # deterministic, explicit format pattern
    "luhn_credit_card": 5,
    "presidio":         4,    # ML-backed, well-trained
    "gliner":           4,    # deep neural zero-shot transformer
    "spacy":            3,    # pretrained statistical NER
    "heuristic_ner":    2,    # built-in gazetteer / pattern fallback
}


# ============================================================
# CONFIDENCE BASELINES
# ============================================================
# Reference values — actual detector modules may override these.
CONFIDENCE_BASELINES: Dict[str, float] = {
    "regex_email":          0.98,
    "regex_phone":          0.92,
    "luhn_credit_card":     0.96,
    "regex_api_key":        0.99,  # varies by sub-pattern; see API_KEY_PATTERNS
    "regex_password":       0.92,
    "regex_bank_account":   0.88,
    "regex_ip":             0.97,
    "regex_id":             0.90,
    "regex_ticket_id":      0.95,
    "regex_address":        0.95,
    "regex_date":           0.90,
    "presidio":             None,  # Presidio provides per-entity scores; use as-is
    "gliner":               0.88,  # GLiNER zero-shot neural NER baseline
    "spacy_ner":            0.85,
    "heuristic_ner":        0.88,
}
