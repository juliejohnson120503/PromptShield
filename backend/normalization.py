"""
PromptShield AI — Normalization Module.
Normalizes detected entity values to canonical representations for
consistent processing, comparison, and policy evaluation.

Phase 1 addition: normalizers for IP_ADDRESS, BANK_ACCOUNT, and
configurable ID types (ORDER_ID, USER_ID, CUSTOMER_ID).
Also adds a type-normalization helper used by the entity fusion engine.
"""

import re
from typing import Optional
from backend.models import EntityType


# ---------------------------------------------------------------------------
# Per-type normalizers
# ---------------------------------------------------------------------------

def normalize_email(raw_val: str) -> str:
    """Lowercase and strip whitespace from email addresses."""
    return raw_val.strip().lower()


def normalize_phone(raw_val: str) -> str:
    """
    Normalize phone number:
    Preserve leading '+' if present, strip all spaces, parentheses, hyphens, and dots.
    """
    cleaned = raw_val.strip()
    has_plus = cleaned.startswith("+")
    digits = re.sub(r"\D", "", cleaned)
    return f"+{digits}" if has_plus else digits


def normalize_credit_card(raw_val: str) -> str:
    """Normalize credit card: strip spaces and hyphens to leave continuous digit string."""
    return re.sub(r"\D", "", raw_val.strip())


def normalize_api_key(raw_val: str) -> str:
    """Normalize API keys by stripping surrounding quotes and whitespace."""
    return raw_val.strip().strip("'\"`")


def normalize_password(raw_val: str) -> str:
    """
    Normalize password/credential:
    If captured in a key-value format (e.g. password: 'xyz' or pwd="xyz"),
    extract only the credential value and strip surrounding delimiters/quotes.
    """
    val = raw_val.strip()
    kv_match = re.search(r'[:=]\s*([\"\`\'"]?)(.+?)\1$', val)
    if kv_match:
        val = kv_match.group(2)
    return val.strip().strip("'\"`")


def normalize_bank_account(raw_val: str) -> str:
    """Normalize IBAN / bank account: uppercase, strip spaces and hyphens."""
    return re.sub(r"[\s\-]", "", raw_val.strip()).upper()


def normalize_ip_address(raw_val: str) -> str:
    """Normalize IP address by stripping surrounding whitespace."""
    return raw_val.strip()


def normalize_id(raw_val: str) -> str:
    """Normalize configurable IDs: strip whitespace and collapse internal spaces."""
    return re.sub(r"\s+", "", raw_val.strip())


def normalize_text_entity(raw_val: str) -> str:
    """Normalize names, organizations, and locations by collapsing redundant spaces."""
    cleaned = re.sub(r"\s+", " ", raw_val.strip())
    cleaned = re.sub(r"[.,;:!]+$", "", cleaned)
    return cleaned


def normalize_date(raw_val: str) -> str:
    """Normalize date strings by cleaning spaces and edge punctuation."""
    cleaned = re.sub(r"\s+", " ", raw_val.strip())
    cleaned = re.sub(r"[.,;:!]+$", "", cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Dispatch function (unchanged public API)
# ---------------------------------------------------------------------------

def normalize_entity_value(entity_type: EntityType, raw_val: str) -> str:
    """Dispatch value to the appropriate normalizer based on its EntityType."""
    if entity_type == EntityType.EMAIL:
        return normalize_email(raw_val)
    elif entity_type == EntityType.PHONE:
        return normalize_phone(raw_val)
    elif entity_type == EntityType.CREDIT_CARD:
        return normalize_credit_card(raw_val)
    elif entity_type in (EntityType.API_KEY, EntityType.ACCESS_TOKEN):
        return normalize_api_key(raw_val)
    elif entity_type == EntityType.PASSWORD:
        return normalize_password(raw_val)
    elif entity_type == EntityType.BANK_ACCOUNT:
        return normalize_bank_account(raw_val)
    elif entity_type == EntityType.IP_ADDRESS:
        return normalize_ip_address(raw_val)
    elif entity_type in (EntityType.ORDER_ID, EntityType.USER_ID, EntityType.CUSTOMER_ID):
        return normalize_id(raw_val)
    elif entity_type in (EntityType.PERSON, EntityType.ORGANIZATION, EntityType.LOCATION):
        return normalize_text_entity(raw_val)
    elif entity_type == EntityType.DATE:
        return normalize_date(raw_val)
    return raw_val.strip()
