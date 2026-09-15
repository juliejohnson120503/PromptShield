"""
Entity normalization module for PromptShield AI.
Normalizes detected values to canonical representations for consistent
processing, comparison, and policy evaluation.
"""

import re
from typing import Optional
from backend.models import EntityType


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
    # Check for key-value assignment patterns
    kv_match = re.search(r"[:=]\s*([\"'`]?)(.+?)\1$", val)
    if kv_match:
        val = kv_match.group(2)
    return val.strip().strip("'\"`")


def normalize_text_entity(raw_val: str) -> str:
    """Normalize names, organizations, and locations by collapsing redundant spaces."""
    cleaned = re.sub(r"\s+", " ", raw_val.strip())
    # Strip any trailing punctuation like commas or periods often caught at token boundaries
    cleaned = re.sub(r"[.,;:!]+$", "", cleaned)
    return cleaned


def normalize_date(raw_val: str) -> str:
    """Normalize date strings by cleaning spaces and edge punctuation."""
    cleaned = re.sub(r"\s+", " ", raw_val.strip())
    cleaned = re.sub(r"[.,;:!]+$", "", cleaned)
    return cleaned


def normalize_entity_value(entity_type: EntityType, raw_val: str) -> str:
    """Dispatch value to the appropriate normalizer based on its EntityType."""
    if entity_type == EntityType.EMAIL:
        return normalize_email(raw_val)
    elif entity_type == EntityType.PHONE:
        return normalize_phone(raw_val)
    elif entity_type == EntityType.CREDIT_CARD:
        return normalize_credit_card(raw_val)
    elif entity_type == EntityType.API_KEY:
        return normalize_api_key(raw_val)
    elif entity_type == EntityType.PASSWORD:
        return normalize_password(raw_val)
    elif entity_type in (EntityType.PERSON, EntityType.ORGANIZATION, EntityType.LOCATION):
        return normalize_text_entity(raw_val)
    elif entity_type == EntityType.DATE:
        return normalize_date(raw_val)
    return raw_val.strip()
