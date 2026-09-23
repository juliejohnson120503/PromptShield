"""
PromptShield AI — Regex / Rule-Based Detector (Phase 1).

This module is the STRUCTURED, DETERMINISTIC detection engine.
It handles entity types that have well-defined, explicit lexical patterns:

    EMAIL, PHONE, CREDIT_CARD (with Luhn validation), API_KEY,
    ACCESS_TOKEN, PASSWORD, BANK_ACCOUNT, IP_ADDRESS,
    ORDER_ID, USER_ID, CUSTOMER_ID, DATE.

Design decisions
----------------
• Uses only Python standard-library `re` — zero ML dependencies.
• All patterns are imported from `backend/config.py`; none are
  defined here.  This keeps the detector logic clean and the patterns
  easy to audit.
• Luhn validation is kept here (not in config) because it is an
  algorithm, not a pattern.
• Confidence values come from `config.CONFIDENCE_BASELINES` so they
  can be tuned in one place.
• Does NOT decide whether a detected entity must be masked.
• Returns `DetectedEntity` objects with `source="regex"`.

Limitations
-----------
• Regex cannot understand semantics.  A 16-digit number that passes
  Luhn is a *candidate* credit card — not a proven one.
• Phone patterns may produce false positives for long numeric IDs.
• API key patterns are signature-based and will miss novel formats.
• Results feed into EntityFusionEngine for deduplication before use.
"""

import re
from typing import List

from backend.models import EntityType, DetectedEntity
from backend.normalization import normalize_entity_value
from backend import config


# ============================================================
# Luhn (mod-10) Validator — Credit Card
# ============================================================

def is_luhn_valid(number_str: str) -> bool:
    """
    Validate a numeric string using the Luhn (mod-10) algorithm.

    Strips all non-digit characters before validation so that
    space/hyphen separated card numbers (e.g. '4111 1111 1111 1111')
    are handled correctly.

    Returns False for strings shorter than 13 or longer than 19 digits
    to exclude arbitrary numeric sequences such as order numbers.
    """
    digits = [int(d) for d in re.sub(r"\D", "", number_str)]
    if len(digits) < 13 or len(digits) > 19:
        return False

    checksum = 0
    for idx, digit in enumerate(reversed(digits)):
        if idx % 2 == 1:
            doubled = digit * 2
            checksum += (doubled - 9) if doubled > 9 else doubled
        else:
            checksum += digit
    return checksum % 10 == 0


def detect_card_brand(card_digits: str) -> str:
    """Identify the credit card issuer brand based on leading digits."""
    clean = re.sub(r"\D", "", card_digits)
    if clean.startswith("4"):
        return "Visa"
    if clean.startswith(("51", "52", "53", "54", "55")) or (
        len(clean) >= 4 and 2221 <= int(clean[:4]) <= 2720
    ):
        return "Mastercard"
    if clean.startswith(("34", "37")):
        return "American Express"
    if clean.startswith(("6011", "65")) or (
        len(clean) >= 6 and 622126 <= int(clean[:6]) <= 622925
    ):
        return "Discover"
    if clean.startswith(("300", "301", "302", "303", "304", "305", "36", "38")):
        return "Diners Club"
    return "Unknown"


# ============================================================
# RegexDetector
# ============================================================

class RegexDetector:
    """
    Structured / deterministic entity detector.

    Applies compiled regex patterns from ``backend.config`` to identify
    explicit-format sensitive entities in a prompt string.

    All results carry ``source="regex"`` so the fusion engine can
    distinguish them from Presidio / spaCy detections.
    """

    def detect(self, prompt: str) -> List[DetectedEntity]:
        """
        Run all structured regex detectors over *prompt*.

        Returns a flat list of ``DetectedEntity`` objects ordered by
        first appearance.  Overlap resolution is handled by the caller
        (EntityFusionEngine), not here.
        """
        if not prompt or not prompt.strip():
            return []

        candidates: List[DetectedEntity] = []
        candidates.extend(self._detect_emails(prompt))
        candidates.extend(self._detect_phones(prompt))
        candidates.extend(self._detect_credit_cards(prompt))
        candidates.extend(self._detect_api_keys(prompt))
        candidates.extend(self._detect_access_tokens(prompt))
        candidates.extend(self._detect_passwords(prompt))
        candidates.extend(self._detect_bank_accounts(prompt))
        candidates.extend(self._detect_ip_addresses(prompt))
        candidates.extend(self._detect_ids(prompt))
        candidates.extend(self._detect_addresses(prompt))
        candidates.extend(self._detect_dates(prompt))
        return candidates

    # ------------------------------------------------------------------
    # Private helpers — one per entity category
    # ------------------------------------------------------------------

    def _detect_emails(self, prompt: str) -> List[DetectedEntity]:
        entities = []
        for match in config.EMAIL_PATTERN.finditer(prompt):
            raw = match.group(0)
            entities.append(
                DetectedEntity(
                    text=raw,
                    entity_type=EntityType.EMAIL,
                    start=match.start(),
                    end=match.end(),
                    normalized_value=normalize_entity_value(EntityType.EMAIL, raw),
                    confidence=config.CONFIDENCE_BASELINES["regex_email"],
                    source="regex",
                    detector="regex_email",
                )
            )
        return entities

    def _detect_phones(self, prompt: str) -> List[DetectedEntity]:
        entities = []
        for pattern, label in config.PHONE_PATTERNS:
            for match in pattern.finditer(prompt):
                raw = match.group(0)
                digits_only = re.sub(r"\D", "", raw)
                # Reject sequences too short or too long to be phone numbers
                if len(digits_only) < 7 or len(digits_only) > 15:
                    continue
                entities.append(
                    DetectedEntity(
                        text=raw,
                        entity_type=EntityType.PHONE,
                        start=match.start(),
                        end=match.end(),
                        normalized_value=normalize_entity_value(EntityType.PHONE, raw),
                        confidence=config.CONFIDENCE_BASELINES["regex_phone"],
                        source="regex",
                        detector=f"regex_{label}",
                    )
                )
        return entities

    def _detect_credit_cards(self, prompt: str) -> List[DetectedEntity]:
        entities = []
        for match in config.CREDIT_CARD_PATTERN.finditer(prompt):
            raw = match.group(0)
            digits = re.sub(r"\D", "", raw)
            if 13 <= len(digits) <= 19 and is_luhn_valid(raw):
                brand = detect_card_brand(digits)
                entities.append(
                    DetectedEntity(
                        text=raw,
                        entity_type=EntityType.CREDIT_CARD,
                        start=match.start(),
                        end=match.end(),
                        normalized_value=normalize_entity_value(EntityType.CREDIT_CARD, raw),
                        confidence=config.CONFIDENCE_BASELINES["luhn_credit_card"],
                        source="regex",
                        detector="luhn_credit_card",
                        metadata={"brand": brand, "digit_count": len(digits)},
                    )
                )
        return entities

    def _detect_api_keys(self, prompt: str) -> List[DetectedEntity]:
        entities = []
        for pattern, label, conf in config.API_KEY_PATTERNS:
            for match in pattern.finditer(prompt):
                if match.groups():
                    raw = match.group(1)
                    start, end = match.start(1), match.end(1)
                else:
                    raw = match.group(0)
                    start, end = match.start(), match.end()
                entities.append(
                    DetectedEntity(
                        text=raw,
                        entity_type=EntityType.API_KEY,
                        start=start,
                        end=end,
                        normalized_value=normalize_entity_value(EntityType.API_KEY, raw),
                        confidence=conf,
                        source="regex",
                        detector=f"regex_{label}",
                    )
                )
        return entities

    def _detect_access_tokens(self, prompt: str) -> List[DetectedEntity]:
        entities = []
        for pattern, label, conf in config.ACCESS_TOKEN_PATTERNS:
            for match in pattern.finditer(prompt):
                raw = match.group(0)
                entities.append(
                    DetectedEntity(
                        text=raw,
                        entity_type=EntityType.ACCESS_TOKEN,
                        start=match.start(),
                        end=match.end(),
                        normalized_value=normalize_entity_value(EntityType.ACCESS_TOKEN, raw),
                        confidence=conf,
                        source="regex",
                        detector=f"regex_{label}",
                    )
                )
        return entities

    def _detect_passwords(self, prompt: str) -> List[DetectedEntity]:
        entities = []
        for pattern, label in config.PASSWORD_PATTERNS:
            for match in pattern.finditer(prompt):
                if match.groups():
                    payload = match.group(len(match.groups()))
                    start = match.start(len(match.groups()))
                    end = match.end(len(match.groups()))
                else:
                    payload = match.group(0)
                    start, end = match.start(), match.end()
                entities.append(
                    DetectedEntity(
                        text=payload,
                        entity_type=EntityType.PASSWORD,
                        start=start,
                        end=end,
                        normalized_value=normalize_entity_value(EntityType.PASSWORD, payload),
                        confidence=config.CONFIDENCE_BASELINES["regex_password"],
                        source="regex",
                        detector=f"regex_{label}",
                    )
                )
        return entities

    def _detect_bank_accounts(self, prompt: str) -> List[DetectedEntity]:
        entities = []
        for pattern, label, conf in config.BANK_ACCOUNT_PATTERNS:
            for match in pattern.finditer(prompt):
                if match.groups():
                    raw = match.group(1)
                    start, end = match.start(1), match.end(1)
                else:
                    raw = match.group(0)
                    start, end = match.start(), match.end()
                entities.append(
                    DetectedEntity(
                        text=raw,
                        entity_type=EntityType.BANK_ACCOUNT,
                        start=start,
                        end=end,
                        normalized_value=normalize_entity_value(EntityType.BANK_ACCOUNT, raw),
                        confidence=conf,
                        source="regex",
                        detector=f"regex_{label}",
                    )
                )
        return entities

    def _detect_ip_addresses(self, prompt: str) -> List[DetectedEntity]:
        entities = []
        for pattern, label, conf in config.IP_ADDRESS_PATTERNS:
            for match in pattern.finditer(prompt):
                raw = match.group(0)
                entities.append(
                    DetectedEntity(
                        text=raw,
                        entity_type=EntityType.IP_ADDRESS,
                        start=match.start(),
                        end=match.end(),
                        normalized_value=normalize_entity_value(EntityType.IP_ADDRESS, raw),
                        confidence=conf,
                        source="regex",
                        detector=f"regex_{label}",
                    )
                )
        return entities

    def _detect_ids(self, prompt: str) -> List[DetectedEntity]:
        """
        Detect configurable project-specific IDs (ORDER_ID, USER_ID, CUSTOMER_ID).
        These are OFF by default in most deployments; patterns are in config.py.
        """
        TYPE_MAP = {
            "order_id": EntityType.ORDER_ID,
            "user_id": EntityType.USER_ID,
            "customer_id": EntityType.CUSTOMER_ID,
            "ticket_id": EntityType.TICKET_ID,
        }
        entities = []
        for pattern, label, conf in config.ID_PATTERNS:
            entity_type = TYPE_MAP.get(label, EntityType.USER_ID)
            for match in pattern.finditer(prompt):
                raw = match.group(1) if match.groups() else match.group(0)
                start = match.start(1) if match.groups() else match.start()
                end = match.end(1) if match.groups() else match.end()
                entities.append(
                    DetectedEntity(
                        text=raw,
                        entity_type=entity_type,
                        start=start,
                        end=end,
                        normalized_value=normalize_entity_value(entity_type, raw),
                        confidence=conf,
                        source="regex",
                        detector=f"regex_{label}",
                    )
                )
        return entities

    def _detect_addresses(self, prompt: str) -> List[DetectedEntity]:
        entities = []
        for pattern, label, conf in config.ADDRESS_PATTERNS:
            for match in pattern.finditer(prompt):
                raw = match.group(0).strip().rstrip(".,;:")
                start = match.start()
                end = start + len(raw)
                entities.append(
                    DetectedEntity(
                        text=raw,
                        entity_type=EntityType.ADDRESS,
                        start=start,
                        end=end,
                        normalized_value=normalize_entity_value(EntityType.ADDRESS, raw),
                        confidence=conf,
                        source="regex",
                        detector=f"regex_{label}",
                    )
                )
        return entities

    def _detect_dates(self, prompt: str) -> List[DetectedEntity]:
        entities = []
        for pattern, label in config.DATE_PATTERNS:
            for match in pattern.finditer(prompt):
                raw = match.group(0)
                entities.append(
                    DetectedEntity(
                        text=raw,
                        entity_type=EntityType.DATE,
                        start=match.start(),
                        end=match.end(),
                        normalized_value=normalize_entity_value(EntityType.DATE, raw),
                        confidence=config.CONFIDENCE_BASELINES["regex_date"],
                        source="regex",
                        detector=f"regex_{label}",
                    )
                )
        return entities
