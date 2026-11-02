"""
PromptShield AI — Normalization Module.
Normalizes detected entity values to canonical representations for
consistent processing, comparison, and policy evaluation.

Phase 1 addition: normalizers for IP_ADDRESS, BANK_ACCOUNT, and
configurable ID types (ORDER_ID, USER_ID, CUSTOMER_ID).
Also adds a type-normalization helper used by the entity fusion engine.
"""

import re
import unicodedata
from typing import Optional, Tuple, List, Dict, Set
from backend.models import EntityType, DetectedEntity


# ---------------------------------------------------------------------------
# Unicode & Homoglyph Evasion Defense Tables
# ---------------------------------------------------------------------------

INVISIBLE_CHARACTERS: Set[str] = {
    "\u200b",  # Zero-width space
    "\u200c",  # Zero-width non-joiner
    "\u200d",  # Zero-width joiner
    "\ufeff",  # Zero-width no-break space / BOM
    "\u00ad",  # Soft hyphen
    "\u200e",  # Left-to-right mark
    "\u200f",  # Right-to-left mark
    "\u202a",  # Left-to-right embedding
    "\u202b",  # Right-to-left embedding
    "\u202c",  # Pop directional formatting
    "\u202d",  # Left-to-right override
    "\u202e",  # Right-to-left override
    "\u2060",  # Word joiner
    "\u2061",  # Function application (invisible math operator)
    "\u2062",  # Invisible times
    "\u2063",  # Invisible separator
    "\u2064",  # Invisible plus
    "\u2066",  # Left-to-right isolate
    "\u2067",  # Right-to-left isolate
    "\u2068",  # First strong isolate
    "\u2069",  # Pop directional isolate
    "\u034f",  # Combining grapheme joiner
    "\u180e",  # Mongolian vowel separator
}

HOMOGLYPH_MAP: Dict[str, str] = {
    # Cyrillic lowercase to Latin
    "\u0430": "a", "\u0441": "c", "\u0435": "e", "\u043e": "o", "\u0440": "p",
    "\u0445": "x", "\u0443": "y", "\u0456": "i", "\u0458": "j", "\u0455": "s",
    "\u0432": "b", "\u0433": "r", "\u0451": "e",
    # Cyrillic uppercase to Latin
    "\u0410": "A", "\u0412": "B", "\u0421": "C", "\u0415": "E", "\u041d": "H",
    "\u0406": "I", "\u0408": "J", "\u041a": "K", "\u041c": "M", "\u041e": "O",
    "\u0420": "P", "\u0405": "S", "\u0422": "T", "\u0425": "X", "\u04ae": "Y",
    # Greek uppercase to Latin
    "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0396": "Z", "\u0397": "H",
    "\u0399": "I", "\u039a": "K", "\u039c": "M", "\u039d": "N", "\u039f": "O",
    "\u03a1": "P", "\u03a4": "T", "\u03a5": "Y", "\u03a7": "X",
    # Greek lowercase to Latin
    "\u03bf": "o", "\u03bd": "v", "\u03ba": "k", "\u03b9": "i", "\u03c1": "p",
    "\u03c4": "t", "\u03c5": "u", "\u03c7": "x",
}


def canonicalize_adversarial_text(
    prompt: str,
) -> Tuple[str, Optional[Tuple[List[int], List[int]]]]:
    """
    Canonicalize user prompt against Unicode evasion, zero-width steganography,
    and homoglyph / confusable spoofing attacks.

    Returns:
    --------
    (clean_prompt, offset_map)
      - If prompt required no changes (e.g. pure ASCII or clean international text),
        returns (prompt, None) for O(1) zero-overhead fast path.
      - If modifications were made, returns (clean_prompt, (orig_starts, orig_ends)),
        where for each character index i in clean_prompt:
          orig_starts[i] is the starting character index in the original prompt.
          orig_ends[i] is the exclusive ending character index in the original prompt.
    """
    if not prompt or prompt.isascii():
        return prompt, None

    clean_chars: List[str] = []
    orig_starts: List[int] = []
    orig_ends: List[int] = []

    for i, ch in enumerate(prompt):
        if ch in INVISIBLE_CHARACTERS:
            continue

        if ch in HOMOGLYPH_MAP:
            sub = HOMOGLYPH_MAP[ch]
        else:
            sub = unicodedata.normalize("NFKC", ch)

        for c in sub:
            clean_chars.append(c)
            orig_starts.append(i)
            orig_ends.append(i + 1)

    clean_prompt = "".join(clean_chars)
    if clean_prompt == prompt:
        return prompt, None

    return clean_prompt, (orig_starts, orig_ends)


def remap_entity_offsets(
    entities: List[DetectedEntity],
    orig_prompt: str,
    offset_map: Tuple[List[int], List[int]],
) -> List[DetectedEntity]:
    """
    Remap detected entity spans from canonicalized text coordinates back to
    exact original prompt coordinates.

    Guarantees strict character-level integrity:
        orig_prompt[ent.start:ent.end] == ent.text
    """
    if not entities or not offset_map:
        return entities

    orig_starts, orig_ends = offset_map
    remapped: List[DetectedEntity] = []

    for ent in entities:
        if ent.start < ent.end and ent.end <= len(orig_ends):
            s_orig = orig_starts[ent.start]
            e_orig = orig_ends[ent.end - 1]
            raw_text = orig_prompt[s_orig:e_orig]

            ent_meta = dict(ent.metadata) if ent.metadata else {}
            if raw_text != ent.text:
                ent_meta["evasion_normalized"] = True
                ent_meta["canonical_text"] = ent.text

            remapped.append(
                DetectedEntity(
                    entity_type=ent.entity_type,
                    text=raw_text,
                    start=s_orig,
                    end=e_orig,
                    confidence=ent.confidence,
                    source=ent.source,
                    detector=ent.detector,
                    normalized_value=ent.normalized_value or normalize_entity_value(ent.entity_type, raw_text),
                    metadata=ent_meta,
                )
            )
        else:
            remapped.append(ent)

    return remapped



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
    """Normalize names, organizations, and locations by collapsing redundant spaces and stripping trailing punctuation/possessives."""
    cleaned = re.sub(r"\s+", " ", raw_val.strip())
    cleaned = re.sub(r"[''’]s?$", "", cleaned)
    cleaned = re.sub(r"[.,;:!]+$", "", cleaned)
    return cleaned



def normalize_date(raw_val: str) -> str:
    """Normalize date strings by cleaning spaces and edge punctuation."""
    cleaned = re.sub(r"\s+", " ", raw_val.strip())
    cleaned = re.sub(r"[.,;:!]+$", "", cleaned)
    return cleaned


def normalize_mac_address(raw_val: str) -> str:
    """Normalize MAC address: uppercase, strip surrounding whitespace."""
    return raw_val.strip().upper()


def normalize_connection_string(raw_val: str) -> str:
    """Normalize connection string: strip surrounding whitespace and quotes."""
    return raw_val.strip().strip("'\"`")


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
    elif entity_type == EntityType.MAC_ADDRESS:
        return normalize_mac_address(raw_val)
    elif entity_type == EntityType.CONNECTION_STRING:
        return normalize_connection_string(raw_val)
    elif entity_type in (
        EntityType.ORDER_ID, EntityType.USER_ID, EntityType.CUSTOMER_ID,
        EntityType.TICKET_ID, EntityType.USERNAME, EntityType.NATIONAL_ID,
        EntityType.PASSPORT, EntityType.DRIVER_LICENSE, EntityType.TAX_ID,
        EntityType.RECOVERY_CODE, EntityType.MEDICAL_RECORD, EntityType.HEALTH_INSURANCE_ID,
    ):
        return normalize_id(raw_val)
    elif entity_type in (EntityType.PERSON, EntityType.ORGANIZATION, EntityType.LOCATION, EntityType.ADDRESS):
        return normalize_text_entity(raw_val)
    elif entity_type in (EntityType.DATE, EntityType.DOB):
        return normalize_date(raw_val)
    elif entity_type == EntityType.PII_OTHER:
        return raw_val.strip()
    return raw_val.strip()

