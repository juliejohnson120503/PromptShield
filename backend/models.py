"""
Data models for PromptShield AI.
Defines entity types, detected entity structures, normalization metadata,
and baseline masking results.

Changelog (Phase 1 — Hybrid Detection Foundation)
--------------------------------------------------
- EntityType extended with BANK_ACCOUNT, IP_ADDRESS, ACCESS_TOKEN,
  ORDER_ID, USER_ID, CUSTOMER_ID.
- DetectedEntity gains a `source` field that records which detector
  subsystem produced the entity ("regex", "presidio", "spacy",
  "heuristic_ner").
  The legacy `detector` field is kept for backward compatibility and
  always mirrors `source` when not set explicitly.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class EntityType(str, Enum):
    """Supported sensitive and named entity types."""
    # ── Structured / Credential Sensitive Data ──────────────────────────
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    CREDIT_CARD = "CREDIT_CARD"
    API_KEY = "API_KEY"
    ACCESS_TOKEN = "ACCESS_TOKEN"       # JWT / OAuth bearer tokens
    PASSWORD = "PASSWORD"
    BANK_ACCOUNT = "BANK_ACCOUNT"       # IBAN, account numbers
    IP_ADDRESS = "IP_ADDRESS"           # IPv4 / IPv6

    # ── Project-Specific Configurable IDs ───────────────────────────────
    ORDER_ID = "ORDER_ID"
    USER_ID = "USER_ID"
    CUSTOMER_ID = "CUSTOMER_ID"

    # ── Unstructured / Named Entities (NER) ─────────────────────────────
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    DATE = "DATE"


@dataclass
class DetectedEntity:
    """
    Represents an entity detected within a user prompt.

    Fields
    ------
    text            : The raw substring from the prompt.
    entity_type     : Canonical PromptShield EntityType.
    start           : Character start offset (inclusive).
    end             : Character end offset (exclusive).
    normalized_value: Canonicalized form of the entity value.
    confidence      : Detection confidence in [0, 1].
                      For regex: rule-defined baseline.
                      For Presidio: analyzer's own score (preserved as-is).
                      For spaCy / heuristic: model/rule baseline.
    source          : Which detector subsystem produced this entity.
                      One of: "regex", "presidio", "spacy", "heuristic_ner".
    detector        : Specific recognizer label within the source
                      (e.g. "openai_api_key", "luhn_credit_card").
                      Kept for backward compatibility.
    metadata        : Additional structured metadata (e.g. card brand,
                      spacy_label, contributing_sources after fusion).
    """
    text: str
    entity_type: EntityType
    start: int
    end: int
    normalized_value: str
    confidence: float = 1.0
    source: str = "regex"        # NEW — which detector subsystem
    detector: str = "rule_based" # legacy label; specific recognizer name
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert detected entity to dictionary representation."""
        return {
            "text": self.text,
            "entity_type": self.entity_type.value,
            "start": self.start,
            "end": self.end,
            "normalized_value": self.normalized_value,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "detector": self.detector,
            "metadata": self.metadata,
        }


@dataclass
class BaselineMaskResult:
    """
    Result of baseline placeholder generation and local mapping.
    NOTE: This is the Phase 1 technical foundation. Context-aware selective
    masking (applying RETAIN/MASK policies) is executed in Phase 4.
    """
    original_prompt: str
    sanitized_prompt: str
    entities: List[DetectedEntity]
    mapping: Dict[str, str]
    session_id: str
    disclaimer: str = (
        "Baseline masking applied for technical foundation. "
        "Context-aware policy decisions will selectively govern masking in Phase 4."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert mask result to dictionary representation."""
        return {
            "original_prompt": self.original_prompt,
            "sanitized_prompt": self.sanitized_prompt,
            "entities": [e.to_dict() for e in self.entities],
            "mapping": self.mapping,
            "session_id": self.session_id,
            "disclaimer": self.disclaimer,
        }


# ==========================================
# PHASE 2: CONTEXT UNDERSTANDING MODELS
# ==========================================

class TaskType(str, Enum):
    """Canonical classification of user prompt task intent."""
    GENERAL_QA = "GENERAL_QA"
    EMAIL_GENERATION = "EMAIL_GENERATION"
    CODE_GENERATION = "CODE_GENERATION"
    SUMMARIZATION = "SUMMARIZATION"
    TRANSLATION = "TRANSLATION"
    DATA_ANALYSIS = "DATA_ANALYSIS"
    DOCUMENT_GENERATION = "DOCUMENT_GENERATION"
    GENERAL_CHAT = "GENERAL_CHAT"


class RoleCategory(str, Enum):
    """Contextual semantic role of a detected entity in the prompt."""
    PERSONAL_IDENTIFIER = "PERSONAL_IDENTIFIER"      # Self-identifying user PII (e.g. "My name is Julie")
    PUBLIC_FIGURE_OR_FACT = "PUBLIC_FIGURE_OR_FACT"  # Well-known public figures/facts (e.g. "Elon Musk", "Paris")
    TASK_RELEVANT_ENTITY = "TASK_RELEVANT_ENTITY"    # Essential to user's requested task (e.g. target employer)
    TECHNICAL_CREDENTIAL = "TECHNICAL_CREDENTIAL"    # Passwords, API keys, tokens, secret values
    GENERAL_REFERENCE = "GENERAL_REFERENCE"          # Ambiguous or third-party background references


@dataclass
class ContextualRole:
    """Detailed contextual evaluation for a detected entity."""
    entity_text: str
    entity_type: EntityType
    role_category: RoleCategory
    is_first_party: bool            # True if associated with user (e.g. "my email", "I am")
    is_public_knowledge: bool       # True if entity is widely known public information
    is_task_relevant: bool          # True if entity is functionally needed for the prompt's task
    context_cue: str                # Explanatory linguistic/structural trigger
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_text": self.entity_text,
            "entity_type": self.entity_type.value,
            "role_category": self.role_category.value,
            "is_first_party": self.is_first_party,
            "is_public_knowledge": self.is_public_knowledge,
            "is_task_relevant": self.is_task_relevant,
            "context_cue": self.context_cue,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class ContextAnalysisResult:
    """Aggregated context understanding output for a prompt."""
    prompt: str
    task_type: TaskType
    task_confidence: float
    task_cues: List[str]
    entity_contexts: List[ContextualRole]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "task_type": self.task_type.value,
            "task_confidence": round(self.task_confidence, 4),
            "task_cues": self.task_cues,
            "entity_contexts": [ec.to_dict() for ec in self.entity_contexts],
        }
