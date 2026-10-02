"""
Data models for PromptShield AI.
Defines entity types, detected entity structures, normalization metadata,
baseline masking results, and Phase 4 semantic masking results.

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
from typing import Dict, List, Optional, Any, Set


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
    TICKET_ID = "TICKET_ID"

    # ── Extended PII & Technical Types ──────────────────────────────────
    USERNAME = "USERNAME"
    DOB = "DOB"
    NATIONAL_ID = "NATIONAL_ID"
    PASSPORT = "PASSPORT"
    DRIVER_LICENSE = "DRIVER_LICENSE"
    TAX_ID = "TAX_ID"
    RECOVERY_CODE = "RECOVERY_CODE"
    MAC_ADDRESS = "MAC_ADDRESS"
    MEDICAL_RECORD = "MEDICAL_RECORD"
    HEALTH_INSURANCE_ID = "HEALTH_INSURANCE_ID"
    CONNECTION_STRING = "CONNECTION_STRING"
    PII_OTHER = "PII_OTHER"

    # ── Unstructured / Named Entities (NER) ─────────────────────────────
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    ADDRESS = "ADDRESS"
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
    INCIDENT_REPORT = "INCIDENT_REPORT"
    GENERAL_CHAT = "GENERAL_CHAT"


class RoleCategory(str, Enum):
    """Contextual semantic role of a detected entity in the prompt."""
    PERSONAL_IDENTIFIER = "PERSONAL_IDENTIFIER"      # Self-identifying user PII (e.g. "My name is Julie")
    PUBLIC_FIGURE_OR_FACT = "PUBLIC_FIGURE_OR_FACT"  # Well-known public figures/facts (e.g. "Elon Musk", "Paris")
    TASK_RELEVANT_ENTITY = "TASK_RELEVANT_ENTITY"    # Essential to user's requested task (e.g. target employer)
    TECHNICAL_CREDENTIAL = "TECHNICAL_CREDENTIAL"    # Passwords, API keys, tokens, secret values
    GENERAL_REFERENCE = "GENERAL_REFERENCE"          # Ambiguous or third-party background references


class EntityTaskRelation(str, Enum):
    """Semantic relationship between a detected entity and the user's task."""
    NONE = "NONE"
    IRRELEVANT = "IRRELEVANT"
    OUTPUT_REFERENCE = "OUTPUT_REFERENCE"
    TARGET_OF_ANALYSIS = "TARGET_OF_ANALYSIS"
    COMPARISON_INPUT = "COMPARISON_INPUT"
    CALCULATION_INPUT = "CALCULATION_INPUT"
    VALIDATION_INPUT = "VALIDATION_INPUT"
    TRANSFORMATION_CONTENT = "TRANSFORMATION_CONTENT"


class OperationType(str, Enum):
    """Broad semantic category of operation requested in the prompt."""
    GENERATION = "GENERATION"
    TRANSFORMATION = "TRANSFORMATION"
    SUMMARIZATION = "SUMMARIZATION"
    VALUE_ANALYSIS = "VALUE_ANALYSIS"
    COMPARISON = "COMPARISON"
    CALCULATION = "CALCULATION"
    VALIDATION = "VALIDATION"
    GENERAL_INFORMATION = "GENERAL_INFORMATION"
    UNKNOWN = "UNKNOWN"


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
    value_required: bool = False    # True if the literal entity characters/value are required for the task
    entity_task_relation: EntityTaskRelation = EntityTaskRelation.NONE  # Specific entity-task relationship
    # Character offsets from the originating DetectedEntity (threaded through for Phase 4)
    start: int = -1
    end: int = -1
    normalized_value: str = ""      # Canonical form of the entity text
    explicit_directive: Optional[str] = None  # "RETAIN", "MASK", or None
    disclosure_allowed: bool = True           # True if allowed to be restored in final output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_text": self.entity_text,
            "entity_type": self.entity_type.value,
            "role_category": self.role_category.value,
            "is_first_party": self.is_first_party,
            "is_public_knowledge": self.is_public_knowledge,
            "is_task_relevant": self.is_task_relevant,
            "value_required": self.value_required,
            "entity_task_relation": (
                self.entity_task_relation.value
                if isinstance(self.entity_task_relation, EntityTaskRelation)
                else str(self.entity_task_relation)
            ),
            "context_cue": self.context_cue,
            "confidence": round(self.confidence, 4),
            "start": self.start,
            "end": self.end,
            "normalized_value": self.normalized_value,
            "explicit_directive": self.explicit_directive,
            "disclosure_allowed": self.disclosure_allowed,
        }


@dataclass
class ContextAnalysisResult:
    """Aggregated context understanding output for a prompt."""
    prompt: str
    task_type: TaskType
    task_confidence: float
    task_cues: List[str]
    entity_contexts: List[ContextualRole]
    operation_type: OperationType = OperationType.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "task_type": self.task_type.value,
            "task_confidence": round(self.task_confidence, 4),
            "task_cues": self.task_cues,
            "operation_type": (
                self.operation_type.value
                if isinstance(self.operation_type, OperationType)
                else str(self.operation_type)
            ),
            "entity_contexts": [ec.to_dict() for ec in self.entity_contexts],
        }


# ==========================================
# PHASE 4: SEMANTIC MASKING MODELS
# ==========================================

@dataclass
class ApprovalRecord:
    """
    Stable approval identifier for a USER_APPROVAL entity.

    Used by the UI/API (Phase 6+) to approve one specific sensitive value
    without affecting other entities of the same or different type.

    Fields
    ------
    entity_id       : Deterministic string key:  "<TYPE>:<normalized_value>"
    entity_type     : The EntityType of the sensitive value.
    entity_text     : The raw literal text from the prompt.
    normalized_value: Canonical form used for deduplication.
    placeholder     : The placeholder currently protecting this entity (e.g. <PASSWORD_1>).
    risk_score      : Risk score from Phase 3 PolicyEngine.
    risk_level      : Risk level string from Phase 3.
    decision_reason : Human-readable Phase 3 reason string.
    """
    entity_id: str          # "<TYPE>:<normalized_value>" — stable within a prompt
    entity_type: EntityType
    entity_text: str
    normalized_value: str
    placeholder: str        # The placeholder assigned by SemanticMasker
    risk_score: float
    risk_level: str
    decision_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id":        self.entity_id,
            "entity_type":      self.entity_type.value,
            "entity_text":      self.entity_text,
            "normalized_value": self.normalized_value,
            "placeholder":      self.placeholder,
            "risk_score":       round(self.risk_score, 4),
            "risk_level":       self.risk_level,
            "decision_reason":  self.decision_reason,
        }


@dataclass
class SemanticMaskingResult:
    """
    Result of Phase 4 — Context-Aware Semantic Masking.

    Produced by SemanticMasker.mask() after consuming Phase 3 PolicyReport.
    Replaces BaselineMaskResult for production use from Phase 4 onwards.

    Fields
    ------
    original_prompt     : Unmodified input prompt.
    sanitized_prompt    : Prompt with MASK/REVIEW/unapproved-USER_APPROVAL
                          placeholders applied. Safe to send to external LLM.
    mapping             : Local-only {placeholder: raw_value} dict.
                          NEVER included in any external payload.
    masked_entities     : Entities processed with MASK decision.
    retained_entities   : Entities processed with RETAIN decision (unchanged).
    approval_required   : Entities requiring user approval before release.
                          Their literal values are protected in sanitized_prompt.
    approved_entities   : Entities that received explicit approval and whose
                          literal values appear in sanitized_prompt.
    review_required     : Entities assigned REVIEW decision; masked by default.
    task_type           : Task classification string from Phase 2.
    overall_risk_level  : Highest risk level across all entities (Phase 3).
    session_id          : Session identifier for the local mapping store.
    """
    original_prompt:   str
    sanitized_prompt:  str
    mapping:           Dict[str, str]          # placeholder → raw value (LOCAL ONLY)
    masked_entities:   List["EntityPolicyResult"]   # forward ref; defined in policy_engine
    retained_entities: List["EntityPolicyResult"]
    approval_required: List[ApprovalRecord]
    approved_entities: List[ApprovalRecord]
    review_required:   List[ApprovalRecord]
    task_type:         str
    overall_risk_level: str
    session_id:        str

    def to_dict(self) -> Dict[str, Any]:
        from backend.policy_engine import EntityPolicyResult  # local import to avoid circular
        return {
            "original_prompt":    self.original_prompt,
            "sanitized_prompt":   self.sanitized_prompt,
            "mapping":            self.mapping,
            "task_type":          self.task_type,
            "overall_risk_level": self.overall_risk_level,
            "session_id":         self.session_id,
            "summary": {
                "masked":           len(self.masked_entities),
                "retained":         len(self.retained_entities),
                "approval_required": len(self.approval_required),
                "approved":         len(self.approved_entities),
                "review_required":  len(self.review_required),
            },
            "masked_entities":    [e.to_dict() for e in self.masked_entities],
            "retained_entities":  [e.to_dict() for e in self.retained_entities],
            "approval_required":  [a.to_dict() for a in self.approval_required],
            "approved_entities":  [a.to_dict() for a in self.approved_entities],
            "review_required":    [a.to_dict() for a in self.review_required],
        }


# ==========================================
# PHASE 5: CONTROLLED RESTORATION & LLM MODELS
# ==========================================

class RestorationStatus(str, Enum):
    """Status of placeholder restoration in LLM response."""
    SUCCESS = "SUCCESS"               # All requested placeholders restored successfully
    PARTIAL = "PARTIAL"               # Some placeholders restored, some blocked or untracked
    DENIED = "DENIED"                 # Restoration rejected due to authorization/policy
    UNMODIFIED = "UNMODIFIED"         # No placeholders were present in LLM response
    ERROR = "ERROR"                   # Error during restoration


@dataclass
class RestoredPlaceholder:
    """Record of an individual placeholder processed during restoration."""
    placeholder: str                  # e.g., "<PERSON_1>"
    original_value: str               # e.g., "Alice"
    entity_type: Optional[EntityType] = None
    restored: bool = True
    reason: str = "Restored from local mapping store"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "placeholder": self.placeholder,
            "original_value": self.original_value if self.restored else "[REDACTED]",
            "entity_type": self.entity_type.value if self.entity_type else "UNKNOWN",
            "restored": self.restored,
            "reason": self.reason,
        }


@dataclass
class RestorationPolicy:
    """Security and authorization policy governing controlled response restoration."""
    allow_restoration: bool = True
    blocked_entity_types: Set[EntityType] = field(default_factory=lambda: {
        EntityType.PASSWORD,
        EntityType.API_KEY,
        EntityType.ACCESS_TOKEN,
        EntityType.RECOVERY_CODE,
    })  # Technical secrets blocked from restoration into output by default
    allow_credentials_restoration: bool = False  # Explicit override if user demands credential reflection
    require_session_match: bool = True
    max_restorations: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow_restoration": self.allow_restoration,
            "blocked_entity_types": [et.value for et in self.blocked_entity_types],
            "allow_credentials_restoration": self.allow_credentials_restoration,
            "require_session_match": self.require_session_match,
            "max_restorations": self.max_restorations,
        }


@dataclass
class RestorationResult:
    """
    Result of Phase 5 — Controlled Placeholder Restoration.

    Produced by ResponseRestorer.restore() after processing LLM generated response.
    """
    restored_text: str
    raw_llm_response: str
    session_id: Optional[str]
    status: RestorationStatus
    restorations: List[RestoredPlaceholder]
    unrestored_placeholders: List[str]
    audit_trail: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "restored_text": self.restored_text,
            "raw_llm_response": self.raw_llm_response,
            "session_id": self.session_id,
            "status": self.status.value,
            "restorations": [r.to_dict() for r in self.restorations],
            "unrestored_placeholders": self.unrestored_placeholders,
            "audit_trail": self.audit_trail,
        }


@dataclass
class ShieldedExchangeResult:
    """
    Complete end-to-end audit bundle of a PromptShield protected LLM exchange (Phases 1–5).
    """
    original_prompt: str
    sanitized_prompt: str
    raw_llm_response: str
    restored_response: str
    session_id: str
    masking_result: SemanticMaskingResult
    restoration_result: RestorationResult
    provider_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "sanitized_prompt": self.sanitized_prompt,
            "raw_llm_response": self.raw_llm_response,
            "restored_response": self.restored_response,
            "session_id": self.session_id,
            "masking_result": self.masking_result.to_dict(),
            "restoration_result": self.restoration_result.to_dict(),
            "provider_name": self.provider_name,
        }

