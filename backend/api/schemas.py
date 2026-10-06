"""
PromptShield AI — Phase 6 REST API Data Schemas (Pydantic models)
Provides robust input validation, serialization, and OpenAPI schema generation.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class EntitySchema(BaseModel):
    text: str = Field(..., description="The detected entity text as it appeared in the prompt")
    entity_type: str = Field(..., description="The canonical EntityType string")
    start: int = Field(..., description="0-indexed start character offset")
    end: int = Field(..., description="0-indexed end character offset (exclusive)")
    confidence: float = Field(..., description="Detector confidence score (0.0 - 1.0)")
    source: str = Field(..., description="Primary detector subsystem (regex, presidio, gliner2, spacy, heuristic_ner)")
    detector: str = Field(..., description="Specific detector rule or component label")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional contextual detector metadata")


class EntityPolicySchema(BaseModel):
    entity_text: str = Field(..., description="The entity text evaluated")
    entity_type: str = Field(..., description="The entity type")
    risk_score: float = Field(..., description="Privacy risk score from 0 to 100")
    decision: str = Field(..., description="Policy decision (MASK, RETAIN, USER_APPROVAL, REPLACE)")
    disclosure_allowed: bool = Field(..., description="Whether restoration in LLM output is permitted")
    rationale: str = Field(..., description="Human-readable explanation of the policy decision")


# ── /analyze ─────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Raw user prompt to analyze")
    security_level: Optional[str] = Field("BALANCED", description="Security level: STRICT, BALANCED, PERMISSIVE")
    context_hint: Optional[str] = Field(None, description="Optional caller-supplied context hint")


class AnalyzeResponse(BaseModel):
    prompt_length: int
    task_type: str
    task_confidence: float
    detected_entities: List[EntitySchema]
    entity_policies: List[EntityPolicySchema]
    overall_risk_score: float
    analysis_latency_ms: float


# ── /sanitize ────────────────────────────────────────────────────────────────

class SanitizeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Raw user prompt to sanitize")
    session_id: Optional[str] = Field(None, description="Session ID for local mapping memory. Generated if omitted.")
    security_level: Optional[str] = Field("BALANCED", description="Security level: STRICT, BALANCED, PERMISSIVE")
    approved_entity_ids: Optional[List[str]] = Field(None, description="Set of entity IDs approved by user ('<TYPE>:<value>')")


class SanitizeResponse(BaseModel):
    session_id: str
    sanitized_prompt: str
    masked_count: int
    retained_count: int
    user_approval_count: int
    placeholders: List[str]
    has_pending_approvals: bool
    pending_approval_entities: List[EntityPolicySchema]
    latency_ms: float


# ── /restore ─────────────────────────────────────────────────────────────────

class RestoreRequest(BaseModel):
    llm_response: str = Field(..., description="Response returned by the LLM containing placeholders")
    session_id: str = Field(..., min_length=1, description="Session ID matching the sanitized prompt's mapping store")
    allow_all_restoration: bool = Field(False, description="If True, bypasses disclosure_allowed policy checks")


class RestoreResponse(BaseModel):
    session_id: str
    restored_response: str
    total_placeholders_found: int
    restored_count: int
    blocked_count: int
    restored_placeholders: List[str]
    blocked_placeholders: List[str]
    leakage_detected: bool
    latency_ms: float


# ── /process ─────────────────────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Raw user prompt")
    session_id: Optional[str] = Field(None, description="Optional existing session ID")
    security_level: Optional[str] = Field("BALANCED", description="Security level: STRICT, BALANCED, PERMISSIVE")
    system_prompt: Optional[str] = Field(None, description="Optional system instruction for LLM")
    llm_provider: Optional[str] = Field("mock", description="LLM provider name ('mock', 'gemini')")


class ProcessResponse(BaseModel):
    session_id: str
    sanitized_prompt: str
    raw_llm_response: str
    restored_response: str
    detected_entities_count: int
    masked_count: int
    restored_count: int
    blocked_count: int
    leakage_detected: bool
    total_latency_ms: float


# ── /policies ────────────────────────────────────────────────────────────────

class PolicyListResponse(BaseModel):
    supported_security_levels: List[str]
    supported_task_types: List[str]
    supported_entity_types: List[str]
    default_security_level: str


# ── /metrics ─────────────────────────────────────────────────────────────────

class MetricsResponse(BaseModel):
    total_requests: int
    requests_by_endpoint: Dict[str, int]
    total_entities_detected: int
    entities_by_type: Dict[str, int]
    total_masks_applied: int
    total_restorations: int
    total_blocked_restorations: int
    avg_latency_ms: float


# ── /health ──────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    detectors: Dict[str, bool]
    spacy_model_loaded: bool
    gliner2_loaded: bool
    presidio_loaded: bool
