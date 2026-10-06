"""
PromptShield AI — Phase 6 Service Manager
Singleton provider for PromptShieldCore and REST endpoint orchestration.
"""

import time
import logging
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from backend.core import PromptShieldCore
from backend.models import TaskType, EntityType, RestorationPolicy
from backend.policy_engine import PolicyDecision, RiskLevel
from backend.api.audit_logger import PrivacySafeAuditLogger
from backend.api.schemas import (
    EntitySchema,
    EntityPolicySchema,
    AnalyzeResponse,
    SanitizeResponse,
    RestoreResponse,
    ProcessResponse,
    PolicyListResponse,
    MetricsResponse,
    HealthResponse,
)

logger = logging.getLogger("promptshield.service")


class SecurityLevel(str, Enum):
    STRICT = "STRICT"
    BALANCED = "BALANCED"
    PERMISSIVE = "PERMISSIVE"


class PromptShieldService:
    """
    High-level facade connecting FastAPI route controllers to PromptShieldCore.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PromptShieldService, cls).__new__(cls)
            cls._instance._shield = None
            cls._instance._audit = PrivacySafeAuditLogger()
        return cls._instance

    @property
    def shield(self) -> PromptShieldCore:
        if self._shield is None:
            logger.info("[PromptShieldService] Initialising PromptShieldCore...")
            self._shield = PromptShieldCore()
        return self._shield

    @property
    def audit(self) -> PrivacySafeAuditLogger:
        return self._audit

    def analyze_prompt(
        self,
        prompt: str,
        security_level_str: str = "BALANCED",
        context_hint: Optional[str] = None,
    ) -> AnalyzeResponse:
        t0 = time.perf_counter()

        # 1. Detection
        entities = self.shield.detector.detect(prompt)
        # 2. Context
        context = self.shield.analyze_context(prompt)
        # 3. Policy
        policy = self.shield.evaluate_policy(prompt)
        latency = (time.perf_counter() - t0) * 1000.0

        # Record audit
        self.audit.record_request("/analyze", latency)
        self.audit.record_entities([e.entity_type.value for e in entities])

        entity_schemas = [
            EntitySchema(
                text=e.text,
                entity_type=e.entity_type.value,
                start=e.start,
                end=e.end,
                confidence=e.confidence,
                source=e.source,
                detector=e.detector,
                metadata=e.metadata or {},
            )
            for e in entities
        ]

        policy_schemas = [
            EntityPolicySchema(
                entity_text=ep.entity_text,
                entity_type=ep.entity_type.value,
                risk_score=ep.risk_score,
                decision=ep.decision.value,
                disclosure_allowed=ep.disclosure_allowed,
                rationale=ep.reason,
            )
            for ep in policy.entity_policies
        ]

        overall_risk = (
            max((ep.risk_score for ep in policy.entity_policies), default=0.0)
        )

        return AnalyzeResponse(
            prompt_length=len(prompt),
            task_type=context.task_type.value,
            task_confidence=context.task_confidence,
            detected_entities=entity_schemas,
            entity_policies=policy_schemas,
            overall_risk_score=round(overall_risk, 2),
            analysis_latency_ms=round(latency, 2),
        )

    def sanitize_prompt(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        security_level_str: str = "BALANCED",
        approved_entity_ids: Optional[List[str]] = None,
    ) -> SanitizeResponse:
        t0 = time.perf_counter()

        approved_set = set(approved_entity_ids) if approved_entity_ids else None
        mask_res = self.shield.sanitize(
            prompt=prompt,
            session_id=session_id,
            approved_entity_ids=approved_set,
        )
        latency = (time.perf_counter() - t0) * 1000.0

        self.audit.record_request("/sanitize", latency)
        self.audit.record_masking(len(mask_res.masked_entities))

        pending_schemas = [
            EntityPolicySchema(
                entity_text=rec.entity_text,
                entity_type=rec.entity_type.value,
                risk_score=rec.risk_score,
                decision="USER_APPROVAL",
                disclosure_allowed=False,
                rationale=rec.decision_reason,
            )
            for rec in mask_res.approval_required
        ]

        return SanitizeResponse(
            session_id=mask_res.session_id,
            sanitized_prompt=mask_res.sanitized_prompt,
            masked_count=len(mask_res.masked_entities),
            retained_count=len(mask_res.retained_entities),
            user_approval_count=len(mask_res.approval_required),
            placeholders=list(mask_res.mapping.keys()),
            has_pending_approvals=len(mask_res.approval_required) > 0,
            pending_approval_entities=pending_schemas,
            latency_ms=round(latency, 2),
        )

    def restore_response(
        self,
        llm_response: str,
        session_id: str,
        allow_all_restoration: bool = False,
    ) -> RestoreResponse:
        t0 = time.perf_counter()

        if allow_all_restoration:
            rest_policy = RestorationPolicy(
                allow_credentials_restoration=True,
                blocked_entity_types=set(),
            )
        else:
            rest_policy = RestorationPolicy()

        res = self.shield.restore_response(
            llm_response=llm_response,
            session_id=session_id,
            policy=rest_policy,
        )
        latency = (time.perf_counter() - t0) * 1000.0

        restored_phs = [r.placeholder for r in res.restorations if r.restored]
        blocked_phs = [r.placeholder for r in res.restorations if not r.restored] + list(res.unrestored_placeholders)
        restored_count = len(restored_phs)
        blocked_count = len(blocked_phs)

        self.audit.record_request("/restore", latency)
        self.audit.record_restoration(restored_count, blocked_count)

        leakage_detected = res.audit_trail.get("leakage_detected", False) if res.audit_trail else False

        return RestoreResponse(
            session_id=session_id,
            restored_response=res.restored_text,
            total_placeholders_found=len(res.restorations) + len(res.unrestored_placeholders),
            restored_count=restored_count,
            blocked_count=blocked_count,
            restored_placeholders=restored_phs,
            blocked_placeholders=blocked_phs,
            leakage_detected=leakage_detected,
            latency_ms=round(latency, 2),
        )

    def process_end_to_end(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        security_level_str: str = "BALANCED",
        system_prompt: Optional[str] = None,
        llm_provider_name: str = "mock",
    ) -> ProcessResponse:
        t0 = time.perf_counter()

        exchange = self.shield.execute_and_restore(
            prompt=prompt,
            session_id=session_id,
            system_prompt=system_prompt,
        )
        latency = (time.perf_counter() - t0) * 1000.0

        restored_phs = [r.placeholder for r in exchange.restoration_result.restorations if r.restored]
        blocked_phs = [r.placeholder for r in exchange.restoration_result.restorations if not r.restored] + list(exchange.restoration_result.unrestored_placeholders)
        restored_count = len(restored_phs)
        blocked_count = len(blocked_phs)

        self.audit.record_request("/process", latency)
        self.audit.record_masking(len(exchange.masking_result.masked_entities))
        self.audit.record_restoration(restored_count, blocked_count)

        detected_count = (
            len(exchange.masking_result.masked_entities)
            + len(exchange.masking_result.retained_entities)
            + len(exchange.masking_result.approval_required)
        )

        leakage_detected = (
            exchange.restoration_result.audit_trail.get("leakage_detected", False)
            if exchange.restoration_result.audit_trail
            else False
        )

        return ProcessResponse(
            session_id=exchange.session_id,
            sanitized_prompt=exchange.sanitized_prompt,
            raw_llm_response=exchange.raw_llm_response,
            restored_response=exchange.restored_response,
            detected_entities_count=detected_count,
            masked_count=len(exchange.masking_result.masked_entities),
            restored_count=restored_count,
            blocked_count=blocked_count,
            leakage_detected=leakage_detected,
            total_latency_ms=round(latency, 2),
        )

    def get_policies(self) -> PolicyListResponse:
        return PolicyListResponse(
            supported_security_levels=[lvl.value for lvl in SecurityLevel],
            supported_task_types=[t.value for t in TaskType],
            supported_entity_types=[e.value for e in EntityType],
            default_security_level=SecurityLevel.BALANCED.value,
        )

    def get_metrics(self) -> MetricsResponse:
        m = self.audit.get_metrics()
        return MetricsResponse(**m)

    def get_health(self) -> HealthResponse:
        detector = self.shield.detector
        spacy_ok = detector._spacy_ner.is_available() if detector._spacy_ner else False
        gliner2_ok = detector._gliner2.is_available() if detector._gliner2 else False
        presidio_ok = detector._presidio.is_available() if detector._presidio else False

        return HealthResponse(
            status="healthy",
            version="1.0.0",
            detectors={
                "regex": True,
                "presidio": presidio_ok,
                "gliner2": gliner2_ok,
                "spacy": spacy_ok,
            },
            spacy_model_loaded=spacy_ok,
            gliner2_loaded=gliner2_ok,
            presidio_loaded=presidio_ok,
        )
