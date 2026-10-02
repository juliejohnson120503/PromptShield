"""
PromptShield AI — Phase 4: Context-Aware Semantic Masker
=========================================================

Executes the policy decisions produced by Phase 3 (PolicyEngine) to produce
a sanitized prompt and a local placeholder-to-secret mapping.

DESIGN PRINCIPLES
-----------------
1. Phase 4 does NOT decide what is sensitive — that is Phase 3's job.
   Phase 4 EXECUTES Phase 3 decisions faithfully.

2. Policy execution rules:
   MASK         -> Replace literal value with <TYPE_N> placeholder.
   RETAIN       -> Leave text unchanged. No placeholder created.
   USER_APPROVAL -> Protect the literal value by default (placeholder).
                   If explicit approval is provided for that entity, release it.
   REVIEW       -> Mask by default; record in review_required.

3. Span-based replacement (right-to-left):
   Replacements are performed from the rightmost entity to the leftmost,
   preventing earlier replacements from shifting character offsets of
   subsequent entities. No naive string.replace() is used.

4. Overlap handling:
   EntityFusionEngine (Phase 1) already guarantees non-overlapping spans
   in the fused entity list. However, if any residual overlaps survive to
   Phase 4 (e.g. from repeated entities or partially-fused results),
   the SemanticMasker resolves them deterministically:
     - Sort all policy entities by start offset ascending.
     - Walk spans greedily: if a span starts before the current cursor
       (last end), skip it (the higher-priority entity already consumed it).
     - Priority: CONNECTION_STRING > API_KEY > PASSWORD > ... (mirrors
       ENTITY_TYPE_PRIORITY in config.py) -- but since fusion already ran,
       we simply keep the first (widest / left-most) non-overlapping span.
   This guarantees no nested placeholders and no partial credential leakage.

5. Stable placeholder IDs:
   The SAME normalized entity value -> SAME placeholder within a prompt.
   Different values of the same type -> different incremented IDs.
   Key: (EntityType, normalized_value)

6. Approval model:
   Each USER_APPROVAL entity is represented by a stable ApprovalRecord
   with entity_id = "<TYPE>:<normalized_value>".
   Callers pass a set of approved entity_ids; matching entities are released
   as literals in the sanitized prompt.

7. MappingStore:
   All MASK / unapproved USER_APPROVAL / REVIEW placeholders are registered
   in the local MappingStore under a session_id. This mapping is NEVER
   included in external payloads -- it is Phase 5 restoration input only.

8. Security invariants (enforced by tests):
   - MASK literals MUST NOT appear in sanitized_prompt.
   - Unapproved USER_APPROVAL literals MUST NOT appear in sanitized_prompt.
   - REVIEW literals MUST NOT appear in sanitized_prompt.
   - Approved USER_APPROVAL literals MAY appear in sanitized_prompt.
   - RETAIN text is unchanged.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

from backend.models import (
    EntityType,
    ApprovalRecord,
    SemanticMaskingResult,
)
from backend.policy_engine import (
    EntityPolicyResult,
    PolicyDecision,
    PolicyReport,
)
from backend.mapping_store import MappingStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed placeholder prefix table
# Mirrors BaselineMasker.TYPE_PREFIXES for consistency.
# ---------------------------------------------------------------------------
PLACEHOLDER_PREFIXES: Dict[EntityType, str] = {
    EntityType.EMAIL:               "EMAIL",
    EntityType.PHONE:               "PHONE",
    EntityType.CREDIT_CARD:         "CREDIT_CARD",
    EntityType.API_KEY:             "API_KEY",
    EntityType.ACCESS_TOKEN:        "TOKEN",
    EntityType.PASSWORD:            "PASSWORD",
    EntityType.BANK_ACCOUNT:        "BANK_ACCOUNT",
    EntityType.IP_ADDRESS:          "IP_ADDRESS",
    EntityType.ORDER_ID:            "ORDER_ID",
    EntityType.USER_ID:             "USER_ID",
    EntityType.USERNAME:            "USERNAME",
    EntityType.CUSTOMER_ID:         "CUSTOMER_ID",
    EntityType.TICKET_ID:           "TICKET_ID",
    EntityType.PERSON:              "PERSON",
    EntityType.ORGANIZATION:        "ORG",
    EntityType.LOCATION:            "LOCATION",
    EntityType.ADDRESS:             "ADDRESS",
    EntityType.DATE:                "DATE",
    EntityType.DOB:                 "DOB",
    EntityType.NATIONAL_ID:         "NATIONAL_ID",
    EntityType.PASSPORT:            "PASSPORT",
    EntityType.DRIVER_LICENSE:      "DRIVER_LICENSE",
    EntityType.TAX_ID:              "TAX_ID",
    EntityType.RECOVERY_CODE:       "RECOVERY_CODE",
    EntityType.MAC_ADDRESS:         "MAC_ADDRESS",
    EntityType.CONNECTION_STRING:   "CONNECTION_STRING",
    EntityType.MEDICAL_RECORD:      "MEDICAL_RECORD",
    EntityType.HEALTH_INSURANCE_ID: "HEALTH_INSURANCE_ID",
    EntityType.PII_OTHER:           "PII",
}


def _placeholder_prefix(entity_type: EntityType) -> str:
    return PLACEHOLDER_PREFIXES.get(entity_type, entity_type.value)


def _entity_id(entity_type: EntityType, normalized_value: str) -> str:
    """Stable approval identifier: '<TYPE>:<normalized_value>'."""
    return f"{entity_type.value}:{normalized_value}"


def _resolve_overlaps(policies: List[EntityPolicyResult]) -> List[EntityPolicyResult]:
    """
    Resolve overlapping entity spans deterministically (greedy left-to-right).

    Entities with invalid spans (start < 0 or end <= start) are excluded.
    When two spans start at the same offset, the wider one takes priority.
    """
    valid = [p for p in policies if p.start >= 0 and p.end > p.start]
    sorted_policies = sorted(valid, key=lambda p: (p.start, -(p.end - p.start)))

    accepted: List[EntityPolicyResult] = []
    cursor = 0
    for p in sorted_policies:
        if p.start >= cursor:
            accepted.append(p)
            cursor = p.end
        else:
            logger.debug(
                "[SemanticMasker] Overlap skipped: '%s' [%d:%d] overlaps cursor %d",
                p.entity_text, p.start, p.end, cursor,
            )
    return accepted


class SemanticMasker:
    """
    Phase 4 -- Context-Aware Semantic Masker.

    Executes PolicyReport decisions to produce a sanitized prompt.
    Does NOT independently decide what is sensitive.

    Parameters
    ----------
    mapping_store : MappingStore, optional
        Shared local mapping store. A fresh store is created if not provided.
    """

    def __init__(self, mapping_store: Optional[MappingStore] = None):
        self._store = mapping_store or MappingStore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mask(
        self,
        prompt: str,
        policy_report: PolicyReport,
        approved_entity_ids: Optional[Set[str]] = None,
        session_id: Optional[str] = None,
    ) -> SemanticMaskingResult:
        """
        Execute Phase 3 policy decisions on *prompt*.

        Parameters
        ----------
        prompt : str
            The original unmodified user prompt.
        policy_report : PolicyReport
            Output from Phase 3 PolicyEngine.evaluate().
        approved_entity_ids : set of str, optional
            Stable entity IDs ('<TYPE>:<normalized_value>') that the user
            has explicitly approved.  Matching USER_APPROVAL entities are
            released as literals in the sanitized prompt.
        session_id : str, optional
            Session identifier for the local MappingStore.

        Returns
        -------
        SemanticMaskingResult
        """
        approved_ids: Set[str] = approved_entity_ids or set()
        sid = self._store.create_session(session_id)

        if not prompt or not prompt.strip():
            return self._empty_result(prompt, policy_report, sid)

        # -----------------------------------------------------------------
        # Step 1: Resolve overlapping spans (defensive)
        # -----------------------------------------------------------------
        resolved = _resolve_overlaps(policy_report.entity_policies)

        invalid_span = [
            p for p in policy_report.entity_policies
            if not (p.start >= 0 and p.end > p.start)
        ]
        if invalid_span:
            logger.warning(
                "[SemanticMasker] %d entities have invalid spans and are excluded "
                "from span-based masking: %s",
                len(invalid_span), [p.entity_text for p in invalid_span],
            )

        # -----------------------------------------------------------------
        # Step 2: Assign placeholders
        # -----------------------------------------------------------------
        placeholder_cache: Dict[Tuple[EntityType, str], str] = {}
        type_counters: Dict[str, int] = {}

        def _get_or_create_placeholder(p: EntityPolicyResult) -> str:
            nv = p.normalized_value or p.entity_text
            key = (p.entity_type, nv)
            if key in placeholder_cache:
                return placeholder_cache[key]
            prefix = _placeholder_prefix(p.entity_type)
            idx = type_counters.get(prefix, 0) + 1
            type_counters[prefix] = idx
            ph = f"<{prefix}_{idx}>"
            placeholder_cache[key] = ph
            return ph

        # -----------------------------------------------------------------
        # Step 3: Classify each resolved policy entity
        # -----------------------------------------------------------------
        masked_entities:   List[EntityPolicyResult] = []
        retained_entities: List[EntityPolicyResult] = []
        approval_required: List[ApprovalRecord] = []
        approved_entities: List[ApprovalRecord] = []
        review_required:   List[ApprovalRecord] = []
        ua_placeholder_map: Dict[str, str] = {}

        for p in resolved:
            decision = p.decision
            nv = p.normalized_value or p.entity_text
            eid = _entity_id(p.entity_type, nv)

            if decision == PolicyDecision.MASK:
                ph = _get_or_create_placeholder(p)
                self._store.store_mapping(
                    session_id=sid, placeholder=ph,
                    raw_value=p.entity_text, entity_type=p.entity_type,
                    normalized_value=nv,
                    entity_id=eid,
                    policy_decision=decision.value,
                    disclosure_allowed=getattr(p, "disclosure_allowed", True),
                )
                masked_entities.append(p)

            elif decision == PolicyDecision.RETAIN:
                retained_entities.append(p)

            elif decision == PolicyDecision.USER_APPROVAL:
                ph = _get_or_create_placeholder(p)
                record = ApprovalRecord(
                    entity_id=eid,
                    entity_type=p.entity_type,
                    entity_text=p.entity_text,
                    normalized_value=nv,
                    placeholder=ph,
                    risk_score=p.risk_score,
                    risk_level=p.risk_level.value,
                    decision_reason=p.reason,
                )
                if eid in approved_ids:
                    approved_entities.append(record)
                    ua_placeholder_map[eid] = "__APPROVED__"
                else:
                    self._store.store_mapping(
                        session_id=sid, placeholder=ph,
                        raw_value=p.entity_text, entity_type=p.entity_type,
                        normalized_value=nv,
                        entity_id=eid,
                        policy_decision=decision.value,
                        disclosure_allowed=getattr(p, "disclosure_allowed", True),
                    )
                    approval_required.append(record)
                    ua_placeholder_map[eid] = ph

            elif decision == PolicyDecision.REVIEW:
                ph = _get_or_create_placeholder(p)
                self._store.store_mapping(
                    session_id=sid, placeholder=ph,
                    raw_value=p.entity_text, entity_type=p.entity_type,
                    normalized_value=nv,
                    entity_id=eid,
                    policy_decision=decision.value,
                    disclosure_allowed=getattr(p, "disclosure_allowed", True),
                )
                review_required.append(ApprovalRecord(
                    entity_id=eid,
                    entity_type=p.entity_type,
                    entity_text=p.entity_text,
                    normalized_value=nv,
                    placeholder=ph,
                    risk_score=p.risk_score,
                    risk_level=p.risk_level.value,
                    decision_reason=p.reason,
                ))

        # -----------------------------------------------------------------
        # Step 4: Build replacement list
        # -----------------------------------------------------------------
        replacements: List[Tuple[int, int, str]] = []

        for p in resolved:
            decision = p.decision
            nv = p.normalized_value or p.entity_text
            eid = _entity_id(p.entity_type, nv)
            key = (p.entity_type, nv)

            if decision == PolicyDecision.RETAIN:
                continue
            elif decision == PolicyDecision.USER_APPROVAL:
                ua_ph = ua_placeholder_map.get(eid, "__APPROVED__")
                if ua_ph == "__APPROVED__":
                    continue
                replacements.append((p.start, p.end, ua_ph))
            elif decision in (PolicyDecision.MASK, PolicyDecision.REVIEW):
                ph = placeholder_cache.get(key)
                if ph:
                    replacements.append((p.start, p.end, ph))

        # -----------------------------------------------------------------
        # Step 5: Apply replacements RIGHT-TO-LEFT
        # -----------------------------------------------------------------
        replacements_rtl = sorted(replacements, key=lambda x: x[0], reverse=True)
        sanitized = prompt
        for start, end, ph in replacements_rtl:
            if sanitized[start:end] != prompt[start:end]:
                logger.warning(
                    "[SemanticMasker] Span mismatch at [%d:%d] -- skipping.",
                    start, end,
                )
                continue
            sanitized = sanitized[:start] + ph + sanitized[end:]

        mapping = self._store.get_mappings(sid)

        return SemanticMaskingResult(
            original_prompt=prompt,
            sanitized_prompt=sanitized,
            mapping=mapping,
            masked_entities=masked_entities,
            retained_entities=retained_entities,
            approval_required=approval_required,
            approved_entities=approved_entities,
            review_required=review_required,
            task_type=policy_report.task_type,
            overall_risk_level=policy_report.overall_risk_level.value,
            session_id=sid,
        )

    def approve_and_remask(
        self,
        prompt: str,
        policy_report: PolicyReport,
        approved_entity_ids: Set[str],
        session_id: Optional[str] = None,
    ) -> SemanticMaskingResult:
        """Re-run masking after user provides explicit approval for specific entities."""
        return self.mask(
            prompt=prompt,
            policy_report=policy_report,
            approved_entity_ids=approved_entity_ids,
            session_id=session_id,
        )

    def _empty_result(
        self,
        prompt: str,
        policy_report: PolicyReport,
        session_id: str,
    ) -> SemanticMaskingResult:
        return SemanticMaskingResult(
            original_prompt=prompt,
            sanitized_prompt=prompt,
            mapping={},
            masked_entities=[],
            retained_entities=[],
            approval_required=[],
            approved_entities=[],
            review_required=[],
            task_type=policy_report.task_type,
            overall_risk_level=policy_report.overall_risk_level.value,
            session_id=session_id,
        )
