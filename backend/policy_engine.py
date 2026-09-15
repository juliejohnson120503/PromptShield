"""
Risk Assessment & Privacy Policy Engine for PromptShield AI (Phase 3).

For every entity that Phase 2 has assigned a ContextualRole, this module
computes a numeric risk score and emits a concrete masking policy decision:

  MASK            — entity must be replaced with a placeholder before sending
                    to the LLM (high privacy risk / personal credential)
  RETAIN          — entity should be kept in the outgoing prompt as-is
                    (public knowledge, or task-essential without personal risk)
  USER_APPROVAL   — borderline; surface to user for an explicit keep/mask choice
  REVIEW          — unusual or ambiguous; flag for manual inspection

The engine operates on ContextualRole objects produced by Phase 2 (context_analyzer).
It is stateless per call; the caller (PromptShieldCore) owns session state.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
from backend.models import (
    EntityType,
    RoleCategory,
    ContextualRole,
    ContextAnalysisResult,
)


# ===========================================================================
# POLICY DECISION ENUM
# ===========================================================================

class PolicyDecision(str, Enum):
    """
    The action PromptShield will apply to a detected entity before the
    sanitised prompt is sent to the external LLM.
    """
    MASK          = "MASK"           # Replace with <TYPE_n> placeholder
    RETAIN        = "RETAIN"         # Pass through unchanged
    USER_APPROVAL = "USER_APPROVAL"  # Ask user before deciding
    REVIEW        = "REVIEW"         # Flag for additional scrutiny


# ===========================================================================
# RISK LEVEL
# ===========================================================================

class RiskLevel(str, Enum):
    CRITICAL = "CRITICAL"   # Must mask; secret / credential / strong personal PII
    HIGH     = "HIGH"       # Should mask; personal identifier in risky context
    MEDIUM   = "MEDIUM"     # Context-dependent; may retain if task-relevant
    LOW      = "LOW"        # Low concern; likely public or general reference
    NONE     = "NONE"       # No privacy risk detected


# ===========================================================================
# OUTPUT MODELS
# ===========================================================================

@dataclass
class EntityPolicyResult:
    """
    Full policy verdict for a single detected entity.
    Carries the full audit trail: role inputs → risk score → decision.
    """
    entity_text:    str
    entity_type:    EntityType
    role_category:  RoleCategory
    is_first_party: bool
    is_public_knowledge: bool
    is_task_relevant: bool

    # Risk assessment
    risk_score:  float          # 0.0 (harmless) → 1.0 (critical)
    risk_level:  RiskLevel
    risk_factors: List[str]     # Human-readable explanations

    # Policy output
    decision:    PolicyDecision
    reason:      str            # Short audit-trail sentence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_text":       self.entity_text,
            "entity_type":       self.entity_type.value,
            "role_category":     self.role_category.value,
            "is_first_party":    self.is_first_party,
            "is_public_knowledge": self.is_public_knowledge,
            "is_task_relevant":  self.is_task_relevant,
            "risk_score":        round(self.risk_score, 4),
            "risk_level":        self.risk_level.value,
            "risk_factors":      self.risk_factors,
            "decision":          self.decision.value,
            "reason":            self.reason,
        }


@dataclass
class PolicyReport:
    """
    Aggregated policy report for an entire prompt's entity set.
    Produced by the PolicyEngine and consumed by Phase 4 (semantic masking).
    """
    prompt:            str
    task_type:         str
    task_confidence:   float
    entity_policies:   List[EntityPolicyResult]

    # Convenience views
    @property
    def entities_to_mask(self) -> List[EntityPolicyResult]:
        return [p for p in self.entity_policies
                if p.decision == PolicyDecision.MASK]

    @property
    def entities_to_retain(self) -> List[EntityPolicyResult]:
        return [p for p in self.entity_policies
                if p.decision == PolicyDecision.RETAIN]

    @property
    def entities_needing_approval(self) -> List[EntityPolicyResult]:
        return [p for p in self.entity_policies
                if p.decision == PolicyDecision.USER_APPROVAL]

    @property
    def overall_risk_level(self) -> RiskLevel:
        """Highest risk level among all evaluated entities."""
        order = [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM,
                 RiskLevel.LOW, RiskLevel.NONE]
        for level in order:
            if any(p.risk_level == level for p in self.entity_policies):
                return level
        return RiskLevel.NONE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt":           self.prompt,
            "task_type":        self.task_type,
            "task_confidence":  round(self.task_confidence, 4),
            "overall_risk":     self.overall_risk_level.value,
            "entity_policies":  [p.to_dict() for p in self.entity_policies],
            "summary": {
                "total":          len(self.entity_policies),
                "mask":           len(self.entities_to_mask),
                "retain":         len(self.entities_to_retain),
                "user_approval":  len(self.entities_needing_approval),
            },
        }


# ===========================================================================
# RISK SCORING RULES
# ===========================================================================

# Base risk scores per entity type (before context modifiers)
BASE_RISK: Dict[EntityType, float] = {
    EntityType.API_KEY:      0.98,
    EntityType.PASSWORD:     0.97,
    EntityType.CREDIT_CARD:  0.95,
    EntityType.EMAIL:        0.65,
    EntityType.PHONE:        0.60,
    EntityType.PERSON:       0.45,
    EntityType.ORGANIZATION: 0.25,
    EntityType.LOCATION:     0.20,
    EntityType.DATE:         0.15,
}

# Modifier weights applied on top of base risk
MODIFIER_FIRST_PARTY_BOOST  = +0.20  # "my email", "my name is" → riskier
MODIFIER_PUBLIC_KNOWLEDGE    = -0.40  # "Elon Musk", "Paris" → less risky
MODIFIER_TASK_RELEVANT       = -0.15  # Needed for the task → slight relaxation
MODIFIER_CREDENTIAL_ROLE     = +0.30  # Explicit TECHNICAL_CREDENTIAL role

# Thresholds mapping score → RiskLevel
RISK_THRESHOLDS = [
    (0.85, RiskLevel.CRITICAL),
    (0.65, RiskLevel.HIGH),
    (0.40, RiskLevel.MEDIUM),
    (0.15, RiskLevel.LOW),
    (0.00, RiskLevel.NONE),
]


def _score_to_level(score: float) -> RiskLevel:
    """Map a continuous [0,1] risk score to a discrete RiskLevel."""
    for threshold, level in RISK_THRESHOLDS:
        if score >= threshold:
            return level
    return RiskLevel.NONE


# ===========================================================================
# POLICY ENGINE
# ===========================================================================

class PolicyEngine:
    """
    Phase 3: Risk Assessment & Privacy Policy Engine.

    Consumes Phase 2 ContextAnalysisResult and produces a PolicyReport
    containing per-entity risk scores and MASK / RETAIN / USER_APPROVAL
    decisions ready for Phase 4 enforcement.
    """

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def evaluate(self, context_result: ContextAnalysisResult) -> PolicyReport:
        """
        Evaluate all entity contexts and produce a full PolicyReport.

        Args:
            context_result: Output from Phase 2 ContextAnalyzer.analyze()

        Returns:
            PolicyReport with per-entity risk scores and policy decisions.
        """
        policies: List[EntityPolicyResult] = []
        for role in context_result.entity_contexts:
            policy = self._evaluate_entity(role)
            policies.append(policy)

        return PolicyReport(
            prompt=context_result.prompt,
            task_type=context_result.task_type.value,
            task_confidence=context_result.task_confidence,
            entity_policies=policies,
        )

    # -----------------------------------------------------------------------
    # Internal per-entity evaluation
    # -----------------------------------------------------------------------

    def _evaluate_entity(self, role: ContextualRole) -> EntityPolicyResult:
        """Compute risk score + policy decision for one entity."""
        score, factors = self._compute_risk_score(role)
        risk_level = _score_to_level(score)
        decision, reason = self._decide_policy(role, score, risk_level)

        return EntityPolicyResult(
            entity_text=role.entity_text,
            entity_type=role.entity_type,
            role_category=role.role_category,
            is_first_party=role.is_first_party,
            is_public_knowledge=role.is_public_knowledge,
            is_task_relevant=role.is_task_relevant,
            risk_score=score,
            risk_level=risk_level,
            risk_factors=factors,
            decision=decision,
            reason=reason,
        )

    def _compute_risk_score(
        self, role: ContextualRole
    ) -> tuple[float, List[str]]:
        """
        Return (score, [factors]) where score ∈ [0.0, 1.0].
        Each factor is a human-readable string explaining a contributing rule.
        """
        base = BASE_RISK.get(role.entity_type, 0.50)
        score = base
        factors: List[str] = [
            f"Base risk for {role.entity_type.value}: {base:.2f}"
        ]

        # --- Context modifiers ---

        if role.role_category == RoleCategory.TECHNICAL_CREDENTIAL:
            score = min(1.0, score + MODIFIER_CREDENTIAL_ROLE)
            factors.append(
                f"Technical credential role (+{MODIFIER_CREDENTIAL_ROLE:.2f})"
            )

        if role.is_first_party:
            score = min(1.0, score + MODIFIER_FIRST_PARTY_BOOST)
            factors.append(
                f"First-party association (user's own data) (+{MODIFIER_FIRST_PARTY_BOOST:.2f})"
            )

        if role.is_public_knowledge:
            score = max(0.0, score + MODIFIER_PUBLIC_KNOWLEDGE)
            factors.append(
                f"Publicly known entity (−{abs(MODIFIER_PUBLIC_KNOWLEDGE):.2f})"
            )

        if role.is_task_relevant and not role.is_first_party:
            score = max(0.0, score + MODIFIER_TASK_RELEVANT)
            factors.append(
                f"Task-essential entity, not first-party (−{abs(MODIFIER_TASK_RELEVANT):.2f})"
            )

        score = round(min(1.0, max(0.0, score)), 4)
        return score, factors

    def _decide_policy(
        self,
        role: ContextualRole,
        score: float,
        risk_level: RiskLevel,
    ) -> tuple[PolicyDecision, str]:
        """
        Map risk score + contextual flags to a concrete PolicyDecision.

        Decision logic (ordered by priority):
        1. Technical credentials → always MASK
        2. Public knowledge and NOT first-party → RETAIN
        3. CRITICAL / HIGH risk → MASK
        4. MEDIUM risk + task-relevant (and not first-party) → USER_APPROVAL
        5. MEDIUM risk otherwise → MASK
        6. LOW / NONE risk → RETAIN
        """

        # Rule 1 — Credentials are always masked, no exceptions
        if role.role_category == RoleCategory.TECHNICAL_CREDENTIAL:
            return (
                PolicyDecision.MASK,
                "Technical credential (API key / password / card) — always masked.",
            )

        # Rule 2 — Public figures / public facts: retain unless the user is
        #           personally associated (e.g., "my company is Google")
        if role.is_public_knowledge and not role.is_first_party:
            return (
                PolicyDecision.RETAIN,
                "Publicly known entity with no personal ownership — retained for task accuracy.",
            )

        # Rule 3 — Critical or high risk → mask unconditionally
        if risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH):
            return (
                PolicyDecision.MASK,
                f"{risk_level.value} risk score ({score:.2f}) — masked to protect personal data.",
            )

        # Rule 4 — Medium risk but task-essential and not personal → user approval
        if risk_level == RiskLevel.MEDIUM and role.is_task_relevant and not role.is_first_party:
            return (
                PolicyDecision.USER_APPROVAL,
                f"Medium risk ({score:.2f}), task-relevant and not first-party — "
                "user approval required before sending.",
            )

        # Rule 5 — Medium risk without task relevance → mask
        if risk_level == RiskLevel.MEDIUM:
            return (
                PolicyDecision.MASK,
                f"Medium risk ({score:.2f}), not clearly task-essential — masked by default.",
            )

        # Rule 6 — Low / no risk → retain
        return (
            PolicyDecision.RETAIN,
            f"Low risk ({score:.2f}) — retained; no significant privacy concern.",
        )
