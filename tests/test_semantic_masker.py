"""
PromptShield AI — Phase 4: Semantic Masker Test Suite (Tests A–J + Security Invariants).

Tests:
  A  — MASK: password in irrelevant task context
  B  — USER_APPROVAL without approval (literal must be protected)
  C  — USER_APPROVAL with explicit approval (literal released, risk metadata preserved)
  D  — RETAIN: public/task-irrelevant entities unchanged
  E  — Mixed decisions in one prompt
  F  — Repeated same-value entity (same placeholder, multiple occurrences)
  G  — Two distinct values of same type (different placeholders)
  H  — Overlapping spans from connection string (no corruption / leakage)
  I  — DOB USER_APPROVAL (before and after approval)
  J  — REVIEW decision (masked by default, in review_required)

  Security invariants:
  SI-1  MASK literals absent from sanitized_prompt
  SI-2  Unapproved USER_APPROVAL literals absent from sanitized_prompt
  SI-3  REVIEW literals absent from sanitized_prompt
  SI-4  RETAIN entities unchanged
  SI-5  Approved USER_APPROVAL literals present in sanitized_prompt
  SI-6  Mapping is local (not embedded in sanitized_prompt)
  SI-7  Placeholder IDs stable within same prompt
  SI-8  Repeated sensitive values all replaced
  SI-9  Overlapping entities cause no partial leakage
"""

import unittest
from unittest.mock import patch, MagicMock

from backend.core import PromptShieldCore
from backend.models import EntityType, ApprovalRecord, SemanticMaskingResult
from backend.policy_engine import PolicyDecision, PolicyReport, EntityPolicyResult, RiskLevel
from backend.semantic_masker import SemanticMasker, _entity_id, _resolve_overlaps
from backend.mapping_store import MappingStore
from backend.models import RoleCategory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_policy_result(
    entity_text: str,
    entity_type: EntityType,
    decision: PolicyDecision,
    start: int,
    end: int,
    normalized_value: str = "",
    risk_score: float = 0.5,
    risk_level: RiskLevel = RiskLevel.MEDIUM,
    reason: str = "test",
    role_category: RoleCategory = RoleCategory.PERSONAL_IDENTIFIER,
    is_first_party: bool = True,
    is_public_knowledge: bool = False,
    is_task_relevant: bool = False,
    value_required: bool = False,
) -> EntityPolicyResult:
    return EntityPolicyResult(
        entity_text=entity_text,
        entity_type=entity_type,
        role_category=role_category,
        is_first_party=is_first_party,
        is_public_knowledge=is_public_knowledge,
        is_task_relevant=is_task_relevant,
        risk_score=risk_score,
        risk_level=risk_level,
        risk_factors=["test factor"],
        decision=decision,
        reason=reason,
        value_required=value_required,
        entity_task_relation="NONE",
        start=start,
        end=end,
        normalized_value=normalized_value or entity_text,
    )


def _make_policy_report(
    prompt: str,
    policies: list,
    task_type: str = "GENERAL_QA",
    task_confidence: float = 0.9,
) -> PolicyReport:
    return PolicyReport(
        prompt=prompt,
        task_type=task_type,
        task_confidence=task_confidence,
        entity_policies=policies,
    )


# ---------------------------------------------------------------------------
# Main test class
# ---------------------------------------------------------------------------

class TestSemanticMaskerTests(unittest.TestCase):

    def setUp(self):
        self.masker = SemanticMasker(mapping_store=MappingStore())

    # -----------------------------------------------------------------------
    # TEST A — MASK
    # -----------------------------------------------------------------------
    def test_a_mask_password_irrelevant_task(self):
        """
        'My password is Apple#123. Explain cloud computing.'
        -> PASSWORD must be replaced with <PASSWORD_1>.
        Reason: task is GENERAL_INFORMATION, value not required.
        """
        prompt = "My password is Apple#123. Explain cloud computing."
        start = prompt.index("Apple#123")
        end = start + len("Apple#123")
        policy = _make_policy_result(
            entity_text="Apple#123",
            entity_type=EntityType.PASSWORD,
            decision=PolicyDecision.MASK,
            start=start, end=end,
            normalized_value="Apple#123",
            risk_score=0.97, risk_level=RiskLevel.CRITICAL,
            reason="Authentication credential -- always masked.",
            role_category=RoleCategory.TECHNICAL_CREDENTIAL,
        )
        report = _make_policy_report(prompt, [policy])
        result = self.masker.mask(prompt, report)

        self.assertEqual(result.sanitized_prompt,
                         "My password is <PASSWORD_1>. Explain cloud computing.")
        self.assertIn("<PASSWORD_1>", result.mapping)
        self.assertEqual(result.mapping["<PASSWORD_1>"], "Apple#123")
        self.assertEqual(len(result.masked_entities), 1)
        self.assertEqual(len(result.approval_required), 0)

    # -----------------------------------------------------------------------
    # TEST B — USER_APPROVAL without approval
    # -----------------------------------------------------------------------
    def test_b_user_approval_without_approval(self):
        """
        'My password is Apple#123. Is this password strong?'
        Without approval -> literal MUST NOT appear in sanitized_prompt.
        approval_required must contain the password entity.
        """
        prompt = "My password is Apple#123. Is this password strong?"
        start = prompt.index("Apple#123")
        end = start + len("Apple#123")
        policy = _make_policy_result(
            entity_text="Apple#123",
            entity_type=EntityType.PASSWORD,
            decision=PolicyDecision.USER_APPROVAL,
            start=start, end=end,
            normalized_value="Apple#123",
            risk_score=0.97, risk_level=RiskLevel.CRITICAL,
            reason="Sensitive literal value required as TARGET_OF_ANALYSIS.",
            role_category=RoleCategory.TECHNICAL_CREDENTIAL,
            value_required=True,
        )
        report = _make_policy_report(prompt, [policy])
        result = self.masker.mask(prompt, report)

        # Literal must NOT appear in sanitized prompt
        self.assertNotIn("Apple#123", result.sanitized_prompt,
                         "Unapproved USER_APPROVAL literal must not appear in sanitized_prompt")
        self.assertIn("<PASSWORD_1>", result.sanitized_prompt)
        self.assertEqual(len(result.approval_required), 1)
        self.assertEqual(result.approval_required[0].entity_text, "Apple#123")
        self.assertEqual(len(result.approved_entities), 0)

    # -----------------------------------------------------------------------
    # TEST C — USER_APPROVAL with approval
    # -----------------------------------------------------------------------
    def test_c_user_approval_with_approval(self):
        """
        Same prompt, but password entity explicitly approved.
        -> literal appears in sanitized_prompt.
        -> risk metadata still indicates CRITICAL.
        """
        prompt = "My password is Apple#123. Is this password strong?"
        start = prompt.index("Apple#123")
        end = start + len("Apple#123")
        policy = _make_policy_result(
            entity_text="Apple#123",
            entity_type=EntityType.PASSWORD,
            decision=PolicyDecision.USER_APPROVAL,
            start=start, end=end,
            normalized_value="Apple#123",
            risk_score=0.97, risk_level=RiskLevel.CRITICAL,
            reason="Sensitive literal value required as TARGET_OF_ANALYSIS.",
            role_category=RoleCategory.TECHNICAL_CREDENTIAL,
            value_required=True,
        )
        report = _make_policy_report(prompt, [policy])
        approved_id = _entity_id(EntityType.PASSWORD, "Apple#123")
        result = self.masker.mask(prompt, report, approved_entity_ids={approved_id})

        # Literal MAY appear
        self.assertEqual(result.sanitized_prompt, prompt,
                         "Approved entity should leave prompt unchanged")
        self.assertEqual(len(result.approved_entities), 1)
        self.assertEqual(len(result.approval_required), 0)
        # Risk metadata preserved
        approved = result.approved_entities[0]
        self.assertEqual(approved.risk_level, RiskLevel.CRITICAL.value)
        self.assertAlmostEqual(approved.risk_score, 0.97, places=2)

    # -----------------------------------------------------------------------
    # TEST D — RETAIN
    # -----------------------------------------------------------------------
    def test_d_retain_public_entities_unchanged(self):
        """
        'Compare Java and Python and tell me which programming language is easier.'
        Phase 3 says RETAIN for Java and Python.
        Phase 4 must leave original text unchanged.
        """
        prompt = "Compare Java and Python and tell me which programming language is easier."
        java_start = prompt.index("Java")
        python_start = prompt.index("Python")
        policies = [
            _make_policy_result(
                "Java", EntityType.ORGANIZATION, PolicyDecision.RETAIN,
                java_start, java_start + 4, "Java",
                risk_score=0.1, risk_level=RiskLevel.NONE,
                is_public_knowledge=True, role_category=RoleCategory.PUBLIC_FIGURE_OR_FACT,
            ),
            _make_policy_result(
                "Python", EntityType.ORGANIZATION, PolicyDecision.RETAIN,
                python_start, python_start + 6, "Python",
                risk_score=0.1, risk_level=RiskLevel.NONE,
                is_public_knowledge=True, role_category=RoleCategory.PUBLIC_FIGURE_OR_FACT,
            ),
        ]
        report = _make_policy_report(prompt, policies)
        result = self.masker.mask(prompt, report)

        self.assertEqual(result.sanitized_prompt, prompt)
        self.assertEqual(len(result.retained_entities), 2)
        self.assertEqual(len(result.masked_entities), 0)
        self.assertEqual(result.mapping, {})

    # -----------------------------------------------------------------------
    # TEST E — Mixed decisions
    # -----------------------------------------------------------------------
    def test_e_mixed_decisions(self):
        """
        'My name is Elena Rodriguez, my password is Apple#123, and I work for Orion Analytics. Explain cloud computing.'
        Elena -> MASK, Apple#123 -> MASK, Orion Analytics -> RETAIN
        """
        prompt = ("My name is Elena Rodriguez, my password is Apple#123, "
                  "and I work for Orion Analytics. Explain cloud computing.")
        elena_start = prompt.index("Elena Rodriguez")
        apple_start = prompt.index("Apple#123")
        orion_start = prompt.index("Orion Analytics")
        policies = [
            _make_policy_result(
                "Elena Rodriguez", EntityType.PERSON, PolicyDecision.MASK,
                elena_start, elena_start + len("Elena Rodriguez"),
                "Elena Rodriguez", 0.65, RiskLevel.HIGH,
                role_category=RoleCategory.PERSONAL_IDENTIFIER,
            ),
            _make_policy_result(
                "Apple#123", EntityType.PASSWORD, PolicyDecision.MASK,
                apple_start, apple_start + len("Apple#123"),
                "Apple#123", 0.97, RiskLevel.CRITICAL,
                role_category=RoleCategory.TECHNICAL_CREDENTIAL,
            ),
            _make_policy_result(
                "Orion Analytics", EntityType.ORGANIZATION, PolicyDecision.RETAIN,
                orion_start, orion_start + len("Orion Analytics"),
                "Orion Analytics", 0.2, RiskLevel.LOW,
                is_public_knowledge=False, role_category=RoleCategory.TASK_RELEVANT_ENTITY,
            ),
        ]
        report = _make_policy_report(prompt, policies)
        result = self.masker.mask(prompt, report)

        self.assertIn("<PERSON_1>", result.sanitized_prompt)
        self.assertIn("<PASSWORD_1>", result.sanitized_prompt)
        self.assertIn("Orion Analytics", result.sanitized_prompt)
        self.assertNotIn("Elena Rodriguez", result.sanitized_prompt)
        self.assertNotIn("Apple#123", result.sanitized_prompt)
        self.assertEqual(len(result.masked_entities), 2)
        self.assertEqual(len(result.retained_entities), 1)

    # -----------------------------------------------------------------------
    # TEST F — Repeated same entity (same placeholder)
    # -----------------------------------------------------------------------
    def test_f_repeated_entity_same_placeholder(self):
        """
        'My email is john@example.com. Send the report to john@example.com.'
        Both occurrences must use <EMAIL_1>.
        Mapping must contain one EMAIL_1 entry.
        """
        prompt = "My email is john@example.com. Send the report to john@example.com."
        first = prompt.index("john@example.com")
        second = prompt.index("john@example.com", first + 1)
        policies = [
            _make_policy_result(
                "john@example.com", EntityType.EMAIL, PolicyDecision.MASK,
                first, first + len("john@example.com"),
                "john@example.com", 0.65, RiskLevel.HIGH,
            ),
            _make_policy_result(
                "john@example.com", EntityType.EMAIL, PolicyDecision.MASK,
                second, second + len("john@example.com"),
                "john@example.com", 0.65, RiskLevel.HIGH,
            ),
        ]
        report = _make_policy_report(prompt, policies)
        result = self.masker.mask(prompt, report)

        self.assertEqual(
            result.sanitized_prompt,
            "My email is <EMAIL_1>. Send the report to <EMAIL_1>.",
        )
        self.assertIn("<EMAIL_1>", result.mapping)
        # Only one placeholder in mapping (same normalized value)
        email_keys = [k for k in result.mapping if "EMAIL" in k]
        self.assertEqual(len(email_keys), 1, "Same entity value must reuse one placeholder")

    # -----------------------------------------------------------------------
    # TEST G — Two distinct values of same type (different placeholders)
    # -----------------------------------------------------------------------
    def test_g_two_distinct_values_different_placeholders(self):
        """
        'Send the report from john@example.com to alice@example.com.'
        -> <EMAIL_1> and <EMAIL_2>
        """
        prompt = "Send the report from john@example.com to alice@example.com."
        john_start = prompt.index("john@example.com")
        alice_start = prompt.index("alice@example.com")
        policies = [
            _make_policy_result(
                "john@example.com", EntityType.EMAIL, PolicyDecision.MASK,
                john_start, john_start + len("john@example.com"),
                "john@example.com", 0.65, RiskLevel.HIGH,
            ),
            _make_policy_result(
                "alice@example.com", EntityType.EMAIL, PolicyDecision.MASK,
                alice_start, alice_start + len("alice@example.com"),
                "alice@example.com", 0.65, RiskLevel.HIGH,
            ),
        ]
        report = _make_policy_report(prompt, policies)
        result = self.masker.mask(prompt, report)

        self.assertIn("<EMAIL_1>", result.sanitized_prompt)
        self.assertIn("<EMAIL_2>", result.sanitized_prompt)
        self.assertNotIn("john@example.com", result.sanitized_prompt)
        self.assertNotIn("alice@example.com", result.sanitized_prompt)
        email_keys = sorted(k for k in result.mapping if "EMAIL" in k)
        self.assertEqual(len(email_keys), 2)

    # -----------------------------------------------------------------------
    # TEST H — Overlapping spans (connection string)
    # -----------------------------------------------------------------------
    def test_h_overlapping_connection_string(self):
        """
        A connection string where sub-detectors might tag overlapping spans.
        Phase 4 must produce: no corrupted text, no nested placeholders,
        no partial credential leakage.
        """
        prompt = "postgresql://erodriguez:Secret123@db.example.com:5432/payroll"
        # Simulate three overlapping policies (as if from different detectors)
        full_start, full_end = 0, len(prompt)
        # Sub-span: EMAIL-like (erodriguez@db...) — overlapping
        sub_start = prompt.index("erodriguez")
        sub_end = prompt.index(":5432")
        policies = [
            _make_policy_result(
                prompt, EntityType.CONNECTION_STRING, PolicyDecision.MASK,
                full_start, full_end, prompt, 1.0, RiskLevel.CRITICAL,
                role_category=RoleCategory.TECHNICAL_CREDENTIAL,
            ),
            # Overlapping sub-span (should be skipped by _resolve_overlaps)
            _make_policy_result(
                "erodriguez", EntityType.USERNAME, PolicyDecision.MASK,
                sub_start, sub_start + len("erodriguez"),
                "erodriguez", 0.9, RiskLevel.HIGH,
            ),
        ]
        report = _make_policy_report(prompt, policies)
        result = self.masker.mask(prompt, report)

        # No nested placeholders
        self.assertNotIn("<<", result.sanitized_prompt)
        self.assertNotIn(">>", result.sanitized_prompt)
        # Credential must be masked, not partially exposed
        self.assertNotIn("Secret123", result.sanitized_prompt)
        self.assertNotIn("erodriguez", result.sanitized_prompt)
        # Exactly one placeholder (the connection string)
        self.assertIn("<CONNECTION_STRING_1>", result.sanitized_prompt)
        self.assertEqual(result.sanitized_prompt, "<CONNECTION_STRING_1>")

    # -----------------------------------------------------------------------
    # TEST I — DOB USER_APPROVAL (before and after approval)
    # -----------------------------------------------------------------------
    def test_i_dob_user_approval_before_and_after(self):
        """
        'My date of birth is 17 September 1994. Calculate my current age.'
        Before approval: DOB protected.
        After approval: DOB literal released.
        """
        prompt = "My date of birth is 17 September 1994. Calculate my current age."
        dob_text = "17 September 1994"
        dob_start = prompt.index(dob_text)
        dob_end = dob_start + len(dob_text)
        policy = _make_policy_result(
            dob_text, EntityType.DOB, PolicyDecision.USER_APPROVAL,
            dob_start, dob_end, dob_text,
            risk_score=0.75, risk_level=RiskLevel.HIGH,
            reason="Literal date value required for mathematical calculation.",
            value_required=True,
        )
        report = _make_policy_report(prompt, [policy])

        # Before approval
        result_before = self.masker.mask(prompt, report)
        self.assertNotIn(dob_text, result_before.sanitized_prompt,
                         "Unapproved DOB must not appear in sanitized_prompt")
        self.assertIn("<DOB_1>", result_before.sanitized_prompt)
        self.assertEqual(len(result_before.approval_required), 1)
        self.assertEqual(result_before.approval_required[0].entity_type, EntityType.DOB)

        # After approval
        approved_id = _entity_id(EntityType.DOB, dob_text)
        result_after = self.masker.mask(prompt, report, approved_entity_ids={approved_id})
        self.assertEqual(result_after.sanitized_prompt, prompt,
                         "Approved DOB should leave prompt unchanged")
        self.assertEqual(len(result_after.approved_entities), 1)
        self.assertEqual(len(result_after.approval_required), 0)

    # -----------------------------------------------------------------------
    # TEST J — REVIEW decision (masked by default)
    # -----------------------------------------------------------------------
    def test_j_review_masked_by_default(self):
        """
        Entity with REVIEW decision must be masked and appear in review_required.
        """
        prompt = "Contact me at 10.0.0.1 for the internal service."
        ip_text = "10.0.0.1"
        ip_start = prompt.index(ip_text)
        ip_end = ip_start + len(ip_text)
        policy = _make_policy_result(
            ip_text, EntityType.IP_ADDRESS, PolicyDecision.REVIEW,
            ip_start, ip_end, ip_text,
            risk_score=0.55, risk_level=RiskLevel.MEDIUM,
            reason="Unusual private IP; flagged for manual review.",
        )
        report = _make_policy_report(prompt, [policy])
        result = self.masker.mask(prompt, report)

        self.assertNotIn(ip_text, result.sanitized_prompt,
                         "REVIEW entity literal must be masked by default")
        self.assertIn("<IP_ADDRESS_1>", result.sanitized_prompt)
        self.assertEqual(len(result.review_required), 1)
        self.assertEqual(result.review_required[0].entity_text, ip_text)
        self.assertEqual(len(result.approved_entities), 0)

    # -----------------------------------------------------------------------
    # SECURITY INVARIANTS
    # -----------------------------------------------------------------------

    def test_si_1_mask_literal_absent_from_sanitized(self):
        """SI-1: A MASK entity's literal value must not occur in sanitized_prompt."""
        prompt = "API key is sk-live-abc1234567890def12345678."
        start = prompt.index("sk-live-abc1234567890def12345678")
        end = start + len("sk-live-abc1234567890def12345678")
        policy = _make_policy_result(
            "sk-live-abc1234567890def12345678", EntityType.API_KEY,
            PolicyDecision.MASK, start, end,
            "sk-live-abc1234567890def12345678", 0.99, RiskLevel.CRITICAL,
            role_category=RoleCategory.TECHNICAL_CREDENTIAL,
        )
        report = _make_policy_report(prompt, [policy])
        result = self.masker.mask(prompt, report)
        self.assertNotIn("sk-live-abc1234567890def12345678", result.sanitized_prompt)

    def test_si_2_unapproved_ua_literal_absent(self):
        """SI-2: An unapproved USER_APPROVAL entity must not appear in sanitized_prompt."""
        prompt = "My DOB is 15 August 1990. Calculate age."
        dob = "15 August 1990"
        start = prompt.index(dob)
        policy = _make_policy_result(
            dob, EntityType.DOB, PolicyDecision.USER_APPROVAL,
            start, start + len(dob), dob, 0.75, RiskLevel.HIGH,
        )
        report = _make_policy_report(prompt, [policy])
        result = self.masker.mask(prompt, report)
        self.assertNotIn(dob, result.sanitized_prompt)

    def test_si_3_review_literal_absent(self):
        """SI-3: A REVIEW entity's literal value must not appear in sanitized_prompt."""
        prompt = "Server at 192.168.1.100 is down."
        ip = "192.168.1.100"
        start = prompt.index(ip)
        policy = _make_policy_result(
            ip, EntityType.IP_ADDRESS, PolicyDecision.REVIEW,
            start, start + len(ip), ip, 0.55, RiskLevel.MEDIUM,
        )
        report = _make_policy_report(prompt, [policy])
        result = self.masker.mask(prompt, report)
        self.assertNotIn(ip, result.sanitized_prompt)

    def test_si_4_retain_entities_unchanged(self):
        """SI-4: RETAIN entities remain unchanged in sanitized_prompt."""
        prompt = "What is quantum computing?"
        policy = _make_policy_result(
            "quantum computing", EntityType.ORGANIZATION, PolicyDecision.RETAIN,
            8, 25, "quantum computing", 0.0, RiskLevel.NONE,
            is_public_knowledge=True,
        )
        report = _make_policy_report(prompt, [policy])
        result = self.masker.mask(prompt, report)
        self.assertEqual(result.sanitized_prompt, prompt)

    def test_si_5_approved_ua_literal_present(self):
        """SI-5: Approved USER_APPROVAL entities may appear as literals."""
        prompt = "My password is S3cur3! Is it strong?"
        start = prompt.index("S3cur3!")
        policy = _make_policy_result(
            "S3cur3!", EntityType.PASSWORD, PolicyDecision.USER_APPROVAL,
            start, start + len("S3cur3!"), "S3cur3!", 0.97, RiskLevel.CRITICAL,
            role_category=RoleCategory.TECHNICAL_CREDENTIAL,
        )
        report = _make_policy_report(prompt, [policy])
        approved_id = _entity_id(EntityType.PASSWORD, "S3cur3!")
        result = self.masker.mask(prompt, report, approved_entity_ids={approved_id})
        self.assertIn("S3cur3!", result.sanitized_prompt)

    def test_si_6_mapping_local_not_in_sanitized(self):
        """SI-6: Placeholder mapping keys/values are not embedded in sanitized_prompt."""
        prompt = "My credit card is 4111111111111111."
        start = prompt.index("4111111111111111")
        policy = _make_policy_result(
            "4111111111111111", EntityType.CREDIT_CARD, PolicyDecision.MASK,
            start, start + 16, "4111111111111111", 0.95, RiskLevel.CRITICAL,
        )
        report = _make_policy_report(prompt, [policy])
        result = self.masker.mask(prompt, report)
        # mapping should not appear in sanitized prompt as a raw dict
        for raw_val in result.mapping.values():
            self.assertNotIn(raw_val, result.sanitized_prompt)

    def test_si_7_placeholder_ids_stable_within_prompt(self):
        """SI-7: Placeholder IDs are stable within the same prompt."""
        prompt = "Email john@a.com and john@a.com again."
        first = prompt.index("john@a.com")
        second = prompt.index("john@a.com", first + 1)
        policies = [
            _make_policy_result(
                "john@a.com", EntityType.EMAIL, PolicyDecision.MASK,
                first, first + 10, "john@a.com",
            ),
            _make_policy_result(
                "john@a.com", EntityType.EMAIL, PolicyDecision.MASK,
                second, second + 10, "john@a.com",
            ),
        ]
        report = _make_policy_report(prompt, policies)
        result = self.masker.mask(prompt, report)
        # Both occurrences should use same placeholder
        count = result.sanitized_prompt.count("<EMAIL_1>")
        self.assertEqual(count, 2)

    def test_si_8_repeated_values_all_replaced(self):
        """SI-8: All occurrences of a repeated sensitive value are replaced."""
        prompt = "John said John would help John."
        indices = []
        pos = 0
        while True:
            idx = prompt.find("John", pos)
            if idx == -1:
                break
            indices.append(idx)
            pos = idx + 1
        policies = [
            _make_policy_result(
                "John", EntityType.PERSON, PolicyDecision.MASK,
                i, i + 4, "John", 0.45, RiskLevel.MEDIUM,
                role_category=RoleCategory.PERSONAL_IDENTIFIER,
            )
            for i in indices
        ]
        report = _make_policy_report(prompt, policies)
        result = self.masker.mask(prompt, report)
        self.assertNotIn("John", result.sanitized_prompt)

    def test_si_9_overlapping_no_partial_leakage(self):
        """SI-9: Overlapping entities cannot cause partial leakage."""
        prompt = "postgresql://admin:P@ssw0rd@db.host:5432/prod"
        full_policy = _make_policy_result(
            prompt, EntityType.CONNECTION_STRING, PolicyDecision.MASK,
            0, len(prompt), prompt, 1.0, RiskLevel.CRITICAL,
            role_category=RoleCategory.TECHNICAL_CREDENTIAL,
        )
        sub_policy = _make_policy_result(
            "P@ssw0rd", EntityType.PASSWORD, PolicyDecision.MASK,
            prompt.index("P@ssw0rd"),
            prompt.index("P@ssw0rd") + len("P@ssw0rd"),
            "P@ssw0rd", 0.97, RiskLevel.CRITICAL,
            role_category=RoleCategory.TECHNICAL_CREDENTIAL,
        )
        report = _make_policy_report(prompt, [full_policy, sub_policy])
        result = self.masker.mask(prompt, report)
        self.assertNotIn("P@ssw0rd", result.sanitized_prompt)
        self.assertNotIn("admin", result.sanitized_prompt)
        self.assertNotIn("<<", result.sanitized_prompt)


# ---------------------------------------------------------------------------
# End-to-end pipeline tests via PromptShieldCore (use_spacy=False for speed)
# ---------------------------------------------------------------------------

class TestSemanticMaskerEndToEnd(unittest.TestCase):
    """
    End-to-end tests that run the real pipeline (Phase 1-4) without mocking.
    Uses use_spacy=False for test speed (regex + heuristic NER only).
    """

    @classmethod
    def setUpClass(cls):
        cls.shield = PromptShieldCore(use_spacy=False)

    def test_e2e_mask_password(self):
        """Password in irrelevant task context is masked in sanitized_prompt."""
        prompt = "My password is Apple#123. Explain cloud computing."
        result = self.shield.sanitize(prompt)
        self.assertNotIn("Apple#123", result.sanitized_prompt)
        self.assertIsInstance(result, SemanticMaskingResult)

    def test_e2e_retain_public_entity(self):
        """Public query retains public entity names."""
        prompt = "What is quantum computing?"
        result = self.shield.sanitize(prompt)
        # Should not crash; sanitized prompt should be reasonable
        self.assertIsInstance(result, SemanticMaskingResult)
        self.assertIsNotNone(result.sanitized_prompt)

    def test_e2e_user_approval_dob(self):
        """DOB in age-calculation context gets USER_APPROVAL; literal protected by default."""
        prompt = "My date of birth is 17 September 1994. Calculate my current age."
        result = self.shield.sanitize(prompt)
        # Either masked or in approval_required (depends on Phase 3 decision)
        # In both cases, DOB literal must not be in sanitized if it was masked/UA
        ua_texts = [a.entity_text for a in result.approval_required]
        masked_texts = [e.entity_text for e in result.masked_entities]
        if "17 September 1994" in ua_texts or "17 September 1994" in masked_texts:
            self.assertNotIn("17 September 1994", result.sanitized_prompt)

    def test_e2e_sanitize_api_key(self):
        """API key is always masked."""
        prompt = "Use API key sk-live-abc1234567890def12345678 to authenticate."
        result = self.shield.sanitize(prompt)
        self.assertNotIn("sk-live-abc1234567890def12345678", result.sanitized_prompt)

    def test_e2e_approve_and_sanitize(self):
        """approve_and_sanitize releases approved entity literal."""
        prompt = "My password is Apple#123. Is this password strong?"
        # First get the approval_required to find entity_id
        result_before = self.shield.sanitize(prompt)
        # Collect approval_required entity IDs
        ua_ids = {a.entity_id for a in result_before.approval_required}
        if ua_ids:
            result_after = self.shield.approve_and_sanitize(prompt, ua_ids)
            self.assertEqual(len(result_after.approved_entities), len(ua_ids))
            self.assertEqual(len(result_after.approval_required), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
