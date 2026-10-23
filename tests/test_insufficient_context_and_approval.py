"""
Tests for Insufficient Context Detection and User Approval Criteria.
Verifies:
1. Prompts lacking context (e.g. 'rahul s7') trigger has_insufficient_context and USER_APPROVAL.
2. Prompts with full context do not trigger false positive context warnings.
3. User approval selective release via approved_entity_ids.
"""

import unittest
from backend.core import PromptShieldCore
from backend.policy_engine import PolicyDecision


class TestInsufficientContextAndUserApproval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shield = PromptShieldCore()

    def test_isolated_entity_triggers_insufficient_context_and_approval(self):
        """'rahul s7' is an isolated 2-word entity prompt with no task verbs."""
        prompt = "rahul s7"
        report = self.shield.evaluate_policy(prompt)

        # 1. Must flag has_insufficient_context
        self.assertTrue(report.has_insufficient_context, "Failed to flag 'rahul s7' as insufficient context")
        self.assertIsNotNone(report.context_warning)
        self.assertIn("rahul s7", report.context_warning)

        # 2. Must request USER_APPROVAL rather than blindly masking or retaining
        policies = [p for p in report.entity_policies if "rahul" in p.entity_text.lower()]
        self.assertGreaterEqual(len(policies), 1)
        for p in policies:
            self.assertEqual(
                p.decision,
                PolicyDecision.USER_APPROVAL,
                f"Expected USER_APPROVAL for isolated entity without context, got {p.decision}"
            )
            self.assertIn("Insufficient prompt context", p.reason)

    def test_contextual_prompt_does_not_trigger_insufficient_context(self):
        """'Draft an email to Rahul regarding project S7 updates.' has clear task verbs."""
        prompt = "Draft an email to Rahul regarding project S7 updates."
        report = self.shield.evaluate_policy(prompt)

        self.assertFalse(report.has_insufficient_context)
        self.assertIsNone(report.context_warning)

    def test_user_approval_selective_release(self):
        """
        When user explicitly approves 'rahul s7', it should appear as a literal
        in sanitized_prompt; if unapproved, it should be protected with a placeholder.
        """
        prompt = "rahul s7"
        eid = "PERSON:rahul s7"

        # 1. Unapproved -> protected as <PERSON_1>
        unapproved_res = self.shield.sanitize(prompt, approved_entity_ids=set())
        self.assertIn("<PERSON_", unapproved_res.sanitized_prompt)
        self.assertNotIn("rahul s7", unapproved_res.sanitized_prompt)
        self.assertEqual(len(unapproved_res.approval_required), 1)

        # 2. Approved -> released as literal 'rahul s7'
        approved_res = self.shield.sanitize(prompt, approved_entity_ids={eid})
        self.assertEqual(approved_res.sanitized_prompt, "rahul s7")
        self.assertEqual(len(approved_res.approved_entities), 1)


if __name__ == "__main__":
    unittest.main()
