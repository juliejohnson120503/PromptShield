"""
Unit tests for Phase 3: Risk Assessment & Privacy Policy Engine.
Verifies that the PolicyEngine produces correct risk scores and
MASK / RETAIN / USER_APPROVAL decisions for all key entity scenarios.
"""

import unittest
from backend.core import PromptShieldCore
from backend.policy_engine import PolicyDecision, RiskLevel


class TestPolicyEngine(unittest.TestCase):

    def setUp(self):
        self.shield = PromptShieldCore(use_spacy=False)

    # -----------------------------------------------------------------------
    # Technical Credentials — always MASK regardless of anything else
    # -----------------------------------------------------------------------

    def test_api_key_always_masked(self):
        prompt = "Use this key: sk-live-abc1234567890def12345678 for the API call."
        report = self.shield.evaluate_policy(prompt)
        api_policies = [p for p in report.entity_policies
                        if "sk-" in p.entity_text or "API" in p.entity_type.value]
        self.assertTrue(len(api_policies) >= 1, "API key not detected")
        for p in api_policies:
            self.assertEqual(p.decision, PolicyDecision.MASK)
            self.assertIn(p.risk_level, (RiskLevel.CRITICAL, RiskLevel.HIGH))

    def test_password_always_masked(self):
        prompt = "Connect using password='SuperSecret!99' to the database."
        report = self.shield.evaluate_policy(prompt)
        pwd_policies = [p for p in report.entity_policies
                        if p.entity_type.value == "PASSWORD"]
        self.assertTrue(len(pwd_policies) >= 1, "Password not detected")
        for p in pwd_policies:
            self.assertEqual(p.decision, PolicyDecision.MASK)
            self.assertEqual(p.risk_level, RiskLevel.CRITICAL)

    def test_credit_card_always_masked(self):
        prompt = "Charge my card 4111-1111-1111-1111 for the renewal."
        report = self.shield.evaluate_policy(prompt)
        card_policies = [p for p in report.entity_policies
                         if p.entity_type.value == "CREDIT_CARD"]
        self.assertTrue(len(card_policies) >= 1, "Credit card not detected")
        for p in card_policies:
            self.assertEqual(p.decision, PolicyDecision.MASK)
            self.assertEqual(p.risk_level, RiskLevel.CRITICAL)

    # -----------------------------------------------------------------------
    # Personal PII — MASK for user-owned data
    # -----------------------------------------------------------------------

    def test_personal_email_masked(self):
        prompt = "My email is julie@example.com, can you help me reset my password?"
        report = self.shield.evaluate_policy(prompt)
        email_policies = [p for p in report.entity_policies
                          if p.entity_type.value == "EMAIL"]
        self.assertTrue(len(email_policies) >= 1, "Email not detected")
        for p in email_policies:
            self.assertEqual(p.decision, PolicyDecision.MASK)

    def test_personal_name_masked(self):
        """'My name is Julie' → Julie should be MASK (personal PII)."""
        prompt = "My name is Julie Johnson and I need help with my application."
        report = self.shield.evaluate_policy(prompt)
        person_policies = [p for p in report.entity_policies
                           if p.entity_type.value == "PERSON"
                           and "julie" in p.entity_text.lower()]
        self.assertTrue(len(person_policies) >= 1, "Julie not detected")
        for p in person_policies:
            self.assertEqual(p.decision, PolicyDecision.MASK)
            self.assertFalse(p.is_public_knowledge)

    def test_informal_name_iam_masked(self):
        """'i am adarsh ...' → adarsh should be MASK."""
        prompt = "i am adarsh and my mail is adarsh@gmail.com, write a mail."
        report = self.shield.evaluate_policy(prompt)
        person_policies = [p for p in report.entity_policies
                           if p.entity_type.value == "PERSON"
                           and "adarsh" in p.entity_text.lower()]
        self.assertTrue(len(person_policies) >= 1, "adarsh not detected as PERSON")
        for p in person_policies:
            self.assertEqual(p.decision, PolicyDecision.MASK)

    # -----------------------------------------------------------------------
    # Public Figures — RETAIN (not first-party, public knowledge)
    # -----------------------------------------------------------------------

    def test_public_figure_retained(self):
        """'Tell me about Elon Musk' → Elon Musk should be RETAIN."""
        prompt = "Tell me about Elon Musk and his work at SpaceX."
        report = self.shield.evaluate_policy(prompt)
        elon_policies = [p for p in report.entity_policies
                         if "elon musk" in p.entity_text.lower()]
        self.assertTrue(len(elon_policies) >= 1, "Elon Musk not detected")
        for p in elon_policies:
            self.assertEqual(p.decision, PolicyDecision.RETAIN)
            self.assertTrue(p.is_public_knowledge)

    def test_public_location_retained(self):
        """'What is the capital of France?' → France should be RETAIN."""
        prompt = "What is the capital of France?"
        report = self.shield.evaluate_policy(prompt)
        loc_policies = [p for p in report.entity_policies
                        if p.entity_type.value == "LOCATION"]
        self.assertTrue(len(loc_policies) >= 1, "France not detected as location")
        for p in loc_policies:
            self.assertEqual(p.decision, PolicyDecision.RETAIN)

    # -----------------------------------------------------------------------
    # Mixed prompt — Julie (MASK) vs Elon Musk (RETAIN)
    # -----------------------------------------------------------------------

    def test_mixed_personal_vs_public_decisions(self):
        """
        Core requirement: in a mixed prompt, personal names must be MASK
        while public figures must be RETAIN — both in the same call.
        """
        prompt = (
            "My name is Julie Johnson and my email is julie@example.com. "
            "Tell me about Elon Musk."
        )
        report = self.shield.evaluate_policy(prompt)
        policy_map = {p.entity_text.lower(): p.decision
                      for p in report.entity_policies}

        # Julie → MASK
        self.assertIn("julie johnson", policy_map)
        self.assertEqual(policy_map["julie johnson"], PolicyDecision.MASK)

        # julie@example.com → MASK
        self.assertIn("julie@example.com", policy_map)
        self.assertEqual(policy_map["julie@example.com"], PolicyDecision.MASK)

        # Elon Musk → RETAIN
        self.assertIn("elon musk", policy_map)
        self.assertEqual(policy_map["elon musk"], PolicyDecision.RETAIN)

    # -----------------------------------------------------------------------
    # Risk scoring sanity checks
    # -----------------------------------------------------------------------

    def test_api_key_risk_score_critical(self):
        prompt = "My API key is sk-live-abc1234567890def12345678."
        report = self.shield.evaluate_policy(prompt)
        api_p = next((p for p in report.entity_policies
                      if p.entity_type.value == "API_KEY"), None)
        self.assertIsNotNone(api_p)
        self.assertGreaterEqual(api_p.risk_score, 0.85)
        self.assertEqual(api_p.risk_level, RiskLevel.CRITICAL)

    def test_overall_risk_reflects_highest(self):
        """A prompt with an API key must report CRITICAL overall risk."""
        prompt = "Use key sk-live-abc1234567890def12345678 to call the endpoint."
        report = self.shield.evaluate_policy(prompt)
        self.assertEqual(report.overall_risk_level, RiskLevel.CRITICAL)

    def test_policy_report_summary_counts(self):
        """summary dict must correctly count mask/retain/approval entities."""
        prompt = (
            "My name is Julie, my email is julie@example.com. "
            "Tell me about Elon Musk."
        )
        report = self.shield.evaluate_policy(prompt)
        d = report.to_dict()
        total = d["summary"]["total"]
        mask  = d["summary"]["mask"]
        retain = d["summary"]["retain"]
        ua    = d["summary"]["user_approval"]
        self.assertEqual(total, mask + retain + ua)
        # At least Julie and her email must be masked
        self.assertGreaterEqual(mask, 2)
        # Elon must be retained
        self.assertGreaterEqual(retain, 1)


if __name__ == "__main__":
    unittest.main()
