"""
Unit and regression tests for Phase 2 Context Understanding (Value-Required Analysis)
and Phase 3 Policy Logic (User Approval on Literal Value Requirement).

Verifies:
1. False-positive elimination ("strong?", "stronger" not detected as PASSWORD).
2. Operation detection (VALUE_ANALYSIS, COMPARISON, CALCULATION, VALIDATION, etc.).
3. Entity-task relationship determination (TARGET_OF_ANALYSIS, COMPARISON_INPUT, etc.).
4. value_required determination and safety fallbacks (prefer privacy when uncertain).
5. PolicyEngine routing to USER_APPROVAL when value_required is True while preserving raw risk scores.
"""

import unittest
from backend.core import PromptShieldCore
from backend.models import (
    EntityType,
    EntityTaskRelation,
    OperationType,
)
from backend.policy_engine import PolicyDecision, RiskLevel
from backend.regex_detector import RegexDetector


class TestValueRequiredContext(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.shield = PromptShieldCore()
        cls.regex_detector = RegexDetector()

    # -----------------------------------------------------------------------
    # STEP 1: False positive regression tests
    # -----------------------------------------------------------------------

    def test_strong_question_mark_not_detected_as_password(self):
        """Verify 'strong?' is never detected as a PASSWORD entity."""
        prompt = "My password is Apple#123. Is this password strong?"
        regex_matches = self.regex_detector._detect_passwords(prompt)
        detected_texts = [m.text for m in regex_matches]
        self.assertNotIn("strong?", detected_texts)
        self.assertNotIn("strong", detected_texts)

        report = self.shield.evaluate_policy(prompt)
        pw_entities = [p for p in report.entity_policies if p.entity_type == EntityType.PASSWORD]
        pw_texts = [p.entity_text for p in pw_entities]
        self.assertIn("Apple#123", pw_texts)
        self.assertNotIn("strong?", pw_texts)
        self.assertNotIn("strong", pw_texts)

    def test_stronger_not_detected_as_password(self):
        """Verify 'stronger' in comparison prompts is not captured as a PASSWORD."""
        prompt = "Compare Apple#123 and Mango#987 and tell me which password is stronger."
        regex_matches = self.regex_detector._detect_passwords(prompt)
        detected_texts = [m.text for m in regex_matches]
        self.assertNotIn("stronger", detected_texts)

        report = self.shield.evaluate_policy(prompt)
        pw_entities = [p for p in report.entity_policies if p.entity_type == EntityType.PASSWORD]
        pw_texts = [p.entity_text for p in pw_entities]
        self.assertNotIn("stronger", pw_texts)

    # -----------------------------------------------------------------------
    # 11 Context Scenarios
    # -----------------------------------------------------------------------

    def test_scenario_1_unrelated_qa(self):
        """Scenario 1: 'My password is Apple#123. Explain cloud computing.' -> MASK"""
        prompt = "My password is Apple#123. Explain cloud computing."
        report = self.shield.evaluate_policy(prompt)
        pw = next(p for p in report.entity_policies if p.entity_text == "Apple#123")
        self.assertEqual(pw.decision, PolicyDecision.MASK)
        self.assertFalse(pw.value_required)
        self.assertEqual(pw.risk_level, RiskLevel.CRITICAL)

    def test_scenario_2_password_reset_email(self):
        """Scenario 2: 'My password is Apple#123. Write a password-reset email.' -> MASK"""
        prompt = "My password is Apple#123. Write a password-reset email."
        report = self.shield.evaluate_policy(prompt)
        pw = next(p for p in report.entity_policies if p.entity_text == "Apple#123")
        self.assertEqual(pw.decision, PolicyDecision.MASK)
        self.assertFalse(pw.value_required)
        self.assertEqual(pw.risk_level, RiskLevel.CRITICAL)

    def test_scenario_3_password_strength_analysis(self):
        """Scenario 3: 'My password is Apple#123. Is this password strong?' -> USER_APPROVAL"""
        prompt = "My password is Apple#123. Is this password strong?"
        report = self.shield.evaluate_policy(prompt)
        pw = next(p for p in report.entity_policies if p.entity_text == "Apple#123")
        self.assertEqual(pw.decision, PolicyDecision.USER_APPROVAL)
        self.assertTrue(pw.value_required)
        self.assertEqual(pw.entity_task_relation, EntityTaskRelation.TARGET_OF_ANALYSIS.value)
        # CRITICAL risk score must remain intact
        self.assertEqual(pw.risk_level, RiskLevel.CRITICAL)
        self.assertEqual(pw.risk_score, 1.0)

    def test_scenario_4_password_digit_count(self):
        """Scenario 4: 'How many digits are in my password Apple#123?' -> USER_APPROVAL"""
        prompt = "How many digits are in my password Apple#123?"
        report = self.shield.evaluate_policy(prompt)
        pw = next(p for p in report.entity_policies if p.entity_text == "Apple#123")
        self.assertEqual(pw.decision, PolicyDecision.USER_APPROVAL)
        self.assertTrue(pw.value_required)
        self.assertIn(
            pw.entity_task_relation,
            (EntityTaskRelation.TARGET_OF_ANALYSIS.value, EntityTaskRelation.CALCULATION_INPUT.value),
        )
        self.assertEqual(pw.risk_level, RiskLevel.CRITICAL)

    def test_scenario_5_password_comparison(self):
        """Scenario 5: 'Compare Apple#123 and Mango#987 and tell me which password is stronger.' -> USER_APPROVAL for both"""
        prompt = "Compare Apple#123 and Mango#987 and tell me which password is stronger."
        report = self.shield.evaluate_policy(prompt)
        apple = next((p for p in report.entity_policies if "Apple#123" in p.entity_text), None)
        mango = next((p for p in report.entity_policies if "Mango#987" in p.entity_text), None)
        self.assertIsNotNone(apple, "Apple#123 was not detected")
        self.assertIsNotNone(mango, "Mango#987 was not detected")

        self.assertEqual(apple.decision, PolicyDecision.USER_APPROVAL)
        self.assertTrue(apple.value_required)
        self.assertEqual(apple.entity_task_relation, EntityTaskRelation.COMPARISON_INPUT.value)
        self.assertEqual(apple.risk_level, RiskLevel.CRITICAL)

        self.assertEqual(mango.decision, PolicyDecision.USER_APPROVAL)
        self.assertTrue(mango.value_required)
        self.assertEqual(mango.entity_task_relation, EntityTaskRelation.COMPARISON_INPUT.value)
        self.assertEqual(mango.risk_level, RiskLevel.CRITICAL)

    def test_scenario_6_unrelated_comparison(self):
        """Scenario 6: 'My password is Apple#123. Compare Java and Python.' -> MASK"""
        prompt = "My password is Apple#123. Compare Java and Python."
        report = self.shield.evaluate_policy(prompt)
        pw = next(p for p in report.entity_policies if p.entity_text == "Apple#123")
        self.assertEqual(pw.decision, PolicyDecision.MASK)
        self.assertFalse(pw.value_required)
        self.assertEqual(pw.risk_level, RiskLevel.CRITICAL)

    def test_scenario_7_rewrite_password(self):
        """Scenario 7: 'Rewrite: My password is Apple#123.' -> MASK"""
        prompt = "Rewrite: My password is Apple#123."
        report = self.shield.evaluate_policy(prompt)
        pw = next(p for p in report.entity_policies if p.entity_text == "Apple#123")
        self.assertEqual(pw.decision, PolicyDecision.MASK)
        self.assertFalse(pw.value_required)
        self.assertEqual(pw.risk_level, RiskLevel.CRITICAL)

    def test_scenario_8_dob_calculate_age(self):
        """Scenario 8: 'My DOB is 17 September 1994. Calculate my age.' -> USER_APPROVAL"""
        prompt = "My DOB is 17 September 1994. Calculate my age."
        report = self.shield.evaluate_policy(prompt)
        dob = next(p for p in report.entity_policies if "17 September 1994" in p.entity_text)
        self.assertEqual(dob.decision, PolicyDecision.USER_APPROVAL)
        self.assertTrue(dob.value_required)
        self.assertEqual(dob.entity_task_relation, EntityTaskRelation.CALCULATION_INPUT.value)

    def test_scenario_9_dob_leave_letter(self):
        """Scenario 9: 'My DOB is 17 September 1994. Write a leave letter.' -> MASK"""
        prompt = "My DOB is 17 September 1994. Write a leave letter."
        report = self.shield.evaluate_policy(prompt)
        dob = next(p for p in report.entity_policies if "17 September 1994" in p.entity_text)
        self.assertEqual(dob.decision, PolicyDecision.MASK)
        self.assertFalse(dob.value_required)

    def test_scenario_10_ip_private_check(self):
        """Scenario 10: 'My IP is 192.168.1.25. Is this a private IPv4 address?' -> USER_APPROVAL"""
        prompt = "My IP is 192.168.1.25. Is this a private IPv4 address?"
        report = self.shield.evaluate_policy(prompt)
        ip = next(p for p in report.entity_policies if p.entity_text == "192.168.1.25")
        self.assertEqual(ip.decision, PolicyDecision.USER_APPROVAL)
        self.assertTrue(ip.value_required)
        self.assertIn(
            ip.entity_task_relation,
            (EntityTaskRelation.TARGET_OF_ANALYSIS.value, EntityTaskRelation.VALIDATION_INPUT.value),
        )

    def test_scenario_11_ip_rewrite(self):
        """Scenario 11: 'My IP is 192.168.1.25. Rewrite this sentence professionally.' -> MASK"""
        prompt = "My IP is 192.168.1.25. Rewrite this sentence professionally."
        report = self.shield.evaluate_policy(prompt)
        ip = next(p for p in report.entity_policies if p.entity_text == "192.168.1.25")
        self.assertEqual(ip.decision, PolicyDecision.MASK)
        self.assertFalse(ip.value_required)


if __name__ == "__main__":
    unittest.main()
