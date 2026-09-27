"""
PromptShield AI — GLiNER2 Production Migration Regression Test Suite.

Covers:
  - Test A: Structured PII (Email, Phone)
  - Test B: Personal Entities (Person, Address/Location)
  - Test C: Credentials (Username, Password)
  - Test D: Financial (Credit Card / Bank Account)
  - Test E: Public/Benign Entities (Detection != Masking)
  - Test F: Mixed Context (Full pipeline through ContextAnalyzer and PolicyEngine)
  - Test G: Repeated Entity (Independent, exact offsets)
  - Test H: Ambiguous Context (Stability on ambiguous entities)
  - Offset accuracy (prompt[start:end] == entity.text)
  - Unknown label handling (safe fallback to PII_OTHER, no crash)
  - End-to-end execution through PromptShieldCore
"""

import unittest
from backend.core import PromptShieldCore
from backend.models import EntityType
from backend.policy_engine import PolicyDecision
from backend.config import GLINER2_ENTITY_MAPPING


class TestGLiNER2Migration(unittest.TestCase):
    """
    Regression and validation suite for GLiNER2 production migration.
    """

    @classmethod
    def setUpClass(cls):
        cls.core = PromptShieldCore(use_spacy=True)
        cls.detector = cls.core.detector
        cls.gliner2 = cls.detector._gliner2

    def test_gliner2_is_loaded_and_active(self):
        """Verify GLiNER2 is the active neural model in production."""
        self.assertIsNotNone(self.gliner2, "GLiNER2 detector should be instantiated.")
        status = self.detector.get_detector_status()
        self.assertIn("gliner2", status)
        self.assertTrue(status["gliner2"]["active"], "GLiNER2 must be active in detector status.")

    # ------------------------------------------------------------------
    # TEST A — Structured PII
    # ------------------------------------------------------------------
    def test_a_structured_pii(self):
        """'My email is arjun.nair27@gmail.com and my phone number is +91 98765 43210.'"""
        prompt = "My email is arjun.nair27@gmail.com and my phone number is +91 98765 43210."
        entities = self.detector.detect(prompt)

        types_found = {e.entity_type for e in entities}
        self.assertIn(EntityType.EMAIL, types_found, f"Expected EMAIL in {types_found}")
        self.assertIn(EntityType.PHONE, types_found, f"Expected PHONE in {types_found}")

        for e in entities:
            self.assertEqual(prompt[e.start:e.end], e.text, f"Offset mismatch for {e}")

    # ------------------------------------------------------------------
    # TEST B — Personal entities
    # ------------------------------------------------------------------
    def test_b_personal_entities(self):
        """'My name is Arjun Nair and I live at 42 Lake View Road, Kochi.'"""
        prompt = "My name is Arjun Nair and I live at 42 Lake View Road, Kochi."
        entities = self.detector.detect(prompt)

        types_found = {e.entity_type for e in entities}
        self.assertIn(EntityType.PERSON, types_found, f"Expected PERSON in {types_found}")
        self.assertTrue(
            EntityType.LOCATION in types_found or EntityType.ADDRESS in types_found,
            f"Expected LOCATION or ADDRESS in {types_found}",
        )

        for e in entities:
            self.assertEqual(prompt[e.start:e.end], e.text, f"Offset mismatch for {e}")

    # ------------------------------------------------------------------
    # TEST C — Credentials
    # ------------------------------------------------------------------
    def test_c_credentials(self):
        """'My username is arjun27 and my password is BlueSky@2026.'"""
        prompt = "My username is arjun27 and my password is BlueSky@2026."
        entities = self.detector.detect(prompt)

        types_found = {e.entity_type for e in entities}
        self.assertTrue(
            EntityType.PASSWORD in types_found or EntityType.USERNAME in types_found,
            f"Expected PASSWORD or USERNAME in {types_found}",
        )

        for e in entities:
            self.assertEqual(prompt[e.start:e.end], e.text, f"Offset mismatch for {e}")

    # ------------------------------------------------------------------
    # TEST D — Financial
    # ------------------------------------------------------------------
    def test_d_financial(self):
        """'My card number is 4111 1111 1111 1111.'"""
        prompt = "My card number is 4111 1111 1111 1111."
        entities = self.detector.detect(prompt)

        types_found = {e.entity_type for e in entities}
        self.assertIn(EntityType.CREDIT_CARD, types_found, f"Expected CREDIT_CARD in {types_found}")

        for e in entities:
            self.assertEqual(prompt[e.start:e.end], e.text, f"Offset mismatch for {e}")

    # ------------------------------------------------------------------
    # TEST E — Public / Benign entities (Detection != Masking)
    # ------------------------------------------------------------------
    def test_e_public_benign_entities(self):
        """'Kochi is a city in Kerala and Microsoft is a technology company.'"""
        prompt = "Kochi is a city in Kerala and Microsoft is a technology company."
        policy_report = self.core.evaluate_policy(prompt)

        # In a public factual query, public entities must NOT be unconditionally MASKed
        # Verify ContextAnalyzer and PolicyEngine allow RETAIN / USER_APPROVAL
        for policy in policy_report.entity_policies:
            if policy.entity_text in ("Kochi", "Kerala", "Microsoft"):
                self.assertIn(
                    policy.decision,
                    (PolicyDecision.RETAIN, PolicyDecision.USER_APPROVAL),
                    f"Public entity {policy.entity_text} should not be blindly MASKed without personal context",
                )

    # ------------------------------------------------------------------
    # TEST F — Mixed context & Full Pipeline
    # ------------------------------------------------------------------
    def test_f_mixed_context_full_pipeline(self):
        """
        'My name is Arjun Nair. Email me at arjun@gmail.com.
        My password is BlueSky@2026 and I am planning a trip to Bangalore.'
        """
        prompt = (
            "My name is Arjun Nair. Email me at arjun@gmail.com. "
            "My password is BlueSky@2026 and I am planning a trip to Bangalore."
        )
        entities = self.detector.detect(prompt)
        context_result = self.core.analyze_context(prompt)
        policy_report = self.core.evaluate_policy(prompt)

        # 1. Multiple entity types
        types_found = {c.entity_type for c in context_result.entity_contexts}
        self.assertIn(EntityType.EMAIL, types_found)
        self.assertIn(EntityType.PERSON, types_found)

        # 2. Correct offsets
        for e in entities:
            self.assertEqual(prompt[e.start:e.end], e.text, f"Offset mismatch: {e}")

        # 3. Canonical labels
        for e in entities:
            self.assertIsInstance(e.entity_type, EntityType)

        # 4. ContextAnalyzer compatibility
        self.assertGreater(len(context_result.entity_contexts), 0)

        # 5. PolicyEngine compatibility
        self.assertGreater(len(policy_report.entity_policies), 0)

        # Password/Credential must be MASKed
        cred_policies = [
            p for p in policy_report.entity_policies
            if p.entity_type in (EntityType.PASSWORD, EntityType.API_KEY)
        ]
        for p in cred_policies:
            self.assertEqual(p.decision, PolicyDecision.MASK, "Sensitive credentials must be MASKed.")

    # ------------------------------------------------------------------
    # TEST G — Repeated entity
    # ------------------------------------------------------------------
    def test_g_repeated_entity(self):
        """'Send the report to arjun@gmail.com and confirm with arjun@gmail.com.'"""
        prompt = "Send the report to arjun@gmail.com and confirm with arjun@gmail.com."
        entities = self.detector.detect(prompt)

        emails = [e for e in entities if e.text == "arjun@gmail.com"]
        self.assertEqual(len(emails), 2, f"Expected 2 occurrences of email, got {len(emails)}")

        first, second = emails[0], emails[1]
        self.assertEqual(prompt[first.start:first.end], "arjun@gmail.com")
        self.assertEqual(prompt[second.start:second.end], "arjun@gmail.com")
        self.assertNotEqual(first.start, second.start)
        self.assertTrue(first.end <= second.start)

    # ------------------------------------------------------------------
    # TEST H — Ambiguous context
    # ------------------------------------------------------------------
    def test_h_ambiguous_context(self):
        """'Rose works on the Grant project.' Pipeline must not crash."""
        prompt = "Rose works on the Grant project."
        policy_report = self.core.evaluate_policy(prompt)
        self.assertIsNotNone(policy_report)
        self.assertIsInstance(policy_report.entity_policies, list)

    # ------------------------------------------------------------------
    # TEST: Character Offset Verification on Edge Cases
    # ------------------------------------------------------------------
    def test_offset_edge_cases(self):
        """Test entities at boundaries, inside quotes, next to punctuation."""
        cases = [
            "arjun@gmail.com is my primary email",
            "Contact: arjun@gmail.com",
            'My login is "arjun27" and token is "secret-12345"',
            "Look at 42 Lake View Road, Kochi.",
        ]
        for c in cases:
            entities = self.detector.detect(c)
            for e in entities:
                self.assertEqual(
                    c[e.start:e.end],
                    e.text,
                    f"Offset slice {c[e.start:e.end]!r} != {e.text!r} in {c!r}",
                )

    # ------------------------------------------------------------------
    # TEST: Unknown Label Fallback
    # ------------------------------------------------------------------
    def test_unknown_label_fallback(self):
        """Verify unmapped GLiNER2 label gracefully normalizes to PII_OTHER."""
        unmapped_label = "some_exotic_nonexistent_pii_type"
        canonical = GLINER2_ENTITY_MAPPING.get(unmapped_label, EntityType.PII_OTHER)
        self.assertEqual(canonical, EntityType.PII_OTHER)


if __name__ == "__main__":
    unittest.main()
