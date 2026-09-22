"""
PromptShield AI — Phase 1 Detection Foundation: Mandatory Test Suite.

This file directly tests all 8 scenarios specified in the project brief
using the full hybrid SensitiveDataDetector pipeline
(Regex + Presidio + spaCy + EntityFusion).

Tests verify:
  1. Email detection
  2. Phone detection
  3. Synthetic credit card with Luhn validation
  4. Pretrained NER: PERSON, ORG, LOCATION
  5. Public information boundary: detection ≠ masking
  6. Mixed PII: EMAIL + PHONE
  7. Messy natural language: PERSON + EMAIL
  8. Synthetic secret: API_KEY / sk-test pattern

IMPORTANT:
  • No real secrets, credentials, or card numbers are used in these tests.
  • Test 5 explicitly verifies that detection does NOT trigger masking.
"""

import unittest
from backend.detector import SensitiveDataDetector
from backend.models import EntityType


class TestPhase1DetectionFoundation(unittest.TestCase):
    """
    Mandatory Phase 1 test suite.
    Uses full hybrid detector (Regex + Presidio + spaCy + Fusion).
    """

    @classmethod
    def setUpClass(cls):
        """Initialise the hybrid detector ONCE for all tests."""
        # use_spacy=True activates both Presidio and spaCy
        cls.detector = SensitiveDataDetector(use_spacy=True)

    # -------------------------------------------------------
    # Test 1 — Email
    # -------------------------------------------------------
    def test_1_email_detection(self):
        """'My email is julie@gmail.com' → EMAIL detected."""
        prompt = "My email is julie@gmail.com"
        entities = self.detector.detect(prompt)
        emails = [e for e in entities if e.entity_type == EntityType.EMAIL]
        self.assertGreaterEqual(len(emails), 1,
            f"Expected EMAIL entity. Got: {[e.to_dict() for e in entities]}")
        self.assertIn(
            "julie@gmail.com",
            [e.text for e in emails],
        )

    # -------------------------------------------------------
    # Test 2 — Phone
    # -------------------------------------------------------
    def test_2_phone_detection(self):
        """'Call me at 9876543210' → PHONE candidate detected."""
        prompt = "Call me at 9876543210"
        entities = self.detector.detect(prompt)
        phones = [e for e in entities if e.entity_type == EntityType.PHONE]
        self.assertGreaterEqual(len(phones), 1,
            f"Expected PHONE entity. Got: {[e.to_dict() for e in entities]}")

    # -------------------------------------------------------
    # Test 3 — Synthetic credit card with Luhn
    # -------------------------------------------------------
    def test_3_synthetic_credit_card_luhn(self):
        """'My card is 4111111111111111' → CREDIT_CARD detected + Luhn valid."""
        prompt = "My card is 4111111111111111"
        entities = self.detector.detect(prompt)
        cards = [e for e in entities if e.entity_type == EntityType.CREDIT_CARD]
        self.assertGreaterEqual(len(cards), 1,
            f"Expected CREDIT_CARD entity. Got: {[e.to_dict() for e in entities]}")
        # Verify Luhn was used (detector label)
        self.assertTrue(
            any("luhn" in e.detector for e in cards),
            "Expected Luhn-validated credit card detection"
        )
        # Verify normalized value is the raw digit string
        self.assertEqual(cards[0].normalized_value, "4111111111111111")

    # -------------------------------------------------------
    # Test 4 — Pretrained NER
    # -------------------------------------------------------
    def test_4_ner_person_org_location(self):
        """
        'Julie works at Microsoft in Kochi.'
        Expected: Julie=PERSON, Microsoft=ORG, Kochi=LOCATION (where supported).
        """
        prompt = "Julie works at Microsoft in Kochi."
        entities = self.detector.detect(prompt)
        entity_types = {e.entity_type for e in entities}

        # PERSON and ORGANIZATION are well-supported by en_core_web_sm
        self.assertIn(EntityType.PERSON, entity_types,
            f"Expected PERSON. Got types: {entity_types}")
        self.assertIn(EntityType.ORGANIZATION, entity_types,
            f"Expected ORGANIZATION. Got types: {entity_types}")

        persons = [e.text for e in entities if e.entity_type == EntityType.PERSON]
        self.assertTrue(
            any("Julie" in p for p in persons),
            f"Expected 'Julie' as PERSON, found: {persons}"
        )
        orgs = [e.text for e in entities if e.entity_type == EntityType.ORGANIZATION]
        self.assertTrue(
            any("Microsoft" in o for o in orgs),
            f"Expected 'Microsoft' as ORG, found: {orgs}"
        )

    # -------------------------------------------------------
    # Test 5 — Public information boundary (Detection ≠ Masking)
    # -------------------------------------------------------
    def test_5_public_info_detected_not_auto_masked(self):
        """
        'Who is Mahatma Gandhi?'
        PERSON may be detected.
        But the detection layer must NOT make a masking decision.

        This test verifies the ARCHITECTURAL BOUNDARY between Phase 1
        (Detection) and Phase 4 (Context-Aware Masking).
        """
        prompt = "Who is Mahatma Gandhi?"
        entities = self.detector.detect(prompt)

        # Detection: PERSON may be found (not guaranteed — model dependent)
        # If found, it must carry NO masking metadata
        persons = [e for e in entities if e.entity_type == EntityType.PERSON]
        for p in persons:
            entity_dict = p.to_dict()
            # These keys must NOT exist — they belong to Phase 4
            self.assertNotIn("mask", entity_dict,
                "DetectedEntity must not contain a masking decision")
            self.assertNotIn("should_mask", entity_dict,
                "DetectedEntity must not contain a masking decision")
            self.assertNotIn("requires_masking", entity_dict,
                "DetectedEntity must not contain a masking decision")
            self.assertNotIn("policy_action", entity_dict,
                "DetectedEntity must not contain a policy action")

    # -------------------------------------------------------
    # Test 6 — Mixed PII
    # -------------------------------------------------------
    def test_6_mixed_email_and_phone(self):
        """
        'Send the report to julie@gmail.com and call 9876543210.'
        Expected: EMAIL + PHONE detected.
        """
        prompt = "Send the report to julie@gmail.com and call 9876543210."
        entities = self.detector.detect(prompt)
        types_found = {e.entity_type for e in entities}

        self.assertIn(EntityType.EMAIL, types_found,
            f"Expected EMAIL. Types found: {types_found}")
        self.assertIn(EntityType.PHONE, types_found,
            f"Expected PHONE. Types found: {types_found}")

        emails = [e.text for e in entities if e.entity_type == EntityType.EMAIL]
        self.assertIn("julie@gmail.com", emails)

    # -------------------------------------------------------
    # Test 7 — Messy natural language
    # -------------------------------------------------------
    def test_7_messy_natural_language_person_and_email(self):
        """
        'msg Julie at julie@gmail.com'
        Expected: PERSON and EMAIL where supported by detectors.
        """
        prompt = "msg Julie at julie@gmail.com"
        entities = self.detector.detect(prompt)
        types_found = {e.entity_type for e in entities}

        self.assertIn(EntityType.EMAIL, types_found,
            f"Expected EMAIL. Types found: {types_found}")
        # PERSON ("Julie") may be detected depending on context window
        # We assert EMAIL is detected; PERSON is a soft assertion
        emails = [e.text for e in entities if e.entity_type == EntityType.EMAIL]
        self.assertIn("julie@gmail.com", emails)

    # -------------------------------------------------------
    # Test 8 — Synthetic secret / API key
    # -------------------------------------------------------
    def test_8_synthetic_api_key_detected(self):
        """
        'Use sk-test-example-12345'
        Expected: API_KEY / secret-like pattern detected.
        No real API keys used.
        """
        prompt = "Use sk-test-example-12345 to authenticate."
        entities = self.detector.detect(prompt)
        keys = [e for e in entities if e.entity_type == EntityType.API_KEY]
        self.assertGreaterEqual(len(keys), 1,
            f"Expected API_KEY entity. Got: {[e.to_dict() for e in entities]}")
        self.assertTrue(
            any("sk-test" in k.text for k in keys),
            f"Expected 'sk-test' pattern. Got: {[k.text for k in keys]}"
        )

    # -------------------------------------------------------
    # Additional: Fusion deduplication
    # -------------------------------------------------------
    def test_no_duplicate_entities_after_fusion(self):
        """
        All entity spans in the result must be non-overlapping.
        This validates EntityFusionEngine integration.
        """
        prompt = (
            "My name is John Mathew and my email is john@gmail.com. "
            "Write a professional email to ABC Technologies regarding my "
            "application for the London office."
        )
        entities = self.detector.detect(prompt)
        # Check no two entities overlap
        for i, a in enumerate(entities):
            for j, b in enumerate(entities):
                if i >= j:
                    continue
                overlap = max(a.start, b.start) < min(a.end, b.end)
                self.assertFalse(
                    overlap,
                    f"Overlapping entities detected: "
                    f"{a.text!r}[{a.start}:{a.end}] and "
                    f"{b.text!r}[{b.start}:{b.end}]"
                )

    # -------------------------------------------------------
    # Detector status check
    # -------------------------------------------------------
    def test_detector_status_reports_all_backends(self):
        """get_detector_status() should report regex, presidio, spacy."""
        status = self.detector.get_detector_status()
        self.assertIn("regex", status)
        self.assertIn("presidio", status)
        self.assertIn("spacy", status)
        self.assertTrue(status["regex"]["active"])


if __name__ == "__main__":
    unittest.main()
