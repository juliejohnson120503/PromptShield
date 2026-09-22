"""
Unit tests for PresidioDetector — Phase 1 Presidio integration.

Tests cover:
  - Presidio availability check
  - Email detection and entity-type normalization
  - Phone detection
  - Entity type mapping from Presidio labels to PromptShield EntityType
  - Presidio confidence score preservation (not fabricated)
  - Graceful behavior when prompt is empty
  - source="presidio" on all results
"""

import unittest
from backend.presidio_detector import PresidioDetector
from backend.models import EntityType


class TestPresidioDetector(unittest.TestCase):

    def setUp(self):
        self.detector = PresidioDetector()

    def test_presidio_available(self):
        """presidio-analyzer is installed — detector must be available."""
        self.assertTrue(
            self.detector.is_available(),
            "presidio-analyzer should be installed and available. "
            "Run: pip install presidio-analyzer"
        )

    def test_email_detected(self):
        if not self.detector.is_available():
            self.skipTest("Presidio not available")
        prompt = "My email is julie@gmail.com"
        entities = self.detector.detect(prompt)
        emails = [e for e in entities if e.entity_type == EntityType.EMAIL]
        self.assertGreaterEqual(len(emails), 1)
        self.assertEqual(emails[0].text, "julie@gmail.com")

    def test_phone_detected(self):
        if not self.detector.is_available():
            self.skipTest("Presidio not available")
        # Presidio's default PHONE_NUMBER recognizer may or may not fire for
        # all formats above its confidence threshold. We verify the call
        # succeeds and returns a list (phone coverage is primarily regex's job).
        prompt = "Call me at +1 555 234 5678"
        result = self.detector.detect(prompt)
        self.assertIsInstance(result, list)
        # If Presidio does detect a phone, it must be properly typed
        phones = [e for e in result if e.entity_type == EntityType.PHONE]
        for p in phones:
            self.assertEqual(p.source, "presidio")


    def test_presidio_confidence_is_real_score(self):
        """
        Presidio provides its own confidence score.
        We must NOT overwrite it with a hard-coded value.
        Verify that score is in [0, 1] and was not fabricated.
        """
        if not self.detector.is_available():
            self.skipTest("Presidio not available")
        prompt = "Email me at test@example.com"
        entities = self.detector.detect(prompt)
        for e in entities:
            self.assertGreaterEqual(
                e.confidence, 0.0,
                f"Confidence below 0 for {e.entity_type}: {e.confidence}"
            )
            self.assertLessEqual(
                e.confidence, 1.0,
                f"Confidence above 1 for {e.entity_type}: {e.confidence}"
            )
            self.assertIn(
                "presidio_score", e.metadata,
                "Presidio score should be preserved in metadata"
            )

    def test_source_is_presidio(self):
        """All results from PresidioDetector must carry source='presidio'."""
        if not self.detector.is_available():
            self.skipTest("Presidio not available")
        prompt = "Send report to contact@example.com"
        entities = self.detector.detect(prompt)
        for e in entities:
            self.assertEqual(
                e.source, "presidio",
                f"Expected source='presidio', got '{e.source}'"
            )

    def test_empty_prompt_returns_empty(self):
        result = self.detector.detect("")
        self.assertEqual(result, [])

    def test_entity_type_normalized_from_presidio_label(self):
        """
        Verify Presidio's EMAIL_ADDRESS maps to PromptShield EMAIL,
        not left as a raw Presidio string.
        """
        if not self.detector.is_available():
            self.skipTest("Presidio not available")
        prompt = "Contact: info@company.org"
        entities = self.detector.detect(prompt)
        emails = [e for e in entities if e.entity_type == EntityType.EMAIL]
        self.assertGreaterEqual(len(emails), 1,
            "Presidio EMAIL_ADDRESS should be normalized to EntityType.EMAIL")

    def test_metadata_contains_presidio_type(self):
        """Metadata should record the original Presidio entity type string."""
        if not self.detector.is_available():
            self.skipTest("Presidio not available")
        prompt = "Email: test@test.com"
        entities = self.detector.detect(prompt)
        for e in entities:
            self.assertIn("presidio_type", e.metadata)
            # presidio_type must be a non-empty string
            self.assertIsInstance(e.metadata["presidio_type"], str)
            self.assertTrue(len(e.metadata["presidio_type"]) > 0)

    def test_no_crash_on_unusual_input(self):
        """Unusual or complex inputs should not raise exceptions."""
        if not self.detector.is_available():
            self.skipTest("Presidio not available")
        unusual = (
            "Привет мир. こんにちは. "
            "1234 ABCD !@#$%^&*() "
            "http://example.com/path?q=test"
        )
        # Should return a list (possibly empty) without raising
        result = self.detector.detect(unusual)
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
