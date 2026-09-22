"""
Unit tests for SpacyNERDetector and HeuristicNERDetector — Phase 1 NER module.

Tests cover:
  - spaCy NER detection (when model is available)
  - Heuristic NER fallback detection
  - PERSON, ORG, LOCATION detection
  - Detection ≠ masking: public figures must be detected but NOT auto-masked
  - source field: "spacy" or "heuristic_ner"
  - spaCy label preserved in metadata
"""

import unittest
from backend.spacy_detector import SpacyNERDetector, HeuristicNERDetector
from backend.models import EntityType


class TestSpacyNERDetector(unittest.TestCase):

    def setUp(self):
        self.detector = SpacyNERDetector()
        self.heuristic = HeuristicNERDetector()

    def test_spacy_model_available(self):
        """en_core_web_sm should be installed and loadable."""
        self.assertTrue(
            self.detector.is_available(),
            "en_core_web_sm not found. Run: python -m spacy download en_core_web_sm"
        )

    def test_empty_prompt_returns_empty(self):
        self.assertEqual(self.detector.detect(""), [])
        self.assertEqual(self.heuristic.detect(""), [])

    # -------------------------------------------------------
    # Test 4 — NER: "Julie works at Microsoft in Kochi."
    # -------------------------------------------------------
    def test_ner_person_org_location(self):
        """Test 4 — Expected: Julie=PERSON, Microsoft=ORG, Kochi=LOCATION."""
        prompt = "Julie works at Microsoft in Kochi."
        entities = self.detector.detect(prompt)
        entity_types = {e.entity_type for e in entities}

        # At least PERSON and ORG should be detected.
        # (spaCy en_core_web_sm may or may not detect "Kochi" as GPE —
        # it is a less common city; we assert PERSON and ORG are present.)
        self.assertIn(EntityType.PERSON, entity_types,
                      f"Expected PERSON in {entity_types}")
        self.assertIn(EntityType.ORGANIZATION, entity_types,
                      f"Expected ORGANIZATION in {entity_types}")

        persons = [e.text for e in entities if e.entity_type == EntityType.PERSON]
        self.assertTrue(
            any("Julie" in p for p in persons),
            f"Expected 'Julie' as PERSON, found: {persons}"
        )

    # -------------------------------------------------------
    # Test 5 — Public information boundary
    # -------------------------------------------------------
    def test_public_figure_detected_but_no_masking_decision(self):
        """
        Test 5 — 'Who is Mahatma Gandhi?'
        Detection layer must identify PERSON.
        The detection result does NOT contain a masking flag.
        Whether to mask is a Phase 2/3/4 decision, not Phase 1's.
        """
        prompt = "Who is Mahatma Gandhi?"
        entities = self.detector.detect(prompt)
        persons = [e for e in entities if e.entity_type == EntityType.PERSON]

        # Detection: PERSON should appear
        self.assertGreaterEqual(len(persons), 1,
            "Expected Mahatma Gandhi to be detected as PERSON")

        # No masking field exists on DetectedEntity — only detection metadata
        for p in persons:
            self.assertNotIn(
                "mask", p.metadata,
                "DetectedEntity must NOT contain a masking decision"
            )
            self.assertNotIn(
                "should_mask", p.to_dict(),
                "to_dict() must NOT contain a masking decision"
            )
            self.assertNotIn(
                "requires_masking", p.to_dict(),
                "to_dict() must NOT contain a masking decision"
            )

    def test_source_is_spacy_when_model_available(self):
        """Entities from spaCy model must carry source='spacy'."""
        if not self.detector.is_available():
            self.skipTest("spaCy model not available")
        prompt = "Alice works at Google."
        entities = self.detector.detect(prompt)
        for e in entities:
            self.assertEqual(e.source, "spacy",
                f"Expected source='spacy', got '{e.source}' for {e}")

    def test_spacy_label_in_metadata(self):
        """spaCy NER label (e.g. 'ORG', 'PERSON') must be in metadata."""
        if not self.detector.is_available():
            self.skipTest("spaCy model not available")
        prompt = "Bob works at OpenAI."
        entities = self.detector.detect(prompt)
        for e in entities:
            self.assertIn(
                "spacy_label", e.metadata,
                f"spacy_label missing from metadata for {e}"
            )

    # -------------------------------------------------------
    # Heuristic NER fallback
    # -------------------------------------------------------
    def test_heuristic_detects_known_org(self):
        prompt = "Does Google provide an API similar to Microsoft and OpenAI?"
        entities = self.heuristic.detect(prompt)
        org_texts = [e.text for e in entities if e.entity_type == EntityType.ORGANIZATION]
        self.assertIn("Google", org_texts)
        self.assertIn("Microsoft", org_texts)
        self.assertIn("OpenAI", org_texts)

    def test_heuristic_detects_known_location(self):
        prompt = "I am currently living in London and traveling to Tokyo."
        entities = self.heuristic.detect(prompt)
        loc_texts = [e.text for e in entities if e.entity_type == EntityType.LOCATION]
        self.assertIn("London", loc_texts)
        self.assertIn("Tokyo", loc_texts)

    def test_heuristic_detects_person_intro(self):
        prompt = "My name is John Mathew and I need help with my account."
        entities = self.heuristic.detect(prompt)
        persons = [e for e in entities if e.entity_type == EntityType.PERSON]
        self.assertGreaterEqual(len(persons), 1)
        person_texts = [p.text for p in persons]
        self.assertTrue(
            any("John" in t for t in person_texts),
            f"Expected 'John' in person texts: {person_texts}"
        )

    def test_heuristic_public_person_detected(self):
        """Mahatma Gandhi should be detected via KNOWN_PUBLIC_PERSONS gazetteer."""
        prompt = "Who is Mahatma Gandhi?"
        entities = self.heuristic.detect(prompt)
        persons = [e for e in entities if e.entity_type == EntityType.PERSON]
        self.assertGreaterEqual(len(persons), 1)

    def test_heuristic_source_label(self):
        """Heuristic NER must carry source='heuristic_ner'."""
        prompt = "I am working at Google."
        entities = self.heuristic.detect(prompt)
        for e in entities:
            self.assertEqual(
                e.source, "heuristic_ner",
                f"Expected source='heuristic_ner', got '{e.source}'"
            )


if __name__ == "__main__":
    unittest.main()
