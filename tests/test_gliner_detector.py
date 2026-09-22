"""
Unit tests for GlinerNERDetector (Deep Neural Open-Vocabulary NER).
"""

import unittest
from backend.gliner_detector import GlinerNERDetector
from backend.models import EntityType


class TestGlinerNERDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = GlinerNERDetector()

    def test_gliner_available(self):
        self.assertTrue(self.detector.is_available(), "GLiNER model should be loaded and available.")

    def test_gliner_empty_prompt(self):
        self.assertEqual(self.detector.detect(""), [])
        self.assertEqual(self.detector.detect("   "), [])

    def test_gliner_detect_person_lowercase(self):
        prompt = "i m julie and this is my message"
        entities = self.detector.detect(prompt)
        person_entities = [e for e in entities if e.entity_type == EntityType.PERSON]
        self.assertTrue(len(person_entities) >= 1)
        self.assertIn("julie", [e.text.lower() for e in person_entities])
        self.assertEqual(person_entities[0].source, "gliner")

    def test_gliner_detect_org_and_location(self):
        prompt = "I applied to OpenAI located in San Francisco."
        entities = self.detector.detect(prompt)
        ent_map = {e.text: e.entity_type for e in entities}
        self.assertIn("OpenAI", ent_map)
        self.assertEqual(ent_map["OpenAI"], EntityType.ORGANIZATION)
        self.assertIn("San Francisco", ent_map)
        self.assertEqual(ent_map["San Francisco"], EntityType.LOCATION)

    def test_gliner_excluded_keywords_not_entities(self):
        prompt = "psswrd is 87654321 and my email is test@test.com"
        entities = self.detector.detect(prompt)
        ent_texts = [e.text.lower() for e in entities]
        self.assertNotIn("psswrd", ent_texts)
        self.assertNotIn("email", ent_texts)


if __name__ == "__main__":
    unittest.main()
