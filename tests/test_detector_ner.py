"""
Unit tests for NER entities:
- PERSON
- ORGANIZATION
- LOCATION
- DATE
"""

import unittest
from backend.detector import SensitiveDataDetector
from backend.models import EntityType


class TestNERDetector(unittest.TestCase):

    def setUp(self):
        self.detector = SensitiveDataDetector(use_spacy=False)

    def test_person_intro_detection(self):
        prompt = "My name is John Mathew and I need help with my account."
        entities = self.detector.detect(prompt)
        persons = [e for e in entities if e.entity_type == EntityType.PERSON]
        self.assertEqual(len(persons), 1)
        self.assertEqual(persons[0].text, "John Mathew")
        self.assertEqual(persons[0].normalized_value, "John Mathew")

    def test_organization_detection(self):
        prompt = "I have an interview with ABC Technologies next Monday."
        entities = self.detector.detect(prompt)
        orgs = [e for e in entities if e.entity_type == EntityType.ORGANIZATION]
        self.assertTrue(len(orgs) >= 1)
        self.assertEqual(orgs[0].text, "ABC Technologies")

    def test_organization_gazetteer(self):
        prompt = "Does Google provide an API similar to Microsoft and OpenAI?"
        entities = self.detector.detect(prompt)
        orgs = [e.text for e in entities if e.entity_type == EntityType.ORGANIZATION]
        self.assertIn("Google", orgs)
        self.assertIn("Microsoft", orgs)
        self.assertIn("OpenAI", orgs)

    def test_location_detection(self):
        prompt = "I am currently living in London and traveling to Tokyo."
        entities = self.detector.detect(prompt)
        locs = [e.text for e in entities if e.entity_type == EntityType.LOCATION]
        self.assertIn("London", locs)
        self.assertIn("Tokyo", locs)

    def test_date_detection_iso(self):
        prompt = "The project report was finalized on 2025-10-24."
        entities = self.detector.detect(prompt)
        dates = [e for e in entities if e.entity_type == EntityType.DATE]
        self.assertEqual(len(dates), 1)
        self.assertEqual(dates[0].text, "2025-10-24")

    def test_date_detection_written(self):
        prompt = "Our symposium date is January 15, 2025 at 10 AM."
        entities = self.detector.detect(prompt)
        dates = [e for e in entities if e.entity_type == EntityType.DATE]
        self.assertEqual(len(dates), 1)
        self.assertEqual(dates[0].text, "January 15, 2025")


if __name__ == "__main__":
    unittest.main()
