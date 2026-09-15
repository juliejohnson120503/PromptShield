"""
Unit tests for BaselineMasker and MappingStore (Phase 1 technical foundation).
"""

import unittest
from backend.models import EntityType, DetectedEntity
from backend.baseline_masker import BaselineMasker
from backend.mapping_store import MappingStore


class TestBaselineMasker(unittest.TestCase):

    def setUp(self):
        self.store = MappingStore()
        self.masker = BaselineMasker(mapping_store=self.store)

    def test_single_entity_masking(self):
        prompt = "My email is john@example.com."
        entity = DetectedEntity(
            text="john@example.com",
            entity_type=EntityType.EMAIL,
            start=12,
            end=28,
            normalized_value="john@example.com",
        )
        res = self.masker.mask(prompt, [entity], session_id="test-session-1")

        self.assertEqual(res.sanitized_prompt, "My email is <EMAIL_1>.")
        self.assertEqual(res.mapping.get("<EMAIL_1>"), "john@example.com")
        self.assertEqual(self.store.get_raw_value("test-session-1", "<EMAIL_1>"), "john@example.com")

    def test_multiple_entities_different_types(self):
        prompt = "Contact John Mathew at john@example.com or 9876543210."
        entities = [
            DetectedEntity(
                text="John Mathew",
                entity_type=EntityType.PERSON,
                start=8,
                end=19,
                normalized_value="John Mathew",
            ),
            DetectedEntity(
                text="john@example.com",
                entity_type=EntityType.EMAIL,
                start=23,
                end=39,
                normalized_value="john@example.com",
            ),
            DetectedEntity(
                text="9876543210",
                entity_type=EntityType.PHONE,
                start=43,
                end=53,
                normalized_value="9876543210",
            ),
        ]
        res = self.masker.mask(prompt, entities, session_id="test-session-2")

        expected = "Contact <PERSON_1> at <EMAIL_1> or <PHONE_1>."
        self.assertEqual(res.sanitized_prompt, expected)
        self.assertEqual(len(res.mapping), 3)
        self.assertEqual(res.mapping["<PERSON_1>"], "John Mathew")
        self.assertEqual(res.mapping["<EMAIL_1>"], "john@example.com")
        self.assertEqual(res.mapping["<PHONE_1>"], "9876543210")

    def test_repeated_entity_reuses_placeholder(self):
        prompt = "Write to john@example.com because john@example.com is the manager."
        entities = [
            DetectedEntity(
                text="john@example.com",
                entity_type=EntityType.EMAIL,
                start=9,
                end=25,
                normalized_value="john@example.com",
            ),
            DetectedEntity(
                text="john@example.com",
                entity_type=EntityType.EMAIL,
                start=34,
                end=50,
                normalized_value="john@example.com",
            ),
        ]
        res = self.masker.mask(prompt, entities, session_id="test-session-3")
        expected = "Write to <EMAIL_1> because <EMAIL_1> is the manager."
        self.assertEqual(res.sanitized_prompt, expected)
        self.assertEqual(len(res.mapping), 1)

    def test_empty_prompt_and_no_entities(self):
        res1 = self.masker.mask("", [])
        self.assertEqual(res1.sanitized_prompt, "")
        self.assertEqual(len(res1.mapping), 0)

        res2 = self.masker.mask("What is the capital of India?", [])
        self.assertEqual(res2.sanitized_prompt, "What is the capital of India?")
        self.assertEqual(len(res2.mapping), 0)

    def test_clear_session(self):
        prompt = "Email: test@test.com"
        entity = DetectedEntity(
            text="test@test.com",
            entity_type=EntityType.EMAIL,
            start=7,
            end=20,
            normalized_value="test@test.com",
        )
        self.masker.mask(prompt, [entity], session_id="sess-to-clear")
        self.assertIsNotNone(self.store.get_raw_value("sess-to-clear", "<EMAIL_1>"))

        cleared = self.store.clear_session("sess-to-clear")
        self.assertTrue(cleared)
        self.assertIsNone(self.store.get_raw_value("sess-to-clear", "<EMAIL_1>"))


if __name__ == "__main__":
    unittest.main()
