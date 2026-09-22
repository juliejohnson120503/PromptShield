"""
Unit tests for PromptShield AI Hybrid Detection: Abbreviations & Shortforms.

Verifies:
1. Informal self-introductions with abbreviations/shortforms:
   - "i m julie", "im amina", "i am adarsh", "myself rahul", "my names david"
2. Password and credential abbreviations:
   - "psswrd is 98765432", "my pswd: Secret123", "pwd is ...", "pin is 4321"
3. Action verbs following names are not swallowed into the name:
   - "i'm amina write a formal letter to elon musk" -> "amina" (not "amina write")
4. 8-digit numbers preceded by password keywords are detected as PASSWORD, not PHONE.
5. Works across both neural mode (use_spacy=True) and pure rule mode (use_spacy=False).
"""

import unittest
from backend.core import PromptShieldCore
from backend.models import EntityType
from backend.policy_engine import PolicyDecision


class TestHybridAbbreviations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.shield_neural = PromptShieldCore(use_spacy=True)
        cls.shield_pure = PromptShieldCore(use_spacy=False)

    def test_user_reported_case_neural(self):
        prompt = "i m julie and my psswrd is 98765432"
        entities = self.shield_neural.analyze(prompt)
        ent_map = {e.text: e.entity_type for e in entities}

        # julie should be detected as PERSON
        self.assertIn("julie", ent_map, f"Expected 'julie' in {ent_map}")
        self.assertEqual(ent_map["julie"], EntityType.PERSON)

        # 98765432 should be detected as PASSWORD, NOT PHONE
        self.assertIn("98765432", ent_map, f"Expected '98765432' in {ent_map}")
        self.assertEqual(ent_map["98765432"], EntityType.PASSWORD)

        # Policy evaluation should flag both for MASK
        report = self.shield_neural.evaluate_policy(prompt)
        decisions = {p.entity_text: p.decision for p in report.entity_policies}
        self.assertEqual(decisions.get("julie"), PolicyDecision.MASK)
        self.assertEqual(decisions.get("98765432"), PolicyDecision.MASK)

    def test_user_reported_case_pure_regex_mode(self):
        prompt = "i m julie and my psswrd is 98765432"
        entities = self.shield_pure.analyze(prompt)
        ent_map = {e.text: e.entity_type for e in entities}

        self.assertIn("julie", ent_map, f"Expected 'julie' in pure mode: {ent_map}")
        self.assertEqual(ent_map["julie"], EntityType.PERSON)

        self.assertIn("98765432", ent_map, f"Expected '98765432' in pure mode: {ent_map}")
        self.assertEqual(ent_map["98765432"], EntityType.PASSWORD)

    def test_informal_name_with_trailing_verb(self):
        prompt = "i'm amina write a formal letter to elon musk"
        entities = self.shield_neural.analyze(prompt)
        ent_texts = [e.text for e in entities]

        # 'amina' should be extracted cleanly without 'write'
        self.assertIn("amina", ent_texts, f"'amina' should be cleanly extracted: {ent_texts}")
        self.assertNotIn("amina write", ent_texts, f"'amina write' should not exist: {ent_texts}")

        # 'elon musk' should be extracted
        self.assertTrue(any("elon musk" in t.lower() for t in ent_texts), f"'elon musk' missing from {ent_texts}")

    def test_various_password_shortforms(self):
        cases = [
            ("psswrd is 87654321", "87654321", EntityType.PASSWORD),
            ("my pswd is Secret123", "Secret123", EntityType.PASSWORD),
            ("pwd: MyPass#99", "MyPass#99", EntityType.PASSWORD),
            ("pin is 4321", "4321", EntityType.PASSWORD),
        ]
        for prompt, expected_text, expected_type in cases:
            entities = self.shield_pure.analyze(prompt)
            ent_map = {e.text: e.entity_type for e in entities}
            self.assertIn(expected_text, ent_map, f"Failed for '{prompt}', got: {ent_map}")
            self.assertEqual(ent_map[expected_text], expected_type)

    def test_various_name_shortforms(self):
        cases = [
            ("im adarsh and email is a@test.com", "adarsh"),
            ("myself rahul from Bangalore", "rahul"),
            ("my names david and I live in Paris", "david"),
        ]
        for prompt, expected_name in cases:
            entities = self.shield_neural.analyze(prompt)
            ent_map = {e.text.lower(): e.entity_type for e in entities}
            self.assertIn(expected_name.lower(), ent_map, f"Failed to detect {expected_name} in '{prompt}': {ent_map}")
            self.assertEqual(ent_map[expected_name.lower()], EntityType.PERSON)


if __name__ == "__main__":
    unittest.main()
