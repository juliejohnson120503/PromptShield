"""
End-to-End integration tests for PromptShield AI Phase 1 Core Pipeline.
Tests the full flow: Raw Prompt -> Detection -> Baseline Masking -> Local Mapping Storage.
"""

import unittest
from backend.core import PromptShieldCore
from backend.models import EntityType


class TestCorePipeline(unittest.TestCase):

    def setUp(self):
        self.shield = PromptShieldCore(use_spacy=False)

    def test_full_pipeline_multi_entity(self):
        prompt = (
            "My name is John Mathew and my email is john@gmail.com. "
            "Write a professional email to ABC Technologies regarding my application "
            "for the London office."
        )

        res = self.shield.sanitize_baseline(prompt, session_id="test-e2e-session")

        # 1. Verify detection
        detected_types = {e.entity_type for e in res.entities}
        self.assertIn(EntityType.PERSON, detected_types)
        self.assertIn(EntityType.EMAIL, detected_types)
        self.assertIn(EntityType.ORGANIZATION, detected_types)
        self.assertIn(EntityType.LOCATION, detected_types)

        # 2. Verify baseline placeholders are present in sanitized prompt
        self.assertIn("<PERSON_1>", res.sanitized_prompt)
        self.assertIn("<EMAIL_1>", res.sanitized_prompt)
        self.assertIn("<ORG_1>", res.sanitized_prompt)
        self.assertIn("<LOCATION_1>", res.sanitized_prompt)

        # 3. Verify sensitive raw values do not appear in the sanitized text
        self.assertNotIn("john@gmail.com", res.sanitized_prompt)
        self.assertNotIn("John Mathew", res.sanitized_prompt)

        # 4. Verify local mapping store holds the original values
        mapping = self.shield.get_local_mappings("test-e2e-session")
        self.assertEqual(mapping.get("<PERSON_1>"), "John Mathew")
        self.assertEqual(mapping.get("<EMAIL_1>"), "john@gmail.com")
        self.assertEqual(mapping.get("<ORG_1>"), "ABC Technologies")
        self.assertEqual(mapping.get("<LOCATION_1>"), "London")

    def test_secrets_pipeline(self):
        prompt = "Connect to database with password='SuperSecretDbPass!2025' and use key sk-live-111122223333444455556666."
        res = self.shield.sanitize_baseline(prompt, session_id="secret-session")

        # Secrets must be detected and masked
        self.assertIn("<PASSWORD_1>", res.sanitized_prompt)
        self.assertIn("<API_KEY_1>", res.sanitized_prompt)
        self.assertNotIn("SuperSecretDbPass!2025", res.sanitized_prompt)
        self.assertNotIn("sk-live-111122223333444455556666", res.sanitized_prompt)

        # Mappings must be saved locally
        mapping = self.shield.get_local_mappings("secret-session")
        self.assertEqual(mapping["<PASSWORD_1>"], "SuperSecretDbPass!2025")
        self.assertEqual(mapping["<API_KEY_1>"], "sk-live-111122223333444455556666")

    def test_analyze_only(self):
        prompt = "Send report to contact@example.com."
        entities = self.shield.analyze(prompt)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].entity_type, EntityType.EMAIL)
        self.assertEqual(entities[0].text, "contact@example.com")


if __name__ == "__main__":
    unittest.main()
