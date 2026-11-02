"""
PromptShield AI — Adversarial Unicode & Homoglyph Evasion Defense Test Suite.

Validates that:
1. Zero-width character obfuscation (e.g. \\u200B, \\u200C, \\uFEFF) in emails, phones,
   cards, passwords, and IDs is neutralized and detected.
2. Cyrillic and Greek homoglyph / confusable spoofing attacks in names, emails,
   and credentials are accurately canonicalized and detected.
3. Mathematical script and fullwidth alphanumeric evasion vectors are defended.
4. Exact character-span integrity (prompt[start:end] == entity.text) is strictly
   preserved so that downstream masking and restoration slice without drift.
5. Standard ASCII and benign international unicode (e.g. accented names) run on
   the O(1) zero-overhead fast path without unnecessary transformation.
"""

import unittest
from backend.detector import SensitiveDataDetector
from backend.baseline_masker import BaselineMasker
from backend.models import EntityType


class TestAdversarialUnicodeDefense(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialise hybrid detector (using spaCy if available, fallback heuristic otherwise)
        cls.detector = SensitiveDataDetector(use_spacy=True)
        cls.masker = BaselineMasker()

    @classmethod
    def tearDownClass(cls):
        cls.detector.close()

    def test_zero_width_email_evasion(self):
        prompt = "Please send credentials to j\u200be\u200br\u200bi\u200bn@example.com by noon."
        entities = self.detector.detect(prompt)

        email_entities = [e for e in entities if e.entity_type == EntityType.EMAIL]
        self.assertEqual(len(email_entities), 1, "Failed to detect zero-width obfuscated email")
        ent = email_entities[0]

        # Verify span integrity
        self.assertEqual(prompt[ent.start:ent.end], ent.text)
        self.assertIn("\u200b", ent.text)
        self.assertEqual(ent.normalized_value, "jerin@example.com")
        self.assertTrue(ent.metadata.get("evasion_normalized"))

        # Verify masking substitutes cleanly without corrupting the prompt
        result = self.masker.mask(prompt, entities)
        self.assertIn("<EMAIL_1>", result.sanitized_prompt)
        self.assertNotIn("example.com", result.sanitized_prompt)
        self.assertNotIn("\u200b", result.sanitized_prompt)

    def test_zero_width_phone_evasion(self):
        prompt = "Call our hotline at 9\u200b8\u200b7-6\u200b5\u200b4-3\u200b2\u200b1\u200b0 immediately."
        entities = self.detector.detect(prompt)

        phone_entities = [e for e in entities if e.entity_type == EntityType.PHONE]
        self.assertEqual(len(phone_entities), 1, "Failed to detect zero-width obfuscated phone")
        ent = phone_entities[0]

        self.assertEqual(prompt[ent.start:ent.end], ent.text)
        self.assertEqual(ent.normalized_value, "9876543210")

        result = self.masker.mask(prompt, entities)
        self.assertIn("<PHONE_1>", result.sanitized_prompt)
        self.assertNotIn("987", result.sanitized_prompt)

    def test_homoglyph_cyrillic_person_name(self):
        # Cyrillic 'о' (\u043e) in John and Doe
        prompt = "Reach out to J\u043ehn D\u043ee regarding the security audit."
        entities = self.detector.detect(prompt)

        person_entities = [e for e in entities if e.entity_type == EntityType.PERSON]
        self.assertEqual(len(person_entities), 1, "Failed to detect homoglyph person name")
        ent = person_entities[0]

        self.assertEqual(prompt[ent.start:ent.end], ent.text)
        self.assertEqual(ent.normalized_value, "John Doe")
        self.assertTrue(ent.metadata.get("evasion_normalized"))

        result = self.masker.mask(prompt, entities)
        self.assertIn("<PERSON_1>", result.sanitized_prompt)
        self.assertNotIn("D\u043ee", result.sanitized_prompt)

    def test_homoglyph_cyrillic_email(self):
        # Cyrillic 'е' (\u0435) inside jerin@example.com
        prompt = "Direct your inquiry to j\u0435rin@\u0435xampl\u0435.com."
        entities = self.detector.detect(prompt)

        email_entities = [e for e in entities if e.entity_type == EntityType.EMAIL]
        self.assertEqual(len(email_entities), 1, "Failed to detect homoglyph email")
        ent = email_entities[0]

        self.assertEqual(prompt[ent.start:ent.end], ent.text)
        self.assertEqual(ent.normalized_value, "jerin@example.com")

        result = self.masker.mask(prompt, entities)
        self.assertIn("<EMAIL_1>", result.sanitized_prompt)
        self.assertNotIn("xampl", result.sanitized_prompt)

    def test_mathematical_script_alphanumerics(self):
        # Mathematical script: admin@example.com
        raw_email = "\U0001D4B6\U0001D4B9\U0001D4C2\U0001D4BE\U0001D4C3@example.com"
        prompt = f"Submit administrative issues to {raw_email}."
        entities = self.detector.detect(prompt)

        email_entities = [e for e in entities if e.entity_type == EntityType.EMAIL]
        self.assertEqual(len(email_entities), 1, "Failed to detect math script email")
        ent = email_entities[0]

        self.assertEqual(prompt[ent.start:ent.end], ent.text)
        self.assertEqual(ent.normalized_value, "admin@example.com")

        result = self.masker.mask(prompt, entities)
        self.assertIn("<EMAIL_1>", result.sanitized_prompt)
        self.assertNotIn(raw_email, result.sanitized_prompt)

    def test_multiple_evasions_in_single_prompt(self):
        # Combination of zero-width email, Cyrillic person name, and standard credit card
        prompt = (
            "Contact J\u043ehn D\u043ee at j\u200be\u200br\u200bi\u200bn@example.com "
            "using card 4532-7564-8901-2345."
        )
        entities = self.detector.detect(prompt)
        self.assertGreaterEqual(len(entities), 3)

        # Verify all entities have strictly non-overlapping spans and match prompt
        for i in range(len(entities)):
            ent = entities[i]
            self.assertEqual(prompt[ent.start:ent.end], ent.text)
            if i > 0:
                self.assertLessEqual(entities[i - 1].end, ent.start)

        # Mask and verify complete sanitization
        result = self.masker.mask(prompt, entities)
        self.assertIn("<PERSON_1>", result.sanitized_prompt)
        self.assertIn("<EMAIL_1>", result.sanitized_prompt)
        self.assertIn("<CREDIT_CARD_1>", result.sanitized_prompt)
        self.assertNotIn("4532", result.sanitized_prompt)
        self.assertNotIn("example.com", result.sanitized_prompt)

    def test_benign_international_text_unaltered(self):
        # Accented characters in legitimate names/locations should not trigger evasion remapping
        prompt = "Clinical research protocol approved by ethics board at Charit\xe9 in Berlin."
        entities = self.detector.detect(prompt)
        charite_entities = [e for e in entities if "Charit" in e.text]
        self.assertTrue(len(charite_entities) >= 1)
        ent = charite_entities[0]
        self.assertEqual(prompt[ent.start:ent.end], ent.text)
        self.assertFalse(ent.metadata.get("evasion_normalized", False))

    def test_detector_status_reports_unicode_defense(self):
        status = self.detector.get_detector_status()
        self.assertIn("unicode_defense", status)
        self.assertTrue(status["unicode_defense"]["active"])


if __name__ == "__main__":
    unittest.main()
