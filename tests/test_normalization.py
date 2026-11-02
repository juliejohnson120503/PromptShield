"""
Unit tests for entity normalization functions.
"""

import unittest
from backend.normalization import (
    normalize_email,
    normalize_phone,
    normalize_credit_card,
    normalize_api_key,
    normalize_password,
    normalize_text_entity,
    normalize_entity_value,
    canonicalize_adversarial_text,
    remap_entity_offsets,
)
from backend.models import EntityType, DetectedEntity


class TestNormalization(unittest.TestCase):

    def test_normalize_email(self):
        self.assertEqual(normalize_email("  User.Name@Example.COM  "), "user.name@example.com")

    def test_normalize_phone(self):
        self.assertEqual(normalize_phone("+91 (987) 654-3210"), "+919876543210")
        self.assertEqual(normalize_phone("987-654-3210"), "9876543210")

    def test_normalize_credit_card(self):
        self.assertEqual(normalize_credit_card("4111-1111-1111-1111"), "4111111111111111")
        self.assertEqual(normalize_credit_card("4111 1111 1111 1111"), "4111111111111111")

    def test_normalize_api_key(self):
        self.assertEqual(normalize_api_key(" 'sk-live-abc123456789' "), "sk-live-abc123456789")
        self.assertEqual(normalize_api_key('"AIzaSyD12345"'), "AIzaSyD12345")

    def test_normalize_password(self):
        self.assertEqual(normalize_password("password='hunter2'"), "hunter2")
        self.assertEqual(normalize_password("pwd: \"SuperSecret!\""), "SuperSecret!")
        self.assertEqual(normalize_password("raw_pass_123"), "raw_pass_123")

    def test_normalize_text_entity(self):
        self.assertEqual(normalize_text_entity("  John   Mathew, "), "John Mathew")
        self.assertEqual(normalize_text_entity("ABC   Technologies. "), "ABC Technologies")

    def test_dispatcher(self):
        self.assertEqual(
            normalize_entity_value(EntityType.EMAIL, " JOHN@TEST.COM "), "john@test.com"
        )
        self.assertEqual(
            normalize_entity_value(EntityType.CREDIT_CARD, "4111-1111-1111-1111"),
            "4111111111111111",
        )

    def test_canonicalize_adversarial_ascii_fast_path(self):
        text = "Contact Alice at alice@example.com"
        clean, offset_map = canonicalize_adversarial_text(text)
        self.assertEqual(clean, text)
        self.assertIsNone(offset_map)

    def test_canonicalize_adversarial_zero_width_chars(self):
        raw = "j\u200be\u200br\u200bi\u200bn@example.com"
        clean, offset_map = canonicalize_adversarial_text(raw)
        self.assertEqual(clean, "jerin@example.com")
        self.assertIsNotNone(offset_map)

    def test_canonicalize_adversarial_homoglyphs(self):
        # Cyrillic 'о' in 'John' and 'е' in 'jerin@example.com'
        raw = "Contact J\u043ehn at j\u0435rin@\u0435xampl\u0435.com"
        clean, offset_map = canonicalize_adversarial_text(raw)
        self.assertEqual(clean, "Contact John at jerin@example.com")
        self.assertIsNotNone(offset_map)

    def test_canonicalize_adversarial_fullwidth_math(self):
        # Mathematical script admin@example.com
        raw = "\U0001D4B6\U0001D4B9\U0001D4C2\U0001D4BE\U0001D4C3@example.com"
        clean, offset_map = canonicalize_adversarial_text(raw)
        self.assertEqual(clean, "admin@example.com")
        self.assertIsNotNone(offset_map)

    def test_remap_entity_offsets_integrity(self):
        raw_prompt = "Contact j\u200be\u200br\u200bi\u200bn@example.com today"
        clean_prompt, offset_map = canonicalize_adversarial_text(raw_prompt)
        self.assertEqual(clean_prompt, "Contact jerin@example.com today")

        # In clean_prompt, 'jerin@example.com' is at [8:25]
        clean_ent = DetectedEntity(
            entity_type=EntityType.EMAIL,
            text="jerin@example.com",
            start=8,
            end=25,
            confidence=1.0,
            source="regex",
            normalized_value="jerin@example.com",
        )

        remapped = remap_entity_offsets([clean_ent], raw_prompt, offset_map)
        self.assertEqual(len(remapped), 1)
        ent = remapped[0]
        self.assertEqual(ent.text, "j\u200be\u200br\u200bi\u200bn@example.com")
        self.assertEqual(raw_prompt[ent.start:ent.end], ent.text)
        self.assertTrue(ent.metadata.get("evasion_normalized"))
        self.assertEqual(ent.metadata.get("canonical_text"), "jerin@example.com")


if __name__ == "__main__":
    unittest.main()

