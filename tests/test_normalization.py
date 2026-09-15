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
)
from backend.models import EntityType


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


if __name__ == "__main__":
    unittest.main()
