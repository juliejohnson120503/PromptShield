"""
Unit tests for structured sensitive entity detection:
- Email detection
- Phone number detection
- Credit card detection & Luhn validation
- API key detection
- Password / credential detection
"""

import unittest
from backend.detector import SensitiveDataDetector, is_luhn_valid, detect_card_brand
from backend.models import EntityType


class TestStructuredDetector(unittest.TestCase):

    def setUp(self):
        # Instantiate detector without requiring spacy for these tests
        self.detector = SensitiveDataDetector(use_spacy=False)

    # ----------------------------------------------------
    # EMAIL TESTS
    # ----------------------------------------------------
    def test_email_single(self):
        prompt = "My email is john.doe@example.com, please contact me."
        entities = self.detector.detect(prompt)
        email_entities = [e for e in entities if e.entity_type == EntityType.EMAIL]
        self.assertEqual(len(email_entities), 1)
        self.assertEqual(email_entities[0].text, "john.doe@example.com")
        self.assertEqual(email_entities[0].normalized_value, "john.doe@example.com")

    def test_email_multiple(self):
        prompt = "Send to support@company.org or admin@sub.domain.co.uk."
        entities = self.detector.detect(prompt)
        emails = [e.text for e in entities if e.entity_type == EntityType.EMAIL]
        self.assertIn("support@company.org", emails)
        self.assertIn("admin@sub.domain.co.uk", emails)
        self.assertEqual(len(emails), 2)

    # ----------------------------------------------------
    # PHONE TESTS
    # ----------------------------------------------------
    def test_phone_international(self):
        prompt = "Call our hotline at +1 (555) 234-5678 immediately."
        entities = self.detector.detect(prompt)
        phones = [e for e in entities if e.entity_type == EntityType.PHONE]
        self.assertTrue(len(phones) >= 1)
        self.assertTrue("555" in phones[0].text)

    def test_phone_indian_format(self):
        prompt = "You can reach me at 9876543210 or +91 98765 43210."
        entities = self.detector.detect(prompt)
        phones = [e for e in entities if e.entity_type == EntityType.PHONE]
        self.assertEqual(len(phones), 2)

    def test_phone_reject_small_numbers(self):
        prompt = "Order 12345 placed in year 2024 with 42 items."
        entities = self.detector.detect(prompt)
        phones = [e for e in entities if e.entity_type == EntityType.PHONE]
        self.assertEqual(len(phones), 0)

    # ----------------------------------------------------
    # CREDIT CARD & LUHN TESTS
    # ----------------------------------------------------
    def test_luhn_algorithm_helper(self):
        # 4111 1111 1111 1111 is Luhn-valid
        self.assertTrue(is_luhn_valid("4111 1111 1111 1111"))
        self.assertTrue(is_luhn_valid("4111-1111-1111-1111"))
        self.assertTrue(is_luhn_valid("4111111111111111"))
        # Invalid checksum
        self.assertFalse(is_luhn_valid("4111 1111 1111 1112"))
        # Too short / too long
        self.assertFalse(is_luhn_valid("12345"))
        self.assertFalse(is_luhn_valid("123456789012345678901"))

    def test_credit_card_detection_valid(self):
        prompt = "Billing card is 4111-1111-1111-1111 expiring next year."
        entities = self.detector.detect(prompt)
        cards = [e for e in entities if e.entity_type == EntityType.CREDIT_CARD]
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].normalized_value, "4111111111111111")
        self.assertEqual(cards[0].metadata.get("brand"), "Visa")

    def test_credit_card_detection_invalid_luhn_rejected(self):
        prompt = "Reference tracking number is 4111-1111-1111-1112 in warehouse."
        entities = self.detector.detect(prompt)
        cards = [e for e in entities if e.entity_type == EntityType.CREDIT_CARD]
        self.assertEqual(len(cards), 0)

    # ----------------------------------------------------
    # API KEY TESTS
    # ----------------------------------------------------
    def test_api_key_openai(self):
        prompt = "Use the OpenAI key sk-live-1234567890abcdef1234567890 for API calls."
        entities = self.detector.detect(prompt)
        keys = [e for e in entities if e.entity_type == EntityType.API_KEY]
        self.assertEqual(len(keys), 1)
        self.assertTrue(keys[0].text.startswith("sk-live-"))

    def test_api_key_google(self):
        prompt = "Google API key AIzaSyD9876543210abcdefghijklmnopqrs is ready."
        entities = self.detector.detect(prompt)
        keys = [e for e in entities if e.entity_type == EntityType.API_KEY]
        self.assertEqual(len(keys), 1)
        self.assertTrue(keys[0].text.startswith("AIza"))

    def test_api_key_github(self):
        prompt = "Deploy token is ghp_1234567890abcdefghijklmnopqrstuvwxyz."
        entities = self.detector.detect(prompt)
        keys = [e for e in entities if e.entity_type == EntityType.API_KEY]
        self.assertEqual(len(keys), 1)
        self.assertTrue(keys[0].text.startswith("ghp_"))

    def test_api_key_generic_assignment(self):
        prompt = "Set api_key = 'abcdef1234567890abcdef' in config."
        entities = self.detector.detect(prompt)
        keys = [e for e in entities if e.entity_type == EntityType.API_KEY]
        self.assertEqual(len(keys), 1)

    # ----------------------------------------------------
    # PASSWORD TESTS
    # ----------------------------------------------------
    def test_password_explicit_assignment(self):
        prompt = "Login with user='admin' and password='SuperSecretPassword!2024'."
        entities = self.detector.detect(prompt)
        passwords = [e for e in entities if e.entity_type == EntityType.PASSWORD]
        self.assertEqual(len(passwords), 1)
        self.assertEqual(passwords[0].normalized_value, "SuperSecretPassword!2024")

    def test_password_natural_language(self):
        prompt = "My password is hunter2_unbreakable! Remember it."
        entities = self.detector.detect(prompt)
        passwords = [e for e in entities if e.entity_type == EntityType.PASSWORD]
        self.assertEqual(len(passwords), 1)
        self.assertIn("hunter2_unbreakable!", passwords[0].normalized_value)


if __name__ == "__main__":
    unittest.main()
