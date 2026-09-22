"""
Unit tests for RegexDetector — Phase 1 structured detection module.

Tests cover:
  - Email detection
  - Phone detection (international + domestic)
  - Credit card Luhn validation and brand detection
  - API key detection (OpenAI, Google, GitHub, AWS, generic)
  - Access token detection (JWT)
  - Password / credential detection
  - Bank account / IBAN detection
  - IP address detection (IPv4)
  - Date detection
"""

import unittest
from backend.regex_detector import RegexDetector, is_luhn_valid, detect_card_brand
from backend.models import EntityType


class TestRegexDetector(unittest.TestCase):

    def setUp(self):
        self.detector = RegexDetector()

    # -------------------------------------------------------
    # EMAIL
    # -------------------------------------------------------
    def test_email_basic(self):
        prompt = "My email is julie@gmail.com"
        entities = self.detector.detect(prompt)
        emails = [e for e in entities if e.entity_type == EntityType.EMAIL]
        self.assertEqual(len(emails), 1)
        self.assertEqual(emails[0].text, "julie@gmail.com")
        self.assertEqual(emails[0].normalized_value, "julie@gmail.com")
        self.assertEqual(emails[0].source, "regex")

    def test_email_multiple(self):
        prompt = "Send to support@company.org or admin@sub.domain.co.uk."
        emails = [e for e in self.detector.detect(prompt)
                  if e.entity_type == EntityType.EMAIL]
        self.assertEqual(len(emails), 2)
        texts = {e.text for e in emails}
        self.assertIn("support@company.org", texts)
        self.assertIn("admin@sub.domain.co.uk", texts)

    def test_email_normalized_lowercase(self):
        prompt = "Contact John at JOHN.DOE@EXAMPLE.COM"
        emails = [e for e in self.detector.detect(prompt)
                  if e.entity_type == EntityType.EMAIL]
        self.assertEqual(len(emails), 1)
        self.assertEqual(emails[0].normalized_value, "john.doe@example.com")

    # -------------------------------------------------------
    # PHONE
    # -------------------------------------------------------
    def test_phone_10digit_indian(self):
        """Test 1 — Phone: 'Call me at 9876543210'"""
        prompt = "Call me at 9876543210"
        phones = [e for e in self.detector.detect(prompt)
                  if e.entity_type == EntityType.PHONE]
        self.assertGreaterEqual(len(phones), 1)
        self.assertIn("9876543210", [p.text for p in phones])

    def test_phone_international(self):
        prompt = "Call our hotline at +1 (555) 234-5678 immediately."
        phones = [e for e in self.detector.detect(prompt)
                  if e.entity_type == EntityType.PHONE]
        self.assertGreaterEqual(len(phones), 1)

    def test_phone_short_number_rejected(self):
        prompt = "Order 12345 placed in year 2024 with 42 items."
        phones = [e for e in self.detector.detect(prompt)
                  if e.entity_type == EntityType.PHONE]
        self.assertEqual(len(phones), 0)

    # -------------------------------------------------------
    # CREDIT CARD & LUHN
    # -------------------------------------------------------
    def test_luhn_valid_synthetic(self):
        """Test 3 — Synthetic credit card Luhn validation."""
        self.assertTrue(is_luhn_valid("4111111111111111"))
        self.assertTrue(is_luhn_valid("4111 1111 1111 1111"))
        self.assertTrue(is_luhn_valid("4111-1111-1111-1111"))

    def test_luhn_invalid_rejected(self):
        self.assertFalse(is_luhn_valid("4111111111111112"))
        self.assertFalse(is_luhn_valid("12345"))
        self.assertFalse(is_luhn_valid("123456789012345678901"))

    def test_card_brand_visa(self):
        self.assertEqual(detect_card_brand("4111111111111111"), "Visa")

    def test_card_brand_mastercard(self):
        self.assertEqual(detect_card_brand("5111111111111118"), "Mastercard")

    def test_credit_card_detection_synthetic(self):
        """Test 3 — 'My card is 4111111111111111' → CREDIT_CARD detected."""
        prompt = "My card is 4111111111111111"
        cards = [e for e in self.detector.detect(prompt)
                 if e.entity_type == EntityType.CREDIT_CARD]
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].normalized_value, "4111111111111111")
        self.assertEqual(cards[0].metadata.get("brand"), "Visa")
        self.assertEqual(cards[0].source, "regex")
        self.assertEqual(cards[0].detector, "luhn_credit_card")

    def test_credit_card_invalid_luhn_rejected(self):
        prompt = "Reference number is 4111-1111-1111-1112 in the system."
        cards = [e for e in self.detector.detect(prompt)
                 if e.entity_type == EntityType.CREDIT_CARD]
        self.assertEqual(len(cards), 0)

    # -------------------------------------------------------
    # API KEY
    # -------------------------------------------------------
    def test_api_key_openai_live(self):
        prompt = "Use key sk-live-1234567890abcdef1234567890 for API calls."
        keys = [e for e in self.detector.detect(prompt)
                if e.entity_type == EntityType.API_KEY]
        self.assertEqual(len(keys), 1)
        self.assertTrue(keys[0].text.startswith("sk-live-"))
        self.assertEqual(keys[0].source, "regex")

    def test_api_key_openai_test(self):
        """Test 8 — Synthetic secret: 'Use sk-test-example-12345'"""
        prompt = "Use sk-test-example-12345 to authenticate."
        keys = [e for e in self.detector.detect(prompt)
                if e.entity_type == EntityType.API_KEY]
        self.assertGreaterEqual(len(keys), 1)

    def test_api_key_google(self):
        prompt = "Google key AIzaSyD9876543210abcdefghijklmnopqrs is ready."
        keys = [e for e in self.detector.detect(prompt)
                if e.entity_type == EntityType.API_KEY]
        self.assertEqual(len(keys), 1)
        self.assertTrue(keys[0].text.startswith("AIza"))

    def test_api_key_github(self):
        prompt = "Deploy token is ghp_1234567890abcdefghijklmnopqrstuvwxyz."
        keys = [e for e in self.detector.detect(prompt)
                if e.entity_type == EntityType.API_KEY]
        self.assertEqual(len(keys), 1)

    def test_api_key_generic_assignment(self):
        prompt = "Set api_key = 'abcdef1234567890abcdef' in config."
        keys = [e for e in self.detector.detect(prompt)
                if e.entity_type == EntityType.API_KEY]
        self.assertEqual(len(keys), 1)

    # -------------------------------------------------------
    # ACCESS TOKEN (JWT)
    # -------------------------------------------------------
    def test_jwt_token_detected(self):
        jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ"
            ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        prompt = f"Authorization header: {jwt}"
        tokens = [e for e in self.detector.detect(prompt)
                  if e.entity_type == EntityType.ACCESS_TOKEN]
        self.assertGreaterEqual(len(tokens), 1)

    # -------------------------------------------------------
    # PASSWORD
    # -------------------------------------------------------
    def test_password_assignment(self):
        prompt = "Login with user='admin' and password='SuperSecretPassword!2024'."
        passwords = [e for e in self.detector.detect(prompt)
                     if e.entity_type == EntityType.PASSWORD]
        self.assertGreaterEqual(len(passwords), 1)
        self.assertIn("SuperSecretPassword!2024",
                      [p.normalized_value for p in passwords])

    def test_password_natural_language(self):
        prompt = "My password is hunter2_unbreakable! Remember it."
        passwords = [e for e in self.detector.detect(prompt)
                     if e.entity_type == EntityType.PASSWORD]
        self.assertGreaterEqual(len(passwords), 1)
        self.assertTrue(
            any("hunter2_unbreakable!" in p.normalized_value for p in passwords)
        )

    # -------------------------------------------------------
    # IP ADDRESS
    # -------------------------------------------------------
    def test_ipv4_detected(self):
        prompt = "Server at 192.168.1.100 is down."
        ips = [e for e in self.detector.detect(prompt)
               if e.entity_type == EntityType.IP_ADDRESS]
        self.assertEqual(len(ips), 1)
        self.assertEqual(ips[0].text, "192.168.1.100")

    def test_ipv4_not_from_unstructured_text(self):
        """A random number string should not become an IP."""
        prompt = "Room number 100 in building 192."
        ips = [e for e in self.detector.detect(prompt)
               if e.entity_type == EntityType.IP_ADDRESS]
        self.assertEqual(len(ips), 0)

    # -------------------------------------------------------
    # DATE
    # -------------------------------------------------------
    def test_date_iso(self):
        prompt = "Report finalized on 2025-10-24."
        dates = [e for e in self.detector.detect(prompt)
                 if e.entity_type == EntityType.DATE]
        self.assertEqual(len(dates), 1)
        self.assertEqual(dates[0].text, "2025-10-24")

    def test_date_written(self):
        prompt = "Symposium on January 15, 2025."
        dates = [e for e in self.detector.detect(prompt)
                 if e.entity_type == EntityType.DATE]
        self.assertEqual(len(dates), 1)
        self.assertIn("January", dates[0].text)

    # -------------------------------------------------------
    # SOURCE FIELD
    # -------------------------------------------------------
    def test_all_regex_entities_have_source_regex(self):
        """Every entity from RegexDetector must carry source='regex'."""
        prompt = (
            "Email: julie@gmail.com, Phone: 9876543210, "
            "Card: 4111111111111111, Key: sk-live-aaa111bbb222ccc333ddd444"
        )
        entities = self.detector.detect(prompt)
        for e in entities:
            self.assertEqual(
                e.source, "regex",
                f"Expected source='regex', got '{e.source}' for {e.entity_type}",
            )


if __name__ == "__main__":
    unittest.main()
