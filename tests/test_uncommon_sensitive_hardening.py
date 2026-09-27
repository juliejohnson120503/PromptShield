"""
Unit & regression tests for uncommon sensitive information hardening (Issues 1-7).
Verifies:
- Canonical entity types (MAC_ADDRESS, MEDICAL_RECORD, HEALTH_INSURANCE_ID, CONNECTION_STRING)
- Deterministic MAC address recognition and 'MAC' label non-sensitivity
- Medical identifiers (MRN, HIN) contextual detection and privacy risk
- Connection strings and prevention of bogus email spans on credential URIs
- Descriptor label suppression (MAC, IBAN, BIC, SWIFT) without blacklisting organizations
- Semantic policy explanation generation distinguishing credentials from identity documents
- Context-sensitive maiden name / security question escalation
"""

import unittest
from backend.models import EntityType, TaskType, RoleCategory
from backend.core import PromptShieldCore
from backend.regex_detector import RegexDetector
from backend.normalization import normalize_entity_value
from backend.baseline_masker import BaselineMasker
from backend.policy_engine import PolicyDecision, RiskLevel


class TestUncommonSensitiveHardening(unittest.TestCase):

    def setUp(self):
        # Initialise core without heavyweight spaCy model for fast, deterministic testing
        self.shield = PromptShieldCore(use_spacy=False)
        self.regex = RegexDetector()

    # -------------------------------------------------------------------------
    # Issue 1: Canonical Entity Types
    # -------------------------------------------------------------------------
    def test_canonical_entity_types_exist(self):
        self.assertEqual(EntityType.MAC_ADDRESS.value, "MAC_ADDRESS")
        self.assertEqual(EntityType.MEDICAL_RECORD.value, "MEDICAL_RECORD")
        self.assertEqual(EntityType.HEALTH_INSURANCE_ID.value, "HEALTH_INSURANCE_ID")
        self.assertEqual(EntityType.CONNECTION_STRING.value, "CONNECTION_STRING")

    def test_new_entity_type_normalizers(self):
        self.assertEqual(
            normalize_entity_value(EntityType.MAC_ADDRESS, " 3c:52:82:4a:91:bf "),
            "3C:52:82:4A:91:BF",
        )
        self.assertEqual(
            normalize_entity_value(EntityType.CONNECTION_STRING, " 'postgresql://usr:pwd@host:5432/db' "),
            "postgresql://usr:pwd@host:5432/db",
        )
        self.assertEqual(
            normalize_entity_value(EntityType.MEDICAL_RECORD, " mrn-84729163 "),
            "mrn-84729163",
        )
        self.assertEqual(
            normalize_entity_value(EntityType.HEALTH_INSURANCE_ID, " HIN-72819456 "),
            "HIN-72819456",
        )

    def test_baseline_masker_prefixes(self):
        masker = BaselineMasker()
        self.assertIn(EntityType.MAC_ADDRESS, masker.TYPE_PREFIXES)
        self.assertIn(EntityType.CONNECTION_STRING, masker.TYPE_PREFIXES)
        self.assertIn(EntityType.MEDICAL_RECORD, masker.TYPE_PREFIXES)
        self.assertIn(EntityType.HEALTH_INSURANCE_ID, masker.TYPE_PREFIXES)

    # -------------------------------------------------------------------------
    # Issue 2: MAC Address Detection
    # -------------------------------------------------------------------------
    def test_mac_address_colon_and_hyphen(self):
        prompt = "Hardware address is 3C:52:82:4A:91:BF and alt is 3C-52-82-4A-91-BF."
        entities = self.regex.detect(prompt)
        mac_entities = [e for e in entities if e.entity_type == EntityType.MAC_ADDRESS]
        self.assertEqual(len(mac_entities), 2)
        self.assertEqual(mac_entities[0].text, "3C:52:82:4A:91:BF")
        self.assertEqual(mac_entities[1].text, "3C-52-82-4A-91-BF")

    def test_mac_label_not_sensitive(self):
        prompt = "device MAC address is 3C:52:82:4A:91:BF"
        fused = self.shield.detector.detect(prompt)
        entity_texts = [e.text for e in fused]
        self.assertIn("3C:52:82:4A:91:BF", entity_texts)
        self.assertNotIn("MAC", entity_texts)

    # -------------------------------------------------------------------------
    # Issue 3: Medical Identifiers (MRN, HIN)
    # -------------------------------------------------------------------------
    def test_medical_identifiers_detection(self):
        prompt = "Her medical record number is MRN-84729163 and health insurance member ID is HIN-72819456."
        fused = self.shield.detector.detect(prompt)
        types = {e.text: e.entity_type for e in fused}
        self.assertEqual(types.get("MRN-84729163"), EntityType.MEDICAL_RECORD)
        self.assertEqual(types.get("HIN-72819456"), EntityType.HEALTH_INSURANCE_ID)

    def test_medical_identifiers_policy_mask(self):
        prompt = "Patient records: medical record number is MRN-84729163 and health insurance member ID is HIN-72819456."
        report = self.shield.evaluate_policy(prompt)
        policies = {p.entity_text: p for p in report.entity_policies}
        self.assertIn("MRN-84729163", policies)
        self.assertIn("HIN-72819456", policies)
        self.assertEqual(policies["MRN-84729163"].decision, PolicyDecision.MASK)
        self.assertEqual(policies["HIN-72819456"].decision, PolicyDecision.MASK)
        self.assertIn(policies["MRN-84729163"].risk_level, (RiskLevel.HIGH, RiskLevel.CRITICAL))
        self.assertIn(policies["HIN-72819456"].risk_level, (RiskLevel.HIGH, RiskLevel.CRITICAL))

    # -------------------------------------------------------------------------
    # Issue 4: Connection Strings / Database Credentials
    # -------------------------------------------------------------------------
    def test_connection_string_detection_and_no_bogus_email(self):
        prompt = "postgresql://erodriguez:RiverStone9472@db.internal.example:5432/payroll"
        fused = self.shield.detector.detect(prompt)
        # Should be a single connection string entity
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].entity_type, EntityType.CONNECTION_STRING)
        self.assertEqual(fused[0].text, prompt)
        # Email entity must NOT be present
        self.assertFalse(any(e.entity_type == EntityType.EMAIL for e in fused))

    def test_connection_string_policy_critical_mask(self):
        prompt = "Connect to database postgresql://erodriguez:RiverStone9472@db.internal.example:5432/payroll."
        report = self.shield.evaluate_policy(prompt)
        conn_p = next((p for p in report.entity_policies if p.entity_type == EntityType.CONNECTION_STRING), None)
        self.assertIsNotNone(conn_p)
        self.assertEqual(conn_p.decision, PolicyDecision.MASK)
        self.assertEqual(conn_p.risk_level, RiskLevel.CRITICAL)

    # -------------------------------------------------------------------------
    # Issue 5: Label Word False Positive Suppression
    # -------------------------------------------------------------------------
    def test_descriptor_words_suppressed_as_labels(self):
        prompt = "IBAN is GB82WEST12345698765432 and SWIFT/BIC is DEUTDEFF500."
        fused = self.shield.detector.detect(prompt)
        texts = [e.text for e in fused]
        self.assertNotIn("IBAN", texts)
        self.assertNotIn("BIC", texts)
        self.assertIn("GB82WEST12345698765432", texts)
        self.assertIn("DEUTDEFF500", texts)

    # -------------------------------------------------------------------------
    # Issue 6: Policy Reason Generation
    # -------------------------------------------------------------------------
    def test_policy_reasons_distinguish_identity_vs_credentials(self):
        prompt = (
            "Her passport number is X12345678, driver's license number is D123-4567-8901, "
            "national identification number is 784-1987-1234567-1, and tax identification "
            "number is 123-45-6789. Password is RiverStone#9472 and API key is sk_test_8Kx29Qa7Lm4Np2Vz."
        )
        report = self.shield.evaluate_policy(prompt)
        policy_map = {p.entity_text: p for p in report.entity_policies}

        # Passport
        passport_p = policy_map.get("X12345678")
        self.assertIsNotNone(passport_p)
        self.assertEqual(passport_p.role_category, RoleCategory.PERSONAL_IDENTIFIER)
        self.assertIn("Government-issued identity document number", passport_p.reason)

        # Driver License
        dl_p = policy_map.get("D123-4567-8901")
        self.assertIsNotNone(dl_p)
        self.assertEqual(dl_p.role_category, RoleCategory.PERSONAL_IDENTIFIER)
        self.assertIn("Government-issued driver's licence identifier", dl_p.reason)

        # National ID
        nid_p = policy_map.get("784-1987-1234567-1")
        self.assertIsNotNone(nid_p)
        self.assertEqual(nid_p.role_category, RoleCategory.PERSONAL_IDENTIFIER)
        self.assertIn("High-risk government-issued personal identifier", nid_p.reason)

        # Tax ID
        tax_p = policy_map.get("123-45-6789")
        self.assertIsNotNone(tax_p)
        self.assertEqual(tax_p.role_category, RoleCategory.PERSONAL_IDENTIFIER)
        self.assertIn("Sensitive tax/government identifier", tax_p.reason)

        # Password
        pwd_p = policy_map.get("RiverStone#9472")
        self.assertIsNotNone(pwd_p)
        self.assertEqual(pwd_p.role_category, RoleCategory.TECHNICAL_CREDENTIAL)
        self.assertIn("Authentication credential", pwd_p.reason)

        # API Key
        api_p = policy_map.get("sk_test_8Kx29Qa7Lm4Np2Vz")
        self.assertIsNotNone(api_p)
        self.assertEqual(api_p.role_category, RoleCategory.TECHNICAL_CREDENTIAL)
        self.assertIn("Technical credential", api_p.reason)

    # -------------------------------------------------------------------------
    # Issue 7: Context-Sensitive Maiden Name
    # -------------------------------------------------------------------------
    def test_maiden_name_contextual_escalation(self):
        prompt_sensitive = "Her mother's maiden name is Fernandez."
        report_sens = self.shield.evaluate_policy(prompt_sensitive)
        fernandez_sens = next((p for p in report_sens.entity_policies if "Fernandez" in p.entity_text), None)
        self.assertIsNotNone(fernandez_sens)
        self.assertEqual(fernandez_sens.decision, PolicyDecision.MASK)
        self.assertEqual(fernandez_sens.role_category, RoleCategory.TECHNICAL_CREDENTIAL)
        self.assertGreaterEqual(fernandez_sens.risk_score, 0.85)

    def test_general_surname_not_high_risk(self):
        prompt_general = "Fernandez is a common surname."
        report_gen = self.shield.evaluate_policy(prompt_general)
        fernandez_gen = next((p for p in report_gen.entity_policies if "Fernandez" in p.entity_text), None)
        if fernandez_gen is not None:
            self.assertNotEqual(fernandez_gen.risk_level, RiskLevel.CRITICAL)
            self.assertNotEqual(fernandez_gen.role_category, RoleCategory.TECHNICAL_CREDENTIAL)
            self.assertEqual(fernandez_gen.decision, PolicyDecision.RETAIN)


if __name__ == "__main__":
    unittest.main()
