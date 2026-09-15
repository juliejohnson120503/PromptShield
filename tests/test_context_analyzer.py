"""
Unit tests for ContextAnalyzer (Phase 2).
Verifies differentiation of personal identifiers (e.g. 'Julie') from public figures (e.g. 'Elon Musk'),
task-relevant entities, and technical credentials.
"""

import unittest
from backend.core import PromptShieldCore
from backend.models import EntityType, RoleCategory, TaskType


class TestContextAnalyzer(unittest.TestCase):

    def setUp(self):
        self.shield = PromptShieldCore(use_spacy=False)

    def test_julie_vs_elon_musk(self):
        """
        Critical requirement test:
        1. 'My name is Julie' -> Julie is PERSONAL_IDENTIFIER (is_first_party=True, is_public_knowledge=False)
        2. 'Tell me about Elon Musk' -> Elon Musk is PUBLIC_FIGURE_OR_FACT (is_first_party=False, is_public_knowledge=True)
        """
        # Case A: "My name is Julie"
        prompt_a = "My name is Julie and I need assistance with my account."
        result_a = self.shield.analyze_context(prompt_a)

        julie_context = next(
            (c for c in result_a.entity_contexts if "julie" in c.entity_text.lower()),
            None
        )
        self.assertIsNotNone(julie_context, "Entity 'Julie' was not detected")
        self.assertEqual(julie_context.entity_type, EntityType.PERSON)
        self.assertEqual(julie_context.role_category, RoleCategory.PERSONAL_IDENTIFIER)
        self.assertTrue(julie_context.is_first_party)
        self.assertFalse(julie_context.is_public_knowledge)

        # Case B: "Tell me about Elon Musk"
        prompt_b = "Tell me about Elon Musk and his role at SpaceX."
        result_b = self.shield.analyze_context(prompt_b)
        self.assertEqual(result_b.task_type, TaskType.GENERAL_QA)

        elon_context = next(
            (c for c in result_b.entity_contexts if "elon musk" in c.entity_text.lower()),
            None
        )
        self.assertIsNotNone(elon_context, "Entity 'Elon Musk' was not detected")
        self.assertEqual(elon_context.entity_type, EntityType.PERSON)
        self.assertEqual(elon_context.role_category, RoleCategory.PUBLIC_FIGURE_OR_FACT)
        self.assertFalse(elon_context.is_first_party)
        self.assertTrue(elon_context.is_public_knowledge)
        self.assertTrue(elon_context.is_task_relevant)

    def test_task_relevant_company_in_email(self):
        prompt = (
            "Write a professional email to ABC Technologies regarding my application for the London office."
        )
        result = self.shield.analyze_context(prompt)
        self.assertEqual(result.task_type, TaskType.EMAIL_GENERATION)

        org_context = next(
            (c for c in result.entity_contexts if c.entity_type == EntityType.ORGANIZATION),
            None
        )
        self.assertIsNotNone(org_context)
        self.assertEqual(org_context.entity_text, "ABC Technologies")
        self.assertTrue(org_context.is_task_relevant)
        self.assertEqual(org_context.role_category, RoleCategory.TASK_RELEVANT_ENTITY)

    def test_technical_credentials_context(self):
        prompt = "Connect with password='SecretPass!123' and api_key='sk-live-1234567890abcdef12345678'."
        result = self.shield.analyze_context(prompt)

        for ec in result.entity_contexts:
            self.assertEqual(ec.role_category, RoleCategory.TECHNICAL_CREDENTIAL)
            self.assertFalse(ec.is_public_knowledge)
            self.assertTrue(ec.is_first_party)

    def test_public_location_qa(self):
        prompt = "What is the capital of France?"
        result = self.shield.analyze_context(prompt)
        self.assertEqual(result.task_type, TaskType.GENERAL_QA)

        loc_context = next(
            (c for c in result.entity_contexts if c.entity_type == EntityType.LOCATION),
            None
        )
        self.assertIsNotNone(loc_context)
        self.assertEqual(loc_context.role_category, RoleCategory.PUBLIC_FIGURE_OR_FACT)
        self.assertTrue(loc_context.is_public_knowledge)


    def test_entire_full_name_masked(self):
        """
        Verify that the ENTIRE full name (first, middle, last name, titles) is detected
        and masked as a single entity, never leaving surnames or middle names behind.
        """
        # 1. First + Last name
        prompt1 = "My name is Julie Johnson and my email is julie@example.com."
        res1 = self.shield.sanitize_baseline(prompt1)
        self.assertEqual(res1.sanitized_prompt, "My name is <PERSON_1> and my email is <EMAIL_1>.")
        self.assertEqual(res1.mapping.get("<PERSON_1>"), "Julie Johnson")
        self.assertNotIn("Johnson", res1.sanitized_prompt)

        # 2. Lowercase full name
        prompt2 = "my name is julie johnson and my phone is 9876543210."
        res2 = self.shield.sanitize_baseline(prompt2)
        self.assertEqual(res2.sanitized_prompt, "my name is <PERSON_1> and my phone is <PHONE_1>.")
        self.assertEqual(res2.mapping.get("<PERSON_1>"), "julie johnson")

        # 3. Full name with title and middle name
        prompt3 = "I am Dr. Julie Ann Johnson from London."
        res3 = self.shield.sanitize_baseline(prompt3)
        self.assertEqual(res3.sanitized_prompt, "I am <PERSON_1> from <LOCATION_1>.")
        self.assertEqual(res3.mapping.get("<PERSON_1>"), "Dr. Julie Ann Johnson")
        self.assertNotIn("Johnson", res3.sanitized_prompt)
        self.assertNotIn("Ann", res3.sanitized_prompt)


if __name__ == "__main__":
    unittest.main()
