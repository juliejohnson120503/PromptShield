"""
Unit tests for TaskClassifier (Phase 2).
Verifies accurate classification across the 8 canonical task categories.
"""

import unittest
from backend.task_classifier import TaskClassifier
from backend.models import TaskType


class TestTaskClassifier(unittest.TestCase):

    def setUp(self):
        self.classifier = TaskClassifier()

    def test_classify_email_generation(self):
        prompts = [
            "Write a professional email to ABC Technologies regarding my application.",
            "Draft a follow-up email to the hiring manager.",
            "Compose a cold email pitching our AI security software.",
        ]
        for p in prompts:
            task_type, conf, _ = self.classifier.classify(p)
            self.assertEqual(task_type, TaskType.EMAIL_GENERATION, f"Failed on prompt: {p}")
            self.assertGreaterEqual(conf, 0.7)

    def test_classify_code_generation(self):
        prompts = [
            "Write python code to implement a binary search tree.",
            "Generate a javascript function to validate an email address.",
            "Debug this react component and fix the state update bug.",
        ]
        for p in prompts:
            task_type, conf, _ = self.classifier.classify(p)
            self.assertEqual(task_type, TaskType.CODE_GENERATION, f"Failed on prompt: {p}")

    def test_classify_summarization(self):
        prompts = [
            "Summarize the key findings of this research paper.",
            "Give me a brief overview and bullet points of the following article.",
            "TLDR of the quarterly financial statement.",
        ]
        for p in prompts:
            task_type, conf, _ = self.classifier.classify(p)
            self.assertEqual(task_type, TaskType.SUMMARIZATION, f"Failed on prompt: {p}")

    def test_classify_translation(self):
        prompts = [
            "Translate the following welcome letter into French.",
            "What is the Spanish translation of 'security layer'?",
        ]
        for p in prompts:
            task_type, conf, _ = self.classifier.classify(p)
            self.assertEqual(task_type, TaskType.TRANSLATION, f"Failed on prompt: {p}")

    def test_classify_data_analysis(self):
        prompts = [
            "Analyze the dataset to identify customer churn trends.",
            "Calculate the mean and standard deviation of these sales metrics.",
        ]
        for p in prompts:
            task_type, conf, _ = self.classifier.classify(p)
            self.assertEqual(task_type, TaskType.DATA_ANALYSIS, f"Failed on prompt: {p}")

    def test_classify_document_generation(self):
        prompts = [
            "Create a resume for an entry-level software engineer.",
            "Draft an agreement for a freelance consulting contract.",
            "Write an essay discussing the future of privacy in AI.",
        ]
        for p in prompts:
            task_type, conf, _ = self.classifier.classify(p)
            self.assertEqual(task_type, TaskType.DOCUMENT_GENERATION, f"Failed on prompt: {p}")

    def test_classify_general_qa(self):
        prompts = [
            "What is the capital of India?",
            "Tell me about Elon Musk and SpaceX.",
            "Who was Albert Einstein and what was his theory of relativity?",
            "Explain how neural networks learn from data.",
        ]
        for p in prompts:
            task_type, conf, _ = self.classifier.classify(p)
            self.assertEqual(task_type, TaskType.GENERAL_QA, f"Failed on prompt: {p}")

    def test_classify_general_chat(self):
        prompts = [
            "Hello, how are you today?",
            "Good morning assistant, nice to meet you.",
        ]
        for p in prompts:
            task_type, conf, _ = self.classifier.classify(p)
            self.assertEqual(task_type, TaskType.GENERAL_CHAT, f"Failed on prompt: {p}")


if __name__ == "__main__":
    unittest.main()
