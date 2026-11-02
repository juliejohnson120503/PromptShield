"""
PromptShield AI — Client-Side Pre-Screening Test Suite (Step 3).

Validates that:
1. Benign general queries (e.g. "what is the speed of light", "how do airplanes fly")
   are safely identified for 0 ms client-side fast-pass.
2. Sensitive prompts across all entity types (emails, phones, credit cards, passwords,
   API keys, person names, IDs, bank accounts, homoglyphs, zero-width spaces) are
   strictly rejected from fast-pass and forwarded to the backend.
3. Across all 600 prompts in the official benchmark dataset, exactly 0 prompts
   with ground-truth sensitive entities are fast-passed (0% False Negative safety guarantee).
"""

import json
import re
import unittest
from pathlib import Path


SAFE_QUERY_PREFIXES = re.compile(
    r"^(what\s+(?:is|are|was|were|causes?|means?)|"
    r"how\s+(?:to|do|does|can|did|works?|much|many)|"
    r"why\s+(?:is|are|do|does|did)|"
    r"explain|summarize|describe|define|translate|"
    r"write\s+a\s+(?:poem|story|script|code|summary|essay)|"
    r"generate\s+a\s+(?:poem|story|script|code|summary|essay)|"
    r"list\s+(?:the|all|some)?|"
    r"help\s+me\s+(?:understand|learn|debug|fix))\b",
    re.IGNORECASE,
)

SENSITIVE_TRIGGER_WORDS = re.compile(
    r"\b("
    r"my|our|his|her|their|your|i\s*am|i'?m|myself|"
    r"name|called|named|who|who'?s|contact|reach|call|email|mail|phone|mobile|cell|"
    r"pass|password|pwd|secret|key|token|credential|auth|login|pin|code|"
    r"card|credit|debit|bank|account|acc|iban|cvv|exp|balance|payment|"
    r"id|ssn|aadhaar|pan|passport|license|licence|ticket|order|customer|user|patient|medical|mrn|"
    r"live|living|reside|address|street|st|ave|avenue|road|rd|city|state|zip|postal|"
    r"mother|father|wife|husband|son|daughter|sister|brother|boss|doctor|manager|"
    r"rohit|john|alice|bob|rahul|deepak|priya|david|sarah"
    r")\b",
    re.IGNORECASE,
)


def can_fast_pass_prompt(prompt: str) -> bool:
    """
    Python mirror of extension/content/content.js canFastPassPrompt().
    """
    if not prompt or not isinstance(prompt, str):
        return False
    trimmed = prompt.strip()
    # Length and single-line bounds
    if len(trimmed) < 5 or len(trimmed) > 120 or "\n" in trimmed:
        return False
    # Non-ASCII (Unicode, homoglyphs, zero-width steganography)
    if any(ord(c) < 32 or ord(c) > 126 for c in trimmed):
        return False
    # Any digit
    if re.search(r"\d", trimmed):
        return False
    # Any uppercase letter (proper nouns, names, orgs)
    if re.search(r"[A-Z]", trimmed):
        return False
    # Any sensitive delimiters or symbols
    if re.search(r"[@#$%^&*+=<>~`|_\\/{}[\]:;\"]", trimmed):
        return False
    # Sensitive trigger words
    if SENSITIVE_TRIGGER_WORDS.search(trimmed):
        return False
    # Must start with a safe query prefix
    if not SAFE_QUERY_PREFIXES.search(trimmed):
        return False

    return True


class TestClientPreScreening(unittest.TestCase):

    def test_benign_queries_fast_pass(self):
        benign_queries = [
            "what is the speed of light",
            "how do airplanes stay in the air",
            "why is the sky blue",
            "explain how photosynthesis works",
            "translate hello to french",
            "write a poem about winter trees",
            "how does gravity work",
            "summarize the main themes of hamlet",
            "describe the water cycle simply",
            "define the term machine learning",
        ]
        for query in benign_queries:
            with self.subTest(query=query):
                self.assertTrue(
                    can_fast_pass_prompt(query),
                    f"Benign query '{query}' should have been fast-passed (0 ms)!",
                )

    def test_sensitive_prompts_rejected_from_fast_pass(self):
        sensitive_cases = [
            ("what is my password", "First person possessive & credential cue"),
            ("how to contact John Doe", "Capital letters & contact cue"),
            ("what is the price of card 4532", "Digits & card cue"),
            ("send an email to test@example.com", "Symbol @ and email cue"),
            ("explain why ORD-9921 failed", "Capital letters, hyphen, digits"),
            ("what is j\u200be\u200br\u200bi\u200bn@example.com", "Non-ASCII & zero-width chars"),
            ("translate this password: secret", "Symbol : and password cue"),
            ("why is my account balance zero", "First person & financial cues"),
            ("call 9876543210 for support", "Digits & call cue"),
            ("my name is Alice", "First person & name cue"),
            ("this is rohit from engineering", "Does not start with safe query prefix"),
        ]
        for prompt, reason in sensitive_cases:
            with self.subTest(prompt=prompt, reason=reason):
                self.assertFalse(
                    can_fast_pass_prompt(prompt),
                    f"Sensitive prompt '{prompt}' ({reason}) MUST NOT be fast-passed!",
                )

    def test_zero_false_negatives_on_benchmark_ground_truth(self):
        benchmark_path = Path("evaluation/promptshield_benchmark/ground_truth.json")
        self.assertTrue(benchmark_path.exists(), "Benchmark ground_truth.json not found")

        with open(benchmark_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        false_passes = []
        for i, item in enumerate(data):
            has_entities = bool(item.get("entities"))
            if has_entities:
                if can_fast_pass_prompt(item["text"]):
                    false_passes.append((i, item["text"]))

        self.assertEqual(
            len(false_passes),
            0,
            f"SAFETY VIOLATION: {len(false_passes)} benchmark prompts with sensitive entities "
            f"were mistakenly fast-passed! {false_passes}",
        )


if __name__ == "__main__":
    unittest.main()
