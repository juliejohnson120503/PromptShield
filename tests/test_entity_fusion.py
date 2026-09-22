"""
Unit tests for EntityFusionEngine — Phase 1 fusion and deduplication.

Tests cover:
  - Exact-span deduplication (same entity type from multiple sources)
  - Exact-span type conflict resolution (structured > NER)
  - Overlapping-span resolution
  - Source priority: regex > presidio > spacy
  - contributing_sources metadata populated after merge
  - Character-offset integrity check (malformed entity discarded)
  - Output sorted by ascending start offset
  - Empty/single candidate edge cases
"""

import unittest
from backend.entity_fusion import EntityFusionEngine
from backend.models import EntityType, DetectedEntity


def make_entity(
    text: str,
    entity_type: EntityType,
    start: int,
    end: int,
    confidence: float = 0.90,
    source: str = "regex",
    detector: str = "test",
) -> DetectedEntity:
    return DetectedEntity(
        text=text,
        entity_type=entity_type,
        start=start,
        end=end,
        normalized_value=text.lower(),
        confidence=confidence,
        source=source,
        detector=detector,
    )


PROMPT = "My email is julie@gmail.com and phone 9876543210."


class TestEntityFusionEngine(unittest.TestCase):

    def setUp(self):
        self.fusion = EntityFusionEngine()

    # -------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------
    def test_empty_candidates(self):
        result = self.fusion.fuse([], PROMPT)
        self.assertEqual(result, [])

    def test_single_candidate_passthrough(self):
        ent = make_entity("julie@gmail.com", EntityType.EMAIL, 12, 27)
        result = self.fusion.fuse([ent], PROMPT)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "julie@gmail.com")

    # -------------------------------------------------------
    # Exact-span deduplication — same type
    # -------------------------------------------------------
    def test_exact_span_same_type_merged_to_one(self):
        """
        Regex and Presidio both detect julie@gmail.com as EMAIL.
        Fusion must produce exactly ONE entity, not two.
        """
        regex_ent = make_entity(
            "julie@gmail.com", EntityType.EMAIL, 12, 27,
            confidence=0.98, source="regex"
        )
        presidio_ent = make_entity(
            "julie@gmail.com", EntityType.EMAIL, 12, 27,
            confidence=0.85, source="presidio"
        )
        result = self.fusion.fuse([regex_ent, presidio_ent], PROMPT)
        email_results = [e for e in result if e.entity_type == EntityType.EMAIL]
        self.assertEqual(len(email_results), 1)
        # Best confidence should win
        self.assertEqual(email_results[0].confidence, 0.98)

    def test_exact_span_merge_records_contributing_sources(self):
        """After merging, metadata must list all contributing detector sources."""
        r1 = make_entity("julie@gmail.com", EntityType.EMAIL, 12, 27,
                         source="regex", confidence=0.98)
        r2 = make_entity("julie@gmail.com", EntityType.EMAIL, 12, 27,
                         source="presidio", confidence=0.85)
        result = self.fusion.fuse([r1, r2], PROMPT)
        email = [e for e in result if e.entity_type == EntityType.EMAIL][0]
        sources = email.metadata.get("contributing_sources", [])
        self.assertIn("regex", sources)
        self.assertIn("presidio", sources)

    # -------------------------------------------------------
    # Exact-span — different types (structured > NER)
    # -------------------------------------------------------
    def test_exact_span_different_type_structured_wins(self):
        """
        If EMAIL and PERSON share the same span, EMAIL (higher priority) wins.
        This is an unusual edge-case but must be deterministic.
        """
        email_ent = make_entity(
            "julie@gmail.com", EntityType.EMAIL, 12, 27,
            confidence=0.98, source="regex"
        )
        person_ent = make_entity(
            "julie@gmail.com", EntityType.PERSON, 12, 27,
            confidence=0.90, source="spacy"
        )
        result = self.fusion.fuse([email_ent, person_ent], PROMPT)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].entity_type, EntityType.EMAIL)

    # -------------------------------------------------------
    # Overlapping spans
    # -------------------------------------------------------
    def test_overlapping_spans_structured_beats_ner(self):
        """
        If a PHONE span and a PERSON span overlap, PHONE (higher priority) wins.
        """
        # Construct a dedicated short prompt for this test
        prompt = "9876543210 is my number"
        phone_ent = make_entity(
            "9876543210", EntityType.PHONE, 0, 10,
            confidence=0.92, source="regex"
        )
        person_ent = make_entity(
            "9876543210", EntityType.PERSON, 0, 10,
            confidence=0.85, source="spacy"
        )
        result = self.fusion.fuse([phone_ent, person_ent], prompt)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].entity_type, EntityType.PHONE)

    def test_non_overlapping_spans_both_kept(self):
        """Non-overlapping entities must both appear in the result."""
        email_ent = make_entity("julie@gmail.com", EntityType.EMAIL, 12, 27)
        phone_ent = make_entity("9876543210", EntityType.PHONE, 38, 48)
        result = self.fusion.fuse([email_ent, phone_ent], PROMPT)
        self.assertEqual(len(result), 2)

    # -------------------------------------------------------
    # Output ordering
    # -------------------------------------------------------
    def test_output_sorted_by_start_offset(self):
        """Fused result must be sorted by ascending start offset."""
        phone_ent = make_entity("9876543210", EntityType.PHONE, 38, 48)
        email_ent = make_entity("julie@gmail.com", EntityType.EMAIL, 12, 27)
        result = self.fusion.fuse([phone_ent, email_ent], PROMPT)
        starts = [e.start for e in result]
        self.assertEqual(starts, sorted(starts))

    # -------------------------------------------------------
    # Integrity check
    # -------------------------------------------------------
    def test_malformed_entity_discarded(self):
        """
        Entity whose text does not match prompt[start:end] must be
        silently discarded by the integrity check.
        """
        bad_entity = make_entity(
            "WRONG_TEXT", EntityType.EMAIL, 12, 27,
            source="regex"
        )
        result = self.fusion.fuse([bad_entity], PROMPT)
        self.assertEqual(len(result), 0)

    def test_out_of_range_entity_discarded(self):
        """Entity with start/end out of prompt length must be discarded."""
        out_of_range = make_entity(
            "julie@gmail.com", EntityType.EMAIL, 200, 215,
            source="regex"
        )
        result = self.fusion.fuse([out_of_range], PROMPT)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
