"""
PromptShield AI — Entity Fusion Engine (Phase 1).

Combines raw detection results from multiple detector subsystems
(regex, presidio, spacy/heuristic_ner) into a single, clean, non-
overlapping list of DetectedEntity objects.

Fusion Strategy (deterministic)
--------------------------------
1. EXACT-SPAN DEDUPLICATION
   When two or more entities share the same [start, end) span:
   a. If entity_types MATCH  → merge into one entity:
      - Use the highest confidence score.
      - Record all contributing sources in metadata["contributing_sources"].
      - Keep the detector label from the highest-confidence source.
   b. If entity_types DIFFER → choose by entity-type priority
      (config.ENTITY_TYPE_PRIORITY).  Structured types beat generic NER.
      If priorities are equal, choose by source priority
      (config.SOURCE_PRIORITY), then by confidence.

2. OVERLAPPING-SPAN RESOLUTION
   When spans partially overlap  max(s1,s2) < min(e1,e2):
   - Keep the entity with the higher entity-type priority.
   - Tie-break: higher source priority → higher confidence → longer span.
   - The losing entity is discarded (not silently — see metadata below).

3. OUTPUT
   - Non-overlapping entities sorted by ascending start offset.
   - Each merged entity carries metadata["contributing_sources"] listing
     every detector subsystem that contributed to it.
   - Character-offset integrity is verified: prompt[start:end] must equal
     entity.text (entities failing this check are discarded with a warning).

Design decisions
----------------
• Structured / credential detections (regex, Luhn, Presidio) are
  preferred over generic NER because they are format-specific and
  have explicit, auditable decision criteria.
• When Presidio and regex agree on the same span and type, confidence is
  upgraded to the higher of the two scores.
• This module does NOT decide whether to mask any entity.  It only
  unifies the detection results for downstream consumption.

Limitations
-----------
• Fusion works at the character-span level; semantic disambiguation
  (e.g. whether "Apple" is the company or the fruit) is a Phase 2 task.
• A PERSON span detected by spaCy and a PASSWORD span detected by regex
  at the same position is an unusual edge-case; the structured type wins.
"""

import logging
from typing import List, Dict, Tuple, Optional

from backend.models import EntityType, DetectedEntity
from backend import config

logger = logging.getLogger(__name__)


def _entity_sort_key(entity: DetectedEntity) -> Tuple:
    """
    Composite sort key for ranking entities when resolving conflicts.
    Higher tuple value = preferred entity.

    Priority order (descending importance):
      1. Entity-type priority (structured > NER)
      2. Source priority     (regex > presidio > spacy > heuristic)
      3. Confidence score
      4. Span length          (longer span wins on tie)
    """
    type_p = config.ENTITY_TYPE_PRIORITY.get(entity.entity_type.value, 0)
    src_p = config.SOURCE_PRIORITY.get(entity.source, 0)
    span_len = entity.end - entity.start
    return (type_p, src_p, entity.confidence, span_len)


def _spans_overlap(a: DetectedEntity, b: DetectedEntity) -> bool:
    """Return True if entities a and b have any character overlap."""
    return max(a.start, b.start) < min(a.end, b.end)


def _spans_exact(a: DetectedEntity, b: DetectedEntity) -> bool:
    """Return True if entities a and b share the identical span."""
    return a.start == b.start and a.end == b.end


class EntityFusionEngine:
    """
    Combines raw candidate entities from multiple detectors into a
    single, non-overlapping, sorted list of DetectedEntity objects.
    """

    def fuse(
        self,
        candidates: List[DetectedEntity],
        prompt: str,
    ) -> List[DetectedEntity]:
        """
        Merge *candidates* from all detectors and return a unified list.

        Parameters
        ----------
        candidates : flat list of DetectedEntity from all detectors
        prompt     : the original prompt string (used for integrity checks)

        Returns
        -------
        List[DetectedEntity] — non-overlapping, sorted by start offset.
        """
        if not candidates:
            return []

        # --- Step 0: integrity filter ---
        valid: List[DetectedEntity] = []
        for ent in candidates:
            if 0 <= ent.start < ent.end <= len(prompt):
                actual = prompt[ent.start: ent.end]
                if actual == ent.text:
                    valid.append(ent)
                else:
                    logger.debug(
                        "[Fusion] Integrity fail: entity text %r != "
                        "prompt[%d:%d]=%r — discarded.",
                        ent.text, ent.start, ent.end, actual,
                    )
            else:
                logger.debug(
                    "[Fusion] Out-of-range span [%d, %d) for text %r — discarded.",
                    ent.start, ent.end, ent.text,
                )

        if not valid:
            return []

        # --- Step 1: group by exact span, merge same-span entities ---
        merged = self._merge_exact_spans(valid)

        # --- Step 2: resolve overlapping spans ---
        resolved = self._resolve_overlaps(merged)

        # --- Step 3: sort by start offset ---
        return sorted(resolved, key=lambda e: e.start)

    # ------------------------------------------------------------------

    def _merge_exact_spans(
        self, candidates: List[DetectedEntity]
    ) -> List[DetectedEntity]:
        """
        Group entities that share the same [start, end) span and merge
        them into a single representative entity.

        Same type:    pick highest confidence, union sources.
        Diff type:    pick by entity-type priority, then source, then conf.
        """
        # Group by span
        span_groups: Dict[Tuple[int, int], List[DetectedEntity]] = {}
        for ent in candidates:
            key = (ent.start, ent.end)
            span_groups.setdefault(key, []).append(ent)

        merged: List[DetectedEntity] = []
        for (start, end), group in span_groups.items():
            if len(group) == 1:
                merged.append(group[0])
                continue

            # Sort group: best entity first
            group_sorted = sorted(group, key=_entity_sort_key, reverse=True)
            winner = group_sorted[0]

            # Collect all contributing sources
            contributing_sources = list(
                dict.fromkeys(e.source for e in group_sorted)
            )

            # If winner and runner-up share the same type, upgrade confidence
            # to the max across all same-type contributors.
            same_type_confs = [
                e.confidence
                for e in group_sorted
                if e.entity_type == winner.entity_type
            ]
            best_conf = max(same_type_confs) if same_type_confs else winner.confidence

            # Build merged entity (copy winner, update metadata)
            merged_meta = dict(winner.metadata)
            merged_meta["contributing_sources"] = contributing_sources
            if len(group) > 1:
                merged_meta["fusion_note"] = (
                    f"Merged {len(group)} detections from "
                    f"{contributing_sources} — kept highest-priority."
                )

            merged_ent = DetectedEntity(
                text=winner.text,
                entity_type=winner.entity_type,
                start=start,
                end=end,
                normalized_value=winner.normalized_value,
                confidence=round(best_conf, 4),
                source=winner.source,
                detector=winner.detector,
                metadata=merged_meta,
            )
            merged.append(merged_ent)

        return merged

    def _resolve_overlaps(
        self, candidates: List[DetectedEntity]
    ) -> List[DetectedEntity]:
        """
        Greedy sweep to eliminate overlapping spans.

        Sort all candidates by descending priority key, then iterate:
        accept a candidate only if it does not overlap with any already
        accepted entity.  This prefers structured / high-confidence
        detections over generic NER.
        """
        priority_sorted = sorted(
            candidates, key=_entity_sort_key, reverse=True
        )
        accepted: List[DetectedEntity] = []

        for cand in priority_sorted:
            overlaps_accepted = any(
                _spans_overlap(cand, acc) for acc in accepted
            )
            if not overlaps_accepted:
                accepted.append(cand)
            else:
                logger.debug(
                    "[Fusion] Overlap dropped: %r (%s @ [%d,%d)) — "
                    "superseded by higher-priority entity.",
                    cand.text, cand.entity_type.value, cand.start, cand.end,
                )

        return accepted
