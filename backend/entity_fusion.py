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
import re
from typing import List, Dict, Tuple, Optional

from backend.models import EntityType, DetectedEntity
from backend import config

logger = logging.getLogger(__name__)


def _is_descriptor_label(ent: DetectedEntity, prompt: str, all_candidates: List[DetectedEntity]) -> bool:
    """
    Check if an entity is a sensitive field descriptor word (e.g. MAC, IBAN, BIC, SWIFT)
    functioning as a label rather than an actual sensitive entity / organization.
    Narrow contextual check to avoid globally blacklisting legitimate organizations.
    """
    LABEL_WORDS = {"MAC", "IBAN", "BIC", "SWIFT", "MRN", "HIN"}
    clean_text = ent.text.strip().upper()
    if clean_text not in LABEL_WORDS:
        return False

    start, end = ent.start, ent.end

    # 1. Check surrounding text within 40 characters
    after_text = prompt[end:end + 40].lower()
    before_text = prompt[max(0, start - 30):start].lower()

    # Check if followed by descriptor markers: e.g. " address", " number", " id", " code", " is ", " :", " = ", " / "
    # or preceded by "device ", "server ", "network ", "her ", "his ", "my ", "your ", "swift/"
    is_label_context = bool(
        re.match(r"^(?:\s*/\s*[a-z]+)?\s*(?:address|number|no\.?|id|code|member\s+id)?\s*(?:is|was|[:\-#=])\s*", after_text)
        or re.search(r"\b(?:device|server|network|client|my|her|his|your|employee|user|swift/|swift\s*/\s*)\s*$", before_text)
    )

    # 2. Check if there is an adjacent target sensitive entity following within 50 characters
    adjacent_sensitive = any(
        0 <= other.start - end <= 50 and other.entity_type in (
            EntityType.MAC_ADDRESS, EntityType.BANK_ACCOUNT, EntityType.MEDICAL_RECORD,
            EntityType.HEALTH_INSURANCE_ID, EntityType.NATIONAL_ID, EntityType.TAX_ID
        )
        for other in all_candidates if other is not ent
    )

    return is_label_context or adjacent_sensitive


def _entity_sort_key(entity: DetectedEntity) -> Tuple:
    """
    Composite sort key for ranking entities when resolving conflicts.
    Higher tuple value = preferred entity.

    Priority order:
      1. Entity-type priority (structured > NER)
      2. For named entities (PERSON, ORG, LOC): span length (longer full name wins over substring token), then source
      3. For other entities: source priority, then span length
      4. Confidence score
    """
    type_p = config.ENTITY_TYPE_PRIORITY.get(entity.entity_type.value, 0)
    src_p = config.SOURCE_PRIORITY.get(entity.source, 0)
    span_len = entity.end - entity.start
    if entity.entity_type in (EntityType.PERSON, EntityType.ORGANIZATION, EntityType.LOCATION):
        return (type_p, span_len, src_p, entity.confidence)
    return (type_p, src_p, span_len, entity.confidence)


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
        from backend.normalization import normalize_entity_value
        valid: List[DetectedEntity] = []
        for ent in candidates:
            # Discard excluded tokens falsely tagged as named entities (e.g. 'IP', 'email', 'password')
            if ent.entity_type in (EntityType.ORGANIZATION, EntityType.PERSON, EntityType.LOCATION):
                clean_norm = ent.text.strip().lower()
                if clean_norm in config.COMMON_EXCLUDED_TOKENS or (
                    clean_norm.startswith("the ") and clean_norm[4:].strip() in config.COMMON_EXCLUDED_TOKENS
                ):
                    continue

            # Reclassify or discard invalid EMAIL without '@'
            if ent.entity_type == EntityType.EMAIL and "@" not in ent.text:
                if re.search(r"(?i)\b(?:inc\.?|corp\.?|corporation|llc|ltd\.?|limited|co\.?|technologies|solutions|group|holdings|enterprises|systems|studios|labs|foundation|university|institute)\b", ent.text):
                    ent.entity_type = EntityType.ORGANIZATION
                    ent.normalized_value = normalize_entity_value(EntityType.ORGANIZATION, ent.text)
                else:
                    continue

            # Discard invalid IP_ADDRESS without '.' or ':'
            if ent.entity_type == EntityType.IP_ADDRESS and ("." not in ent.text and ":" not in ent.text):
                continue

            # Reclassify PERSON entities with corporate suffixes to ORGANIZATION
            if ent.entity_type == EntityType.PERSON:
                if re.search(r"(?i)\b(?:inc\.?|corp\.?|corporation|llc|ltd\.?|limited|co\.?|technologies|solutions|group|holdings|enterprises|systems|studios|labs|foundation|university|institute)\b", ent.text):
                    ent.entity_type = EntityType.ORGANIZATION
                    ent.normalized_value = normalize_entity_value(EntityType.ORGANIZATION, ent.text)

            # Format validation for BANK_ACCOUNT: must contain digits (do not relabel to ORGANIZATION)
            if ent.entity_type == EntityType.BANK_ACCOUNT:
                has_digits = any(c.isdigit() for c in ent.text)
                if not has_digits:
                    continue

            # Format validation for project-specific IDs: must contain digits and not be common words
            if ent.entity_type in (EntityType.CUSTOMER_ID, EntityType.ORDER_ID, EntityType.TICKET_ID):
                if not any(c.isdigit() for c in ent.text):
                    continue
                if re.search(r"(?i)\b(?:support|service|center|team|help|care|portal|desk|agent|group|staff)\b", ent.text):
                    continue

            # Format validation for DATE: reject isolated month words without date expression context
            if ent.entity_type == EntityType.DATE:
                MONTH_WORDS = {
                    "january", "february", "march", "april", "may", "june",
                    "july", "august", "september", "october", "november", "december",
                    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
                }
                if ent.text.strip().lower() in MONTH_WORDS:
                    w_before = prompt[max(0, ent.start - 25):ent.start]
                    w_after = prompt[ent.end:min(len(prompt), ent.end + 25)]
                    has_nearby_digits = bool(re.search(r"\b\d{1,4}(?:st|nd|rd|th)?\b", w_before + " " + w_after))
                    has_date_prep = bool(re.search(r"(?i)\b(?:in|on|during|dated|since|until|by|before|after|of)\s*$", w_before.strip()))
                    if not (has_nearby_digits or has_date_prep):
                        continue

            # Format validation for EMAIL: strip trailing sentence words (.The) and trailing punctuation
            if ent.entity_type == EntityType.EMAIL:
                email_sent_match = re.match(r"^([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\.([A-Z][a-z]+.*)$", ent.text)
                if email_sent_match:
                    ent.text = email_sent_match.group(1)
                    ent.end = ent.start + len(ent.text)
                    ent.normalized_value = normalize_entity_value(EntityType.EMAIL, ent.text)
                elif ent.text.endswith("."):
                    ent.text = ent.text.rstrip(".")
                    ent.end = ent.start + len(ent.text)
                    ent.normalized_value = normalize_entity_value(EntityType.EMAIL, ent.text)

            # Generalized span boundary cleaning for named/structured entities
            if ent.entity_type in (EntityType.PERSON, EntityType.ORGANIZATION, EntityType.LOCATION, EntityType.ORDER_ID, EntityType.CUSTOMER_ID, EntityType.USER_ID, EntityType.TICKET_ID):
                # 1. Glued lowercase prefix (e.g. "involvingRahul" -> "Rahul")
                glued_match = re.match(r"^([a-z]{2,})([A-Z].*)$", ent.text)
                if glued_match:
                    prefix_len = len(glued_match.group(1))
                    ent.start += prefix_len
                    ent.text = glued_match.group(2)
                    ent.normalized_value = normalize_entity_value(ent.entity_type, ent.text)

                # 2. Leading whitespace
                l_stripped = ent.text.lstrip()
                if len(l_stripped) < len(ent.text):
                    diff = len(ent.text) - len(l_stripped)
                    ent.start += diff
                    ent.text = l_stripped
                    ent.normalized_value = normalize_entity_value(ent.entity_type, ent.text)

                # 3. Trailing whitespace
                r_stripped = ent.text.rstrip()
                if len(r_stripped) < len(ent.text):
                    diff = len(ent.text) - len(r_stripped)
                    ent.end -= diff
                    ent.text = r_stripped
                    ent.normalized_value = normalize_entity_value(ent.entity_type, ent.text)

                # 4. Leading connector/preposition words (e.g. "involving Rahul", "for Infosys", "order ORD-78291")
                CONNECTOR_WORDS = {"involving", "regarding", "concerning", "about", "from", "with", "for", "to", "at", "by", "order", "ticket", "user", "customer", "the", "a", "an"}
                words = ent.text.split(None, 1)
                if len(words) > 1 and words[0].lower() in CONNECTOR_WORDS:
                    sub_idx = ent.text.find(words[1], len(words[0]))
                    if sub_idx != -1:
                        ent.start += sub_idx
                        ent.text = ent.text[sub_idx:]
                        ent.normalized_value = normalize_entity_value(ent.entity_type, ent.text)

            # Clean PERSON entities
            if ent.entity_type == EntityType.PERSON:
                # Strip possessives (e.g. "Priya Nair's" -> "Priya Nair")
                if re.search(r"['’]s?$", ent.text):
                    trimmed = re.sub(r"['’]s?$", "", ent.text).rstrip()
                    if trimmed:
                        ent.text = trimmed
                        ent.end = ent.start + len(trimmed)
                        ent.normalized_value = normalize_entity_value(EntityType.PERSON, trimmed)

                # Reject malformed PERSON spans containing sentence breaks, punctuation sequences, or directive keywords
                if re.search(r"[.!?;:\n]{1,}\s*[-–—]|\.[a-zA-Z]{2,}|\b(?<![A-Z])\.[A-Z]", ent.text):
                    continue
                disqualifying = {
                    "api", "key", "token", "password", "secret", "credential", "database",
                    "server", "ip", "endpoint", "must", "should", "shall", "cannot", "can",
                    "will", "would", "could", "please", "report", "explain", "recommend",
                    "incident", "order", "ticket", "customer", "support",
                }
                words_clean = [re.sub(r"^\W+|\W+$", "", w).lower() for w in ent.text.split()]
                if any(w in disqualifying for w in words_clean):
                    continue

                from backend.spacy_detector import PERSON_STOP_WORDS
                words = ent.text.split()
                while len(words) > 1 and words[-1].lower() in PERSON_STOP_WORDS:
                    words.pop()
                trimmed = " ".join(words)
                if trimmed != ent.text and trimmed:
                    ent.text = trimmed
                    ent.end = ent.start + len(trimmed)
                    ent.normalized_value = normalize_entity_value(EntityType.PERSON, trimmed)

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

        # --- Step 0.5: filter descriptor label words (e.g. MAC, IBAN, BIC) functioning as field labels ---
        filtered: List[DetectedEntity] = []
        for ent in valid:
            if _is_descriptor_label(ent, prompt, valid):
                logger.debug(
                    "[Fusion] Filtered descriptor label %r (%s) at [%d, %d) — acts as field label.",
                    ent.text, ent.entity_type.value, ent.start, ent.end,
                )
            else:
                filtered.append(ent)

        if not filtered:
            return []

        # --- Step 1: group by exact span, merge same-span entities ---
        merged = self._merge_exact_spans(filtered)

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

    @staticmethod
    def format_explainable_report(entities: List[DetectedEntity]) -> str:
        """
        Format detected entities as an explainable report:
        ENTITY | TYPE | SOURCE | CONFIDENCE
        """
        if not entities:
            return "No entities detected."
        lines = [
            f"{'ENTITY':<30} | {'TYPE':<15} | {'SOURCE':<15} | {'CONFIDENCE':<10}",
            "-" * 78,
        ]
        for e in entities:
            src = getattr(e, "source", "unknown")
            conf = f"{e.confidence:.2f}"
            lines.append(f"{e.text:<30} | {e.entity_type.value:<15} | {src:<15} | {conf:<10}")
        return "\n".join(lines)
