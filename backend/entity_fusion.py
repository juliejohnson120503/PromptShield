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
from backend import entity_validators as validators

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

        # Identify existing placeholder spans (e.g. {{PERSON_1}}, <PERSON_1>) to prevent re-detecting them
        placeholder_spans = [
            (m.start(), m.end()) for m in re.finditer(r"(?:\{\{|\<)[A-Za-z0-9_]+(?:\}\}|\>)", prompt)
        ]

        loc_lookup = {loc.lower(): loc for loc in config.KNOWN_LOCATIONS}
        org_lookup = {org.lower(): org for org in config.KNOWN_ORGANIZATIONS}

        # Step 0a: Expand compound Organization + Location spans (e.g. "Cognizant Hyderabad" -> "Cognizant" + "Hyderabad")
        expanded_candidates: List[DetectedEntity] = []
        for ent in candidates:
            if ent.entity_type in (EntityType.ORGANIZATION, EntityType.PERSON):
                text_clean = ent.text.strip().rstrip(",")
                words = text_clean.split()
                if len(words) >= 2:
                    last_word_lower = words[-1].lower()
                    last_two_words_lower = " ".join(words[-2:]).lower() if len(words) >= 3 else None
                    split_loc_text = None
                    split_org_text = None

                    if last_two_words_lower and last_two_words_lower in loc_lookup:
                        split_loc_text = " ".join(words[-2:])
                        split_org_text = " ".join(words[:-2])
                    elif last_word_lower in loc_lookup:
                        split_loc_text = words[-1]
                        split_org_text = " ".join(words[:-1])

                    if split_loc_text and split_org_text:
                        org_idx = ent.text.find(split_org_text)
                        loc_idx = ent.text.rfind(split_loc_text)
                        if org_idx != -1 and loc_idx != -1 and org_idx + len(split_org_text) <= loc_idx:
                            org_start = ent.start + org_idx
                            org_end = org_start + len(split_org_text)
                            loc_start = ent.start + loc_idx
                            loc_end = loc_start + len(split_loc_text)

                            org_ent = DetectedEntity(
                                text=split_org_text,
                                entity_type=EntityType.ORGANIZATION,
                                start=org_start,
                                end=org_end,
                                confidence=ent.confidence,
                                source=ent.source,
                                detector=ent.detector,
                                normalized_value=normalize_entity_value(EntityType.ORGANIZATION, split_org_text),
                                metadata=dict(ent.metadata) if ent.metadata else {},
                            )
                            loc_ent = DetectedEntity(
                                text=split_loc_text,
                                entity_type=EntityType.LOCATION,
                                start=loc_start,
                                end=loc_end,
                                confidence=ent.confidence,
                                source=ent.source,
                                detector="fusion_compound_split_loc",
                                normalized_value=normalize_entity_value(EntityType.LOCATION, split_loc_text),
                                metadata=dict(ent.metadata) if ent.metadata else {},
                            )
                            expanded_candidates.append(org_ent)
                            expanded_candidates.append(loc_ent)
                            continue

            expanded_candidates.append(ent)

        # Step 0b: Compound Address Merging
        # Merge street address + adjacent (comma/postal-code separated) city/country/postal_code
        # e.g. "10 Downing Street" + ", London, SW1A 2AA" -> "10 Downing Street, London, SW1A 2AA"
        # e.g. "12 Rue de la Paix" + ", 75002 Paris, France" -> "12 Rue de la Paix, 75002 Paris, France"
        sorted_cands = sorted(expanded_candidates, key=lambda e: (e.start, -(e.end - e.start)))
        absorbed_indices = set()
        merged_address_cands: List[DetectedEntity] = []

        for i, ent in enumerate(sorted_cands):
            if i in absorbed_indices:
                continue

            if ent.entity_type == EntityType.ADDRESS:
                current_start = ent.start
                current_end = ent.end
                current_text = ent.text
                best_conf = ent.confidence

                j = i + 1
                while j < len(sorted_cands):
                    next_ent = sorted_cands[j]
                    if j in absorbed_indices:
                        j += 1
                        continue

                    # If next entity is contained within current address span
                    if next_ent.start >= current_start and next_ent.end <= current_end:
                        if next_ent.entity_type in (EntityType.LOCATION, EntityType.ADDRESS):
                            absorbed_indices.add(j)
                        j += 1
                        continue

                    # If next entity is adjacent, separated only by commas, spaces, or postal code
                    # Only merge compound address into city/country if the prompt section actually contains postal code digits
                    if next_ent.start >= current_end:
                        gap_text = prompt[current_end:next_ent.start]
                        is_valid_gap = bool(re.fullmatch(r"[,\s]*(?:\d{4,6}|[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})?[,\s]*", gap_text))
                        has_postal_nearby = bool(re.search(r"\b(?:\d{4,6}|[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})\b", prompt[current_start:min(len(prompt), current_start + 120)]))
                        if is_valid_gap and has_postal_nearby and len(gap_text) <= 25 and next_ent.entity_type in (EntityType.LOCATION, EntityType.ADDRESS):
                            absorbed_indices.add(j)
                            current_end = next_ent.end
                            current_text = prompt[current_start:current_end]
                            best_conf = max(best_conf, next_ent.confidence)
                            j += 1
                            continue
                        else:
                            break
                    else:
                        break

                if current_end != ent.end:
                    # Also absorb any trailing postal code digits directly following the address components
                    post_match = re.match(r"^([,\s]+(?:\d{4,6}|[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}))\b", prompt[current_end:])
                    if post_match:
                        current_end += len(post_match.group(1))
                        current_text = prompt[current_start:current_end]

                    merged_addr = DetectedEntity(
                        text=current_text,
                        entity_type=EntityType.ADDRESS,
                        start=current_start,
                        end=current_end,
                        confidence=best_conf,
                        source=ent.source,
                        detector="fusion_compound_address",
                        normalized_value=normalize_entity_value(EntityType.ADDRESS, current_text),
                        metadata=dict(ent.metadata) if ent.metadata else {},
                    )
                    merged_address_cands.append(merged_addr)
                    continue

            merged_address_cands.append(ent)

        # Step 0c: Compound Person Merging
        # Merge adjacent PERSON components (e.g. "elon" + "musk" -> "elon musk", "rahul" + "s7" -> "rahul s7")
        # separated only by whitespace or a single hyphen/apostrophe
        sorted_person_cands = sorted(merged_address_cands, key=lambda e: (e.start, -(e.end - e.start)))
        absorbed_person_indices = set()
        merged_person_cands: List[DetectedEntity] = []

        for i, ent in enumerate(sorted_person_cands):
            if i in absorbed_person_indices:
                continue

            if ent.entity_type == EntityType.PERSON:
                current_start = ent.start
                current_end = ent.end
                current_text = ent.text
                best_conf = ent.confidence

                j = i + 1
                while j < len(sorted_person_cands):
                    next_ent = sorted_person_cands[j]
                    if j in absorbed_person_indices:
                        j += 1
                        continue

                    # If next entity is contained within current span
                    if next_ent.start >= current_start and next_ent.end <= current_end:
                        if next_ent.entity_type == EntityType.PERSON:
                            absorbed_person_indices.add(j)
                        j += 1
                        continue

                    # If next entity is adjacent PERSON separated only by space or hyphen
                    if next_ent.start >= current_end:
                        gap_text = prompt[current_end:next_ent.start]
                        if re.fullmatch(r"[\s\-']+", gap_text) and len(gap_text) <= 5 and next_ent.entity_type == EntityType.PERSON:
                            combined = prompt[current_start:next_ent.end]
                            is_valid, _ = validators.validate_person(combined)
                            if is_valid:
                                absorbed_person_indices.add(j)
                                current_end = next_ent.end
                                current_text = combined
                                best_conf = max(best_conf, next_ent.confidence)
                                j += 1
                                continue
                            else:
                                break
                        else:
                            break
                    else:
                        break

                if current_end != ent.end:
                    merged_person = DetectedEntity(
                        text=current_text,
                        entity_type=EntityType.PERSON,
                        start=current_start,
                        end=current_end,
                        confidence=best_conf,
                        source=ent.source,
                        detector="fusion_compound_person",
                        normalized_value=normalize_entity_value(EntityType.PERSON, current_text),
                        metadata=dict(ent.metadata) if ent.metadata else {},
                    )
                    merged_person_cands.append(merged_person)
                    continue

            merged_person_cands.append(ent)

        for ent in merged_person_cands:
            # Skip any detection that overlaps with an existing placeholder token
            if any(not (ent.end <= ps_start or ent.start >= ps_end) for ps_start, ps_end in placeholder_spans):
                continue

            # Discard excluded tokens falsely tagged as named entities (e.g. 'IP', 'email', 'password')
            if validators.is_excluded_token(ent.text):
                continue

            # Trim trailing generic team words from ORGANIZATION spans (e.g. "Infosys Support Team" -> "Infosys")
            if ent.entity_type == EntityType.ORGANIZATION:
                for suffix in (" support team", " network team", " operations team", " security team", " team"):
                    if ent.text.lower().endswith(suffix):
                        prefix = ent.text[:-len(suffix)].rstrip()
                        if prefix.lower() in org_lookup or len(prefix) >= 3:
                            ent.text = prefix
                            ent.end = ent.start + len(prefix)
                            ent.normalized_value = normalize_entity_value(EntityType.ORGANIZATION, prefix)
                        break

            # Gazetteer-assisted relabeling for known entities misclassified by statistical NER
            text_norm = ent.text.strip().lower()
            if text_norm in loc_lookup and ent.entity_type in (EntityType.PERSON, EntityType.ORGANIZATION):
                ent.entity_type = EntityType.LOCATION
                ent.normalized_value = normalize_entity_value(EntityType.LOCATION, ent.text)
            elif text_norm in org_lookup and ent.entity_type in (EntityType.PERSON, EntityType.LOCATION):
                ent.entity_type = EntityType.ORGANIZATION
                ent.normalized_value = normalize_entity_value(EntityType.ORGANIZATION, ent.text)

            # Reclassify or discard invalid EMAIL without '@'
            if ent.entity_type == EntityType.EMAIL:
                is_valid, cleaned_email = validators.validate_email(ent.text)
                if not is_valid:
                    if validators.has_corporate_suffix(ent.text):
                        ent.entity_type = EntityType.ORGANIZATION
                        ent.normalized_value = normalize_entity_value(EntityType.ORGANIZATION, ent.text)
                    else:
                        continue
                elif cleaned_email != ent.text:
                    ent.text = cleaned_email
                    ent.end = ent.start + len(cleaned_email)
                    ent.normalized_value = normalize_entity_value(EntityType.EMAIL, ent.text)

            # Discard invalid IP_ADDRESS without '.' or ':'
            if ent.entity_type == EntityType.IP_ADDRESS and not validators.validate_ip_address(ent.text):
                continue

            # Reclassify PERSON entities with corporate suffixes to ORGANIZATION
            if ent.entity_type == EntityType.PERSON:
                if validators.has_corporate_suffix(ent.text):
                    ent.entity_type = EntityType.ORGANIZATION
                    ent.normalized_value = normalize_entity_value(EntityType.ORGANIZATION, ent.text)

            # Reclassify ticket/authorization prefixes falsely tagged as DRIVER_LICENSE by neural NER
            if ent.entity_type == EntityType.DRIVER_LICENSE and re.match(r"^(?:TKT|TICKET|SR|INC|CASE|DOCK|LIC)[-_]\d+$", ent.text, re.IGNORECASE):
                ent.entity_type = EntityType.TICKET_ID
                ent.normalized_value = normalize_entity_value(EntityType.TICKET_ID, ent.text)

            # Format validation for CREDIT_CARD
            if ent.entity_type == EntityType.CREDIT_CARD and not validators.validate_credit_card(ent.text):
                continue

            # Format validation for PHONE
            if ent.entity_type == EntityType.PHONE:
                before_window = prompt[max(0, ent.start - 25):ent.start].lower()
                if re.search(r"\brouting\s*(?:number|no\.?|#)?\s*$", before_window):
                    continue
                if not validators.validate_phone(ent.text):
                    continue

            # Format validation for LOCATION
            if ent.entity_type == EntityType.LOCATION and not validators.validate_location(ent.text):
                continue

            # Format validation for BANK_ACCOUNT
            if ent.entity_type == EntityType.BANK_ACCOUNT:
                before_window = prompt[max(0, ent.start - 25):ent.start].lower()
                if re.search(r"\brouting\s*(?:number|no\.?|#)?\s*$", before_window):
                    continue
                if not validators.validate_bank_account(ent.text):
                    continue

            # Format validation for project-specific IDs and usernames
            if ent.entity_type in (EntityType.CUSTOMER_ID, EntityType.ORDER_ID, EntityType.TICKET_ID, EntityType.USER_ID, EntityType.USERNAME):
                if not validators.validate_structured_id(ent.text, prompt, ent.start):
                    continue

            # Format validation for DATE
            if ent.entity_type == EntityType.DATE:
                if not validators.validate_date(ent.text, prompt, ent.start, ent.end):
                    continue

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

                # 4. Leading connector/preposition/salutation words
                CONNECTOR_WORDS = {
                    "involving", "regarding", "concerning", "about", "from", "with", "for", "to", "at", "by",
                    "order", "ticket", "user", "customer", "the", "a", "an",
                    "dear", "email", "mail", "contact", "call", "reach", "send", "tell", "attn", "attention",
                    "update", "delete", "deploy", "check", "fetch", "prepare", "myself", "im", "i'm",
                }
                words = ent.text.split(None, 1)
                if len(words) > 1 and words[0].lower() in CONNECTOR_WORDS:
                    sub_idx = ent.text.find(words[1], len(words[0]))
                    if sub_idx != -1:
                        first_word = words[0].lower()
                        ent.start += sub_idx
                        ent.text = ent.text[sub_idx:]
                        if first_word in ("dear", "attn", "attention") and ent.entity_type == EntityType.ORGANIZATION:
                            ent.entity_type = EntityType.PERSON
                        ent.normalized_value = normalize_entity_value(ent.entity_type, ent.text)

                # 5. Trailing action verbs and prepositions on named entities
                if ent.entity_type in (EntityType.PERSON, EntityType.ORGANIZATION):
                    TRAILING_ACTION_VERBS = {
                        "leads", "lead", "leading", "works", "worked", "working", "said",
                        "contacted", "called", "emailed", "wrote", "joins", "joined", "resides",
                        "likes", "liked", "loves", "loved",
                        "in", "at", "for", "to", "from", "on", "with", "by", "of",
                    }
                    words_all = ent.text.split()
                    if len(words_all) > 1 and words_all[-1].lower() in TRAILING_ACTION_VERBS:
                        last_word = words_all[-1]
                        sub_idx = ent.text.rfind(last_word)
                        if sub_idx != -1:
                            trimmed_span = ent.text[:sub_idx].rstrip()
                            ent.end = ent.start + len(trimmed_span)
                            ent.text = trimmed_span
                            if ent.entity_type == EntityType.ORGANIZATION and not validators.has_corporate_suffix(ent.text) and ent.text.lower() not in org_lookup:
                                ent.entity_type = EntityType.PERSON
                            ent.normalized_value = normalize_entity_value(ent.entity_type, ent.text)

            # Clean PERSON entities
            if ent.entity_type == EntityType.PERSON:
                is_valid, cleaned_p = validators.validate_person(ent.text)
                if not is_valid or validators.is_excluded_token(cleaned_p):
                    continue

                from backend.spacy_detector import PERSON_STOP_WORDS
                trimmed = validators.trim_person_stop_words(cleaned_p, PERSON_STOP_WORDS)
                if not trimmed:
                    continue
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

        # --- Step 2.5: ADDRESS subsumption ---
        # Suppress any LOCATION whose span is contained inside an accepted ADDRESS containing postal digits
        address_spans = [(e.start, e.end) for e in resolved if e.entity_type == EntityType.ADDRESS and re.search(r"\d{4,6}|[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}", e.text)]
        if address_spans:
            resolved = [
                e for e in resolved
                if not (e.entity_type == EntityType.LOCATION and any(a_start <= e.start and e.end <= a_end for a_start, a_end in address_spans))
            ]

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
