"""
Baseline Masker module for PromptShield AI.
Provides foundational placeholder generation and local mapping registration.

IMPORTANT ARCHITECTURAL NOTE:
This module establishes the mechanical baseline for placeholder substitution
and local mapping storage. It blindly masks all detected entities provided to it.
In Phase 4, the full Context-Aware Semantic Masker will supersede this by
evaluating Phase 2 context understanding and Phase 3 policy decisions
to selectively retain low-risk/task-relevant entities and mask only sensitive items.
"""

import uuid
from typing import Dict, List, Optional
from backend.models import EntityType, DetectedEntity, BaselineMaskResult
from backend.mapping_store import MappingStore


class BaselineMasker:
    """
    Foundational baseline masking engine.
    Converts detected entities into structured placeholders (<TYPE_index>)
    and records associations in the local MappingStore.
    """

    # Placeholder tag prefixes
    TYPE_PREFIXES = {
        EntityType.EMAIL: "EMAIL",
        EntityType.PHONE: "PHONE",
        EntityType.CREDIT_CARD: "CREDIT_CARD",
        EntityType.API_KEY: "API_KEY",
        EntityType.PASSWORD: "PASSWORD",
        EntityType.PERSON: "PERSON",
        EntityType.ORGANIZATION: "ORG",
        EntityType.LOCATION: "LOCATION",
        EntityType.DATE: "DATE",
    }

    def __init__(self, mapping_store: Optional[MappingStore] = None):
        self.mapping_store = mapping_store or MappingStore()

    def mask(
        self,
        prompt: str,
        entities: List[DetectedEntity],
        session_id: Optional[str] = None,
    ) -> BaselineMaskResult:
        """
        Generate placeholders for all detected entities, replace them in the prompt,
        and register the local mapping.

        Replaces matches by slicing the original text in ascending order (or reverse)
        to prevent offset drift. Reuses the same placeholder for identical normalized entities.
        """
        sid = self.mapping_store.create_session(session_id)

        if not prompt or not entities:
            return BaselineMaskResult(
                original_prompt=prompt,
                sanitized_prompt=prompt,
                entities=[],
                mapping={},
                session_id=sid,
            )

        # Ensure entities are sorted by start position ascending without overlaps
        sorted_entities = sorted(entities, key=lambda e: e.start)

        # Maintain counters per entity type prefix
        type_counters: Dict[str, int] = {}
        # Track already assigned placeholders for identical normalized values
        seen_entity_map: Dict[tuple, str] = {}
        entity_placeholders: List[str] = []

        for entity in sorted_entities:
            prefix = self.TYPE_PREFIXES.get(entity.entity_type, entity.entity_type.value)
            cache_key = (entity.entity_type, entity.normalized_value)

            if cache_key in seen_entity_map:
                ph = seen_entity_map[cache_key]
            else:
                counter = type_counters.get(prefix, 0) + 1
                type_counters[prefix] = counter
                ph = f"<{prefix}_{counter}>"
                seen_entity_map[cache_key] = ph

                # Register in local mapping store
                self.mapping_store.store_mapping(
                    session_id=sid,
                    placeholder=ph,
                    raw_value=entity.text,
                    entity_type=entity.entity_type,
                    normalized_value=entity.normalized_value,
                )

            entity_placeholders.append(ph)

        # Construct sanitized prompt using character slicing to avoid offset drift
        sanitized_parts = []
        last_idx = 0

        for entity, ph in zip(sorted_entities, entity_placeholders):
            # Append plain text between previous match and current match
            sanitized_parts.append(prompt[last_idx:entity.start])
            # Append placeholder
            sanitized_parts.append(ph)
            last_idx = entity.end

        # Append remaining trailing prompt text
        sanitized_parts.append(prompt[last_idx:])
        sanitized_prompt = "".join(sanitized_parts)

        # Fetch the registered session mappings
        session_mappings = self.mapping_store.get_mappings(sid)

        return BaselineMaskResult(
            original_prompt=prompt,
            sanitized_prompt=sanitized_prompt,
            entities=sorted_entities,
            mapping=session_mappings,
            session_id=sid,
        )
