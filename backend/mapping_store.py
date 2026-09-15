"""
Secure Local Mapping Store for PromptShield AI.
Maintains session-isolated mappings between semantic placeholders and their
raw sensitive values on the local user side.

CRITICAL SECURITY PRINCIPLE:
The mapping store resides exclusively on the client/local environment
and is never transmitted to external LLMs.
"""

import uuid
from typing import Dict, Optional, Any
from backend.models import EntityType


class MappingStore:
    """
    In-memory session-isolated mapping repository for PromptShield AI.
    Maps placeholders (e.g. <EMAIL_1>) to raw sensitive text.
    """

    def __init__(self):
        # Format: {session_id: {placeholder: {"raw_value": str, "entity_type": EntityType, ...}}}
        self._store: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session container if it does not exist."""
        sid = session_id or str(uuid.uuid4())
        if sid not in self._store:
            self._store[sid] = {}
        return sid

    def store_mapping(
        self,
        session_id: str,
        placeholder: str,
        raw_value: str,
        entity_type: EntityType,
        normalized_value: Optional[str] = None,
    ) -> None:
        """Register a placeholder to raw value association under a session."""
        if session_id not in self._store:
            self.create_session(session_id)

        self._store[session_id][placeholder] = {
            "raw_value": raw_value,
            "entity_type": entity_type,
            "normalized_value": normalized_value or raw_value,
        }

    def get_mappings(self, session_id: str) -> Dict[str, str]:
        """
        Return a simple dictionary of {placeholder: raw_value} for a session.
        Returns an empty dict if the session does not exist.
        """
        if session_id not in self._store:
            return {}
        return {
            ph: data["raw_value"]
            for ph, data in self._store[session_id].items()
        }

    def get_mapping_details(self, session_id: str) -> Dict[str, Dict[str, Any]]:
        """Return full metadata mapping for a session."""
        return self._store.get(session_id, {})

    def get_raw_value(self, session_id: str, placeholder: str) -> Optional[str]:
        """Retrieve the raw sensitive value for a given placeholder in a session."""
        session_data = self._store.get(session_id)
        if not session_data:
            return None
        item = session_data.get(placeholder)
        return item["raw_value"] if item else None

    def clear_session(self, session_id: str) -> bool:
        """Delete session data from local store."""
        if session_id in self._store:
            del self._store[session_id]
            return True
        return False

    def session_count(self) -> int:
        """Number of active sessions stored locally."""
        return len(self._store)
