"""
Secure Local Mapping Store for PromptShield AI.
Maintains session-isolated mappings between semantic placeholders and their
raw sensitive values on the local user side.

CRITICAL SECURITY PRINCIPLE:
The mapping store resides exclusively on the client/local environment
and is never transmitted to external LLMs.
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, Optional, Any
from backend.models import EntityType

logger = logging.getLogger(__name__)

MAPPINGS_FILE = Path(__file__).parent / ".session_mappings.json"


class MappingStore:
    """
    Session-isolated mapping repository for PromptShield AI.
    Maps placeholders (e.g. <EMAIL_1>) to raw sensitive text.
    Maintains in-memory state with local disk persistence so sessions
    survive server restarts and hot-reloads.
    """

    def __init__(self, persist: bool = True):
        # Format: {session_id: {placeholder: {"raw_value": str, "entity_type": EntityType, ...}}}
        self._store: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._last_session_id: Optional[str] = None
        self._persist: bool = persist
        if self._persist:
            self._load_from_disk()

    def _persist_to_disk(self) -> None:
        """Atomically persist in-memory mappings to local JSON store."""
        if not self._persist:
            return
        try:
            serializable_store = {}
            for sid, ph_map in self._store.items():
                serializable_store[sid] = {}
                for ph, data in ph_map.items():
                    entry = dict(data)
                    et = entry.get("entity_type")
                    if isinstance(et, EntityType):
                        entry["entity_type"] = et.value
                    serializable_store[sid][ph] = entry
            with open(MAPPINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(serializable_store, f, indent=2)
        except Exception as e:
            logger.debug(f"[MappingStore] Could not persist to disk: {e}")

    def _load_from_disk(self) -> None:
        """Load persisted session mappings from disk on startup."""
        if not MAPPINGS_FILE.exists():
            return
        try:
            with open(MAPPINGS_FILE, "r", encoding="utf-8") as f:
                raw_store = json.load(f)
            for sid, ph_map in raw_store.items():
                self._store[sid] = {}
                for ph, data in ph_map.items():
                    entry = dict(data)
                    et_val = entry.get("entity_type")
                    if et_val:
                        try:
                            entry["entity_type"] = EntityType(et_val)
                        except ValueError:
                            entry["entity_type"] = EntityType.PII_OTHER
                    self._store[sid][ph] = entry
                self._last_session_id = sid
        except Exception as e:
            logger.debug(f"[MappingStore] Could not load from disk: {e}")

    def get_most_recent_session_id(self) -> Optional[str]:
        """Return the most recently created or updated session ID."""
        if self._last_session_id and self._last_session_id in self._store:
            return self._last_session_id
        if self._store:
            return list(self._store.keys())[-1]
        return None

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session container if it does not exist."""
        sid = session_id or str(uuid.uuid4())
        if sid not in self._store:
            self._store[sid] = {}
        self._last_session_id = sid
        return sid

    def store_mapping(
        self,
        session_id: str,
        placeholder: str,
        raw_value: str,
        entity_type: EntityType,
        normalized_value: Optional[str] = None,
        entity_id: Optional[str] = None,
        native_label: Optional[str] = None,
        policy_decision: Optional[str] = None,
        disclosure_allowed: bool = True,
    ) -> None:
        """Register a placeholder to raw value association with security metadata under a session."""
        if session_id not in self._store:
            self.create_session(session_id)

        self._store[session_id][placeholder] = {
            "placeholder": placeholder,
            "raw_value": raw_value,
            "entity_type": entity_type,
            "normalized_value": normalized_value or raw_value,
            "entity_id": entity_id or f"{entity_type.value}:{normalized_value or raw_value}",
            "native_label": native_label or entity_type.value.lower(),
            "policy_decision": policy_decision or "MASK",
            "disclosure_allowed": disclosure_allowed,
        }
        self._last_session_id = session_id
        self._persist_to_disk()

    def register_mapping(
        self,
        session_id: str,
        placeholder: str,
        raw_value: str,
        entity_type: Optional[EntityType] = None,
        normalized_value: Optional[str] = None,
        entity_id: Optional[str] = None,
        native_label: Optional[str] = None,
        policy_decision: Optional[str] = None,
        disclosure_allowed: bool = True,
    ) -> None:
        """Convenience alias for store_mapping."""
        et = entity_type or EntityType.PII_OTHER
        self.store_mapping(
            session_id=session_id,
            placeholder=placeholder,
            raw_value=raw_value,
            entity_type=et,
            normalized_value=normalized_value,
            entity_id=entity_id,
            native_label=native_label,
            policy_decision=policy_decision,
            disclosure_allowed=disclosure_allowed,
        )

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
            if self._last_session_id == session_id:
                self._last_session_id = list(self._store.keys())[-1] if self._store else None
            self._persist_to_disk()
            return True
        return False

    def session_count(self) -> int:
        """Number of active sessions stored locally."""
        return len(self._store)

