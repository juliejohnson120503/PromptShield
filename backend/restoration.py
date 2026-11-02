"""
PromptShield AI — Controlled Response Restoration Engine (Phase 5).
==================================================================
Safely restores placeholder tags (<TYPE_N>) in external LLM responses
using the local client-side MappingStore, subject to policy controls
and authorization checks.

Security Invariants Enforced:
----------------------------
R-1: Local Isolation: Raw secrets are fetched exclusively from the local
     in-memory MappingStore (isolated by session_id).
R-2: Single-Pass Non-Recursive: Substituted values can never trigger secondary
     placeholder expansions (prevents recursive injection attacks).
R-3: Credential Quarantine: By default, high-risk technical credentials
     (passwords, API keys, tokens) are blocked from being restored into
     downstream text unless explicitly authorized by policy.
R-4: Hallucination Tracking: Untracked or fabricated placeholders generated
     by the LLM are identified, preserved, and flagged in the audit report.
R-5: Policy Gate: If restoration is disabled by policy, the response remains
     unaltered with status=DENIED.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Set, Tuple

from backend.models import (
    EntityType,
    RestorationPolicy,
    RestorationResult,
    RestoredPlaceholder,
    RestorationStatus,
)
from backend.mapping_store import MappingStore

logger = logging.getLogger(__name__)

# Pattern to capture PromptShield placeholder formats, e.g.: {{PERSON_1}}, <PERSON_1>, [PERSON_1], {PERSON_1}, &lt;PERSON_1&gt;
PLACEHOLDER_REGEX = re.compile(
    r"(?:\{\{|\{|\<|\[|&lt;)\s*([A-Za-z_]+)_(\d+)\s*(?:\}\}|\}|\>|\]|&gt;)",
    re.IGNORECASE,
)

# Common LLM email template slot aliases (e.g. [Your Name], [Name], [Your Full Name])
NAME_SLOT_REGEX = re.compile(
    r"\[\s*(?:Your\s+Name|Your\s+Full\s+Name|Full\s+Name|Candidate\s+Name|Applicant\s+Name|Sender\s+Name|Name)\s*\]",
    re.IGNORECASE,
)


class ResponseRestorer:
    """
    Phase 5 Controlled Response Restoration Engine.

    Coordinates policy checks, local mapping store lookups, and safe
    atomic substitution of placeholders inside LLM responses.
    """

    def __init__(
        self,
        mapping_store: Optional[MappingStore] = None,
        default_policy: Optional[RestorationPolicy] = None,
    ):
        self.mapping_store: MappingStore = mapping_store or MappingStore()
        self.default_policy: RestorationPolicy = default_policy or RestorationPolicy()

    def restore(
        self,
        llm_response: str,
        session_id: Optional[str] = None,
        policy: Optional[RestorationPolicy] = None,
        mapping_override: Optional[Dict[str, str]] = None,
    ) -> RestorationResult:
        """
        Scan *llm_response* for placeholders and restore them using the
        authorized secrets stored in the local MappingStore.

        Parameters
        ----------
        llm_response : str
            Raw text received from the downstream LLM.
        session_id : str, optional
            Session ID for client-side local mapping lookup.
        policy : RestorationPolicy, optional
            Overrides default restoration security policy.
        mapping_override : dict, optional
            Explicit placeholder-to-secret mapping (bypasses store lookup).

        Returns
        -------
        RestorationResult
            Complete bundle containing restored text, status, and audit records.
        """
        active_policy = policy or self.default_policy

        # Edge case: empty response
        if not llm_response:
            return RestorationResult(
                restored_text="",
                raw_llm_response="",
                session_id=session_id,
                status=RestorationStatus.UNMODIFIED,
                restorations=[],
                unrestored_placeholders=[],
                audit_trail={"total_placeholders": 0, "restored_count": 0},
            )

        # Invariant R-5: Policy gate
        if not active_policy.allow_restoration:
            logger.info("[ResponseRestorer] Restoration denied by policy (allow_restoration=False).")
            return RestorationResult(
                restored_text=llm_response,
                raw_llm_response=llm_response,
                session_id=session_id,
                status=RestorationStatus.DENIED,
                restorations=[],
                unrestored_placeholders=[],
                audit_trail={
                    "reason": "Restoration denied by policy (allow_restoration=False)",
                    "blocked_count": 0,
                },
            )

        # Find all placeholders in LLM response
        matches = list(PLACEHOLDER_REGEX.finditer(llm_response))
        name_slot_matches = list(NAME_SLOT_REGEX.finditer(llm_response))

        if not matches and not name_slot_matches:
            return RestorationResult(
                restored_text=llm_response,
                raw_llm_response=llm_response,
                session_id=session_id,
                status=RestorationStatus.UNMODIFIED,
                restorations=[],
                unrestored_placeholders=[],
                audit_trail={"total_placeholders": 0, "restored_count": 0},
            )

        # Retrieve local mappings
        local_mapping: Dict[str, str] = {}
        mapping_details: Dict[str, Dict[str, Any]] = {}
        if mapping_override is not None:
            local_mapping = dict(mapping_override)
        elif session_id is not None:
            local_mapping = self.mapping_store.get_mappings(session_id)
            mapping_details = self.mapping_store.get_mapping_details(session_id)
        else:
            fallback_sid = self.mapping_store.get_most_recent_session_id() or "default"
            local_mapping = self.mapping_store.get_mappings(fallback_sid)
            mapping_details = self.mapping_store.get_mapping_details(fallback_sid)

        # Build bracket-agnostic and case-agnostic normalized lookups
        def _clean_token(t: str) -> str:
            return re.sub(r"[\{\}\<\>\[\]]|&lt;|&gt;", "", t).strip().upper()

        norm_mapping: Dict[str, str] = {}
        norm_details: Dict[str, Dict[str, Any]] = {}
        for k, v in local_mapping.items():
            norm_k = _clean_token(k)
            norm_mapping[norm_k] = v
            norm_mapping[norm_k.lower()] = v
            norm_mapping[f"{{{{{norm_k}}}}}"] = v
            norm_mapping[f"{{{{{norm_k.lower()}}}}}"] = v
            norm_mapping[f"<{norm_k}>"] = v
            norm_mapping[f"<{norm_k.lower()}>"] = v
            norm_mapping[f"[{norm_k}]"] = v
            norm_mapping[f"[{norm_k.lower()}]"] = v
            norm_mapping[f"&lt;{norm_k}&gt;"] = v
            norm_mapping[f"&lt;{norm_k.lower()}&gt;"] = v
            norm_mapping[k] = v
            norm_mapping[k.lower()] = v

        for k, v in mapping_details.items():
            norm_k = _clean_token(k)
            norm_details[norm_k] = v
            norm_details[norm_k.lower()] = v
            norm_details[f"{{{{{norm_k}}}}}"] = v
            norm_details[f"{{{{{norm_k.lower()}}}}}"] = v
            norm_details[f"<{norm_k}>"] = v
            norm_details[f"<{norm_k.lower()}>"] = v
            norm_details[f"[{norm_k}]"] = v
            norm_details[f"[{norm_k.lower()}]"] = v
            norm_details[f"&lt;{norm_k}&gt;"] = v
            norm_details[f"&lt;{norm_k.lower()}&gt;"] = v
            norm_details[k] = v
            norm_details[k.lower()] = v

        # Identify primary persona for email template slot aliases (e.g. [Your Name] -> "Jerin")
        primary_person_secret: Optional[str] = None
        for ph, val in local_mapping.items():
            ph_upper = ph.upper()
            det = mapping_details.get(ph, {})
            et = det.get("entity_type")
            if (et == EntityType.PERSON or "PERSON" in ph_upper) and det.get("disclosure_allowed", True):
                primary_person_secret = val
                break

        # Track unique placeholders and plan substitutions
        unique_placeholders: Set[str] = set()
        approved_substitutions: Dict[str, str] = {}
        restoration_records: List[RestoredPlaceholder] = []
        unrestored_list: List[str] = []

        total_restoration_budget = active_policy.max_restorations

        for m in matches:
            ph = m.group(0)
            if ph in unique_placeholders:
                continue
            unique_placeholders.add(ph)

            type_name = m.group(1).upper()
            clean_ph = _clean_token(ph)

            # Map placeholder type name to EntityType enum
            entity_type: Optional[EntityType] = None
            try:
                entity_type = EntityType(type_name)
            except ValueError:
                # Type name could be an alias or custom ID (e.g. ORG -> ORGANIZATION)
                if type_name == "ORG":
                    entity_type = EntityType.ORGANIZATION
                elif type_name == "LOC":
                    entity_type = EntityType.LOCATION
                else:
                    entity_type = None

            # 1. Check if placeholder exists in local mapping (bracket-agnostic and case-agnostic)
            raw_secret = (
                norm_mapping.get(clean_ph)
                or norm_mapping.get(clean_ph.lower())
                or local_mapping.get(ph)
            )
            details = (
                norm_details.get(clean_ph)
                or norm_details.get(clean_ph.lower())
                or mapping_details.get(ph, {})
            )

            if raw_secret is None:
                unrestored_list.append(ph)
                restoration_records.append(
                    RestoredPlaceholder(
                        placeholder=ph,
                        original_value="",
                        entity_type=entity_type,
                        restored=False,
                        reason="Placeholder not found in local mapping store (untracked or model hallucination)",
                    )
                )
                continue

            # 2. Check Explicit Non-Disclosure Policy (Controlled Restoration Gate)
            if not details.get("disclosure_allowed", True):
                unrestored_list.append(ph)
                restoration_records.append(
                    RestoredPlaceholder(
                        placeholder=ph,
                        original_value=raw_secret,
                        entity_type=entity_type,
                        restored=False,
                        reason="Restoration blocked by policy: explicit non-disclosure directive (disclosure_allowed=False)",
                    )
                )
                continue

            # 3. Check Credential Quarantine Policy (Invariant R-3)
            is_credential = entity_type in active_policy.blocked_entity_types
            if is_credential and not active_policy.allow_credentials_restoration:
                unrestored_list.append(ph)
                restoration_records.append(
                    RestoredPlaceholder(
                        placeholder=ph,
                        original_value=raw_secret,
                        entity_type=entity_type,
                        restored=False,
                        reason=f"Blocked by policy: {type_name} credential restoration is restricted",
                    )
                )
                continue

            # 4. Budget limit check
            restored_count_so_far = len([r for r in restoration_records if r.restored])
            if restored_count_so_far >= total_restoration_budget:
                unrestored_list.append(ph)
                restoration_records.append(
                    RestoredPlaceholder(
                        placeholder=ph,
                        original_value=raw_secret,
                        entity_type=entity_type,
                        restored=False,
                        reason="Exceeded maximum restoration budget",
                    )
                )
                continue

            # Authorized for restoration
            approved_substitutions[ph] = raw_secret
            approved_substitutions[ph.lower()] = raw_secret
            approved_substitutions[clean_ph] = raw_secret
            approved_substitutions[clean_ph.lower()] = raw_secret
            restoration_records.append(
                RestoredPlaceholder(
                    placeholder=ph,
                    original_value=raw_secret,
                    entity_type=entity_type,
                    restored=True,
                    reason="Authorized and restored from local session store",
                )
            )

        # Name slot alias restoration for email templates (e.g. [Your Name] -> "Jerin")
        if primary_person_secret and name_slot_matches:
            for sm in name_slot_matches:
                slot_ph = sm.group(0)
                if slot_ph in unique_placeholders:
                    continue
                unique_placeholders.add(slot_ph)

                restored_count_so_far = len([r for r in restoration_records if r.restored])
                if restored_count_so_far >= total_restoration_budget:
                    unrestored_list.append(slot_ph)
                    restoration_records.append(
                        RestoredPlaceholder(
                            placeholder=slot_ph,
                            original_value=primary_person_secret,
                            entity_type=EntityType.PERSON,
                            restored=False,
                            reason="Exceeded maximum restoration budget",
                        )
                    )
                    continue

                approved_substitutions[slot_ph] = primary_person_secret
                approved_substitutions[slot_ph.lower()] = primary_person_secret
                restoration_records.append(
                    RestoredPlaceholder(
                        placeholder=slot_ph,
                        original_value=primary_person_secret,
                        entity_type=EntityType.PERSON,
                        restored=True,
                        reason="Restored LLM email template slot alias to authorized persona name",
                    )
                )

        # Invariant R-2: Single-pass atomic substitution prevents recursive injection
        sub_pattern = re.compile(
            rf"{PLACEHOLDER_REGEX.pattern}|{NAME_SLOT_REGEX.pattern}",
            re.IGNORECASE,
        )

        def _replace_callback(match):
            token = match.group(0)
            if token in approved_substitutions:
                return approved_substitutions[token]
            clean = _clean_token(token)
            if clean in approved_substitutions:
                return approved_substitutions[clean]
            for k, val in approved_substitutions.items():
                if k.lower() == token.lower() or _clean_token(k) == clean:
                    return val
            if primary_person_secret and NAME_SLOT_REGEX.fullmatch(token):
                return primary_person_secret
            return token

        restored_text = sub_pattern.sub(_replace_callback, llm_response)

        # Determine overall status
        total_restored = len([r for r in restoration_records if r.restored])
        if unrestored_list:
            status = RestorationStatus.PARTIAL if total_restored > 0 else RestorationStatus.DENIED
        elif total_restored > 0:
            status = RestorationStatus.SUCCESS
        else:
            status = RestorationStatus.UNMODIFIED

        audit_trail = {
            "total_placeholders_found": len(matches) + len(name_slot_matches),
            "unique_placeholders": len(unique_placeholders),
            "restored_count": len(approved_substitutions),
            "unrestored_count": len(unrestored_list),
            "blocked_credentials_count": sum(
                1 for r in restoration_records if not r.restored and "credential" in r.reason
            ),
            "blocked_non_disclosure_count": sum(
                1 for r in restoration_records if not r.restored and "non-disclosure" in r.reason
            ),
            "hallucinated_placeholders": [
                r.placeholder for r in restoration_records if "untracked" in r.reason
            ],
        }

        return RestorationResult(
            restored_text=restored_text,
            raw_llm_response=llm_response,
            session_id=session_id,
            status=status,
            restorations=restoration_records,
            unrestored_placeholders=unrestored_list,
            audit_trail=audit_trail,
        )
