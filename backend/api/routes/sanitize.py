"""
Prompt Sanitization Route
"""

from fastapi import APIRouter, HTTPException
from backend.api.schemas import SanitizeRequest, SanitizeResponse
from backend.api.service import PromptShieldService

router = APIRouter(tags=["Sanitization"])


@router.post("/sanitize", response_model=SanitizeResponse)
def sanitize_prompt(request: SanitizeRequest):
    """
    Perform context-aware semantic masking (Phase 4).
    Replaces sensitive tokens with type-safe placeholders, retains public/task-relevant entities,
    and stores the mapping in local session memory.
    """
    try:
        service = PromptShieldService()
        return service.sanitize_prompt(
            prompt=request.prompt,
            session_id=request.session_id,
            security_level_str=request.security_level or "BALANCED",
            approved_entity_ids=request.approved_entity_ids,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
