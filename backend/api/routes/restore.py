"""
Response Restoration Route
"""

from fastapi import APIRouter, HTTPException
from backend.api.schemas import RestoreRequest, RestoreResponse
from backend.api.service import PromptShieldService

router = APIRouter(tags=["Restoration"])


@router.post("/restore", response_model=RestoreResponse)
def restore_response(request: RestoreRequest):
    """
    Perform controlled response restoration (Phase 5).
    Substitutes authorized placeholders in an LLM response using the session mapping
    while keeping prohibited or credential tokens strictly masked.
    """
    try:
        service = PromptShieldService()
        return service.restore_response(
            llm_response=request.llm_response,
            session_id=request.session_id,
            allow_all_restoration=request.allow_all_restoration,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
