"""
End-to-End Process Route
"""

from fastapi import APIRouter, HTTPException
from backend.api.schemas import ProcessRequest, ProcessResponse
from backend.api.service import PromptShieldService

router = APIRouter(tags=["End-to-End Orchestration"])


@router.post("/process", response_model=ProcessResponse)
def process_prompt(request: ProcessRequest):
    """
    Execute complete end-to-end prompt protection and response restoration (Phases 1-5).
    Pipeline: Sanitize -> LLM Generation -> Controlled Restoration.
    """
    try:
        service = PromptShieldService()
        return service.process_end_to_end(
            prompt=request.prompt,
            session_id=request.session_id,
            security_level_str=request.security_level or "BALANCED",
            system_prompt=request.system_prompt,
            llm_provider_name=request.llm_provider or "mock",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
