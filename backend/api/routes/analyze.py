"""
Prompt Analysis Route
"""

from fastapi import APIRouter, HTTPException
from backend.api.schemas import AnalyzeRequest, AnalyzeResponse
from backend.api.service import PromptShieldService

router = APIRouter(tags=["Analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_prompt(request: AnalyzeRequest):
    """
    Perform hybrid detection (Phases 1-3) on prompt text.
    Extracts entities, classifies task type, assesses privacy risk,
    and returns policy decisions without modifying the prompt.
    """
    try:
        service = PromptShieldService()
        return service.analyze_prompt(
            prompt=request.prompt,
            security_level_str=request.security_level or "BALANCED",
            context_hint=request.context_hint,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
