"""
Operational Metrics and Privacy-Safe Telemetry Route
"""

from fastapi import APIRouter
from backend.api.schemas import MetricsResponse
from backend.api.service import PromptShieldService

router = APIRouter(tags=["Metrics"])


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """
    Retrieve operational metrics and detection audit totals.
    Zero raw PII is exposed in these statistics.
    """
    service = PromptShieldService()
    return service.get_metrics()
