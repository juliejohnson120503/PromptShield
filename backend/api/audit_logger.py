"""
PromptShield AI — Phase 6 Privacy-Safe Audit Logger
Maintains aggregated operational metrics and produces anonymized audit trails.

Guarantees:
  - ZERO raw PII or secret tokens are ever persisted or logged.
  - Aggregations track only counts, entity types, policy action counts, and latency statistics.
"""

import logging
import threading
import time
from typing import Dict, List, Any

logger = logging.getLogger("promptshield.audit")


class PrivacySafeAuditLogger:
    """
    Thread-safe audit logger and in-memory operational metrics collector.
    Never stores sensitive prompt text or secret payload values.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PrivacySafeAuditLogger, cls).__new__(cls)
                cls._instance._init_state()
            return cls._instance

    def _init_state(self) -> None:
        self._metrics_lock = threading.Lock()
        self.total_requests = 0
        self.requests_by_endpoint: Dict[str, int] = {}
        self.total_entities_detected = 0
        self.entities_by_type: Dict[str, int] = {}
        self.total_masks_applied = 0
        self.total_restorations = 0
        self.total_blocked_restorations = 0
        self._latencies: List[float] = []

    def record_request(self, endpoint: str, latency_ms: float) -> None:
        with self._metrics_lock:
            self.total_requests += 1
            self.requests_by_endpoint[endpoint] = (
                self.requests_by_endpoint.get(endpoint, 0) + 1
            )
            self._latencies.append(latency_ms)
            if len(self._latencies) > 2000:
                self._latencies = self._latencies[-1000:]
        logger.info(
            "[Audit] Endpoint=%s Latency=%.2fms TotalRequests=%d",
            endpoint,
            latency_ms,
            self.total_requests,
        )

    def record_entities(self, entity_types: List[str]) -> None:
        if not entity_types:
            return
        with self._metrics_lock:
            self.total_entities_detected += len(entity_types)
            for et in entity_types:
                self.entities_by_type[et] = self.entities_by_type.get(et, 0) + 1

    def record_masking(self, count: int) -> None:
        if count <= 0:
            return
        with self._metrics_lock:
            self.total_masks_applied += count

    def record_restoration(self, restored: int, blocked: int) -> None:
        with self._metrics_lock:
            self.total_restorations += restored
            self.total_blocked_restorations += blocked

    def get_metrics(self) -> Dict[str, Any]:
        with self._metrics_lock:
            avg_lat = (
                sum(self._latencies) / len(self._latencies)
                if self._latencies
                else 0.0
            )
            return {
                "total_requests": self.total_requests,
                "requests_by_endpoint": dict(self.requests_by_endpoint),
                "total_entities_detected": self.total_entities_detected,
                "entities_by_type": dict(self.entities_by_type),
                "total_masks_applied": self.total_masks_applied,
                "total_restorations": self.total_restorations,
                "total_blocked_restorations": self.total_blocked_restorations,
                "avg_latency_ms": round(avg_lat, 2),
            }

    def reset(self) -> None:
        with self._metrics_lock:
            self._init_state()
