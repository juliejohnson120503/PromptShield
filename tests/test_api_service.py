"""
Tests for Phase 6 FastAPI REST API Service
Validates all endpoints (/health, /analyze, /sanitize, /restore, /process, /policies, /metrics)
and ensures privacy-safe operational audit logging.
"""

import pytest
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.api.service import PromptShieldService


@pytest.fixture(scope="module")
def client():
    """Create a test client sharing the lifespan of the FastAPI app."""
    svc = PromptShieldService()
    _ = svc.shield
    if svc.shield.detector._gliner2:
        svc.shield.detector._gliner2.is_available()
    with TestClient(app) as c:
        yield c


class TestHealthAndMetadataRoutes:
    def test_root_welcome(self, client):
        res = client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert data["service"] == "PromptShield AI REST Service"
        assert data["version"] == "1.0.0"
        assert "/docs" in data["docs"]

    def test_health_check(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "detectors" in data
        assert data["detectors"]["regex"] is True

    def test_policies_list(self, client):
        res = client.get("/policies")
        assert res.status_code == 200
        data = res.json()
        assert "BALANCED" in data["supported_security_levels"]
        assert "STRICT" in data["supported_security_levels"]
        assert "EMAIL" in data["supported_entity_types"]
        assert "INCIDENT_REPORT" in data["supported_task_types"] or "GENERAL_QA" in data["supported_task_types"]
        assert data["default_security_level"] == "BALANCED"


class TestAnalyzeEndpoint:
    def test_analyze_empty_prompt_validation(self, client):
        res = client.post("/analyze", json={"prompt": ""})
        assert res.status_code == 422  # Pydantic validation error

    def test_analyze_sensitive_prompt(self, client):
        payload = {
            "prompt": "Contact Alice at alice@example.com regarding order ORD-99120.",
            "security_level": "BALANCED",
        }
        res = client.post("/analyze", json=payload)
        assert res.status_code == 200
        data = res.json()

        assert data["prompt_length"] == len(payload["prompt"])
        assert "task_type" in data
        assert data["analysis_latency_ms"] >= 0.0

        # Check detected entities
        detected_texts = [e["text"] for e in data["detected_entities"]]
        assert "alice@example.com" in detected_texts
        assert "ORD-99120" in detected_texts

        # Check entity policies
        policy_map = {p["entity_text"]: p for p in data["entity_policies"]}
        assert "alice@example.com" in policy_map
        assert policy_map["alice@example.com"]["decision"] in ("MASK", "USER_APPROVAL")


class TestSanitizeEndpoint:
    def test_sanitize_masks_sensitive_data(self, client):
        prompt = "My email is john.doe@example.com and API key is sk-test-SECRET123456."
        res = client.post("/sanitize", json={"prompt": prompt, "security_level": "BALANCED"})
        assert res.status_code == 200
        data = res.json()

        assert "session_id" in data
        sanitized = data["sanitized_prompt"]
        assert "john.doe@example.com" not in sanitized
        assert "sk-test-SECRET123456" not in sanitized
        assert "<EMAIL_1>" in sanitized or any(p.startswith("<EMAIL") for p in data["placeholders"])
        assert data["masked_count"] >= 1

    def test_sanitize_with_explicit_session_id(self, client):
        prompt = "Reach me at test@example.com"
        session_id = "custom-test-session-123"
        res = client.post(
            "/sanitize",
            json={"prompt": prompt, "session_id": session_id, "security_level": "BALANCED"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["session_id"] == session_id


class TestRestoreEndpoint:
    def test_restore_authorized_placeholders(self, client):
        # 1. Sanitize prompt to populate session mapping
        prompt = "Support contact is agent.smith@matrix.com with API key sk-live-SECRET99."
        san_res = client.post("/sanitize", json={"prompt": prompt, "security_level": "BALANCED"})
        assert san_res.status_code == 200
        san_data = san_res.json()
        session_id = san_data["session_id"]
        placeholders = san_data["placeholders"]

        # 2. Simulate LLM response containing placeholders
        email_ph = next((p for p in placeholders if "EMAIL" in p), "<EMAIL_1>")
        llm_response = f"Acknowledged. We will reach out to {email_ph} immediately."

        # 3. Restore response
        res = client.post(
            "/restore",
            json={
                "llm_response": llm_response,
                "session_id": session_id,
                "allow_all_restoration": True,
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["session_id"] == session_id
        assert "agent.smith@matrix.com" in data["restored_response"]
        assert data["restored_count"] >= 1
        assert data["leakage_detected"] is False

    def test_restore_invalid_session(self, client):
        res = client.post(
            "/restore",
            json={
                "llm_response": "Hello <EMAIL_1>",
                "session_id": "non-existent-session-xyz",
            },
        )
        assert res.status_code == 200
        data = res.json()
        # Non-existent session cannot restore anything
        assert data["restored_count"] == 0
        assert data["restored_response"] == "Hello <EMAIL_1>"


class TestProcessEndpoint:
    def test_process_end_to_end_shielded_exchange(self, client):
        prompt = (
            "Write a short response confirming that Priya Nair from TechNova Solutions "
            "contacted support at priya@example.com."
        )
        res = client.post(
            "/process",
            json={
                "prompt": prompt,
                "security_level": "BALANCED",
                "llm_provider": "mock",
            },
        )
        assert res.status_code == 200
        data = res.json()

        assert "session_id" in data
        assert "sanitized_prompt" in data
        assert "raw_llm_response" in data
        assert "restored_response" in data
        assert data["total_latency_ms"] >= 0.0

        # Sanitized prompt protected email
        assert "priya@example.com" not in data["sanitized_prompt"]


class TestMetricsAndAuditEndpoint:
    def test_metrics_collection_and_privacy_safety(self, client):
        # Trigger an analyze request to register telemetry
        client.post("/analyze", json={"prompt": "Ping metrics test with test@email.com"})

        res = client.get("/metrics")
        assert res.status_code == 200
        data = res.json()

        assert data["total_requests"] >= 1
        assert "/analyze" in data["requests_by_endpoint"]
        assert data["total_entities_detected"] >= 1
        assert "EMAIL" in data["entities_by_type"]
        assert data["avg_latency_ms"] >= 0.0

        # Verify zero raw PII is stored in metrics
        data_str = str(data)
        assert "test@email.com" not in data_str
