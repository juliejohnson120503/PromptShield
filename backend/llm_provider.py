"""
PromptShield AI — Pluggable LLM Provider Layer (Phase 5).
=========================================================
Defines the abstract interface for downstream LLM integrations and provides:
1. BaseLLMProvider: Abstract base class for all LLM connectors.
2. MockLLMProvider: Deterministic, offline provider for unit tests, CI/CD, and reproducible benchmarks.
3. GeminiProvider: Google Gemini API integration (via HTTP REST).
4. OpenAIProvider: OpenAI API integration (via HTTP REST).
"""

import os
import re
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """
    Abstract Base Class for LLM providers.
    All external or simulated models integrate through this uniform interface.
    """

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate a text response given a prompt and optional system instructions.

        Parameters
        ----------
        prompt : str
            The prompt (typically sanitized by PromptShield Phase 4).
        system_prompt : str, optional
            System-level prompt instructions.

        Returns
        -------
        str
            The model's text completion.
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return human-readable identifier of the provider and model."""
        pass


class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic, offline mock provider.
    Enables comprehensive testing of Phase 5 restoration without external APIs,
    costs, or network latency.

    Features:
    - Pre-configured fixed response mappings.
    - Context-aware template generation that mirrors and propagates placeholders
      found in the sanitized prompt so restoration can be verified.
    """

    def __init__(
        self,
        custom_responses: Optional[Dict[str, str]] = None,
        default_template: Optional[str] = None,
    ):
        self.custom_responses: Dict[str, str] = custom_responses or {}
        self.default_template: Optional[str] = default_template
        self.call_history: List[Dict[str, Any]] = []

    def set_response(self, prompt_match: str, response: str) -> None:
        """Register a custom response for a specific prompt substring."""
        self.custom_responses[prompt_match] = response

    def get_provider_name(self) -> str:
        return "MockLLMProvider (Deterministic Local Offline)"

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        self.call_history.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "kwargs": kwargs,
        })

        # 1. Exact or substring match in custom_responses
        for pattern, canned in self.custom_responses.items():
            if pattern in prompt:
                return canned

        # 2. Explicit default template
        if self.default_template:
            return self.default_template

        # 3. Dynamic context-aware mock response preserving placeholders
        # Extract any placeholders present in the prompt (e.g. <PERSON_1>, <ORG_1>)
        placeholders = re.findall(r"<[A-Z_]+_\d+>", prompt)
        prompt_lower = prompt.lower()

        # Task: Incident / Security / Investigation Report
        if re.search(r"(?i)\b(?:incident|postmortem|investigation|breach|outage)\b", prompt_lower):
            # Dynamic incident title based on input context
            if "customer support" in prompt_lower or "customer-support" in prompt_lower:
                topic = "Customer Support Incident"
            elif "payment" in prompt_lower or "transaction" in prompt_lower:
                topic = "Payment Transaction Incident"
            else:
                topic = "Operational Incident"

            # Dynamic extraction of mentioned parties/context from input prompt
            context_summary = []
            mention_match = re.search(r"(?i)\b(?:must\s+mention|mentioning)\s+([^.?!;\n]+)", prompt)
            if not mention_match:
                mention_match = re.search(r"(?i)\b(?:involving|regarding)\s+([^.?!;\n]+)", prompt)
            if mention_match:
                context_summary.append(f"Incident context includes {mention_match.group(1).strip()}.")
            else:
                context_summary.append(f"An investigation was conducted regarding the reported {topic.lower()}.")

            # Extract any order / ticket references directly present
            order_refs = re.findall(r"\b(?:ORD|ORDER|TKT|TICKET|SR|INC)[-_][A-Za-z0-9]+\b", prompt, re.IGNORECASE)
            if order_refs:
                context_summary.append(f"Related transaction reference: {', '.join(order_refs)}.")

            # Identify contact and technical placeholders to propagate
            email_phs = [p for p in placeholders if "EMAIL" in p]
            phone_phs = [p for p in placeholders if "PHONE" in p]
            acc_phs = [p for p in placeholders if any(k in p for k in ("BANK_ACCOUNT", "CUSTOMER_ID", "ACCOUNT", "USER_ID"))]
            net_phs = [p for p in placeholders if any(k in p for k in ("IP_ADDRESS", "API_KEY", "PASSWORD", "ACCESS_TOKEN"))]

            ph_sections = []
            if email_phs:
                ph_sections.append(f"Recorded contact emails: {', '.join(email_phs)}")
            if phone_phs:
                ph_sections.append(f"Telephone records: {', '.join(phone_phs)}")
            if acc_phs:
                ph_sections.append(f"Account/customer identifiers: {', '.join(acc_phs)}")
            if net_phs:
                ph_sections.append(f"System access and authentication tokens: {', '.join(net_phs)}")

            ph_all_str = ", ".join(placeholders) if placeholders else "all recorded telemetry"
            if not ph_sections:
                ph_details_block = f"Recorded operational parameters: {ph_all_str}."
            else:
                ph_details_block = "; ".join(ph_sections) + f" (References: {ph_all_str})."

            exec_summary_text = " ".join(context_summary)

            return (
                f"{topic} Report\n\n"
                f"1. Executive Summary:\n"
                f"{exec_summary_text}\n\n"
                f"2. Incident Findings & Technical Parameters:\n"
                f"{ph_details_block}\n\n"
                f"3. Root Cause Analysis:\n"
                f"The exact root cause is not established by the provided information.\n\n"
                f"4. Recommended Preventative Actions:\n"
                f"- Implement automated idempotent transaction handling and retry policies.\n"
                f"- Enforce strict token rotation and network access allowlisting.\n"
                f"- Configure automated alerts for support escalation and anomaly detection."
            )
        # Task: Support Response
        elif re.search(r"(?i)\b(?:support|ticket|complaint|customer\s+response)\b", prompt_lower):
            ph_str = ", ".join(placeholders) if placeholders else "your ticket"
            return (
                f"Dear Customer,\n\n"
                f"Thank you for contacting customer support. We have reviewed your request regarding {ph_str}. "
                f"Our team is actively addressing the issue and will provide an update shortly.\n\n"
                f"Best regards,\nCustomer Support Team"
            )
        # Task: Email Drafting (only when explicit draft/write intent exists)
        elif re.search(r"(?i)\b(?:draft|write|compose|send)\s+(?:a\s+|an\s+)?(?:professional\s+|formal\s+|follow-up\s+)?(?:email|letter)\b", prompt_lower) or "dear " in prompt_lower:
            sender = placeholders[0] if placeholders else "Sender"
            target = placeholders[1] if len(placeholders) > 1 else "the team"
            return (
                f"Subject: Formal Communication\n\n"
                f"Dear {target},\n\n"
                f"I am writing to formally follow up on our previous discussion regarding the project requirements. "
                f"Please let me know your availability for a brief sync.\n\n"
                f"Best regards,\n{sender}"
            )
        elif "summarize" in prompt_lower or "summary" in prompt_lower:
            entities_str = ", ".join(placeholders) if placeholders else "the documented items"
            return f"Summary:\nThe primary subject concerns {entities_str}, highlighting key operational milestones and deliverables."
        elif "fix" in prompt_lower or "debug" in prompt_lower or "code" in prompt_lower:
            return (
                "Here is the corrected code implementation:\n\n"
                "```python\ndef process_data():\n    # Secure operational routine\n    return True\n```"
            )
        else:
            if placeholders:
                ph_summary = " ".join(placeholders)
                return f"Thank you for reaching out. I have processed the request regarding {ph_summary} successfully."
            return f"I have received your prompt: \"{prompt.strip()}\" and processed it successfully."


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini API connector via REST.
    Uses google-genai / requests with graceful error messaging when keys are absent.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model_name = model_name

    def get_provider_name(self) -> str:
        return f"GeminiProvider ({self.model_name})"

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        if not self.api_key:
            raise ValueError(
                "GeminiProvider requires a valid GEMINI_API_KEY. "
                "Set the GEMINI_API_KEY environment variable or pass api_key to constructor."
            )

        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        contents = []
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": f"System Instructions: {system_prompt}"}]
            })
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })

        payload = {"contents": contents}
        timeout = kwargs.get("timeout", 30)

        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if response.status_code != 200:
            raise RuntimeError(f"Gemini API error ({response.status_code}): {response.text}")

        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as err:
            raise RuntimeError(f"Failed to parse Gemini API response: {data}") from err


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API connector via REST.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name

    def get_provider_name(self) -> str:
        return f"OpenAIProvider ({self.model_name})"

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        if not self.api_key:
            raise ValueError(
                "OpenAIProvider requires a valid OPENAI_API_KEY. "
                "Set the OPENAI_API_KEY environment variable or pass api_key to constructor."
            )

        import requests
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
        }
        timeout = kwargs.get("timeout", 30)

        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI API error ({response.status_code}): {response.text}")

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as err:
            raise RuntimeError(f"Failed to parse OpenAI API response: {data}") from err
