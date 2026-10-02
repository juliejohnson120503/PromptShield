from typing import List, Optional, Dict, Set, Tuple, Any
from backend.models import (
    DetectedEntity,
    BaselineMaskResult,
    SemanticMaskingResult,
    TaskType,
    ContextAnalysisResult,
    RestorationPolicy,
    RestorationResult,
    ShieldedExchangeResult,
)
from backend.detector import SensitiveDataDetector
from backend.mapping_store import MappingStore
from backend.baseline_masker import BaselineMasker
from backend.semantic_masker import SemanticMasker
from backend.restoration import ResponseRestorer
from backend.llm_provider import BaseLLMProvider, MockLLMProvider
from backend.task_classifier import TaskClassifier
from backend.context_analyzer import ContextAnalyzer
from backend.policy_engine import PolicyEngine, PolicyReport


class PromptShieldCore:
    """
    Primary interface for PromptShield AI (Phases 1, 2, 3, 4 & 5).
    Coordinates:
    - Phase 1: Entity detection, normalization, baseline mapping.
    - Phase 2: Task classification and contextual entity role analysis.
    - Phase 3: Risk assessment and privacy policy decisions (MASK/RETAIN/USER_APPROVAL/REVIEW).
    - Phase 4: Context-aware semantic masking — executes Phase 3 decisions.
    - Phase 5: Downstream LLM interaction and controlled response restoration.
    """

    def __init__(self, use_spacy: bool = True, llm_provider: Optional[BaseLLMProvider] = None):
        self.detector = SensitiveDataDetector(use_spacy=use_spacy)
        self.mapping_store = MappingStore()
        self.masker = BaselineMasker(mapping_store=self.mapping_store)  # Phase 1 baseline
        self.semantic_masker = SemanticMasker(mapping_store=self.mapping_store)  # Phase 4
        self.restorer = ResponseRestorer(mapping_store=self.mapping_store)  # Phase 5
        self.llm_provider: BaseLLMProvider = llm_provider or MockLLMProvider()
        self.task_classifier = TaskClassifier()
        self.context_analyzer = ContextAnalyzer()
        self.policy_engine = PolicyEngine()

    def analyze(self, prompt: str) -> List[DetectedEntity]:
        """
        Detect sensitive structured entities and named entities in prompt.
        Does not alter the prompt.
        """
        return self.detector.detect(prompt)

    def classify_task(self, prompt: str) -> Tuple[TaskType, float, List[str]]:
        """
        Identify the prompt's task category (e.g. GENERAL_QA, EMAIL_GENERATION).
        """
        return self.task_classifier.classify(prompt)

    def analyze_context(self, prompt: str) -> ContextAnalysisResult:
        """
        Execute Phase 2 context understanding:
        1. Detect entities in prompt
        2. Classify task intent
        3. Determine contextual roles (e.g., personal PII vs public figures/facts)
        """
        entities = self.detector.detect(prompt)
        task_type, conf, cues = self.task_classifier.classify(prompt)
        return self.context_analyzer.analyze(
            prompt=prompt,
            entities=entities,
            task_type=task_type,
            task_confidence=conf,
            task_cues=cues,
        )

    def evaluate_policy(self, prompt: str) -> PolicyReport:
        """
        Execute Phase 3 Risk Assessment & Privacy Policy:
        1. Detect entities (Phase 1)
        2. Classify task and analyze contextual roles (Phase 2)
        3. Score risk and decide MASK / RETAIN / USER_APPROVAL per entity (Phase 3)

        Returns a PolicyReport — the authoritative input for Phase 4 masking.
        """
        context_result = self.analyze_context(prompt)
        return self.policy_engine.evaluate(context_result)

    def sanitize_baseline(
        self, prompt: str, session_id: Optional[str] = None
    ) -> BaselineMaskResult:
        """
        [Phase 1 BASELINE] Detection + blind placeholder substitution.
        Masks ALL detected entities regardless of context or policy.
        Retained for evaluation / debugging comparison against Phase 4.

        NOTE: Use sanitize() for production context-aware masking.
        """
        entities = self.detector.detect(prompt)
        return self.masker.mask(prompt, entities, session_id=session_id)

    def sanitize(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        approved_entity_ids: Optional[Set[str]] = None,
    ) -> SemanticMaskingResult:
        """
        [Phase 4] Context-Aware Semantic Masking.

        Full pipeline: Detect (P1) -> Context (P2) -> Policy (P3) -> Semantic Mask (P4).
        Respects MASK / RETAIN / USER_APPROVAL / REVIEW decisions from Phase 3.

        Parameters
        ----------
        prompt : str
            The user prompt to sanitize.
        session_id : str, optional
            Session identifier for the local MappingStore.
        approved_entity_ids : set of str, optional
            Set of stable entity IDs approved by the user. Approved USER_APPROVAL
            entities are released as literals in the sanitized prompt.

        Returns
        -------
        SemanticMaskingResult
            Contains sanitized_prompt, local mapping, and per-decision entity lists.
        """
        policy_report = self.evaluate_policy(prompt)
        return self.semantic_masker.mask(
            prompt=prompt,
            policy_report=policy_report,
            approved_entity_ids=approved_entity_ids,
            session_id=session_id,
        )

    def approve_and_sanitize(
        self,
        prompt: str,
        approved_entity_ids: Set[str],
        session_id: Optional[str] = None,
    ) -> SemanticMaskingResult:
        """
        [Phase 4] Re-run semantic masking after user provides explicit approval.

        Approved entity IDs ('<TYPE>:<normalized_value>') are released as
        literal values in the resulting sanitized prompt.
        """
        policy_report = self.evaluate_policy(prompt)
        return self.semantic_masker.approve_and_remask(
            prompt=prompt,
            policy_report=policy_report,
            approved_entity_ids=approved_entity_ids,
            session_id=session_id,
        )

    def get_local_mappings(self, session_id: str) -> Dict[str, str]:
        """Retrieve the local mapping dictionary for a given session."""
        return self.mapping_store.get_mappings(session_id)

    def clear_session(self, session_id: str) -> bool:
        """Remove a session from local mapping memory."""
        return self.mapping_store.clear_session(session_id)

    # -------------------------------------------------------------------------
    # PHASE 5: CONTROLLED RESTORATION & LLM INTEGRATION
    # -------------------------------------------------------------------------

    def set_llm_provider(self, provider: BaseLLMProvider) -> None:
        """Attach a pluggable LLM provider (e.g. GeminiProvider, OpenAIProvider, MockLLMProvider)."""
        self.llm_provider = provider

    def restore_response(
        self,
        llm_response: str,
        session_id: Optional[str] = None,
        policy: Optional[RestorationPolicy] = None,
    ) -> RestorationResult:
        """
        [Phase 5] Restore placeholder values in an external LLM response using local session mapping.
        """
        return self.restorer.restore(
            llm_response=llm_response,
            session_id=session_id,
            policy=policy,
        )

    def execute_and_restore(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        approved_entity_ids: Optional[Set[str]] = None,
        restoration_policy: Optional[RestorationPolicy] = None,
        system_prompt: Optional[str] = None,
        **llm_kwargs,
    ) -> ShieldedExchangeResult:
        """
        [Phase 5 End-to-End Orchestrator] Complete prompt protection, generation, and response restoration.

        Pipeline:
        1. Sanitize prompt using Phase 4 Context-Aware Semantic Masking.
        2. Dispatch protected/sanitized prompt to active LLMProvider.
        3. Restore placeholders in LLM response using authorized local mapping.
        4. Return full ShieldedExchangeResult audit bundle.
        """
        # Step 1: Sanitize prompt (Phases 1-4)
        masking_result = self.sanitize(
            prompt=prompt,
            session_id=session_id,
            approved_entity_ids=approved_entity_ids,
        )
        active_session_id = masking_result.session_id

        # Step 2: Forward sanitized prompt to LLM (raw secrets never leave client)
        raw_response = self.llm_provider.generate(
            prompt=masking_result.sanitized_prompt,
            system_prompt=system_prompt,
            **llm_kwargs,
        )

        # Step 3: Controlled restoration of LLM response using client-side local mapping
        restoration_result = self.restorer.restore(
            llm_response=raw_response,
            session_id=active_session_id,
            policy=restoration_policy,
        )

        # Step 4: Return unified audit bundle
        return ShieldedExchangeResult(
            original_prompt=prompt,
            sanitized_prompt=masking_result.sanitized_prompt,
            raw_llm_response=raw_response,
            restored_response=restoration_result.restored_text,
            session_id=active_session_id,
            masking_result=masking_result,
            restoration_result=restoration_result,
            provider_name=self.llm_provider.get_provider_name(),
        )
