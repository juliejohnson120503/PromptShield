from typing import List, Optional, Dict, Tuple
from backend.models import (
    DetectedEntity,
    BaselineMaskResult,
    TaskType,
    ContextAnalysisResult,
)
from backend.detector import SensitiveDataDetector
from backend.mapping_store import MappingStore
from backend.baseline_masker import BaselineMasker
from backend.task_classifier import TaskClassifier
from backend.context_analyzer import ContextAnalyzer
from backend.policy_engine import PolicyEngine, PolicyReport


class PromptShieldCore:
    """
    Primary interface for PromptShield AI (Phases 1, 2 & 3).
    Coordinates:
    - Phase 1: Entity detection, normalization, baseline mapping.
    - Phase 2: Task classification and contextual entity role analysis.
    - Phase 3: Risk assessment and privacy policy decisions (MASK/RETAIN/USER_APPROVAL).
    """

    def __init__(self, use_spacy: bool = True):
        self.detector = SensitiveDataDetector(use_spacy=use_spacy)
        self.mapping_store = MappingStore()
        self.masker = BaselineMasker(mapping_store=self.mapping_store)
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
        Execute Phase 1 detection and baseline placeholder substitution.
        Stores the placeholder-to-secret mapping strictly within the local store.

        NOTE: This performs foundational baseline masking.
        Context-aware selective retention/masking is introduced in Phase 4.
        """
        entities = self.detector.detect(prompt)
        return self.masker.mask(prompt, entities, session_id=session_id)

    def get_local_mappings(self, session_id: str) -> Dict[str, str]:
        """Retrieve the local mapping dictionary for a given session."""
        return self.mapping_store.get_mappings(session_id)

    def clear_session(self, session_id: str) -> bool:
        """Remove a session from local mapping memory."""
        return self.mapping_store.clear_session(session_id)

