"""
PromptShield AI — GLiNER2-PII Experimental Detector
=====================================================
This is an EXPERIMENTAL variant of SensitiveDataDetector where the ONLY
change is replacing urchade/gliner_small-v2.1 with fastino/gliner2-pii-v1.

Architecture (unchanged from production):
    Regex → Presidio → GLiNER2-PII → spaCy → EntityFusion

ALL other components are identical to the production system:
    - RegexDetector        ← unchanged
    - PresidioDetector     ← unchanged
    - SpacyNERDetector     ← unchanged
    - EntityFusionEngine   ← unchanged

The only substitution:
    GlinerNERDetector (gliner_small-v2.1) → GLiNER2PIIDetector (gliner2-pii-v1)

GLiNER2 entities with UNSUPPORTED canonical labels (e.g. national_id_number,
passport_number) are forwarded to the fusion engine with entity_type=PERSON
as a placeholder. The real label is always stored in metadata["gliner2_label"].
The fusion engine filters them by confidence/priority as normal; they will
appear in predictions with their native metadata intact.

IMPORTANT: Do NOT import this in production code. It is evaluation-only.
"""

import logging
from typing import Any, Dict, List, Optional

from backend.models import DetectedEntity, EntityType
from backend.regex_detector import RegexDetector, is_luhn_valid, detect_card_brand
from backend.presidio_detector import PresidioDetector
from backend.gliner2_pii_detector import GLiNER2PIIDetector
from backend.spacy_detector import SpacyNERDetector
from backend.entity_fusion import EntityFusionEngine

logger = logging.getLogger(__name__)


class SensitiveDataDetectorGLiNER2:
    """
    Experimental PromptShield detection pipeline with GLiNER2-PII.

    Identical to SensitiveDataDetector except GlinerNERDetector is replaced
    with GLiNER2PIIDetector using fastino/gliner2-pii-v1 and all 42 PII labels.
    """

    def __init__(
        self,
        use_spacy: bool = True,
        use_presidio: bool = True,
        use_gliner2: bool = True,
    ):
        self._use_spacy    = use_spacy
        self._use_presidio = use_presidio and use_spacy
        self._use_gliner2  = use_gliner2 and use_spacy

        self._regex    = RegexDetector()
        self._presidio: Optional[PresidioDetector] = (
            PresidioDetector() if self._use_presidio else None
        )
        self._gliner2: Optional[GLiNER2PIIDetector] = (
            GLiNER2PIIDetector() if self._use_gliner2 else None
        )
        self._spacy_ner: Optional[SpacyNERDetector] = (
            SpacyNERDetector() if self._use_spacy else None
        )
        self._fusion = EntityFusionEngine()

    def detect(self, prompt: str) -> List[DetectedEntity]:
        """Detect PII using the GLiNER2-PII experimental pipeline."""
        if not prompt or not prompt.strip():
            return []

        candidates: List[DetectedEntity] = []

        # 1. Regex
        regex_results = self._regex.detect(prompt)
        candidates.extend(regex_results)

        # 2. Presidio
        if self._presidio is not None:
            candidates.extend(self._presidio.detect(prompt))

        # 3. GLiNER2-PII (replaces old GLiNER)
        if self._gliner2 is not None and self._gliner2.is_available():
            candidates.extend(self._gliner2.detect(prompt))

        # 4. spaCy
        if self._spacy_ner is not None:
            candidates.extend(self._spacy_ner.detect(prompt))
        else:
            from backend.spacy_detector import HeuristicNERDetector
            candidates.extend(HeuristicNERDetector().detect(prompt))

        # 5. Fusion
        return self._fusion.fuse(candidates, prompt)

    def get_detector_status(self) -> Dict[str, Any]:
        gliner2_ok = self._gliner2.is_available() if self._gliner2 else False
        spacy_ok   = self._spacy_ner.is_available() if self._spacy_ner else False
        presidio_ok= self._presidio.is_available() if self._presidio else False
        return {
            "regex":      {"active": True, "note": "Always active; zero-dependency"},
            "presidio":   {"active": presidio_ok, "note": "presidio-analyzer" if presidio_ok else "unavailable"},
            "gliner2_pii": {
                "active":       gliner2_ok,
                "model_loaded": gliner2_ok,
                "model_name":   "fastino/gliner2-pii-v1",
                "labels_count": len(self._gliner2.labels) if self._gliner2 else 0,
                "note": "GLiNER2-PII (42 labels)" if gliner2_ok else "unavailable",
            },
            "spacy": {
                "active":       self._use_spacy,
                "model_loaded": spacy_ok,
                "note": "en_core_web_sm loaded" if spacy_ok else "heuristic fallback",
            },
        }
