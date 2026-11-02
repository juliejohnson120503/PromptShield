"""
PromptShield AI — Detection Engine (Phase 1 — Hybrid Refactor).

This module is the TOP-LEVEL ORCHESTRATOR of the detection pipeline.
It does NOT contain patterns, ML logic, or conflict-resolution code —
those responsibilities have been delegated to dedicated modules:

    backend/regex_detector.py    — structured / deterministic detection
    backend/presidio_detector.py — Microsoft Presidio PII detection
    backend/spacy_detector.py    — spaCy pretrained NER (+ heuristic fallback)
    backend/entity_fusion.py     — deduplication & overlap resolution

Pipeline (per prompt)
---------------------
    User Prompt
        │
        ▼
    RegexDetector.detect()          ← EMAIL, PHONE, CREDIT_CARD (Luhn),
        │                              API_KEY, ACCESS_TOKEN, PASSWORD,
        │                              BANK_ACCOUNT, IP_ADDRESS, IDs, DATE
        ▼
    PresidioDetector.detect()       ← Presidio PII (if available)
        │
        ▼
    SpacyNERDetector.detect()       ← spaCy en_core_web_sm NER
        │                              (falls back to heuristic gazetteer)
        ▼
    EntityFusionEngine.fuse()       ← merge, deduplicate, resolve overlaps
        │
        ▼
    List[DetectedEntity]            ← unified, sorted, non-overlapping

Backward-compatibility guarantee
---------------------------------
• SensitiveDataDetector.detect(prompt) still works exactly as before.
• is_luhn_valid() and detect_card_brand() remain importable from this
  module so existing tests do not need updating.
• The constructor signature SensitiveDataDetector(use_spacy=True) is
  preserved; use_spacy=False disables both spaCy and Presidio and forces
  the heuristic NER fallback (mirrors previous behaviour).

Detection ≠ Masking
--------------------
This detector identifies candidate sensitive / named entities.
It does NOT decide whether any entity must be masked.
That decision belongs to:
    Phase 2 — Context Analysis
    Phase 3 — Risk Assessment & Privacy Policy
    Phase 4 — Context-Aware Semantic Masking
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from backend.models import DetectedEntity, EntityType
from backend.regex_detector import RegexDetector, is_luhn_valid, detect_card_brand
from backend.presidio_detector import PresidioDetector
from backend.gliner2_pii_detector import GLiNER2PIIDetector
from backend.spacy_detector import SpacyNERDetector
from backend.entity_fusion import EntityFusionEngine
from backend.normalization import canonicalize_adversarial_text, remap_entity_offsets

# Re-export for backward compatibility with existing tests that import
# is_luhn_valid and detect_card_brand from backend.detector.
__all__ = [
    "SensitiveDataDetector",
    "is_luhn_valid",
    "detect_card_brand",
]

logger = logging.getLogger(__name__)


class SensitiveDataDetector:
    """
    Hybrid detection engine for PromptShield AI.

    Orchestrates complementary detector subsystems across 3 architectural tiers:
      1. RegexDetector       — fast, deterministic, zero-ML (API keys, cards+Luhn, passwords, emails, phones)
      2. PresidioDetector    — enterprise compliance PII (Microsoft Presidio)
      3. GLiNER2PIIDetector  — deep neural PII privacy filter model (zero-shot, 42 native PII labels)
      4. SpacyNERDetector    — pretrained statistical NER (+ contextual heuristics fallback)

    Results are merged by EntityFusionEngine into a single sorted,
    non-overlapping list of DetectedEntity objects.

    Parameters
    ----------
    use_spacy : bool
        If False, spaCy and Presidio are disabled; the detector runs in
        pure regex + heuristic NER mode (zero external dependencies).
        Default: True.
    use_presidio : bool
        Explicitly enable or disable Presidio regardless of use_spacy.
        Default: True (active when use_spacy is True).
    use_gliner2 : bool
        Enable or disable GLiNER2 neural PII detector.
        Default: True (active when available and use_spacy is True).
    use_gliner : Optional[bool]
        Backward-compatibility alias for use_gliner2.
    """

    def __init__(
        self,
        use_spacy: bool = True,
        use_presidio: bool = True,
        use_gliner2: bool = True,
        use_gliner: Optional[bool] = None,
    ):
        if use_gliner is not None:
            use_gliner2 = use_gliner

        self._use_spacy = use_spacy
        self._use_presidio = use_presidio and use_spacy  # Presidio needs NLP stack
        self._use_gliner2 = use_gliner2 and use_spacy

        self._regex = RegexDetector()
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
        self._use_gliner = self._use_gliner2
        self._gliner = self._gliner2
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="ps_detector")

        logger.info(
            "[SensitiveDataDetector] Initialised — "
            "regex=ON, presidio=%s, gliner2=%s, spacy=%s",
            "ON" if self._presidio else "OFF",
            "ON" if (self._gliner2 and self._gliner2.is_installed()) else "OFF/fallback",
            "ON (+" + ("spacy" if (self._spacy_ner and self._spacy_ner.is_available())
                       else "heuristic") + ")" if self._use_spacy else "OFF",
        )

    def close(self) -> None:
        """Shut down the worker thread pool."""
        if hasattr(self, "_executor") and self._executor:
            self._executor.shutdown(wait=False)

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, prompt: str) -> List[DetectedEntity]:
        """
        Analyse *prompt* and return all detected entities.

        The returned list is:
          • non-overlapping (overlaps resolved by EntityFusionEngine)
          • sorted by ascending start offset
          • free of duplicate detections for the same span+type
        """
        if not prompt or not prompt.strip():
            return []

        # 0. Unicode & Homoglyph Evasion Defense (canonicalize adversarial text)
        clean_prompt, offset_map = canonicalize_adversarial_text(prompt)
        target_prompt = clean_prompt if offset_map is not None else prompt

        candidates: List[DetectedEntity] = []

        # 1. Structured / deterministic regex detection (fast, synchronous on main thread < 1 ms)
        regex_results = self._regex.detect(target_prompt)
        candidates.extend(regex_results)
        logger.debug("[Detector] Regex found %d candidates.", len(regex_results))

        # 2. Concurrently execute heavy detectors (Presidio, GLiNER2, and spaCy)
        futures = []

        if self._presidio is not None:
            futures.append(self._executor.submit(self._presidio.detect, target_prompt))

        if self._gliner2 is not None and self._gliner2.is_available():
            futures.append(self._executor.submit(self._gliner2.detect, target_prompt))

        if self._spacy_ner is not None:
            futures.append(self._executor.submit(self._spacy_ner.detect, target_prompt))
        else:
            from backend.spacy_detector import HeuristicNERDetector
            heuristic_det = HeuristicNERDetector()
            futures.append(self._executor.submit(heuristic_det.detect, target_prompt))

        # Collect results from all concurrent subsystems
        for future in futures:
            try:
                sub_results = future.result()
                candidates.extend(sub_results)
            except Exception as exc:
                logger.warning("[Detector] Subsystem detection failed in worker thread: %s", exc)

        # 3. Fusion: deduplicate, resolve overlaps, sort
        fused = self._fusion.fuse(candidates, target_prompt)

        # 4. Remap entity offsets back to original prompt if adversarial defense transformed text
        if offset_map is not None:
            fused = remap_entity_offsets(fused, prompt, offset_map)

        logger.debug(
            "[Detector] Fusion reduced %d candidates → %d entities.",
            len(candidates), len(fused),
        )
        return fused

    @staticmethod
    def format_explainable_report(entities: List[DetectedEntity]) -> str:
        """
        Format detected entities as an explainable report:
        ENTITY | TYPE | SOURCE | CONFIDENCE
        """
        if not entities:
            return "No entities detected."
        lines = [
            f"{'ENTITY':<30} | {'TYPE':<15} | {'SOURCE':<15} | {'CONFIDENCE':<10}",
            "-" * 78,
        ]
        for e in entities:
            src = e.source if hasattr(e, "source") else "unknown"
            conf = f"{e.confidence:.2f}"
            lines.append(f"{e.text:<30} | {e.entity_type.value:<15} | {src:<15} | {conf:<10}")
        return "\n".join(lines)

    def get_detector_status(self) -> Dict[str, Any]:
        """
        Return a status dictionary showing which backends are active.
        Useful for diagnostics and the final implementation report.
        """
        presidio_ok = (
            self._presidio.is_available() if self._presidio else False
        )
        gliner2_ok = (
            self._gliner2.is_available() if self._gliner2 else False
        )
        spacy_ok = (
            self._spacy_ner.is_available() if self._spacy_ner else False
        )
        gliner2_status = {
            "active": self._use_gliner2,
            "model_loaded": gliner2_ok,
            "note": (
                "GLiNER2 multi-category PII transformer loaded"
                if gliner2_ok
                else "unavailable or disabled"
            ),
        }
        return {
            "regex": {"active": True, "note": "Always active; zero-dependency"},
            "presidio": {
                "active": presidio_ok,
                "note": (
                    "presidio-analyzer loaded"
                    if presidio_ok
                    else "unavailable or disabled"
                ),
            },
            "gliner2": gliner2_status,
            "gliner": gliner2_status,  # Backwards compatibility alias
            "unicode_defense": {
                "active": True,
                "note": "Zero-width stripping, Cyrillic/Greek homoglyph mapping, NFKC canonicalization",
            },
            "spacy": {
                "active": self._use_spacy,
                "model_loaded": spacy_ok,
                "note": (
                    "en_core_web_sm loaded"
                    if spacy_ok
                    else "heuristic NER fallback active"
                ),
            },
        }
