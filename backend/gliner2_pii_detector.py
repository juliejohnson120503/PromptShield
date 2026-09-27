"""
PromptShield AI — GLiNER2-PII Detector (Production)
===================================================
Wraps the production GLiNER2-PII model (fastino/gliner2-privacy-filter-PII-multi)
using the official gliner2 package API.

KEY DESIGN PRINCIPLES
---------------------
• Preserves ALL 42 native GLiNER2-PII labels — never drops or collapses
  them inside the detector itself.
• Provides an ADDITIONAL canonical PromptShield label via backend.config.GLINER2_ENTITY_MAPPING.
• Unknown / unmapped native labels map to EntityType.PII_OTHER (safe fallback with audit trail).
• Identifies itself as source="gliner2" with detector="gliner2_<native_label>".
• Graceful degradation: if gliner2 is not installed or the model fails to
  load, detect() returns [] safely without raising exceptions.
• Span integrity: every entity is verified with text[start:end] == text
  before being returned.
"""

import logging
import importlib.util
from typing import Any, Dict, List, Optional

from backend.models import DetectedEntity, EntityType
from backend.normalization import normalize_entity_value
from backend import config

logger = logging.getLogger(__name__)

# Expose constants for backward compatibility
GLINER2_PII_LABELS = config.GLINER2_PII_LABELS
GLINER2_TO_CANONICAL = config.GLINER2_ENTITY_MAPPING
GLINER2_MODEL_NAME = config.GLINER2_MODEL_NAME
GLINER2_DEFAULT_THRESHOLD = config.GLINER2_DEFAULT_THRESHOLD


def _load_gliner2_model(model_name: str = config.GLINER2_MODEL_NAME) -> Optional[Any]:
    """
    Attempt to load GLiNER2 model using the official gliner2 package.
    Returns the model instance on success, or None on failure.
    """
    try:
        import sys
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        if hasattr(sys.stderr, "reconfigure"):
            try:
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

        from gliner2 import GLiNER2
        logger.info("[GLiNER2PIIDetector] Loading production GLiNER2 model: %s ...", model_name)
        model = GLiNER2.from_pretrained(model_name)
        logger.info("[GLiNER2PIIDetector] Model %s loaded successfully.", model_name)
        return model
    except ImportError:
        logger.warning(
            "[GLiNER2PIIDetector] 'gliner2' package not installed. "
            "Install with: pip install \"gliner2[local]\""
        )
        return None
    except Exception as exc:
        logger.warning(
            "[GLiNER2PIIDetector] Failed to load model '%s': %s",
            model_name, exc,
        )
        return None


class GLiNER2PIIDetector:
    """
    GLiNER2-PII Open-Vocabulary PII Detector.

    Uses fastino/gliner2-privacy-filter-PII-multi with the complete 42-label taxonomy.
    Preserves native GLiNER2 labels; maps them cleanly into PromptShield's
    canonical EntityType schema.

    The detector identifies itself as source="gliner2".
    """

    _model: Optional[Any] = None
    _initialised: bool = False

    def __init__(
        self,
        labels: Optional[List[str]] = None,
        threshold: float = config.GLINER2_DEFAULT_THRESHOLD,
        model_name: str = config.GLINER2_MODEL_NAME,
    ):
        self.labels = labels or list(config.GLINER2_PII_LABELS)
        self.threshold = threshold
        self.model_name = model_name

    @classmethod
    def _get_model(cls, model_name: str = config.GLINER2_MODEL_NAME) -> Optional[Any]:
        if not cls._initialised:
            cls._model = _load_gliner2_model(model_name)
            cls._initialised = True
        return cls._model

    @classmethod
    def is_installed(cls) -> bool:
        """Return True if the gliner2 package is installed."""
        try:
            return importlib.util.find_spec("gliner2") is not None
        except Exception:
            return False

    def is_available(self) -> bool:
        """Return True if the GLiNER2 model is loaded and ready."""
        return self._get_model(self.model_name) is not None

    def detect(self, prompt: str) -> List[DetectedEntity]:
        """
        Extract PII entities from *prompt* using GLiNER2-PII.

        For every detection:
          - native GLiNER2 label is preserved in entity.metadata["gliner2_label"]
          - canonical PromptShield label is stored in entity.entity_type
          - entities with unknown native labels map to EntityType.PII_OTHER
          - source is "gliner2"
          - span integrity is verified: prompt[start:end] == entity.text

        Returns [] gracefully if model is unavailable or prompt is empty.
        """
        if not prompt or not prompt.strip():
            return []

        model = self._get_model(self.model_name)
        if model is None:
            return []

        raw_entities = []
        if hasattr(model, "extract_entities"):
            try:
                extraction = model.extract_entities(
                    prompt,
                    self.labels,
                    threshold=self.threshold,
                    include_confidence=True,
                    include_spans=True,
                )
                entity_dict = extraction.get("entities", extraction) if isinstance(extraction, dict) else {}
                for lbl, items in entity_dict.items():
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                raw_entities.append({
                                    "label": lbl,
                                    "text": item.get("text", ""),
                                    "start": item.get("start", -1),
                                    "end": item.get("end", -1),
                                    "score": item.get("confidence", self.threshold),
                                })
            except Exception as exc:
                logger.warning("[GLiNER2PIIDetector] extract_entities failed: %s", exc)
                return []
        elif hasattr(model, "predict_entities"):
            try:
                raw_entities = model.predict_entities(
                    prompt,
                    self.labels,
                    flat_ner=True,
                    threshold=self.threshold,
                )
            except Exception as exc:
                logger.warning("[GLiNER2PIIDetector] predict_entities failed: %s", exc)
                return []

        entities: List[DetectedEntity] = []

        for ent in raw_entities:
            gliner2_label = ent.get("label", "").lower().strip()
            text          = ent.get("text", "")
            start         = ent.get("start", -1)
            end           = ent.get("end", -1)
            raw_score     = float(ent.get("score", self.threshold))

            # --- Span integrity check ---
            if start < 0 or end <= start or end > len(prompt):
                logger.debug(
                    "[GLiNER2PIIDetector] Invalid span [%d:%d] for '%s' — skipping.",
                    start, end, text,
                )
                continue

            actual = prompt[start:end]
            if actual != text:
                logger.debug(
                    "[GLiNER2PIIDetector] Span mismatch: prompt[%d:%d]='%s' != '%s' — skipping.",
                    start, end, actual, text,
                )
                continue

            # --- Canonical label lookup ---
            canonical_str = config.GLINER2_ENTITY_MAPPING.get(gliner2_label)
            is_unsupported = False

            if canonical_str is not None:
                try:
                    entity_type = EntityType(canonical_str)
                except ValueError:
                    logger.warning(
                        "[GLiNER2PIIDetector] Canonical string '%s' not a valid EntityType",
                        canonical_str,
                    )
                    entity_type = EntityType.PII_OTHER
                    is_unsupported = True
            else:
                logger.warning(
                    "[GLiNER2PIIDetector] Unknown native label '%s' mapped to PII_OTHER",
                    gliner2_label,
                )
                entity_type = EntityType.PII_OTHER
                is_unsupported = True

            entities.append(
                DetectedEntity(
                    text=text,
                    entity_type=entity_type,
                    start=start,
                    end=end,
                    normalized_value=normalize_entity_value(entity_type, text),
                    confidence=round(raw_score, 4),
                    source="gliner2",
                    detector=f"gliner2_{gliner2_label}",
                    metadata={
                        "gliner2_label":     gliner2_label,
                        "canonical_label":   entity_type.value,
                        "is_unsupported":    is_unsupported,
                        "raw_score":         raw_score,
                    },
                )
            )

        return entities

    def get_status(self) -> Dict[str, Any]:
        """Return a status dict for diagnostics."""
        loaded = self.is_available()
        return {
            "active":       loaded,
            "model_loaded": loaded,
            "model_name":   self.model_name,
            "labels_count": len(self.labels),
            "threshold":    self.threshold,
            "note": (
                f"GLiNER2-PII loaded ({len(self.labels)} labels)"
                if loaded
                else "GLiNER2-PII unavailable or not installed"
            ),
        }
