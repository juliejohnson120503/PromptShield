"""
PromptShield AI — GLiNER Open-Vocabulary Neural NER Detector.

Wraps the GLiNER zero-shot transformer model (urchade/gliner_small-v2.1)
as a class-level singleton to perform open-vocabulary contextual entity
extraction on user prompts.

Key Strengths:
--------------
• Overcomes spaCy's rigid capitalization requirement: accurately detects
  lowercase names, abbreviations, and informal phrasing (e.g. "i m julie",
  "im amina write to elon musk").
• Open-vocabulary: queries custom entity labels dynamically without
  retraining from scratch: ["person", "organization", "location", "password", "username"].
• Pure PyTorch / HuggingFace Transformers inference: runs on CPU in ~50-100ms.
• Graceful degradation: if GLiNER or PyTorch is not available, returns []
  safely, allowing the hybrid orchestrator to fall back to spaCy and heuristics.
"""

import logging
from typing import Any, Dict, List, Optional

from backend.models import DetectedEntity, EntityType
from backend.normalization import normalize_entity_value
from backend import config

logger = logging.getLogger(__name__)


def _load_gliner_model(model_name: str = config.GLINER_MODEL_NAME) -> Optional[Any]:
    """
    Attempt to load GLiNER model.
    Returns the GLiNER instance on success, or None on failure.
    """
    try:
        from gliner import GLiNER
        logger.info("[GlinerNERDetector] Loading GLiNER model: %s ...", model_name)
        model = GLiNER.from_pretrained(model_name)
        logger.info("[GlinerNERDetector] GLiNER model %s loaded successfully.", model_name)
        return model
    except ImportError:
        logger.warning(
            "[GlinerNERDetector] 'gliner' package not installed. "
            "Install with: pip install gliner\n"
            "Falling back to spaCy / heuristic NER."
        )
        return None
    except Exception as exc:
        logger.warning(
            "[GlinerNERDetector] Failed to load GLiNER model '%s': %s. "
            "Falling back to spaCy / heuristic NER.",
            model_name,
            exc,
        )
        return None


COMMON_EXCLUDED_TOKENS = config.COMMON_EXCLUDED_TOKENS


class GlinerNERDetector:
    """
    GLiNER-backed Open-Vocabulary Neural NER Detector.

    Singleton wrapper around GLiNER model. Performs zero-shot entity
    recognition for person, organization, location, password, and username.
    """

    _model: Optional[Any] = None
    _initialised: bool = False

    def __init__(
        self,
        labels: Optional[List[str]] = None,
        threshold: float = 0.50,
        model_name: str = config.GLINER_MODEL_NAME,
    ):
        self.labels = labels or config.GLINER_DEFAULT_LABELS
        self.threshold = threshold
        self.model_name = model_name

    @classmethod
    def _get_model(cls, model_name: str = config.GLINER_MODEL_NAME) -> Optional[Any]:
        if not cls._initialised:
            cls._model = _load_gliner_model(model_name)
            cls._initialised = True
        return cls._model

    @classmethod
    def is_installed(cls) -> bool:
        """Return True if gliner package is installed."""
        try:
            import importlib.util
            return importlib.util.find_spec("gliner") is not None
        except Exception:
            return False

    def is_available(self) -> bool:
        """Return True if the GLiNER model is loaded and available."""
        return self._get_model(self.model_name) is not None

    def detect(self, prompt: str) -> List[DetectedEntity]:
        """
        Extract named and sensitive entities from *prompt* using GLiNER.

        Returns an empty list gracefully if the model is not loaded or
        prompt is empty.
        """
        if not prompt or not prompt.strip():
            return []

        model = self._get_model(self.model_name)
        if model is None:
            return []

        try:
            raw_entities = model.predict_entities(
                prompt,
                self.labels,
                flat_ner=True,
                threshold=self.threshold,
            )
        except Exception as exc:
            logger.warning("[GlinerNERDetector] predict_entities failed: %s", exc)
            return []

        entities: List[DetectedEntity] = []
        for ent in raw_entities:
            label = ent.get("label", "").lower()
            ps_type_str = config.GLINER_ENTITY_MAPPING.get(label)
            if not ps_type_str:
                continue

            try:
                entity_type = EntityType(ps_type_str)
            except ValueError:
                continue

            text = ent.get("text", "")
            start = ent.get("start", -1)
            end = ent.get("end", -1)

            if start < 0 or end <= start or end > len(prompt):
                continue

            # Verify span text matches prompt slice
            if prompt[start:end] != text:
                continue

            # Filter out common keywords and prompt labels falsely tagged as entities
            if text.lower().strip() in COMMON_EXCLUDED_TOKENS:
                continue

            score = float(ent.get("score", config.CONFIDENCE_BASELINES.get("gliner", 0.88)))

            entities.append(
                DetectedEntity(
                    text=text,
                    entity_type=entity_type,
                    start=start,
                    end=end,
                    normalized_value=normalize_entity_value(entity_type, text),
                    confidence=round(score, 3),
                    source="gliner",
                    detector=f"gliner_{label}",
                    metadata={"gliner_label": label, "raw_score": score},
                )
            )

        return entities
