"""
PromptShield AI — Microsoft Presidio Detector (Phase 1).

Wraps the presidio_analyzer.AnalyzerEngine as a singleton and exposes
Presidio PII detections in the same DetectedEntity format used by all
other PromptShield detectors.

Design decisions
----------------
• The AnalyzerEngine is loaded ONCE at first access (singleton) so
  model initialisation overhead is paid only once per process lifetime.
• If presidio-analyzer is not installed or fails to load, the detector
  sets _available=False, logs a clear warning, and returns [] on every
  call — it NEVER raises an exception into the calling pipeline.
• Presidio's own confidence score (result.score in [0,1]) is preserved
  AS-IS; we do NOT substitute a hard-coded value.
• Entity types are normalised from Presidio labels to PromptShield
  EntityType using the mapping in backend.config.PRESIDIO_ENTITY_MAPPING.
  Unknown Presidio types are silently ignored.
• source="presidio" is set on all returned entities.
• This detector does NOT decide whether an entity must be masked.

Limitations
-----------
• Presidio's built-in recognisers are primarily trained on US/UK data.
  Non-English or regional formats may have lower recall.
• Presidio's spaCy-based PERSON recogniser requires a language model.
  If en_core_web_sm is not available, Presidio falls back to its own
  pattern-based heuristics for PERSON detection.
• Confidence scores from Presidio reflect its internal threshold and
  should not be compared directly to regex confidence values.
"""

import logging
from typing import List, Optional, Any

from backend.models import EntityType, DetectedEntity
from backend.normalization import normalize_entity_value
from backend import config

logger = logging.getLogger(__name__)


def _load_analyzer() -> Optional[Any]:
    """
    Attempt to import and initialise presidio_analyzer.AnalyzerEngine.
    Returns the engine instance on success or None on failure.
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        engine = AnalyzerEngine()
        logger.info("[PresidioDetector] AnalyzerEngine initialised successfully.")
        return engine
    except ImportError:
        logger.warning(
            "[PresidioDetector] presidio-analyzer is not installed. "
            "Install it with:  pip install presidio-analyzer\n"
            "Presidio detection will be SKIPPED — other detectors remain active."
        )
        return None
    except Exception as exc:
        logger.warning(
            "[PresidioDetector] AnalyzerEngine failed to initialise: %s\n"
            "Presidio detection will be SKIPPED — other detectors remain active.",
            exc,
        )
        return None


class PresidioDetector:
    """
    Microsoft Presidio-backed PII detector.

    The AnalyzerEngine is a class-level singleton so it is shared
    across all instances created during a process lifetime.
    """

    _engine: Optional[Any] = None
    _initialised: bool = False

    @classmethod
    def _get_engine(cls) -> Optional[Any]:
        if not cls._initialised:
            cls._engine = _load_analyzer()
            cls._initialised = True
        return cls._engine

    def is_available(self) -> bool:
        """Return True if the Presidio AnalyzerEngine is loaded and ready."""
        return self._get_engine() is not None

    def detect(self, prompt: str) -> List[DetectedEntity]:
        """
        Run Presidio analysis over *prompt* and return normalised
        DetectedEntity objects.

        Returns an empty list (gracefully) if Presidio is unavailable or
        the prompt is empty.
        """
        engine = self._get_engine()
        if engine is None or not prompt or not prompt.strip():
            return []

        try:
            results = engine.analyze(
                text=prompt,
                entities=config.PRESIDIO_REQUESTED_ENTITIES,
                language="en",
            )
        except Exception as exc:
            logger.warning("[PresidioDetector] analyze() raised: %s", exc)
            return []

        entities: List[DetectedEntity] = []
        for result in results:
            # Normalise Presidio entity type → PromptShield EntityType string
            ps_type_str = config.PRESIDIO_ENTITY_MAPPING.get(result.entity_type)
            if ps_type_str is None:
                # Unknown / unmapped Presidio type — ignore gracefully
                continue

            try:
                entity_type = EntityType(ps_type_str)
            except ValueError:
                logger.debug(
                    "[PresidioDetector] Cannot map '%s' to EntityType — skipping.",
                    ps_type_str,
                )
                continue

            raw_text = prompt[result.start: result.end]
            entities.append(
                DetectedEntity(
                    text=raw_text,
                    entity_type=entity_type,
                    start=result.start,
                    end=result.end,
                    normalized_value=normalize_entity_value(entity_type, raw_text),
                    # Presidio provides its own score — preserve it AS-IS.
                    # Do NOT fabricate or override this value.
                    confidence=round(result.score, 4),
                    source="presidio",
                    detector=f"presidio_{result.entity_type.lower()}",
                    metadata={
                        "presidio_type": result.entity_type,
                        "presidio_score": result.score,
                        "recognition_metadata": (
                            result.recognition_metadata
                            if hasattr(result, "recognition_metadata")
                            else {}
                        ),
                    },
                )
            )

        return entities
