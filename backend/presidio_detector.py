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
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
        try:
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
            })
            nlp_engine = provider.create_engine()
            engine = AnalyzerEngine(nlp_engine=nlp_engine)
        except Exception:
            engine = AnalyzerEngine()

        # Custom structured recognizers for enterprise IDs with strict word boundaries
        cust_recognizer = PatternRecognizer(
            supported_entity="CUSTOMER_ID",
            patterns=[Pattern("cust_id_pat", r"\b(?:CUST|CUSTOMER)[-_][A-Za-z0-9]{4,16}\b", 0.95)],
        )
        ord_recognizer = PatternRecognizer(
            supported_entity="ORDER_ID",
            patterns=[Pattern("ord_id_pat", r"\b(?:ORD|ORDER)[-_][A-Za-z0-9]{4,16}\b", 0.95)],
        )
        tkt_recognizer = PatternRecognizer(
            supported_entity="TICKET_ID",
            patterns=[Pattern("tkt_id_pat", r"\b(?:TKT|TICKET|SR|INC)[-_][A-Za-z0-9]{4,16}\b", 0.95)],
        )
        engine.registry.add_recognizer(cust_recognizer)
        engine.registry.add_recognizer(ord_recognizer)
        engine.registry.add_recognizer(tkt_recognizer)

        logger.info("[PresidioDetector] AnalyzerEngine initialised with custom ID recognizers.")
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
        import re
        for result in results:
            # Score filter: drop low-confidence Presidio noise (< 0.40)
            if result.score < 0.40:
                continue

            raw_text = prompt[result.start: result.end]
            clean_text = raw_text.strip().lower()

            # Exclude common tokens
            if clean_text in config.COMMON_EXCLUDED_TOKENS:
                continue

            # Normalise Presidio entity type → PromptShield EntityType string
            ps_type_str = config.PRESIDIO_ENTITY_MAPPING.get(result.entity_type)
            if ps_type_str is None:
                continue

            try:
                entity_type = EntityType(ps_type_str)
            except ValueError:
                logger.debug(
                    "[PresidioDetector] Cannot map '%s' to EntityType — skipping.",
                    ps_type_str,
                )
                continue

            # Filter false positive dates that actually describe street addresses or roads
            if entity_type == EntityType.DATE:
                if re.search(r"(?i)\b(?:road|rd\.?|street|st\.?|avenue|ave\.?|lane|ln\.?|drive|dr\.?|mg|marg|nagar|sector)\b", raw_text):
                    continue
                # Reject isolated month words like "MAY", "march" that do not form a date expression
                MONTH_WORDS = {
                    "january", "february", "march", "april", "may", "june",
                    "july", "august", "september", "october", "november", "december",
                    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
                }
                if clean_text in MONTH_WORDS:
                    w_before = prompt[max(0, result.start - 25):result.start]
                    w_after = prompt[result.end:min(len(prompt), result.end + 25)]
                    has_nearby_digits = bool(re.search(r"\b\d{1,4}(?:st|nd|rd|th)?\b", w_before + " " + w_after))
                    has_date_prep = bool(re.search(r"(?i)\b(?:in|on|during|dated|since|until|by|before|after|of)\s*$", w_before.strip()))
                    if not (has_nearby_digits or has_date_prep):
                        continue

            # Clean EMAIL trailing sentence text (e.g. .The) or trailing punctuation
            if entity_type == EntityType.EMAIL:
                email_sent_match = re.match(r"^([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\.([A-Z][a-z]+.*)$", raw_text)
                if email_sent_match:
                    raw_text = email_sent_match.group(1)
                    result.end = result.start + len(raw_text)
                elif raw_text.endswith("."):
                    raw_text = raw_text.rstrip(".")
                    result.end = result.start + len(raw_text)

            # Strip trailing action verbs/stop words for PERSON entities (e.g. "amina write" -> "amina")
            if entity_type == EntityType.PERSON:
                # Strip trailing possessive suffixes (e.g. "Priya Nair's" -> "Priya Nair")
                if re.search(r"['’]s?$", raw_text):
                    trimmed = re.sub(r"['’]s?$", "", raw_text).rstrip()
                    if trimmed:
                        raw_text = trimmed
                        result.end = result.start + len(trimmed)

                # Reject malformed PERSON spans containing sentence boundaries or punctuation sequences
                if re.search(r"[.!?;:\n]{1,}\s*[-–—]|\.[a-zA-Z]{2,}|\b(?<![A-Z])\.[A-Z]", raw_text):
                    continue

                # Reject spans containing technical/directive words (e.g. "API key.- You MUST")
                disqualifying = {
                    "api", "key", "token", "password", "secret", "credential", "database",
                    "server", "ip", "endpoint", "must", "should", "shall", "cannot", "can",
                    "will", "would", "could", "please", "report", "explain", "recommend",
                    "incident", "order", "ticket", "customer", "support",
                }
                words_clean = [re.sub(r"^\W+|\W+$", "", w).lower() for w in raw_text.split()]
                if any(w in disqualifying for w in words_clean):
                    continue

                from backend.spacy_detector import PERSON_STOP_WORDS
                words = raw_text.split()
                while len(words) > 1 and words[-1].lower() in PERSON_STOP_WORDS:
                    words.pop()
                if not words:
                    continue
                trimmed = " ".join(words)
                if trimmed != raw_text:
                    raw_text = trimmed
                    result.end = result.start + len(trimmed)

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
