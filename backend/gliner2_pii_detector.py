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
import re
from typing import Any, Dict, List, Optional

from backend.models import DetectedEntity, EntityType
from backend.normalization import normalize_entity_value
from backend import config

logger = logging.getLogger(__name__)

CORPORATE_SUFFIX_PATTERN = re.compile(
    r"(?i)\b(?:inc\.?|corp\.?|corporation|llc|ltd\.?|limited|co\.?|technologies|solutions|group|holdings|enterprises|systems|studios|labs|foundation|university|institute)\b"
)

# Expose constants for backward compatibility
GLINER2_PII_LABELS = config.GLINER2_PII_LABELS
GLINER2_TO_CANONICAL = config.GLINER2_ENTITY_MAPPING
GLINER2_MODEL_NAME = config.GLINER2_MODEL_NAME
GLINER2_DEFAULT_THRESHOLD = config.GLINER2_DEFAULT_THRESHOLD


def _check_memory_available(min_mb: float = 650.0) -> bool:
    """Ensure sufficient free physical RAM before loading large transformer weights."""
    import platform
    if platform.system() == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                avail_mb = stat.ullAvailPhys / (1024 * 1024)
                if avail_mb < min_mb:
                    logger.warning(
                        "[GLiNER2PIIDetector] Low physical memory (%.1f MB available, need >= %.1f MB). "
                        "Skipping GLiNER2 neural model to prevent OS pagefile crash; hybrid detectors remain active.",
                        avail_mb,
                        min_mb,
                    )
                    return False
        except Exception:
            pass
    return True


def _load_gliner2_model(model_name: str = config.GLINER2_MODEL_NAME) -> Optional[Any]:
    """
    Attempt to load GLiNER2 model using the official gliner2 package.
    Returns the model instance on success, or None on failure.
    """
    if not _check_memory_available():
        return None

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

            # --- Excluded tokens filter (keywords falsely tagged as entities) ---
            clean_norm = text.lower().strip()
            if clean_norm in config.COMMON_EXCLUDED_TOKENS or (
                clean_norm.startswith("the ") and clean_norm[4:].strip() in config.COMMON_EXCLUDED_TOKENS
            ):
                logger.debug(
                    "[GLiNER2PIIDetector] Excluded token '%s' in COMMON_EXCLUDED_TOKENS — skipping.",
                    text,
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

            # If classified as EMAIL but lacks '@', skip
            if entity_type == EntityType.EMAIL and "@" not in text:
                continue

            # Clean EMAIL trailing sentence text (e.g. .The) or trailing punctuation
            if entity_type == EntityType.EMAIL:
                email_sent_match = re.match(r"^([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?:\.([A-Z][a-z]+.*))?$", text)
                if email_sent_match and email_sent_match.group(2):
                    text = email_sent_match.group(1)
                    end = start + len(text)
                elif text.endswith("."):
                    text = text.rstrip(".")
                    end = start + len(text)

            # If classified as IP_ADDRESS but lacks '.' or ':', skip
            if entity_type == EntityType.IP_ADDRESS and ("." not in text and ":" not in text):
                continue

            # If classified as PERSON:
            if entity_type == EntityType.PERSON:
                # Strip trailing possessive suffixes (e.g. "Priya Nair's" -> "Priya Nair")
                if re.search(r"['’]s?$", text):
                    trimmed = re.sub(r"['’]s?$", "", text).rstrip()
                    if trimmed:
                        text = trimmed
                        end = start + len(text)

                # Reject malformed PERSON spans containing sentence breaks, punctuation sequences, or directive keywords
                if re.search(r"[.!?;:\n]{1,}\s*[-–—]|\.[a-zA-Z]{2,}|\b(?<![A-Z])\.[A-Z]", text):
                    continue
                words = [re.sub(r"^\W+|\W+$", "", w).lower() for w in text.split()]
                disqualifying = {
                    "api", "key", "token", "password", "secret", "credential", "database",
                    "server", "ip", "endpoint", "must", "should", "shall", "cannot", "can",
                    "will", "would", "could", "please", "report", "explain", "recommend",
                    "incident", "order", "ticket", "customer", "support",
                }
                if any(w in disqualifying for w in words):
                    continue

                if CORPORATE_SUFFIX_PATTERN.search(text):
                    continue

            # Format validation for BANK_ACCOUNT / IBAN:
            # Structured account numbers and IBANs must contain digits. Purely alphabetic text cannot be a bank account.
            # Do NOT silently relabel an incorrect native GLiNER2 label as ORGANIZATION.
            if entity_type == EntityType.BANK_ACCOUNT:
                has_digits = any(c.isdigit() for c in text)
                if not has_digits:
                    continue

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
