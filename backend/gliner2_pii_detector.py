"""
PromptShield AI — GLiNER2-PII Detector
========================================
Wraps the GLiNER2-PII model (fastino/gliner2-pii-v1) using the official
gliner2 package API.

KEY DESIGN PRINCIPLES
---------------------
• Preserves ALL 42 native GLiNER2-PII labels — never drops or collapses
  them inside the detector itself.
• Provides an ADDITIONAL canonical PromptShield label for downstream fusion
  compatibility, but the native label is always retained in metadata.
• Identifies itself as source="gliner2_pii" so it is distinguishable from
  the old urchade/gliner_small-v2.1 detector (source="gliner").
• Graceful degradation: if gliner2 is not installed or the model fails to
  load, detect() returns [] safely.
• Span integrity: every entity is verified with text[start:end] == text
  before being returned.

IMPORTANT: This file does NOT modify or replace backend/gliner_detector.py.
The original detector (urchade/gliner_small-v2.1) remains available for
baseline reproduction.
"""

import logging
import importlib.util
from typing import Any, Dict, List, Optional, Tuple

from backend.models import DetectedEntity, EntityType
from backend.normalization import normalize_entity_value

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Full 42-label GLiNER2-PII taxonomy
# (MUST remain complete — do not remove any label)
# ---------------------------------------------------------------------------
GLINER2_PII_LABELS: List[str] = [
    # PERSON / NAMES
    "person",
    "full_name",
    "first_name",
    "middle_name",
    "last_name",
    "date_of_birth",
    # CONTACT / ADDRESS
    "email",
    "phone_number",
    "address",
    "street_address",
    "city",
    "state_or_region",
    "postal_code",
    "country",
    # GOVERNMENT / TAX IDs
    "government_id",
    "national_id_number",
    "passport_number",
    "drivers_license_number",
    "license_number",
    "tax_id",
    "tax_number",
    # BANKING / PAYMENT
    "bank_account",
    "account_number",
    "routing_number",
    "iban",
    "payment_card",
    "card_number",
    "card_expiry",
    "card_cvv",
    # DIGITAL IDENTITY
    "username",
    "ip_address",
    "account_id",
    "sensitive_account_id",
    # SECRETS / CREDENTIALS
    "password",
    "secret",
    "api_key",
    "access_token",
    "recovery_code",
    # SENSITIVE DATES
    "sensitive_date",
    "document_date",
    "expiration_date",
    "transaction_date",
]

# ---------------------------------------------------------------------------
# Mapping: GLiNER2 native label → PromptShield canonical EntityType string
# None = no equivalent in current PromptShield taxonomy (UNSUPPORTED)
# ---------------------------------------------------------------------------
GLINER2_TO_CANONICAL: Dict[str, Optional[str]] = {
    # PERSON / NAMES
    "person":                   "PERSON",
    "full_name":                "PERSON",
    "first_name":               "PERSON",
    "middle_name":              "PERSON",
    "last_name":                "PERSON",
    "date_of_birth":            "DATE",
    # CONTACT / ADDRESS
    "email":                    "EMAIL",
    "phone_number":             "PHONE",
    "address":                  "ADDRESS",
    "street_address":           "ADDRESS",
    "city":                     "LOCATION",
    "state_or_region":          "LOCATION",
    "postal_code":              "LOCATION",
    "country":                  "LOCATION",
    # GOVERNMENT / TAX IDs — no exact PromptShield equivalent
    "government_id":            None,   # UNSUPPORTED
    "national_id_number":       None,   # UNSUPPORTED
    "passport_number":          None,   # UNSUPPORTED
    "drivers_license_number":   None,   # UNSUPPORTED
    "license_number":           None,   # UNSUPPORTED
    "tax_id":                   None,   # UNSUPPORTED
    "tax_number":               None,   # UNSUPPORTED
    # BANKING / PAYMENT
    "bank_account":             "BANK_ACCOUNT",
    "account_number":           "BANK_ACCOUNT",
    "routing_number":           "BANK_ACCOUNT",
    "iban":                     "BANK_ACCOUNT",
    "payment_card":             "CREDIT_CARD",
    "card_number":              "CREDIT_CARD",
    "card_expiry":              "CREDIT_CARD",
    "card_cvv":                 "CREDIT_CARD",
    # DIGITAL IDENTITY
    "username":                 "USER_ID",
    "ip_address":               "IP_ADDRESS",
    "account_id":               "USER_ID",
    "sensitive_account_id":     "USER_ID",
    # SECRETS / CREDENTIALS
    "password":                 "PASSWORD",
    "secret":                   "API_KEY",
    "api_key":                  "API_KEY",
    "access_token":             "ACCESS_TOKEN",
    "recovery_code":            "ACCESS_TOKEN",
    # SENSITIVE DATES
    "sensitive_date":           "DATE",
    "document_date":            "DATE",
    "expiration_date":          "DATE",
    "transaction_date":         "DATE",
}

# Entity types that have no PromptShield canonical equivalent
UNSUPPORTED_GLINER2_LABELS = {
    lbl for lbl, canon in GLINER2_TO_CANONICAL.items() if canon is None
}

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
GLINER2_MODEL_NAME = "fastino/gliner2-pii-v1"
GLINER2_DEFAULT_THRESHOLD = 0.50


def _load_gliner2_model(model_name: str = GLINER2_MODEL_NAME) -> Optional[Any]:
    """
    Attempt to load GLiNER2 model using the official gliner2 package.
    Returns the model instance on success, or None on failure.
    """
    try:
        from gliner2 import GLiNER2
        logger.info("[GLiNER2PIIDetector] Loading model: %s ...", model_name)
        model = GLiNER2.from_pretrained(model_name)
        logger.info("[GLiNER2PIIDetector] Model %s loaded successfully.", model_name)
        return model
    except ImportError:
        logger.warning(
            "[GLiNER2PIIDetector] 'gliner2' package not installed. "
            "Install with: pip install gliner2"
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

    Uses fastino/gliner2-pii-v1 with the complete 42-label taxonomy.
    Preserves native GLiNER2 labels; provides canonical PromptShield labels
    as supplementary metadata for fusion engine compatibility.

    The detector identifies itself as source="gliner2_pii".
    """

    _model: Optional[Any] = None
    _initialised: bool = False

    def __init__(
        self,
        labels: Optional[List[str]] = None,
        threshold: float = GLINER2_DEFAULT_THRESHOLD,
        model_name: str = GLINER2_MODEL_NAME,
    ):
        # Always use the full 42-label list unless explicitly overridden
        self.labels = labels or list(GLINER2_PII_LABELS)
        self.threshold = threshold
        self.model_name = model_name

    @classmethod
    def _get_model(cls, model_name: str = GLINER2_MODEL_NAME) -> Optional[Any]:
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
          - canonical PromptShield label (if mappable) is in metadata["canonical_label"]
          - entities with no canonical mapping (UNSUPPORTED) are still returned
            using EntityType.PERSON as a placeholder; the true type is in metadata
          - source is always "gliner2_pii"
          - span integrity is verified: text[start:end] == entity text

        Returns [] gracefully if model is unavailable or prompt is empty.
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
            canonical_str = GLINER2_TO_CANONICAL.get(gliner2_label)
            is_unsupported = canonical_str is None

            if not is_unsupported:
                try:
                    entity_type = EntityType(canonical_str)
                except ValueError:
                    logger.debug(
                        "[GLiNER2PIIDetector] Unknown EntityType '%s' for label '%s'",
                        canonical_str, gliner2_label,
                    )
                    continue
            else:
                # Unsupported label: still report with PERSON as placeholder
                # The REAL type is in metadata["gliner2_label"]
                entity_type = EntityType.PERSON

            entities.append(
                DetectedEntity(
                    text=text,
                    entity_type=entity_type,
                    start=start,
                    end=end,
                    normalized_value=normalize_entity_value(entity_type, text),
                    confidence=round(raw_score, 4),
                    source="gliner2_pii",
                    detector=f"gliner2_{gliner2_label}",
                    metadata={
                        "gliner2_label":     gliner2_label,
                        "canonical_label":   canonical_str,       # None = UNSUPPORTED
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
