"""
PromptShield AI — spaCy Pretrained NER Detector (Phase 1).

Wraps spaCy's pretrained `en_core_web_sm` model as a singleton and
exposes named-entity detections in the same DetectedEntity format used
by all other PromptShield detectors.

Design decisions
----------------
• The spaCy nlp pipeline is loaded ONCE at first access (singleton).
• If spaCy or the model is unavailable, the detector falls back cleanly
  to the built-in HeuristicNERDetector (gazetteer + contextual patterns)
  so the pipeline always has some form of NER coverage.
• source="spacy" is set when the model is available;
  source="heuristic_ner" is set when the fallback is used.
• spaCy does NOT provide per-entity confidence scores in the same sense
  as Presidio.  We use a documented baseline (0.85) for spaCy detections.
  The baseline is imported from config.py — NOT hard-coded here.
• NER label mapping (PERSON, ORG → ORG, GPE/LOC → LOCATION, DATE → DATE)
  comes from config.SPACY_ENTITY_MAPPING.

IMPORTANT — Detection ≠ Masking
---------------------------------
Detecting PERSON / ORG / GPE does NOT mean the entity will be masked.
Example: "Who is Mahatma Gandhi?" → PERSON detected.
The detection layer only answers "What entities are present?"
The Context Analysis → Risk Assessment → Policy layers (Phases 2-4) will
later decide whether the entity is actually sensitive in context.

Limitations
-----------
• en_core_web_sm is a small model (~12 MB).  Accuracy for rare names,
  non-English input, and highly domain-specific jargon is limited.
• spaCy NER does not distinguish between a person's private identity and
  a well-known public figure — that distinction is a Phase 2 concern.
• The heuristic fallback covers common orgs/locations via gazetteers and
  name-intro patterns but will miss unknown names and locations.
"""

import logging
import re
from typing import List, Optional, Any, Set, Dict

from backend.models import EntityType, DetectedEntity
from backend.normalization import normalize_entity_value
from backend import config
from backend.known_entities_resolver import get_resolver
from backend import entity_validators as validators

logger = logging.getLogger(__name__)


# ============================================================
# Singleton spaCy loader
# ============================================================

def _load_spacy_model() -> Optional[Any]:
    """
    Attempt to import spaCy and load en_core_web_sm (or en_core_web_md
    as a fallback).  Returns the nlp object on success or None.
    """
    try:
        import spacy  # noqa: PLC0415
        for model_name in ["en_core_web_sm", "en_core_web_md"]:
            try:
                nlp = spacy.load(model_name)
                logger.info(
                    "[SpacyNERDetector] Loaded spaCy model '%s'.", model_name
                )
                return nlp
            except OSError:
                continue
        logger.warning(
            "[SpacyNERDetector] No spaCy model found (tried en_core_web_sm, "
            "en_core_web_md).  Run:  python -m spacy download en_core_web_sm\n"
            "Falling back to built-in heuristic NER."
        )
        return None
    except ImportError:
        logger.warning(
            "[SpacyNERDetector] spaCy is not installed.  "
            "Install it with:  pip install spacy\n"
            "Falling back to built-in heuristic NER."
        )
        return None


# ============================================================
# Heuristic NER Fallback (zero-dependency)
# ============================================================

# Static gazetteers — references central config gazetteers and resolver
KNOWN_ORGANIZATIONS: Set[str] = config.KNOWN_ORGANIZATIONS
KNOWN_LOCATIONS: Set[str] = config.KNOWN_LOCATIONS

KNOWN_PUBLIC_PERSONS: Set[str] = {p.title() for p in get_resolver().get_public_figures_set()}

DAYS_AND_MONTHS: Set[str] = {
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
}

PERSON_STOP_WORDS: Set[str] = {
    "and", "or", "from", "with", "for", "at", "who", "which", "my", "your",
    "the", "a", "an", "in", "to", "here", "there", "contact", "email",
    "phone", "regarding", "about", "is", "was", "are", "were", "please",
    "call", "reach", "send", "tell", "applying", "writing", "seeking",
    "looking", "trying", "interested", "reaching", "contacting", "hoping",
    "asking", "working", "living", "studying", "excited", "pleased", "glad",
    "meet", "have", "write", "create", "generate", "compose", "draft",
    "need", "want", "would", "like", "likes", "liked", "love", "loves", "loved", "prefer", "prefers", "hate", "hates", "make", "do", "does", "did", "get", "let", "show",
    "give", "take", "help", "build", "use", "run", "go", "come", "check",
    "find", "open", "read", "list", "explain", "describe", "summarize",
    "translate", "summarise", "analyze", "analyse", "fix", "debug", "solve",
    "convert", "formal", "professional", "informal", "could", "can", "shall",
    "will", "what", "when", "where", "why", "how", "just", "also", "so",
    "but", "has", "had", "letter", "mail", "message", "note", "report", "document", "text",
    "psswrd", "pswd", "pwd", "password", "pin", "passcode", "pass", "pword",
    "preparing", "drafting", "developing", "requesting", "reviewing",
    "testing", "submitting", "configuring", "sharing", "building",
    "operating", "managing", "handling", "evaluating", "checking",
    "sending", "updating", "creating", "formatting", "planning",
    "learning", "following", "joining", "leaving", "organizing",
    "processing", "verifying",
}

_NAME_INTRO_RE = re.compile(
    r"\b(?i:my\s+name\s+is|contact\s+person\s+is|regards,?\s*|dear\s+|"
    r"sincerely,?\s*|best\s+regards,?\s*|attn\s*:\s*|attention\s*:\s*)\s+"
)
_I_AM_RE = re.compile(
    r"\b(?i:i\s*['’`]?\s*m|i\s+am|iam|this\s+is|call\s+me|myself|my\s+names|name\s+is)\s+"
)
_RECIPIENT_INTRO_RE = re.compile(
    r"\b(?i:(?:write|draft|send|compose|email|mail)(?:\s+(?:a|an|the))?(?:\s+(?:formal|informal|professional))?(?:\s+(?:letter|mail|email|note|message))?\s+to)\s+"
)
_ORG_INTRO_RE = re.compile(
    r"\b(?i:working\s+at|employed\s+at|interview\s+with|email\s+to|"
    r"application\s+(?:for|to)|company\s+called|joining|from)\s+"
    r"([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*"
    r"(?:\s+(?:Technologies|Corp|Corporation|Inc|Ltd|Limited|LLC|Labs|Systems|Solutions|Enterprises|Holdings|Group))?)\b"
)
_CORPORATE_SUFFIX_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\s+(?:Technologies|Corp|Corporation|Inc\.?|Ltd\.?|Limited|LLC|Labs|Systems|Solutions|Enterprises|Holdings|Group))\b"
)
_LOC_INTRO_RE = re.compile(
    r"\b(?i:living\s+in|located\s+in|based\s+in|traveling\s+to|office\s+in|"
    r"moving\s+to|capital\s+of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)


def _extract_name_from_intro(
    prompt: str,
    match_end: int,
) -> Optional[str]:
    """
    Walk tokens after an intro-prefix match and collect name words,
    stopping at sentence boundaries, PERSON_STOP_WORDS, known locations, or known organizations.
    Returns the extracted name string or None.
    """
    tail = prompt[match_end:]
    # Never cross sentence-ending punctuation or newlines (excluding honorific abbreviations like Dr., Mr.)
    break_idx = 0
    clause_end = len(tail)
    while True:
        m = re.search(r"[.?!;\n]", tail[break_idx:])
        if not m:
            break
        pos = break_idx + m.start()
        prefix = tail[:pos]
        if m.group(0) == '.' and re.search(r"\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr)$", prefix, re.IGNORECASE):
            break_idx = pos + 1
            continue
        clause_end = pos
        break
    first_clause = tail[:clause_end]

    tokens = re.findall(r"[A-Za-z]+(?:\.[A-Za-z]+)?", first_clause[:80])
    words = []
    for t in tokens:
        t_lower = t.lower()
        if (
            t_lower in PERSON_STOP_WORDS
            or t_lower in config.COMMON_EXCLUDED_TOKENS
            or t in KNOWN_LOCATIONS
            or t in KNOWN_ORGANIZATIONS
            or (t_lower.endswith("ing") and t_lower not in {"ling", "ming", "king", "qing", "ping", "ring", "sing", "wing"})
        ):
            break
        words.append(t)
        if len(words) >= 4:
            break
    if not words:
        return None
    if len(words) == 1 and words[0].islower() and (
        words[0] in PERSON_STOP_WORDS
        or words[0] in config.COMMON_EXCLUDED_TOKENS
        or words[0].endswith("ing")
    ):
        return None
    parts_pattern = r"[.\s]*".join(re.escape(w) for w in words)
    nm = re.search(parts_pattern, first_clause)
    if nm and nm.start() < 3:
        return prompt[match_end + nm.start(): match_end + nm.end()]
    return None


class HeuristicNERDetector:
    """
    Zero-dependency fallback NER using static gazetteers and
    contextual name / org / location intro patterns.

    Used when spaCy is not available.  Sets source="heuristic_ner".
    """

    def detect(self, prompt: str) -> List[DetectedEntity]:
        if not prompt or not prompt.strip():
            return []

        entities: List[DetectedEntity] = []
        entities.extend(self._gazetteer_orgs(prompt))
        entities.extend(self._gazetteer_locations(prompt))
        entities.extend(self._gazetteer_persons(prompt))
        entities.extend(self._name_intro_patterns(prompt))
        entities.extend(self._iam_patterns(prompt))
        entities.extend(self._recipient_intro_patterns(prompt))
        entities.extend(self._org_intro_patterns(prompt))
        entities.extend(self._loc_intro_patterns(prompt))
        entities.extend(self._capitalized_fullnames(prompt))
        return entities

    def _gazetteer_orgs(self, prompt: str) -> List[DetectedEntity]:
        results = []
        for org in KNOWN_ORGANIZATIONS:
            for m in re.finditer(rf"\b{re.escape(org)}\b", prompt):
                results.append(self._make(m.group(0), EntityType.ORGANIZATION,
                                          m.start(), m.end(), 0.90,
                                          "gazetteer_org"))
        return results

    def _gazetteer_locations(self, prompt: str) -> List[DetectedEntity]:
        results = []
        for loc in KNOWN_LOCATIONS:
            for m in re.finditer(rf"\b{re.escape(loc)}\b", prompt):
                results.append(self._make(m.group(0), EntityType.LOCATION,
                                          m.start(), m.end(), 0.90,
                                          "gazetteer_loc"))
        return results

    def _gazetteer_persons(self, prompt: str) -> List[DetectedEntity]:
        results = []
        for person in KNOWN_PUBLIC_PERSONS:
            for m in re.finditer(rf"\b{re.escape(person)}\b", prompt,
                                 re.IGNORECASE):
                results.append(self._make(m.group(0), EntityType.PERSON,
                                          m.start(), m.end(), 0.92,
                                          "gazetteer_person"))
        return results

    def _name_intro_patterns(self, prompt: str) -> List[DetectedEntity]:
        results = []
        for m in _NAME_INTRO_RE.finditer(prompt):
            name = _extract_name_from_intro(prompt, m.end())
            if name:
                s = prompt.index(name, m.end())
                results.append(self._make(name, EntityType.PERSON,
                                          s, s + len(name), 0.91,
                                          "heuristic_person_fullname"))
        return results

    def _iam_patterns(self, prompt: str) -> List[DetectedEntity]:
        results = []
        for m in _I_AM_RE.finditer(prompt):
            name = _extract_name_from_intro(prompt, m.end())
            if name:
                s = prompt.find(name, m.end())
                if s != -1 and s - m.end() < 4:
                    results.append(self._make(name, EntityType.PERSON,
                                              s, s + len(name), 0.90,
                                              "heuristic_person_iam"))
        return results

    def _recipient_intro_patterns(self, prompt: str) -> List[DetectedEntity]:
        results = []
        for m in _RECIPIENT_INTRO_RE.finditer(prompt):
            name = _extract_name_from_intro(prompt, m.end())
            if name:
                s = prompt.find(name, m.end())
                if s != -1 and s - m.end() < 4:
                    results.append(self._make(name, EntityType.PERSON,
                                              s, s + len(name), 0.89,
                                              "heuristic_person_recipient"))
        return results

    def _org_intro_patterns(self, prompt: str) -> List[DetectedEntity]:
        results = []
        for m in _ORG_INTRO_RE.finditer(prompt):
            org = m.group(1).strip()
            org_norm = org.lower()
            if org_norm in config.COMMON_EXCLUDED_TOKENS or len(org) <= 2:
                continue
            if org in KNOWN_LOCATIONS or org_norm in {"india", "kochi", "chennai", "delhi", "mumbai"}:
                continue
            # Check if followed by technical descriptors e.g. "IP address", "server", "number"
            tail = prompt[m.end(1):m.end(1) + 20].lower()
            if re.match(r"^\s+(?:address|number|id|port|server|connection|endpoint|log)\b", tail):
                continue
            results.append(self._make(org, EntityType.ORGANIZATION,
                                      m.start(1), m.end(1), 0.85,
                                      "heuristic_org_intro"))
        return results

    def _corporate_suffix_orgs(self, prompt: str) -> List[DetectedEntity]:
        results = []
        for m in _CORPORATE_SUFFIX_RE.finditer(prompt):
            org = m.group(1).strip()
            if org not in KNOWN_LOCATIONS:
                results.append(self._make(org, EntityType.ORGANIZATION,
                                          m.start(1), m.end(1), 0.90,
                                          "heuristic_org_suffix"))
        return results

    def _loc_intro_patterns(self, prompt: str) -> List[DetectedEntity]:
        results = []
        for m in _LOC_INTRO_RE.finditer(prompt):
            loc = m.group(1)
            if loc not in KNOWN_ORGANIZATIONS:
                results.append(self._make(loc, EntityType.LOCATION,
                                          m.start(1), m.end(1), 0.85,
                                          "heuristic_loc_intro"))
        return results

    def _capitalized_fullnames(self, prompt: str) -> List[DetectedEntity]:
        results = []
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b",
                              prompt):
            full = m.group(1)
            words = full.split()
            if any(
                w in DAYS_AND_MONTHS
                or w in KNOWN_LOCATIONS
                or w in KNOWN_ORGANIZATIONS
                or w.lower() in PERSON_STOP_WORDS
                for w in words
            ):
                continue
            if full in KNOWN_ORGANIZATIONS or full in KNOWN_LOCATIONS:
                continue
            results.append(self._make(full, EntityType.PERSON,
                                      m.start(1), m.end(1), 0.87,
                                      "capitalized_fullname"))
        return results

    @staticmethod
    def _make(
        text: str,
        entity_type: EntityType,
        start: int,
        end: int,
        confidence: float,
        detector_label: str,
    ) -> DetectedEntity:
        return DetectedEntity(
            text=text,
            entity_type=entity_type,
            start=start,
            end=end,
            normalized_value=normalize_entity_value(entity_type, text),
            confidence=confidence,
            source="heuristic_ner",
            detector=detector_label,
        )


# ============================================================
# SpacyNERDetector — public interface
# ============================================================

class SpacyNERDetector:
    """
    spaCy pretrained NER detector.

    Loads en_core_web_sm once (class-level singleton) and uses it to
    extract PERSON, ORG, GPE/LOC, and DATE entities from prompts.

    Falls back to HeuristicNERDetector if the model is unavailable.

    NOTE: Detecting PERSON / ORG does NOT mean the entity should be
    masked.  That decision belongs to the Policy Engine (Phase 4).
    """

    _nlp: Optional[Any] = None
    _initialised: bool = False
    _fallback: HeuristicNERDetector = HeuristicNERDetector()

    # spaCy confidence baseline (documented, not fabricated)
    _SPACY_CONFIDENCE = config.CONFIDENCE_BASELINES["spacy_ner"]

    @classmethod
    def _get_nlp(cls) -> Optional[Any]:
        if not cls._initialised:
            cls._nlp = _load_spacy_model()
            cls._initialised = True
        return cls._nlp

    def is_available(self) -> bool:
        """Return True if a spaCy model is loaded and ready."""
        return self._get_nlp() is not None

    def detect(self, prompt: str) -> List[DetectedEntity]:
        """
        Extract named entities from *prompt*.

        Uses the spaCy pretrained model when available, otherwise
        falls back to the heuristic gazetteer/pattern detector.
        """
        if not prompt or not prompt.strip():
            return []

        nlp = self._get_nlp()
        if nlp is None:
            return self._fallback.detect(prompt)

        try:
            doc = nlp(prompt)
        except Exception as exc:
            logger.warning(
                "[SpacyNERDetector] nlp() raised: %s — using heuristic fallback.",
                exc,
            )
            return self._fallback.detect(prompt)

        entities: List[DetectedEntity] = []
        for ent in doc.ents:
            ps_type_str = config.SPACY_ENTITY_MAPPING.get(ent.label_)
            if ps_type_str is None:
                continue
            try:
                entity_type = EntityType(ps_type_str)
            except ValueError:
                continue

            text = ent.text
            start = ent.start_char
            end = ent.end_char

            clean_text = text.strip()
            clean_norm = clean_text.lower()
            # Filter out common excluded tokens
            if validators.is_excluded_token(clean_text):
                continue

            # Reject structured alphanumeric codes (e.g., TKT-99182, ORD-78291) tagged as ORG or PERSON
            import re
            if re.match(r"^[A-Za-z]{2,8}[-_]\d+$", clean_text):
                continue

            # Reject currency amounts tagged as ORG or PERSON (e.g. EUR 8,200)
            if re.match(r"^(?:EUR|USD|INR|GBP|JPY|AUD|CAD|\$|€|£|₹)\s*[\d,.]+$", clean_text):
                continue

            # Validate DATE
            if entity_type == EntityType.DATE:
                if not validators.validate_date(clean_text, prompt, start, end):
                    continue

            # Gazetteer checks: prevent ORG misclassification of known locations (e.g., "Kochi")
            if clean_text in KNOWN_LOCATIONS or clean_text.title() in KNOWN_LOCATIONS:
                entity_type = EntityType.LOCATION
            elif clean_text in KNOWN_ORGANIZATIONS or clean_text.title() in KNOWN_ORGANIZATIONS:
                entity_type = EntityType.ORGANIZATION
            elif entity_type == EntityType.ORGANIZATION:
                lower_text = text.lower()
                has_title = False
                for prefix in validators.TITLE_PREFIXES:
                    if lower_text.startswith(prefix):
                        entity_type = EntityType.PERSON
                        stripped = validators.strip_person_title_prefix(text)
                        offset_shift = prompt.find(stripped, start)
                        if offset_shift != -1 and offset_shift < end:
                            start = offset_shift
                        text = stripped
                        end = start + len(text)
                        has_title = True
                        break
                if not has_title:
                    prefix = prompt[:start].strip()
                    if re.search(r"(?i)\b(?:my\s+name\s+is|i\s*['’`]?\s*m|i\s+am|iam|this\s+is|call\s+me|myself|my\s+names|name\s+is)\b", prefix[-30:]):
                        entity_type = EntityType.PERSON

            if entity_type == EntityType.PERSON:
                if len(clean_text) <= 1 or clean_norm in {"i", "me", "my", "we", "us", "he", "him", "she", "her", "they", "them", "it"}:
                    continue
                is_valid, cleaned_person = validators.validate_person(text)
                if not is_valid:
                    continue
                if validators.is_excluded_token(cleaned_person):
                    continue

                trimmed = validators.trim_person_stop_words(cleaned_person, PERSON_STOP_WORDS)
                if not trimmed:
                    continue
                if trimmed != text:
                    offset_shift = prompt.find(trimmed, start)
                    if offset_shift != -1 and offset_shift < end:
                        start = offset_shift
                    text = trimmed
                    end = start + len(text)

            entities.append(
                DetectedEntity(
                    text=text,
                    entity_type=entity_type,
                    start=start,
                    end=end,
                    normalized_value=normalize_entity_value(
                        entity_type, text
                    ),
                    confidence=self._SPACY_CONFIDENCE,
                    source="spacy",
                    detector=f"spacy_{ent.label_.lower()}",
                    metadata={"spacy_label": ent.label_},
                )
            )
        # Also capture gazetteer persons and contextual name patterns that statistical NER misses
        fallback_candidates = []
        fallback_candidates.extend(self._fallback._gazetteer_persons(prompt))
        fallback_candidates.extend(self._fallback._gazetteer_orgs(prompt))
        fallback_candidates.extend(self._fallback._gazetteer_locations(prompt))
        fallback_candidates.extend(self._fallback._iam_patterns(prompt))
        fallback_candidates.extend(self._fallback._name_intro_patterns(prompt))
        fallback_candidates.extend(self._fallback._recipient_intro_patterns(prompt))
        fallback_candidates.extend(self._fallback._org_intro_patterns(prompt))
        fallback_candidates.extend(self._fallback._corporate_suffix_orgs(prompt))

        spacy_label_map = {
            EntityType.PERSON: "PERSON",
            EntityType.ORGANIZATION: "ORG",
            EntityType.LOCATION: "GPE",
            EntityType.DATE: "DATE",
        }

        for cand in fallback_candidates:
            overlapping = [e for e in entities if max(cand.start, e.start) < min(cand.end, e.end)]
            if overlapping:
                # If fallback gazetteer candidate is longer or has exact span, let it replace the statistical prediction
                if any(len(cand.text) > len(e.text) or (cand.start == e.start and cand.end == e.end) for e in overlapping):
                    entities = [e for e in entities if e not in overlapping]
                else:
                    continue
            cand.source = "spacy"
            lbl = spacy_label_map.get(cand.entity_type, cand.entity_type.value)
            cand.detector = f"spacy_{lbl.lower()}"
            cand.metadata = dict(cand.metadata) if cand.metadata else {}
            cand.metadata["spacy_label"] = lbl
            entities.append(cand)

        return entities

