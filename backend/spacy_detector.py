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

# Static gazetteers — kept in this module for the fallback path.
# These are NOT the primary detection mechanism when spaCy is available.
KNOWN_ORGANIZATIONS: Set[str] = {
    "Google", "Microsoft", "Apple", "Amazon", "Meta", "Facebook", "Netflix",
    "OpenAI", "Anthropic", "IBM", "Intel", "Cisco", "Oracle", "Nvidia",
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
    "ABC Technologies", "XYZ Corp", "Acme Corp", "Tech Solutions",
    "Harvard University", "MIT", "Stanford University", "Oxford University",
}

KNOWN_LOCATIONS: Set[str] = {
    "India", "United States", "USA", "UK", "United Kingdom", "Canada",
    "Australia", "Germany", "France", "Japan", "China", "Singapore",
    "New York", "London", "San Francisco", "Paris", "Tokyo", "Berlin",
    "Sydney", "Toronto", "Delhi", "New Delhi", "Mumbai", "Bangalore",
    "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune", "Kochi",
    "California", "Texas", "Washington",
}

KNOWN_PUBLIC_PERSONS: Set[str] = {
    "Elon Musk", "Bill Gates", "Steve Jobs", "Albert Einstein",
    "Sundar Pichai", "Satya Nadella", "Sam Altman", "Jeff Bezos",
    "Mark Zuckerberg", "Barack Obama", "Narendra Modi", "Alan Turing",
    "Marie Curie", "Isaac Newton", "Charles Babbage", "Ada Lovelace",
    "Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio", "Andrew Ng",
    "Warren Buffett", "Tim Cook", "Jensen Huang", "Linus Torvalds",
    "Mahatma Gandhi", "Nelson Mandela", "Abraham Lincoln",
}

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
    "need", "want", "would", "like", "make", "do", "get", "let", "show",
    "give", "take", "help", "build", "use", "run", "go", "come", "check",
    "find", "open", "read", "list", "explain", "describe", "summarize",
    "translate", "summarise", "analyze", "analyse", "fix", "debug", "solve",
    "convert", "formal", "professional", "informal", "could", "can", "shall",
    "will", "what", "when", "where", "why", "how", "just", "also", "so",
    "but", "letter", "mail", "message", "note", "report", "document", "text",
    "psswrd", "pswd", "pwd", "password", "pin", "passcode", "pass", "pword",
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
    r"application\s+(?:for|to)|company\s+called|joining)\s+"
    r"([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*"
    r"(?:\s+(?:Technologies|Corp|Corporation|Inc|Ltd|Limited|LLC|Labs|Systems|Solutions))?)\\b"
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
    stopping at PERSON_STOP_WORDS, known locations, or known organizations.
    Returns the extracted name string or None.
    """
    tail = prompt[match_end:]
    tokens = re.findall(r"[A-Za-z]+(?:\.[A-Za-z]+)?", tail[:80])
    words = []
    for t in tokens:
        t_lower = t.lower()
        if (
            t_lower in PERSON_STOP_WORDS
            or t in KNOWN_LOCATIONS
            or t in KNOWN_ORGANIZATIONS
        ):
            break
        words.append(t)
        if len(words) >= 4:
            break
    if not words:
        return None
    parts_pattern = r"[.\s]*".join(re.escape(w) for w in words)
    nm = re.search(parts_pattern, tail)
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
            org = m.group(1)
            if org not in KNOWN_LOCATIONS:
                results.append(self._make(org, EntityType.ORGANIZATION,
                                          m.start(1), m.end(1), 0.85,
                                          "heuristic_org_intro"))
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
            # Filter out common excluded tokens (e.g., 'ip', 'email', 'password', etc.)
            if clean_text.lower() in config.COMMON_EXCLUDED_TOKENS:
                continue

            # Gazetteer checks: prevent ORG misclassification of known locations (e.g., "Kochi")
            if clean_text in KNOWN_LOCATIONS or clean_text.title() in KNOWN_LOCATIONS:
                entity_type = EntityType.LOCATION
            elif clean_text in KNOWN_ORGANIZATIONS or clean_text.title() in KNOWN_ORGANIZATIONS:
                entity_type = EntityType.ORGANIZATION

            if entity_type == EntityType.PERSON:
                words = text.split()
                while len(words) > 1 and words[-1].lower() in PERSON_STOP_WORDS:
                    words.pop()
                if not words:
                    continue
                trimmed_text = " ".join(words)
                if trimmed_text != text:
                    text = trimmed_text
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

        return entities
