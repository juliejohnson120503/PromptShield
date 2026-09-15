"""
Detection Engine for PromptShield AI (Phase 1).
Provides deterministic regex/rule detection for structured sensitive entities:
- EMAIL
- PHONE
- CREDIT_CARD (with Luhn validation)
- API_KEY
- PASSWORD/CREDENTIAL
And hybrid NER detection for unstructured entities:
- PERSON
- ORGANIZATION
- LOCATION
- DATE
Includes conflict resolution for overlapping spans and entity normalization.
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from backend.models import EntityType, DetectedEntity
from backend.normalization import normalize_entity_value


# ==========================================
# LUHN VALIDATOR (Credit Card Verification)
# ==========================================

def is_luhn_valid(number_str: str) -> bool:
    """
    Validate a credit card number string using the Luhn (mod 10) algorithm.
    Filters out arbitrary digit sequences and order numbers.
    """
    digits = [int(d) for d in re.sub(r"\D", "", number_str)]
    if len(digits) < 13 or len(digits) > 19:
        return False

    checksum = 0
    reverse_digits = digits[::-1]
    for idx, digit in enumerate(reverse_digits):
        if idx % 2 == 1:
            doubled = digit * 2
            checksum += (doubled - 9) if doubled > 9 else doubled
        else:
            checksum += digit
    return checksum % 10 == 0


def detect_card_brand(card_digits: str) -> str:
    """Identify the credit card issuer brand based on leading digits."""
    clean = re.sub(r"\D", "", card_digits)
    if clean.startswith("4"):
        return "Visa"
    elif clean.startswith(("51", "52", "53", "54", "55")) or (
        len(clean) >= 4 and 2221 <= int(clean[:4]) <= 2720
    ):
        return "Mastercard"
    elif clean.startswith(("34", "37")):
        return "American Express"
    elif clean.startswith(("6011", "65")) or (
        len(clean) >= 6 and 622126 <= int(clean[:6]) <= 622925
    ):
        return "Discover"
    elif clean.startswith(("300", "301", "302", "303", "304", "305", "36", "38")):
        return "Diners Club"
    return "Unknown"


# ==========================================
# STRUCTURED REGEX DETECTORS
# ==========================================

# Email regex (RFC 5322 compliant subset)
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# Phone number regex: supports international codes, brackets, dashes, spaces
PHONE_PATTERNS = [
    # International formatted: e.g., +91 98765 43210, +1 (555) 123-4567, +44 20 7946 0958
    re.compile(r"(?:\+\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,5}\b"),
    # Standard 10-digit number without punctuation: e.g. 9876543210, 8005551234
    re.compile(r"\b(?:[6-9]\d{9}|[2-9]\d{2}[2-9]\d{6})\b"),
]

# Credit Card Candidate pattern (digits separated by hyphens or spaces, or contiguous 13-19 digits)
CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,4}|\d{4}[ -]?\d{6}[ -]?\d{4,5}|\d{13,19})\b"
)

# API Key Patterns
API_KEY_PATTERNS = [
    # OpenAI API keys: sk-..., sk-proj-..., sk-live-...
    (re.compile(r"\b(?:sk-(?:proj-|live-|test-)?[A-Za-z0-9_\-]{20,})\b"), "openai_api_key", 0.99),
    # Google API Key: AIza... (typically 39 characters total)
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{30,45}\b"), "google_api_key", 0.99),
    # GitHub Tokens: ghp_..., github_pat_...
    (re.compile(r"\b(?:ghp_[0-9a-zA-Z]{36}|github_pat_[0-9a-zA-Z_]{22,})\b"), "github_pat", 0.99),
    # AWS Access Key ID: AKIA...
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "aws_access_key", 0.98),
    # Generic Key/Secret assignment: api_key = "...", token: '...'
    (re.compile(r"(?i)\b(?:api[_-]?key|secret[_-]?token|auth[_-]?token|access[_-]?token)\s*[:=]\s*['\"]?([A-Za-z0-9_\-\.]{16,})['\"]?"), "generic_key_assignment", 0.95),
    # Bearer tokens
    (re.compile(r"(?i)\bbearer\s+([A-Za-z0-9_\-\.]{20,})\b"), "bearer_token", 0.93),
]

# Password / Credential Patterns
PASSWORD_PATTERNS = [
    # password = "...", pwd: "...", pass: '...'
    re.compile(r"(?i)\b(?:password|passwd|pwd|passphrase|userpass)\s*[:=]\s*['\"]?([^'\"\s\r\n]{4,})['\"]?"),
    # "my password is ..."
    re.compile(r"(?i)\b(?:my\s+(?:login\s+)?password\s+is\s+)(['\"]?([^'\"\s\r\n]+)['\"]?)"),
    # "credentials: username/password"
    re.compile(r"(?i)\b(?:credentials?\s*[:=]\s*['\"]?([^'\"\s\r\n]+)['\"]?)"),
]

# Date patterns
DATE_PATTERNS = [
    # ISO-like YYYY-MM-DD or YYYY/MM/DD
    re.compile(r"\b(?:19|20)\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])\b"),
    # DD-MM-YYYY or DD/MM/YYYY
    re.compile(r"\b(?:0[1-9]|[12]\d|3[01])[-/.](?:0[1-9]|1[0-2])[-/.](?:19|20)\d{2}\b"),
    # Month DD, YYYY or DD Month YYYY: e.g. January 15, 2025 or 15th August 1947
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+(?:19|20)\d{2}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(?:,)?\s+(?:19|20)\d{2}\b",
        re.IGNORECASE,
    ),
]


# ==========================================
# BUILT-IN NER / GAZETTEER FALLBACK
# ==========================================

KNOWN_ORGANIZATIONS = {
    "Google", "Microsoft", "Apple", "Amazon", "Meta", "Facebook", "Netflix",
    "OpenAI", "Anthropic", "IBM", "Intel", "Cisco", "Oracle", "Nvidia",
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
    "ABC Technologies", "XYZ Corp", "Acme Corp", "Tech Solutions",
    "Harvard University", "MIT", "Stanford University", "Oxford University"
}

KNOWN_LOCATIONS = {
    "India", "United States", "USA", "UK", "United Kingdom", "Canada", "Australia",
    "Germany", "France", "Japan", "China", "Singapore", "New York", "London",
    "San Francisco", "Paris", "Tokyo", "Berlin", "Sydney", "Toronto",
    "Delhi", "New Delhi", "Mumbai", "Bangalore", "Bengaluru", "Hyderabad",
    "Chennai", "Kolkata", "Pune", "California", "Texas", "Washington"
}

KNOWN_PERSONS = {
    "Elon Musk", "Bill Gates", "Steve Jobs", "Albert Einstein",
    "Sundar Pichai", "Satya Nadella", "Sam Altman", "Jeff Bezos",
    "Mark Zuckerberg", "Barack Obama", "Narendra Modi", "Alan Turing",
    "Marie Curie", "Isaac Newton", "Charles Babbage", "Ada Lovelace",
    "Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio", "Andrew Ng",
    "Warren Buffett", "Tim Cook", "Jensen Huang", "Linus Torvalds"
}

DAYS_AND_MONTHS = {
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December"
}

PERSON_STOP_WORDS = {
    # Conjunctions, prepositions, articles
    "and", "or", "from", "with", "for", "at", "who", "which", "my", "your",
    "the", "a", "an", "in", "to", "here", "there", "contact", "email", "phone",
    "regarding", "about", "is", "was", "are", "were", "please", "call", "reach",
    "send", "tell", "applying", "writing", "seeking", "looking", "trying",
    "interested", "reaching", "contacting", "hoping", "asking", "working",
    "living", "studying", "excited", "pleased", "glad", "meet", "have",
    # Task / action verbs — terminate name sequence in informal prompts
    # e.g. "i'm amina write a formal..." must stop after "amina"
    "write", "create", "generate", "compose", "draft", "need", "want",
    "would", "like", "make", "do", "get", "let", "show", "give", "take",
    "help", "build", "use", "run", "go", "come", "check", "find", "open",
    "read", "list", "explain", "describe", "summarize", "translate",
    "summarise", "analyze", "analyse", "fix", "debug", "solve", "convert",
    "formal", "professional", "informal", "could", "can", "shall", "will",
    "what", "when", "where", "why", "how", "just", "also", "so", "but",
    "letter", "mail", "message", "note", "report", "document", "text",
}

# Explicit name introduction prefixes
NAME_INTRO_PREFIX_REGEX = re.compile(
    r"\b(?i:my\s+name\s+is|contact\s+person\s+is|regards,?\s*|dear\s+|sincerely,?\s*|best\s+regards,?\s*)\s+"
)
I_AM_NAME_PREFIX_REGEX = re.compile(
    r"\b(?i:i\s+am|i\'m|this\s+is|call\s+me|myself)\s+"
)

ORG_INTRO_PATTERN = re.compile(
    r"\b(?i:working\s+at|employed\s+at|interview\s+with|email\s+to|application\s+(?:for|to)|company\s+called|joining)\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*(?:\s+(?:Technologies|Corp|Corporation|Inc|Ltd|Limited|LLC|Labs|Systems|Solutions))?)\b"
)

LOC_INTRO_PATTERN = re.compile(
    r"\b(?i:living\s+in|located\s+in|based\s+in|traveling\s+to|office\s+in|moving\s+to|capital\s+of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)


class SensitiveDataDetector:
    """
    Unified detection engine for PromptShield AI.
    Executes structured regex rules, Luhn card verification, and hybrid NER
    to detect sensitive entities in raw user prompts.
    """

    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy
        self._spacy_nlp = None
        if self.use_spacy:
            self._init_spacy()

    def _init_spacy(self):
        """Optionally load spaCy pipeline if installed and available."""
        try:
            import spacy
            for model_name in ["en_core_web_sm", "en_core_web_md"]:
                try:
                    self._spacy_nlp = spacy.load(model_name)
                    break
                except Exception:
                    continue
        except ImportError:
            self._spacy_nlp = None

    def detect(self, prompt: str) -> List[DetectedEntity]:
        """
        Analyze prompt string and return all detected entities,
        with overlaps resolved and values normalized.
        """
        if not prompt or not prompt.strip():
            return []

        candidates: List[DetectedEntity] = []

        # 1. Detect Structured Sensitive Entities
        candidates.extend(self._detect_emails(prompt))
        candidates.extend(self._detect_credit_cards(prompt))
        candidates.extend(self._detect_api_keys(prompt))
        candidates.extend(self._detect_passwords(prompt))
        candidates.extend(self._detect_phones(prompt))
        candidates.extend(self._detect_dates(prompt))

        # 2. Detect Named Entities (NER)
        if self._spacy_nlp is not None:
            candidates.extend(self._detect_spacy_ner(prompt))
        else:
            candidates.extend(self._detect_heuristic_ner(prompt))

        # 3. Resolve overlapping spans & sort
        resolved = self._resolve_conflicts(candidates)
        return resolved

    def _detect_emails(self, prompt: str) -> List[DetectedEntity]:
        entities = []
        for match in EMAIL_PATTERN.finditer(prompt):
            raw = match.group(0)
            entities.append(
                DetectedEntity(
                    text=raw,
                    entity_type=EntityType.EMAIL,
                    start=match.start(),
                    end=match.end(),
                    normalized_value=normalize_entity_value(EntityType.EMAIL, raw),
                    confidence=0.98,
                    detector="regex_email",
                )
            )
        return entities

    def _detect_phones(self, prompt: str) -> List[DetectedEntity]:
        entities = []
        for pattern in PHONE_PATTERNS:
            for match in pattern.finditer(prompt):
                raw = match.group(0)
                # Filter out pure numbers with fewer than 7 digits or standard 4-digit years
                digits_only = re.sub(r"\D", "", raw)
                if len(digits_only) < 7 or len(digits_only) > 15:
                    continue
                # Skip if already identified as a card pattern
                entities.append(
                    DetectedEntity(
                        text=raw,
                        entity_type=EntityType.PHONE,
                        start=match.start(),
                        end=match.end(),
                        normalized_value=normalize_entity_value(EntityType.PHONE, raw),
                        confidence=0.92,
                        detector="regex_phone",
                    )
                )
        return entities

    def _detect_credit_cards(self, prompt: str) -> List[DetectedEntity]:
        entities = []
        for match in CREDIT_CARD_PATTERN.finditer(prompt):
            raw = match.group(0)
            digits = re.sub(r"\D", "", raw)
            if 13 <= len(digits) <= 19 and is_luhn_valid(raw):
                brand = detect_card_brand(digits)
                entities.append(
                    DetectedEntity(
                        text=raw,
                        entity_type=EntityType.CREDIT_CARD,
                        start=match.start(),
                        end=match.end(),
                        normalized_value=normalize_entity_value(EntityType.CREDIT_CARD, raw),
                        confidence=0.96,
                        detector="luhn_credit_card",
                        metadata={"brand": brand, "digit_count": len(digits)},
                    )
                )
        return entities

    def _detect_api_keys(self, prompt: str) -> List[DetectedEntity]:
        entities = []
        for pattern, detector_name, conf in API_KEY_PATTERNS:
            for match in pattern.finditer(prompt):
                # If pattern has a capturing group for the key itself
                if match.groups():
                    raw = match.group(1)
                    start = match.start(1)
                    end = match.end(1)
                else:
                    raw = match.group(0)
                    start = match.start()
                    end = match.end()

                entities.append(
                    DetectedEntity(
                        text=raw,
                        entity_type=EntityType.API_KEY,
                        start=start,
                        end=end,
                        normalized_value=normalize_entity_value(EntityType.API_KEY, raw),
                        confidence=conf,
                        detector=f"rule_{detector_name}",
                    )
                )
        return entities

    def _detect_passwords(self, prompt: str) -> List[DetectedEntity]:
        entities = []
        for pattern in PASSWORD_PATTERNS:
            for match in pattern.finditer(prompt):
                if match.groups():
                    # Extract the password payload
                    payload = match.group(len(match.groups()))
                    start = match.start(len(match.groups()))
                    end = match.end(len(match.groups()))
                else:
                    payload = match.group(0)
                    start = match.start()
                    end = match.end()

                entities.append(
                    DetectedEntity(
                        text=payload,
                        entity_type=EntityType.PASSWORD,
                        start=start,
                        end=end,
                        normalized_value=normalize_entity_value(EntityType.PASSWORD, payload),
                        confidence=0.92,
                        detector="regex_password",
                    )
                )
        return entities

    def _detect_dates(self, prompt: str) -> List[DetectedEntity]:
        entities = []
        for pattern in DATE_PATTERNS:
            for match in pattern.finditer(prompt):
                raw = match.group(0)
                entities.append(
                    DetectedEntity(
                        text=raw,
                        entity_type=EntityType.DATE,
                        start=match.start(),
                        end=match.end(),
                        normalized_value=normalize_entity_value(EntityType.DATE, raw),
                        confidence=0.90,
                        detector="regex_date",
                    )
                )
        return entities

    def _detect_spacy_ner(self, prompt: str) -> List[DetectedEntity]:
        """Extract entities using spaCy's pretrained language model."""
        entities = []
        if self._spacy_nlp is None:
            return entities

        doc = self._spacy_nlp(prompt)
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "DATE": EntityType.DATE,
        }
        for ent in doc.ents:
            if ent.label_ in mapping:
                target_type = mapping[ent.label_]
                entities.append(
                    DetectedEntity(
                        text=ent.text,
                        entity_type=target_type,
                        start=ent.start_char,
                        end=ent.end_char,
                        normalized_value=normalize_entity_value(target_type, ent.text),
                        confidence=0.88,
                        detector="ner_spacy",
                        metadata={"spacy_label": ent.label_},
                    )
                )
        return entities

    def _detect_heuristic_ner(self, prompt: str) -> List[DetectedEntity]:
        """
        Robust gazetteer and contextual heuristic NER fallback
        when spaCy is not installed or offline.
        """
        entities = []

        # 1. Gazetteer exact matches
        for org in KNOWN_ORGANIZATIONS:
            for match in re.finditer(rf"\b{re.escape(org)}\b", prompt):
                entities.append(
                    DetectedEntity(
                        text=match.group(0),
                        entity_type=EntityType.ORGANIZATION,
                        start=match.start(),
                        end=match.end(),
                        normalized_value=normalize_entity_value(EntityType.ORGANIZATION, match.group(0)),
                        confidence=0.90,
                        detector="gazetteer_org",
                    )
                )

        for loc in KNOWN_LOCATIONS:
            for match in re.finditer(rf"\b{re.escape(loc)}\b", prompt):
                entities.append(
                    DetectedEntity(
                        text=match.group(0),
                        entity_type=EntityType.LOCATION,
                        start=match.start(),
                        end=match.end(),
                        normalized_value=normalize_entity_value(EntityType.LOCATION, match.group(0)),
                        confidence=0.90,
                        detector="gazetteer_loc",
                    )
                )

        # 2. Known Public Figures / Persons gazetteer
        for person in KNOWN_PERSONS:
            for match in re.finditer(rf"\b{re.escape(person)}\b", prompt, re.IGNORECASE):
                entities.append(
                    DetectedEntity(
                        text=match.group(0),
                        entity_type=EntityType.PERSON,
                        start=match.start(),
                        end=match.end(),
                        normalized_value=normalize_entity_value(EntityType.PERSON, match.group(0)),
                        confidence=0.92,
                        detector="gazetteer_person",
                    )
                )

        # 3. Explicit Name Intro Patterns (capturing ENTIRE full name: first, middle, last, titles)
        for match in NAME_INTRO_PREFIX_REGEX.finditer(prompt):
            start_pos = match.end()
            tail = prompt[start_pos:]
            tokens = re.findall(r"([A-Za-z]+(?:\.[A-Za-z]+)?|\b[A-Za-z]\.?)", tail[:80])
            words = []
            for t in tokens:
                clean = t.rstrip(".").lower()
                if clean in PERSON_STOP_WORDS or t in KNOWN_LOCATIONS or t in KNOWN_ORGANIZATIONS:
                    break
                words.append(t)
                if len(words) >= 4:
                    break
            if words:
                pattern = re.escape(words[0]) + (
                    r"(?:\s+" + r"\s+".join(re.escape(w) for w in words[1:]) + r")"
                    if len(words) > 1 else ""
                )
                nm = re.search(pattern, tail)
                if nm and nm.start() < 3:
                    s = start_pos + nm.start()
                    e = start_pos + nm.end()
                    extracted_name = prompt[s:e]
                    entities.append(
                        DetectedEntity(
                            text=extracted_name,
                            entity_type=EntityType.PERSON,
                            start=s,
                            end=e,
                            normalized_value=normalize_entity_value(EntityType.PERSON, extracted_name),
                            confidence=0.91,
                            detector="heuristic_person_fullname",
                        )
                    )

        # 4. "I am [Title] [Name]" / "I'm [Name]" pattern
        #    Handles informal, all-lowercase prompts like "i am adarsh" or
        #    "i'm amina write a formal letter..."
        for match in I_AM_NAME_PREFIX_REGEX.finditer(prompt):
            start_pos = match.end()
            tail = prompt[start_pos:]

            # Tokenize the words immediately following the prefix.
            # [A-Za-z]+ accepts both "Adarsh" and "adarsh" (case-insensitive).
            tokens = re.findall(r"[A-Za-z]+(?:\.[A-Za-z]+)?", tail[:80])

            # Walk tokens and BREAK on the first stop-word (same strategy as
            # NAME_INTRO_PREFIX_REGEX).  This prevents task verbs like "write"
            # from bleeding into the captured name.
            name_words: list = []
            for t in tokens:
                t_lower = t.lower()
                if (
                    t_lower in PERSON_STOP_WORDS
                    or t in KNOWN_LOCATIONS
                    or t in KNOWN_ORGANIZATIONS
                ):
                    break
                name_words.append(t)
                if len(name_words) >= 4:  # names rarely exceed 4 words
                    break

            if name_words:
                # Build a regex that allows optional punctuation between parts
                # so "Dr. Julie Ann Johnson" is matched with the period intact.
                parts_pattern = r"[\.\s]*".join(re.escape(w) for w in name_words)
                nm = re.search(parts_pattern, tail)
                if nm and nm.start() < 3:
                    s = start_pos + nm.start()
                    e = start_pos + nm.end()
                    actual_text = prompt[s:e]
                    entities.append(
                        DetectedEntity(
                            text=actual_text,
                            entity_type=EntityType.PERSON,
                            start=s,
                            end=e,
                            normalized_value=normalize_entity_value(
                                EntityType.PERSON, actual_text
                            ),
                            confidence=0.90,
                            detector="heuristic_person_iam",
                        )
                    )

        # 5. General Capitalized Full Names (e.g., "Julie Johnson", "John Mathew" without explicit intros)
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", prompt):
            full_cand = match.group(1)
            cand_words = full_cand.split()
            # Must not contain days, months, common action words, or known orgs/locations
            if any(w in DAYS_AND_MONTHS or w in KNOWN_LOCATIONS or w.lower() in PERSON_STOP_WORDS for w in cand_words):
                continue
            if full_cand in KNOWN_ORGANIZATIONS or full_cand in KNOWN_LOCATIONS:
                continue
            entities.append(
                DetectedEntity(
                    text=full_cand,
                    entity_type=EntityType.PERSON,
                    start=match.start(1),
                    end=match.end(1),
                    normalized_value=normalize_entity_value(EntityType.PERSON, full_cand),
                    confidence=0.87,
                    detector="capitalized_fullname",
                )
            )

        # 3. Contextual intro patterns for ORGANIZATION
        for match in ORG_INTRO_PATTERN.finditer(prompt):
            org = match.group(1)
            if org not in KNOWN_LOCATIONS:
                entities.append(
                    DetectedEntity(
                        text=org,
                        entity_type=EntityType.ORGANIZATION,
                        start=match.start(1),
                        end=match.end(1),
                        normalized_value=normalize_entity_value(EntityType.ORGANIZATION, org),
                        confidence=0.85,
                        detector="heuristic_org_intro",
                    )
                )

        # 4. Contextual intro patterns for LOCATION
        for match in LOC_INTRO_PATTERN.finditer(prompt):
            loc = match.group(1)
            if loc not in KNOWN_ORGANIZATIONS:
                entities.append(
                    DetectedEntity(
                        text=loc,
                        entity_type=EntityType.LOCATION,
                        start=match.start(1),
                        end=match.end(1),
                        normalized_value=normalize_entity_value(EntityType.LOCATION, loc),
                        confidence=0.85,
                        detector="heuristic_loc_intro",
                    )
                )

        return entities

    def _resolve_conflicts(self, candidates: List[DetectedEntity]) -> List[DetectedEntity]:
        """
        Disambiguate overlapping entity spans.
        Prioritizes:
        1. Higher confidence score
        2. Specific structured types (API_KEY, CREDIT_CARD, EMAIL) over generic NER
        3. Longer span length
        Returns non-overlapping entities sorted in ascending start offset order.
        """
        if not candidates:
            return []

        # Type specificity priority
        type_priority = {
            EntityType.API_KEY: 10,
            EntityType.CREDIT_CARD: 9,
            EntityType.PASSWORD: 8,
            EntityType.EMAIL: 7,
            EntityType.PHONE: 6,
            EntityType.DATE: 5,
            EntityType.PERSON: 4,
            EntityType.ORGANIZATION: 3,
            EntityType.LOCATION: 2,
        }

        # Sort candidates by quality
        def sort_key(e: DetectedEntity):
            return (
                e.confidence,
                type_priority.get(e.entity_type, 0),
                (e.end - e.start),
            )

        sorted_candidates = sorted(candidates, key=sort_key, reverse=True)
        accepted: List[DetectedEntity] = []

        for cand in sorted_candidates:
            # Check overlap with any accepted entity
            overlap = False
            for acc in accepted:
                # Two spans [s1, e1) and [s2, e2) overlap if max(s1, s2) < min(e1, e2)
                if max(cand.start, acc.start) < min(cand.end, acc.end):
                    overlap = True
                    break
            if not overlap:
                accepted.append(cand)

        # Return sorted by appearance in text (start offset ascending)
        return sorted(accepted, key=lambda e: e.start)
