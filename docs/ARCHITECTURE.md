# PromptShield AI — Architecture Documentation

**Phase 1: Modular Hybrid Entity-Detection Foundation**  
*B.Tech Computer Science and Engineering Final-Year Project*

---

## 1. Detection ≠ Masking — The Core Principle

Before reading anything else, understand this architectural boundary:

```
DETECTION  ─────────────────────────────────────────────►  Phase 1
   │   "What potentially sensitive / named entities exist in the prompt?"
   │
   ▼
CONTEXT ANALYSIS  ──────────────────────────────────────►  Phase 2
   │   "What is the user trying to do? Is this entity private or public?"
   │
   ▼
RISK ASSESSMENT & POLICY  ──────────────────────────────►  Phase 3
   │   "What is the risk score? Should this entity be MASKED or RETAINED?"
   │
   ▼
CONTEXT-AWARE MASKING  ─────────────────────────────────►  Phase 4
       "Apply the policy decision to the prompt text."
```

The detection layer **only produces candidate entities**. It **never decides** whether an entity must be masked. For example:

| Prompt | Detection Result | Masking Decision |
|---|---|---|
| "Who is Mahatma Gandhi?" | `PERSON` detected | Phase 4 decision — NOT here |
| "My email is john@gmail.com" | `EMAIL` detected | Phase 4 decision — NOT here |
| "Use sk-live-abc123" | `API_KEY` detected | Phase 4 decision — NOT here |

---

## 2. Phase 1 Detection Pipeline

```
User Prompt
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  RegexDetector  (backend/regex_detector.py)              │
│  ─────────────────────────────────────────────────────── │
│  • EMAIL         — RFC 5322 pattern                      │
│  • PHONE         — International E.164, 10-digit         │
│  • CREDIT_CARD   — Format regex + Luhn mod-10 validation │
│  • API_KEY       — sk-, AIza, ghp_, AKIA, bearer tokens  │
│  • ACCESS_TOKEN  — JWT (ey*.ey*.sig*)                    │
│  • PASSWORD      — key=value, "my password is"           │
│  • BANK_ACCOUNT  — IBAN standard, account number trigger │
│  • IP_ADDRESS    — IPv4, IPv6                            │
│  • ORDER_ID / USER_ID / CUSTOMER_ID  — configurable IDs  │
│  • DATE          — ISO, DD-MM-YYYY, written month names  │
│  source = "regex"                                        │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│  PresidioDetector  (backend/presidio_detector.py)        │
│  ─────────────────────────────────────────────────────── │
│  • Microsoft Presidio AnalyzerEngine (singleton)         │
│  • Detects: EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD,    │
│    IP_ADDRESS, IBAN_CODE, PERSON, LOCATION, DATE_TIME    │
│  • Confidence score: Presidio's own score (preserved     │
│    AS-IS — not fabricated or overridden)                 │
│  • Gracefully returns [] if unavailable                  │
│  source = "presidio"                                     │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│  SpacyNERDetector  (backend/spacy_detector.py)           │
│  ─────────────────────────────────────────────────────── │
│  • spaCy en_core_web_sm pretrained model (singleton)     │
│  • Detects: PERSON, ORG → ORGANIZATION, GPE/LOC → LOC,  │
│    DATE                                                  │
│  • Confidence: documented baseline 0.85 (not fabricated) │
│  • Falls back to HeuristicNERDetector if model missing   │
│  source = "spacy" (or "heuristic_ner" in fallback)       │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│  EntityFusionEngine  (backend/entity_fusion.py)          │
│  ─────────────────────────────────────────────────────── │
│  1. Integrity Check   — prompt[start:end] == entity.text │
│  2. Exact-Span Merge  — same [s,e): merge, union sources │
│  3. Overlap Resolution— structured > NER by priority     │
│  4. Sort              — ascending start offset           │
│  metadata["contributing_sources"] records all sources    │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
               List[DetectedEntity]
         (unified, deduplicated, sorted)
```

---

## 3. Why Each Detector Exists

### 3.1 Regex / Rule-Based (RegexDetector)

**Why**: Structured sensitive data (email addresses, API keys, credit card numbers, IBANs) follows explicit, well-known lexical patterns. Regex is:
- **Deterministic**: same input always gives same output.
- **Explainable**: every match can be traced to a specific pattern.
- **Fast**: no ML inference overhead.
- **Auditable**: academic examiners can read the exact pattern.

**Limitation**: Regex cannot understand context or semantics. A Luhn-valid 16-digit number is a *credit card candidate*, not a proven card. The `PHONE` pattern may match long numeric IDs.

**Luhn Algorithm**: For credit card candidates, after regex identifies the format, the Luhn mod-10 checksum is applied. This eliminates arbitrary numeric sequences (e.g. order numbers, tracking IDs) that happen to be 16 digits.

### 3.2 Microsoft Presidio

**Why**: Presidio provides a battle-tested, ML-backed PII detection framework. It:
- Covers entity types that regex misses (e.g. US SSN, passport numbers, medical licenses).
- Uses pattern matching + context (e.g. the word "email" near an email address raises confidence).
- Provides per-entity confidence scores from its own analysis.
- Has been used in production privacy pipelines.

**Difference from regex**: Presidio uses a combination of pattern recognisers, NLP context, and pre-trained recognisers. It is NOT pure regex — it evaluates surrounding context.

**What it does NOT do**: Presidio does not decide whether to mask. It produces `RecognizerResult` objects — PromptShield converts these to `DetectedEntity` objects with `source="presidio"`.

### 3.3 spaCy Pretrained NER

**Why**: Natural-language entities (PERSON, ORGANIZATION, LOCATION) do not have explicit lexical patterns. "Julie" is a PERSON because of its linguistic context in a sentence — not because it matches a regex. spaCy's pretrained `en_core_web_sm` provides:
- State-of-the-art NER for English text.
- PERSON, ORG, GPE (geo-political entity), LOC, DATE detection.
- Zero training required — uses an existing pretrained model.

**Difference from Presidio**: Presidio focuses on **PII** (personally identifiable information with legal/privacy implications). spaCy focuses on **named entity recognition** (identifying real-world entities in text). They are complementary:
- Presidio: "Is this a phone number?" → pattern + context.
- spaCy: "Is 'Julie' a person's name in this sentence?" → linguistic NER.

**Limitation**: NER ≠ PII detection. spaCy detecting "Mahatma Gandhi" as PERSON does not mean it is private information. That distinction is Phase 2's job.

---

## 4. NER vs PII Detection — Key Distinction

| | NER (spaCy) | PII Detection (Presidio / Regex) |
|---|---|---|
| **Goal** | Identify real-world entity categories | Identify information that could identify/harm a person |
| **Examples** | "Elon Musk" = PERSON, "Apple" = ORG | "john@gmail.com" = EMAIL, "4111..." = CREDIT_CARD |
| **Contextual?** | Yes — linguistic context | Partially — format + context |
| **Implies Masking?** | **No** | **No** — detection ≠ masking |
| **Used for** | Seeding Phase 2 context analysis | High-confidence structured PII candidates |

---

## 5. Entity Type Normalization

Different detectors use different entity type names. A centralized mapping in `backend/config.py` converts all labels to PromptShield's canonical `EntityType`:

```
Presidio:  EMAIL_ADDRESS  →  PromptShield:  EMAIL
Presidio:  PHONE_NUMBER   →  PromptShield:  PHONE
Presidio:  IBAN_CODE      →  PromptShield:  BANK_ACCOUNT
spaCy:     ORG            →  PromptShield:  ORGANIZATION
spaCy:     GPE            →  PromptShield:  LOCATION
spaCy:     PERSON         →  PromptShield:  PERSON
```

---

## 6. Entity Fusion Strategy

The `EntityFusionEngine` resolves the situation where multiple detectors find the same entity:

### Exact-Span Deduplication
When detectors share the same `[start, end)` span:
- **Same type** (e.g. both detect EMAIL at [12,27)): Merge into one entity with the highest confidence. Record both sources in `metadata["contributing_sources"]`.
- **Different types** (e.g. EMAIL vs PERSON at same span): Apply entity-type priority. Structured types beat generic NER.

### Overlapping-Span Resolution
When spans partially overlap:
1. Entity-type priority (API_KEY=10 > EMAIL=7 > PERSON=3)
2. Source priority (regex=5 > presidio=4 > spacy=3 > heuristic=2)
3. Confidence score
4. Span length (longer wins on tie)

The losing entity is **discarded** (not silently — a debug log is written).

### Priority Table
```
API_KEY / ACCESS_TOKEN  = 10  (highest — explicit credential)
CREDIT_CARD             =  9
PASSWORD                =  8
EMAIL                   =  7
PHONE / BANK_ACCOUNT    =  6
IP_ADDRESS / IDs        =  5
DATE                    =  4
PERSON                  =  3
ORGANIZATION / LOCATION =  2  (lowest — generic NER)
```

---

## 7. Model Loading Strategy

To avoid re-loading large ML models on every request:
- **SpacyNERDetector**: Class-level `_nlp` singleton. Loaded on first `detect()` call.
- **PresidioDetector**: Class-level `_engine` singleton. Loaded on first `detect()` call.
- **RegexDetector**: No models — all patterns are pre-compiled at import time.

If a model fails to load:
1. A clear warning is logged (not suppressed).
2. The detector returns `[]` gracefully.
3. The pipeline continues with the remaining detectors.

---

## 8. Unified DetectedEntity Format

Every detector produces `DetectedEntity` objects with the same schema:

```python
@dataclass
class DetectedEntity:
    text: str            # Raw substring from prompt
    entity_type: EntityType
    start: int           # Character start offset (inclusive)
    end: int             # Character end offset (exclusive)
    normalized_value: str
    confidence: float    # [0, 1] — see notes below
    source: str          # "regex" | "presidio" | "spacy" | "heuristic_ner"
    detector: str        # Specific recognizer (e.g. "luhn_credit_card")
    metadata: dict       # brand, spacy_label, contributing_sources, etc.
```

**Confidence notes**:
- `regex`: rule-defined baseline from `config.CONFIDENCE_BASELINES`.
- `presidio`: Presidio's own score — preserved AS-IS.
- `spacy`: documented baseline (0.85) — spaCy does not provide per-span scores.
- `heuristic_ner`: varies by recognizer type (0.85–0.92).

Confidence values are **never fabricated**. If a source does not provide a meaningful confidence value, a documented baseline is used and noted.

---

## 9. Configuration

All detection configuration is centralized in `backend/config.py`:

| Config Key | Purpose |
|---|---|
| `EMAIL_PATTERN` | Compiled regex for email detection |
| `PHONE_PATTERNS` | List of (pattern, label) tuples |
| `CREDIT_CARD_PATTERN` | Credit card candidate pattern |
| `API_KEY_PATTERNS` | List of (pattern, label, confidence) |
| `ACCESS_TOKEN_PATTERNS` | JWT and OAuth token patterns |
| `PASSWORD_PATTERNS` | Credential assignment patterns |
| `BANK_ACCOUNT_PATTERNS` | IBAN and account number patterns |
| `IP_ADDRESS_PATTERNS` | IPv4 and IPv6 patterns |
| `ID_PATTERNS` | Configurable project ID patterns |
| `DATE_PATTERNS` | ISO, DMY, written date patterns |
| `PRESIDIO_ENTITY_MAPPING` | Presidio label → EntityType mapping |
| `SPACY_ENTITY_MAPPING` | spaCy label → EntityType mapping |
| `ENTITY_TYPE_PRIORITY` | Conflict resolution priority weights |
| `SOURCE_PRIORITY` | Source reliability ordering |
| `CONFIDENCE_BASELINES` | Documented confidence values by source |

---

## 10. Directory Structure (Phase 1)

```
Prompshield/
├── README.md
├── requirements.txt
├── backend/
│   ├── __init__.py          — package exports (v0.3.0)
│   ├── config.py            — [NEW] centralized configuration
│   ├── models.py            — [UPDATED] extended EntityType + source field
│   ├── normalization.py     — [UPDATED] new normalizers for new entity types
│   ├── detector.py          — [REFACTORED] hybrid pipeline orchestrator
│   ├── regex_detector.py    — [NEW] modular structured detector
│   ├── presidio_detector.py — [NEW] Microsoft Presidio wrapper
│   ├── spacy_detector.py    — [NEW] spaCy NER + heuristic fallback
│   ├── entity_fusion.py     — [NEW] fusion / conflict resolution
│   ├── baseline_masker.py   — [UPDATED] new entity type prefixes
│   ├── mapping_store.py
│   ├── core.py
│   ├── context_analyzer.py
│   ├── task_classifier.py
│   └── policy_engine.py
├── tests/
│   ├── test_regex_detector.py              — [NEW]
│   ├── test_presidio_detector.py           — [NEW]
│   ├── test_spacy_detector.py              — [NEW]
│   ├── test_entity_fusion.py               — [NEW]
│   ├── test_phase1_detection_foundation.py — [NEW] mandatory 8 scenarios
│   ├── test_detector_structured.py         — [EXISTING — preserved]
│   ├── test_detector_ner.py                — [EXISTING — preserved]
│   ├── test_normalization.py               — [EXISTING — preserved]
│   ├── test_baseline_masker.py             — [EXISTING — preserved]
│   ├── test_core_pipeline.py               — [EXISTING — preserved]
│   ├── test_context_analyzer.py            — [EXISTING — preserved]
│   ├── test_policy_engine.py               — [EXISTING — preserved]
│   └── test_task_classifier.py             — [EXISTING — preserved]
├── docs/
│   ├── ARCHITECTURE.md      — this file
│   └── DEVELOPMENT_PHASES.md
└── cli/
    └── demo.py
```

---

## 11. Known Limitations

1. **Phone false positives**: Long numeric IDs and ZIP codes may be misidentified as phone numbers.
2. **spaCy model accuracy**: `en_core_web_sm` is a small model. Less common names, non-English input, and highly technical text may have lower NER accuracy.
3. **Presidio US-centric**: Presidio's built-in recognizers are primarily trained on US/UK data. International PII formats may have lower recall.
4. **NER ≠ PII**: spaCy detecting "Mahatma Gandhi" as PERSON does not imply it is private. That is a Phase 2 decision.
5. **Credit card false negatives**: Exotic card formats not covered by the regex pattern will not be detected.
6. **No semantic understanding**: A password-like string in a comment block may be incorrectly flagged.
7. **Luhn validation is necessary but not sufficient**: Some non-card numbers happen to pass Luhn by coincidence.
8. **Context disambiguation**: Distinguishing between the company "Apple" and the fruit "apple" is a Phase 2 task.

---

## 12. What is NOT Implemented (Phase 1 Non-Goals)

- Context-aware semantic masking
- Risk scoring
- Privacy policy decisions (MASK / RETAIN / USER_APPROVAL)
- Sentence Transformers / semantic similarity
- External LLM integration
- REST API / FastAPI
- Browser extension
- Authentication / RBAC
- PostgreSQL / Redis
- Dashboard
- Docker / Cloud deployment
