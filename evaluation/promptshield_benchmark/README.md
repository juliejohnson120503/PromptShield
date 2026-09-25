# PromptShield Baseline Benchmark

**Pipeline:** CURRENT PromptShield Detection Pipeline (v0.3.0)  
**Purpose:** Quantitative baseline for Precision / Recall / F1 / Latency before any model upgrade.  
**Do not use this README's numbers to modify the system.** This is a read-only evaluation artifact.

---

## 1. Current PromptShield Implementation

### Detection Pipeline

The full pipeline is orchestrated by `backend/detector.py` → `SensitiveDataDetector.detect()`.

Four detector subsystems run in order, then results are merged by `EntityFusionEngine`:

```
User Prompt
    │
    ▼
1. RegexDetector.detect()          ← zero-ML, deterministic
    │
    ▼
2. PresidioDetector.detect()       ← Microsoft Presidio ML PII
    │
    ▼
3. GlinerNERDetector.detect()      ← Deep neural open-vocabulary NER
    │
    ▼
4. SpacyNERDetector.detect()       ← spaCy pretrained NER
    │
    ▼
5. EntityFusionEngine.fuse()       ← dedup + overlap resolution
    │
    ▼
List[DetectedEntity]               ← sorted, non-overlapping
```

### Models Used

| Detector  | Model / Library | Version |
|-----------|----------------|---------|
| Regex     | Python `re` (stdlib) | N/A |
| Presidio  | `presidio-analyzer` | ≥ 2.2.350 |
| GLiNER    | `urchade/gliner_small-v2.1` | `gliner ≥ 0.2.0` |
| spaCy     | `en_core_web_sm` (fallback: `en_core_web_md`) | `spacy ≥ 3.7.0` |

### Detector Configuration

**Regex Detector** (`backend/regex_detector.py` + `backend/config.py`):
- EMAIL: `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b`
- PHONE: Three patterns — international (+XX …), separated, 10-digit Indian/US
- CREDIT_CARD: Pattern + **Luhn (mod-10) validation** (rejects non-valid candidates)
- API_KEY: OpenAI (`sk-*`), Google (`AIza*`), GitHub (`ghp_*`), AWS (`AKIA*`), generic assignment, bearer
- ACCESS_TOKEN: JWT three-segment pattern (`eyXXX.eyXXX.XXX`)
- PASSWORD: Assignment (`password = …`), natural language (`my password is …`), PIN, credentials
- BANK_ACCOUNT: IBAN pattern + keyword-triggered account numbers
- IP_ADDRESS: IPv4 strict, IPv6 canonical
- ORDER_ID / CUSTOMER_ID / TICKET_ID: Prefix patterns + keyword context patterns
- ADDRESS: Full postal address (with postal code) + street address (without)
- DATE: ISO (YYYY-MM-DD), DMY (DD/MM/YYYY), Month DD YYYY, DD Month YYYY

**Presidio Detector** (`backend/presidio_detector.py`):
- Engine: `AnalyzerEngine()` with custom `PatternRecognizer` for CUSTOMER_ID, ORDER_ID, TICKET_ID
- Requested entities: all `PRESIDIO_ENTITY_MAPPING` keys except `ORGANIZATION`
- Score threshold: results with `score < 0.40` are discarded
- ORGANIZATION excluded from Presidio requests (covered by spaCy)

**GLiNER Detector** (`backend/gliner_detector.py`):
- Model: `urchade/gliner_small-v2.1` (class-level singleton)
- Labels: `["person", "organization", "location", "password", "username"]`
- Threshold: `0.50`
- `flat_ner=True`
- Post-filtering: common excluded tokens, pronoun filter, single-char filter,
  alphanumeric ID pattern filter, lowercase-without-corporate-indicator filter

**spaCy Detector** (`backend/spacy_detector.py`):
- Model: `en_core_web_sm` (fallback to `en_core_web_md`, then `HeuristicNERDetector`)
- Mapped labels: `PERSON`, `ORG` → ORGANIZATION, `GPE` / `LOC` → LOCATION, `DATE`
- Post-filtering: `COMMON_EXCLUDED_TOKENS`, structured ID code rejection, PIN/postal code DATE rejection
- Confidence baseline: `0.85` (documented, not fabricated)

**Entity Fusion** (`backend/entity_fusion.py`):
- Step 0: Integrity filter (span bounds + `prompt[start:end] == text`)
- Step 1: Exact-span deduplication (same span, same/different type → priority-based winner)
- Step 2: Overlapping-span resolution (greedy by entity-type priority → source priority → confidence → span length)
- Step 3: Sort by ascending start offset

**Conflict Resolution Priorities** (`backend/config.py`):
- Entity type: API_KEY=10, ACCESS_TOKEN=10, CREDIT_CARD=9, PASSWORD=8, EMAIL=7, PHONE/BANK_ACCOUNT/IP/ORDER/CUSTOMER/TICKET=6, USER_ID/ADDRESS=5, DATE=4, PERSON/ORGANIZATION=3, LOCATION=2
- Source: regex=5, luhn=5, presidio=4, gliner=4, spacy=3, heuristic=2

**Common Excluded Tokens** (enforced in all NER detectors):
- Technical labels: `password`, `email`, `phone`, `ip`, `token`, `key`, `api`, etc.
- Pronouns: `i`, `me`, `my`, `we`, `us`, `he`, `she`, `they`, etc.
- Generic nouns: `team`, `company`, `organization`, `model`, `dataset`, etc.

---

## 2. Dataset

- **File:** `evaluation/promptshield_benchmark/ground_truth.json`
- **Total prompts:** 105
- **Ground-truth methodology:** Manually annotated by the developer.
  Entity spans verified character-by-character against the original prompt text.
  No model predictions were used to create the ground truth.

### Dataset Coverage

| Category | Prompt IDs |
|----------|-----------|
| Single entity | test_002, test_004, test_006, test_007, test_021, etc. |
| Multiple entities | test_001, test_009, test_018, test_034, test_100, etc. |
| Multiple same-type entities | test_019, test_033, test_039, test_059, test_063, test_098, etc. |
| Repeated entities | test_099 (3 ORGs), test_019, test_059 |
| Mixed PII + normal text | test_073, test_085, test_102, test_105 |
| Technical text | test_011, test_021, test_026, test_077, test_088 |
| Informal text | test_087 (lowercase) |
| Long prompts | test_105 |
| Short prompts | test_013, test_035, test_075 |
| Negative (no entities) | test_013, test_014, test_016, test_017, test_035, test_041, test_046, test_049, test_057, test_062, test_066, test_067, test_072, test_075, test_084, test_086, test_090, test_095, test_097, test_101, test_104 |
| Numbers that are NOT PII | test_066, test_067, test_072, test_090, test_095 |
| Ambiguous context | test_060 (public figure), test_015 (org vs. product) |
| Difficult address parsing | test_012, test_030, test_045, test_055, test_089 |
| Unsupported entity types | test_008 (PASSPORT_NUMBER), test_010 (MAC_ADDRESS), test_015, test_022, etc. |

### Unsupported Ground-Truth Entity Types

The following entity types appear in ground truth but are NOT defined in
`backend/models.py` `EntityType`. They are **excluded from TP/FP/FN scoring**
and tracked separately in `coverage.csv`:

- `PASSPORT_NUMBER` — PromptShield has no matching EntityType
- `MAC_ADDRESS` — not an EntityType
- `UUID` — not an EntityType
- `IFSC` — not an EntityType
- `NATIONALITY` — not an EntityType
- `MEDICAL_INFORMATION` — not an EntityType
- `EMPLOYEE_ID` — not an EntityType (may be captured as CUSTOMER_ID by regex)

---

## 3. Ground-Truth Methodology

- All entity spans were manually verified using Python slice notation:
  `assert text[start:end] == entity_text`
- Start offsets are **inclusive**, end offsets are **exclusive** (Python convention).
- Labels use the canonical PromptShield EntityType values where a mapping exists,
  or the actual semantic type where no EntityType exists (these are excluded from scoring).
- No entity was inferred from model output. All annotations were done a priori.

---

## 4. Label Normalization

See `label_mapping.json` for the complete normalization rules for all four detectors.

Key decisions:
- `USERNAME` (GT) → `USER_ID` (canonical) — PromptShield has no USERNAME EntityType
- `PHONE_NUMBER` (GT) → `PHONE`
- `PASSPORT_NUMBER`, `MAC_ADDRESS`, `UUID`, `IFSC`, `NATIONALITY`,
  `MEDICAL_INFORMATION`, `EMPLOYEE_ID` → `null` (unsupported, excluded from scoring)

---

## 5. Matching Methodology

**STRICT entity-level matching** is used:

A prediction is a **True Positive (TP)** only when ALL THREE of these hold:
1. `prediction.start == ground_truth.start`
2. `prediction.end == ground_truth.end`
3. `canonical(prediction.label) == canonical(ground_truth.label)`

Any mismatch in span boundaries OR label → NOT a TP.

- **False Positive (FP):** A prediction with no matching GT entity.
- **False Negative (FN):** A GT entity with no matching prediction.
- **Partial span:** Counted as both FP + FN (not a TP).
- **Wrong label (correct span):** Counted as both FP + FN.

---

## 6. Precision / Recall / F1 Formulas

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)

Micro F1  = 2 × Precision × Recall / (Precision + Recall)
            [computed from aggregate TP, FP, FN across all entity types]

Macro F1  = mean(F1 per entity type)
            [computed as simple average of per-entity F1 scores]
```

---

## 7. Latency Methodology

- **Timer:** `time.perf_counter()` (high-resolution, sub-millisecond)
- **Model loading time** is measured separately (`model_init_time_ms`) and
  **excluded** from the reported latency statistics.
- **Warm-up:** 5 prompts are run before timing begins to ensure JIT, model
  caches, and OS page caches are warm.
- **Measured runs:** All 105 dataset prompts (one pass).
- **What is measured:** Only `SensitiveDataDetector.detect(prompt)` — no file I/O,
  no UI, no LLM calls, no network communication.
- **Component latency:** Measured by calling each sub-detector individually
  in the same process after the full pipeline measurement. This does not
  change production pipeline behavior.
- Note: Fusion component latency is measured on regex-only candidate lists
  as a proxy (full fusion time is embedded in total pipeline latency).

---

## 8. Reproducibility

### Exact Command

```bash
# From the project root directory:
python evaluation/promptshield_benchmark/run_benchmark.py
```

### Prerequisites

```bash
pip install presidio-analyzer>=2.2.350 spacy>=3.7.0 gliner>=0.2.0
python -m spacy download en_core_web_sm
```

### Hardware / Software

- **OS:** Windows (see `metrics.json` for exact platform string)
- **Python:** 3.x (see `metrics.json`)
- **Timer:** `time.perf_counter()` — OS high-resolution timer
- **Packages:** Recorded in `metrics.json` → `detector_status`

---

## 9. Output Files

| File | Description |
|------|-------------|
| `ground_truth.json` | 105 manually-annotated prompts |
| `predictions.json` | System output for each prompt |
| `metrics.json` | All summary metrics (precision, recall, F1, latency) |
| `per_entity_metrics.csv` | TP/FP/FN/Precision/Recall/F1 per entity type |
| `latency_results.csv` | Per-prompt latency + length category |
| `component_latency.csv` | Mean/median/P95 per detector component |
| `detector_contribution.csv` | Which detector contributed each entity |
| `coverage.csv` | GT count / detected / missed / recall per type |
| `error_analysis.md` | FP and FN analysis with classifications |
| `label_mapping.json` | Complete label normalization rules |
| `run_benchmark.py` | Benchmark runner script |
| `README.md` | This file |

---

## 10. Limitations

1. **Strict span matching** penalizes the system for any off-by-one character
   differences (e.g., trailing punctuation). This is intentional — it measures
   precision of the final user-facing output.
2. **Dataset size (105 prompts)** is sufficient for a baseline but not for
   statistically robust confidence intervals. Results may shift with a larger dataset.
3. **Unsupported entity types:** PASSPORT_NUMBER, MAC_ADDRESS, UUID, IFSC,
   NATIONALITY, MEDICAL_INFORMATION, EMPLOYEE_ID have no PromptShield EntityType.
   They cannot be detected or scored by this system.
4. **GLiNER labels are limited** to: person, organization, location, password, username.
   The model cannot detect EMAIL, PHONE, CREDIT_CARD, API_KEY, etc. directly —
   these are covered by Regex and Presidio.
5. **spaCy en_core_web_sm** is a small (12 MB) model. Recall for rare or
   non-English names may be lower than larger models.
6. **Component latency is approximate** — individual sub-detector timings are
   measured in parallel with the full pipeline (not inside it), so they
   do not perfectly decompose the full pipeline time (which includes Python
   overhead, list operations, and fusion).
7. **Latency is single-run (105 samples)** per component — for production-grade
   latency benchmarking, consider 500+ runs per prompt.

---

## 11. Future Comparison

This benchmark is designed so that the **exact same**:
- Prompts (`ground_truth.json`)
- GT annotations
- Matching rules (strict span + label)
- Metrics (Precision / Recall / Micro F1 / Macro F1)
- Latency methodology

can be reused to evaluate GLiNER2-PII or any replacement model.
Simply run a new detector against the same `ground_truth.json` and apply
the same scoring logic from `run_benchmark.py`.
