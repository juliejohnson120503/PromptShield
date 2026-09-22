# PromptShield AI — Development Phases

*B.Tech Computer Science and Engineering Final-Year Project*

---

## Project Vision

PromptShield AI is a **context-aware security layer** that sits between a user and an external LLM. It detects potentially sensitive information in prompts, evaluates whether that information is actually sensitive in context, applies configurable privacy policies, and selectively sanitizes the prompt before sending it to the LLM — all without disrupting task utility.

---

## Development Roadmap

| Phase | Title | Status | Architectural Boundary |
|:---:|:---|:---:|:---|
| **1** | **Modular Hybrid Detection Foundation** | ✅ Completed | Detects candidates; does NOT mask |
| **2** | **Context Understanding & Task Classification** | ✅ Completed | Classifies task intent; determines entity role (private vs public, first-party vs reference) |
| **3** | **Risk Assessment & Privacy Policy Engine** | ✅ Completed | Scores risk (0–100); decides MASK / RETAIN / REPLACE / USER_APPROVAL |
| **4** | **Context-Aware Semantic Masking** | 🔜 Next | Executes policy; replaces sensitive text with `<TYPE_N>` placeholders |
| **5** | **Controlled Restoration & LLM Layer** | Planned | Pluggable LLM interface; authorized placeholder restoration |
| **6** | **FastAPI Backend Service** | Planned | REST endpoints; privacy-safe audit logging |
| **7** | **Browser Extension (Manifest V3)** | Planned | Chrome extension; popup UI; prompt injection |
| **8** | **Dashboard & Empirical Evaluation** | Planned | Interactive dashboard; 3-way comparative benchmark |

---

## Phase 1 — Modular Hybrid Detection Foundation

### Objective
Build the detection foundation. The pipeline identifies what sensitive or named entities *potentially* exist in a prompt — without deciding what to do about them.

### What Phase 1 Does
- Structured detection (regex + Luhn): EMAIL, PHONE, CREDIT_CARD, API_KEY, ACCESS_TOKEN, PASSWORD (with shortforms & abbreviations), BANK_ACCOUNT, IP_ADDRESS, IDs.
- Transformer-based Open-Vocabulary NER (GLiNER urchade/gliner_small-v2.1): zero-shot bidirectional neural extraction for informal, lowercase, and conversational text.
- Pretrained statistical NER (spaCy en_core_web_sm): PERSON, ORGANIZATION, LOCATION, DATE.
- ML-backed PII detection (Microsoft Presidio).
- Entity type normalization across all detector vocabularies.
- Entity fusion: priority-based overlap resolution, deduplication, integrity validation.
- Heuristic NER fallback: gazetteer + contextual pattern matching for ultra-fast zero-ML environments.

### What Phase 1 Does NOT Do
- Does NOT decide whether an entity must be masked.
- Does NOT evaluate whether "Julie" in "Who is Julie Roberts?" is private or public.
- Does NOT call any external LLM.
- Does NOT implement any privacy policy.

### Key Modules
```
backend/config.py           — centralized patterns, type maps, priorities
backend/regex_detector.py   — deterministic structured detection
backend/presidio_detector.py— Microsoft Presidio PII detection
backend/gliner_detector.py  — GLiNER transformer-based neural NER
backend/spacy_detector.py   — spaCy NER + heuristic fallback
backend/entity_fusion.py    — deduplication & conflict resolution
backend/detector.py         — hybrid orchestrator (uses all above)
```

### Packages Installed
- `gliner==0.2.29` (with `torch==2.14.0`, `transformers==5.16.1`)
- `urchade/gliner_small-v2.1` (open-vocabulary transformer model)
- `presidio-analyzer==2.2.364`
- `spacy==3.8.16`
- `en_core_web_sm==3.8.0` (pretrained model)

---

## Phase 2 — Context Understanding & Task Classification

### Objective
Understand WHAT the user is trying to do and HOW each detected entity relates to that task.

### What Phase 2 Does
- **Task Classification**: Classifies the prompt into one of 8 task types (GENERAL_QA, EMAIL_GENERATION, CODE_GENERATION, SUMMARIZATION, TRANSLATION, DATA_ANALYSIS, DOCUMENT_GENERATION, GENERAL_CHAT).
- **Contextual Role Analysis**: For each detected entity, assigns a `RoleCategory`:
  - `PERSONAL_IDENTIFIER` — Self-identifying user PII ("My name is Julie")
  - `PUBLIC_FIGURE_OR_FACT` — Well-known public figures ("Elon Musk", "Paris")
  - `TASK_RELEVANT_ENTITY` — Needed for the task ("ABC Technologies" in an email task)
  - `TECHNICAL_CREDENTIAL` — Passwords, API keys
  - `GENERAL_REFERENCE` — Ambiguous third-party references

### Key Modules
```
backend/task_classifier.py  — rule-based 8-category task classification
backend/context_analyzer.py — entity role analysis using linguistic cues
```

### What Phase 2 Does NOT Do
- Does NOT decide whether to mask.
- Does NOT use Sentence Transformers (planned for Phase 2 enhancement).

---

## Phase 3 — Risk Assessment & Privacy Policy Engine

### Objective
Score the privacy risk of each entity and make explainable policy decisions.

### What Phase 3 Does
- **Risk Scoring**: Assigns a transparent risk score (0–100) per entity based on:
  - Entity type sensitivity weight
  - First-party possession indicator
  - Task context relevance
  - Public knowledge indicator
- **Policy Decisions**: Produces one of four actions per entity:
  - `RETAIN` — Entity is safe or task-essential; do not mask.
  - `MASK` — Entity is sensitive; replace with `<TYPE_N>` placeholder.
  - `REPLACE` — Replace with a safe synthetic substitute.
  - `USER_APPROVAL` — Requires user confirmation before proceeding.

### Key Modules
```
backend/policy_engine.py    — risk scorer + policy decision engine
```

---

## Phase 4 — Context-Aware Semantic Masking (Next)

### Objective
Execute the policy decisions from Phase 3 to produce a sanitized prompt.

### Planned Implementation
- Read `PolicyReport` from Phase 3.
- For each entity with `MASK` / `REPLACE` action, substitute with `<TYPE_N>` placeholder.
- For each entity with `RETAIN` action, leave the text unchanged.
- Maintain `MappingStore` for local user-side placeholder-to-secret association.

### Why Phase 4 Is NOT Phase 1
Phase 1 `BaselineMasker` performs **blind masking** of all detected entities — it does not distinguish between private PII and public figures. This is intentional scaffolding for later phases. Phase 4 will supersede this with **selective masking** driven by Phase 2 context and Phase 3 policy.

---

## Phase 5 — Controlled Restoration & LLM Layer (Planned)

### Objective
After the LLM processes the sanitized prompt and returns a response, restore placeholder values under authorization.

### Planned Implementation
- Pluggable `LLMProvider` interface (MockLLMProvider for testing, GeminiProvider for production).
- Scan LLM response for `<TYPE_N>` placeholders.
- Restore values from `MappingStore` only when authorized.
- Apply post-restoration policy checks.

---

## Phase 6 — FastAPI Backend Service (Planned)

### Endpoints (Planned)
```
POST /analyze     — detect entities in prompt
POST /sanitize    — detect + apply masking + return protected prompt
POST /restore     — restore placeholders in LLM response
GET  /policies    — list current privacy policies
GET  /metrics     — detection statistics (privacy-safe)
```

---

## Phase 7 — Browser Extension (Planned)

### Implementation
- Chrome Manifest V3 extension.
- Capture prompt text from LLM web UIs (ChatGPT, Claude, Gemini).
- Show risk preview popup before submission.
- Allow user to approve/reject entity masking per entity.
- Inject sanitized prompt into LLM input field.

---

## Phase 8 — Dashboard & Empirical Evaluation (Planned)

### Benchmark
3-way comparative evaluation:
1. **No Protection** — raw prompt sent to LLM.
2. **Blind Masking** — all entities masked unconditionally.
3. **PromptShield** — context-aware selective masking.

Metrics: Precision, Recall, F1, Utility Preservation Score, False Positive Rate.

---

## Non-Goals (All Phases)

- Training a custom NER model.
- Training a custom LLM.
- Hosting or proxying LLM API calls.
- Storing sensitive data in any database.
- Claiming 100% detection accuracy.
- Replacing end-to-end encryption.
- Legal compliance certification.
