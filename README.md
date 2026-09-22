# PromptShield AI 🛡️

**A Context-Aware Security Layer for Safe, Responsible, and Enterprise-Ready AI Systems**  
*B.Tech Computer Science and Engineering Final-Year Project*

---

## 1. Overview & Problem Statement

When interacting with Large Language Models (LLMs), users frequently include sensitive personal, financial, and organizational data in their prompts (e.g. personal names, emails, phone numbers, API credentials, passwords, and payment card numbers). Sending this raw information to external LLMs creates severe privacy and security risks.

However, **blindly masking everything harms utility and task semantic meaning**. For example:
- *"What is the capital of India?"* $\rightarrow$ Public knowledge; should **never** be masked.
- *"Write an internship email to ABC Technologies."* $\rightarrow$ The organization name is essential to the task context; safe to **retain**.
- *"My email is john@gmail.com and password is 'secret123'."* $\rightarrow$ Highly sensitive private data; must be **masked**.

PromptShield AI addresses this **Privacy–Utility trade-off** by acting as a user-side context-aware security layer.

---

## 2. End-to-End Architectural Pipeline

```text
       USER PROMPT
            │
            ▼
┌──────────────────────────────────────────────────────────┐
│ PHASE 1: SENSITIVE INFORMATION DETECTION                │
│ ├── Structured: Email, Phone, Credit Card (Luhn),       │
│ │   API Keys, Passwords/Credentials                      │
│ └── Unstructured / NER: Person, Org, Location, Date      │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│ PHASE 2: CONTEXT UNDERSTANDING & TASK CLASSIFICATION     │
│ ├── Task Classifier (Q&A, Email, Coding, Summary, etc.)  │
│ └── Contextual Role Analyzer (Public vs Personal,        │
│     Task-Relevant vs Irrelevant)                         │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│ PHASE 3: RISK ASSESSMENT & PRIVACY POLICY ENGINE         │
│ ├── Transparent Risk Scoring (0–100)                     │
│ └── Explainable Policy Engine:                           │
│     RETAIN / MASK / REPLACE / USER_APPROVAL              │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│ PHASE 4: CONTEXT-AWARE SEMANTIC MASKING                  │
│ ├── Policy Execution (Retains safe, Masks sensitive)     │
│ ├── Semantic Placeholders (<PERSON_1>, <EMAIL_1>)        │
│ └── Local Mapping Store (User-Side ONLY)                 │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
     PROTECTED PROMPT ──► EXTERNAL LLM ──► LLM RESPONSE
                                                  │
                                                  ▼
┌──────────────────────────────────────────────────────────┐
│ PHASE 5: CONTROLLED RESTORATION                          │
│ ├── Policy & Authorization Check                         │
│ └── Controlled Placeholder Restoration                   │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
                   FINAL OUTPUT TO USER
```

---

## 3. Project Roadmap

| Phase | Title | Status | Description |
| :--- | :--- | :---: | :--- |
| **Phase 1** | **Detection Foundation** | **Completed ✅** | Deterministic detection for structured & NER entities, normalization, foundational baseline placeholder generator, local mapping store, unit tests & CLI. |
| **Phase 2** | **Context Understanding** | **Completed ✅** | 8-category task classification and entity contextual role analysis (public vs personal, task relevance, distinguishing "Julie" from "Elon Musk"). |
| **Phase 3** | **Risk & Privacy Policy** | *Next 🔜* | Transparent multi-factor risk scoring (0–100) and deterministic explainable policy decisions (`RETAIN`, `MASK`, `REPLACE`, `USER_APPROVAL`). |
| **Phase 4** | **Context-Aware Semantic Masking** | *Planned* | Policy-driven selective masking, assigning `<TYPE_idx>` placeholders while retaining task-essential/public entities. |
| **Phase 5** | **Controlled Restoration & LLM Layer** | *Planned* | Pluggable LLM interface (`MockLLMProvider`, `GeminiProvider`) and controlled, authorized response placeholder restoration. |
| **Phase 6** | **FastAPI Backend Service** | *Planned* | REST API endpoints (`/analyze`, `/sanitize`, `/restore`, `/policies`, `/metrics`) with privacy-safe audit logging. |
| **Phase 7** | **Browser Extension (Manifest V3)** | *Planned* | Chrome extension for prompt capture, risk preview popup, user approval prompts, and sanitized prompt injection. |
| **Phase 8** | **Dashboard & Empirical Evaluation** | *Planned* | Interactive dashboard & 3-way comparative benchmark (No Protection vs Blind Masking vs PromptShield). |

> **Crucial Architectural Boundary**: Phase 1 establishes the detection foundation and local mapping plumbing. The system does not claim context awareness at Phase 1. True context-aware selective masking occurs in Phase 4 once context understanding (Phase 2) and policy engines (Phase 3) are active.

---

## 4. Phase 1 Features — Modular Hybrid Detection Foundation

Phase 1 has been refactored from a monolithic detector into a modular, hybrid detection architecture.

### Detection Pipeline

```
User Prompt
    │
    ▼
RegexDetector         ← Deterministic structured detection (zero-ML)
    │                   EMAIL, PHONE, CREDIT_CARD (Luhn), API_KEY,
    │                   ACCESS_TOKEN, PASSWORD, BANK_ACCOUNT, IP_ADDRESS,
    │                   ORDER_ID/USER_ID/CUSTOMER_ID, DATE
    ▼
PresidioDetector      ← Microsoft Presidio PII detection
    │                   EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD,
    │                   IBAN_CODE, IP_ADDRESS, PERSON, LOCATION, DATE_TIME
    │                   Confidence score preserved AS-IS from Presidio
    ▼
SpacyNERDetector      ← spaCy en_core_web_sm pretrained NER
    │                   PERSON, ORG → ORGANIZATION, GPE/LOC → LOCATION, DATE
    │                   Falls back to HeuristicNERDetector (gazetteer +
    │                   contextual patterns) when model is unavailable
    ▼
EntityFusionEngine    ← Unified deduplication & conflict resolution
    │                   • Exact-span merge (multi-source → contributing_sources)
    │                   • Overlapping-span resolution (structured > NER)
    │                   • Character-offset integrity validation
    │                   • Output sorted by start offset
    ▼
List[DetectedEntity]  ← Unified, non-overlapping, sorted
```

### Key Principles

1. **Detection ≠ Masking**: The detector identifies candidate entities only. Whether an entity is masked is decided by Phases 2–4.
2. **Modular**: Each detector is an independent module; adding or replacing a detector does not affect others.
3. **Graceful degradation**: If Presidio or spaCy is unavailable, the system logs a clear warning and continues with the remaining detectors.
4. **No fabricated confidence**: Presidio's real scores are preserved. spaCy uses a documented baseline (not a made-up number).

### New Modules (Phase 1 Refactor)

| File | Role |
|:---|:---|
| `backend/config.py` | Centralized regex patterns, type maps, confidence baselines, conflict-resolution priorities |
| `backend/regex_detector.py` | Modular `RegexDetector` class (all structured detection) |
| `backend/presidio_detector.py` | `PresidioDetector` singleton wrapper |
| `backend/spacy_detector.py` | `SpacyNERDetector` + `HeuristicNERDetector` fallback |
| `backend/entity_fusion.py` | `EntityFusionEngine` — deduplication & overlap resolution |

---

## 5. Directory Structure

```text
Prompshield/
├── README.md                          # Master documentation & viva guide
├── requirements.txt                   # Dependency specifications
├── backend/
│   ├── __init__.py                    # Public package exports
│   ├── models.py                      # Data models (EntityType, DetectedEntity, BaselineMaskResult)
│   ├── normalization.py               # Value normalization functions
│   ├── detector.py                    # SensitiveDataDetector with Luhn verification & NER
│   ├── mapping_store.py               # Local session-isolated MappingStore
│   ├── baseline_masker.py             # Baseline placeholder substitute & mapping registrar
│   └── core.py                        # PromptShieldCore primary orchestration interface
├── cli/
│   ├── __init__.py
│   └── demo.py                        # Interactive CLI demonstration tool
└── tests/
    ├── __init__.py
    ├── test_detector_structured.py    # Structured detector unit tests (Email, Phone, Card, Keys)
    ├── test_detector_ner.py           # NER detector unit tests (Person, Org, Loc, Date)
    ├── test_normalization.py          # Entity normalization unit tests
    ├── test_baseline_masker.py        # Baseline masker & mapping store tests
    └── test_core_pipeline.py          # End-to-end integration tests
```

---

## 6. How to Run

### Requirements
- Python 3.10+ (Phase 1 uses standard library modules with zero mandatory external packages).

### Running Automated Tests
Run the test suite using Python's built-in `unittest` runner:
```powershell
python -m unittest discover -s tests -v
```

### Running the Interactive CLI Demo
Launch the demonstration tool:
```powershell
python -m cli.demo
```
The CLI provides pre-configured sample prompts and allows you to test custom inputs:
```text
Sample Input:
"My name is John Mathew and my email is john@gmail.com. Write a professional email to ABC Technologies regarding my application for the London office."

Output:
[1] DETECTED:
    - PERSON: "John Mathew" (conf: 0.89)
    - EMAIL: "john@gmail.com" (conf: 0.98)
    - ORGANIZATION: "ABC Technologies" (conf: 0.90)
    - LOCATION: "London" (conf: 0.90)

[2] SANITIZED PROMPT:
    "My name is <PERSON_1> and my email is <EMAIL_1>. Write a professional email to <ORG_1> regarding my application for the <LOCATION_1> office."

[3] LOCAL USER-SIDE MAPPING:
    <PERSON_1>   ===> "John Mathew"
    <EMAIL_1>    ===> "john@gmail.com"
    <ORG_1>      ===> "ABC Technologies"
    <LOCATION_1> ===> "London"
```

---

## 7. Security Principles Enforced
1. **User-Side Isolation**: The mapping between `<PLACEHOLDER>` and raw secrets is kept strictly on the local machine and is never sent to external LLMs.
2. **Fail-Safe Integrity**: Detection conflict resolution ensures entities are not corrupted by partial or overlapping substitutions.
3. **No Blind Claims**: Explicit separation between Phase 1 baseline masking infrastructure and Phase 4 context-aware selective masking.
