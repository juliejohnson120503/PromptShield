# PromptShield Baseline — Error Analysis

**Pipeline:** CURRENT PromptShield Detection Pipeline
**Dataset:** 250 prompts — `ground_truth.json`

---

## False Positives

Total FPs: **21**

| Prompt ID | Predicted Text | Predicted Type | Source | Error Classification |
|-----------|---------------|----------------|--------|---------------------|
| test_016 | `yesterday` | DATE | gliner2 | false positive (hallucinated entity) |
| test_023 | `admin_root` | CUSTOMER_ID | gliner2 | false positive (hallucinated entity) |
| test_049 | `2022` | DATE | presidio | false positive (hallucinated entity) |
| test_050 | `15 Hauptstrasse, Berlin, Germany` | ADDRESS | gliner2 | wrong entity type |
| test_066 | `1234567890` | PHONE | gliner2 | false positive (hallucinated entity) |
| test_071 | `456789` | CUSTOMER_ID | presidio | false positive (hallucinated entity) |
| test_075 | `Today` | DATE | presidio | false positive (hallucinated entity) |
| test_077 | `admin` | CUSTOMER_ID | gliner2 | false positive (hallucinated entity) |
| test_087 | `priya` | PERSON | gliner2 | false positive (hallucinated entity) |
| test_087 | `delhi` | LOCATION | gliner2 | false positive (hallucinated entity) |
| test_090 | `May` | DATE | presidio | false positive (hallucinated entity) |
| test_111 | `admin` | CUSTOMER_ID | gliner2 | false positive (hallucinated entity) |
| test_141 | `London` | LOCATION | gliner2 | wrong entity type |
| test_141 | `SW1A 2AA` | LOCATION | gliner2 | wrong entity type |
| test_142 | `Marylebone` | LOCATION | gliner2 | wrong entity type |
| test_142 | `London` | LOCATION | gliner2 | wrong entity type |
| test_142 | `NW1 6XE` | LOCATION | gliner2 | wrong entity type |
| test_144 | `Paris` | LOCATION | gliner2 | wrong entity type |
| test_144 | `France` | LOCATION | gliner2 | wrong entity type |
| test_152 | `Paris` | LOCATION | gliner2 | wrong entity type |
| test_160 | `fox` | PERSON | gliner2 | false positive (hallucinated entity) |

---

## False Negatives

Total FNs: **4**

| Prompt ID | Expected Text | Expected Type | System Predicted Instead | Error Classification |
|-----------|--------------|----------------|--------------------------|---------------------|
| test_050 | `Berlin` | LOCATION | `15 Hauptstrasse, Berlin, Germany` (ADDRESS) | partial span / incorrect boundary |
| test_050 | `Germany` | LOCATION | `15 Hauptstrasse, Berlin, Germany` (ADDRESS) | partial span / incorrect boundary |
| test_055 | `Springfield` | LOCATION | `742 Evergreen Terrace, Springfield` (ADDRESS) | partial span / incorrect boundary |
| test_089 | `New York` | LOCATION | `45 Park Avenue, New York, NY 10022` (ADDRESS) | partial span / incorrect boundary |

---

## Error Summary by Type

### Most Over-Detected Types (FP)

| Entity Type | FP Count |
|-------------|----------|
| LOCATION | 9 |
| DATE | 4 |
| CUSTOMER_ID | 4 |
| PERSON | 2 |
| ADDRESS | 1 |
| PHONE | 1 |

### Most Frequently Missed Types (FN)

| Entity Type | FN Count |
|-------------|----------|
| LOCATION | 4 |

---

## Unsupported Ground-Truth Entity Types

The following GT entity types have no matching PromptShield EntityType.
They are excluded from TP/FP/FN scoring.

| GT Label | Count |
|----------|-------|
