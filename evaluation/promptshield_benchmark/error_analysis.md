# PromptShield Baseline — Error Analysis

**Pipeline:** CURRENT PromptShield Detection Pipeline
**Dataset:** 105 prompts — `ground_truth.json`

---

## False Positives

Total FPs: **54**

| Prompt ID | Predicted Text | Predicted Type | Source | Error Classification |
|-----------|---------------|----------------|--------|---------------------|
| test_003 | `noon` | DATE | presidio | false positive (hallucinated entity) |
| test_006 | `OpenAI` | ORGANIZATION | gliner | false positive (hallucinated entity) |
| test_015 | `last quarter` | DATE | presidio | false positive (hallucinated entity) |
| test_016 | `yesterday` | DATE | presidio | false positive (hallucinated entity) |
| test_021 | `AWS` | ORGANIZATION | gliner | false positive (hallucinated entity) |
| test_021 | `public repository` | LOCATION | gliner | false positive (hallucinated entity) |
| test_022 | `IBAN` | ORGANIZATION | spacy | false positive (hallucinated entity) |
| test_023 | `admin_root` | USER_ID | gliner | false positive (hallucinated entity) |
| test_026 | `IPv6` | ORGANIZATION | spacy | false positive (hallucinated entity) |
| test_028 | `GitHub` | ORGANIZATION | gliner | false positive (hallucinated entity) |
| test_028 | `CI` | ORGANIZATION | spacy | false positive (hallucinated entity) |
| test_031 | `employee` | PERSON | gliner | false positive (hallucinated entity) |
| test_031 | `ID EMP-7712` | USER_ID | gliner | wrong entity type |
| test_031 | `quarterly` | DATE | presidio | false positive (hallucinated entity) |
| test_032 | `Indian` | PERSON | presidio | false positive (hallucinated entity) |
| test_035 | `2024 fiscal year` | DATE | presidio | false positive (hallucinated entity) |
| test_035 | `Q3` | LOCATION | presidio | false positive (hallucinated entity) |
| test_040 | `hotel` | LOCATION | gliner | false positive (hallucinated entity) |
| test_041 | `regions` | LOCATION | gliner | false positive (hallucinated entity) |
| test_043 | `Google` | ORGANIZATION | gliner | false positive (hallucinated entity) |
| test_047 | `Bearer` | ORGANIZATION | spacy | false positive (hallucinated entity) |
| test_049 | `2022` | DATE | spacy | false positive (hallucinated entity) |
| test_050 | `15 Hauptstrasse` | LOCATION | gliner | false positive (hallucinated entity) |
| test_052 | `next year` | DATE | presidio | false positive (hallucinated entity) |
| test_053 | `123456789012` | PASSWORD | gliner | wrong entity type |
| test_055 | `Evergreen Terrace` | ORGANIZATION | spacy | wrong entity type |
| test_056 | `EMP-5501` | PERSON | presidio | wrong entity type |
| test_061 | `Pass#2024` | PASSWORD | gliner | partial span / incorrect boundary |
| test_062 | `quarterly` | DATE | presidio | false positive (hallucinated entity) |
| test_063 | `project Alpha` | ORGANIZATION | gliner | false positive (hallucinated entity) |
| test_064 | `Visa` | ORGANIZATION | spacy | false positive (hallucinated entity) |
| test_066 | `1234567890` | PASSWORD | gliner | false positive (hallucinated entity) |
| test_067 | `postal zone` | LOCATION | gliner | false positive (hallucinated entity) |
| test_071 | `456789` | USER_ID | presidio | false positive (hallucinated entity) |
| test_075 | `Today` | DATE | presidio | false positive (hallucinated entity) |
| test_075 | `a good day` | DATE | presidio | false positive (hallucinated entity) |
| test_077 | `admin` | USER_ID | gliner | false positive (hallucinated entity) |
| test_078 | `project Orion` | ORGANIZATION | gliner | false positive (hallucinated entity) |
| test_079 | `Db` | USER_ID | gliner | wrong entity type |
| test_080 | `three engineers` | PERSON | gliner | false positive (hallucinated entity) |
| test_084 | `Ubuntu 22.04 LTS` | ORGANIZATION | spacy | false positive (hallucinated entity) |
| test_086 | `every 90 days` | DATE | presidio | false positive (hallucinated entity) |
| test_087 | `priya` | PERSON | gliner | false positive (hallucinated entity) |
| test_087 | `delhi` | LOCATION | gliner | false positive (hallucinated entity) |
| test_090 | `May` | DATE | presidio | false positive (hallucinated entity) |
| test_093 | `AWS` | ORGANIZATION | gliner | false positive (hallucinated entity) |
| test_096 | `IBAN` | ORGANIZATION | spacy | false positive (hallucinated entity) |
| test_097 | `nightly` | DATE | presidio | false positive (hallucinated entity) |
| test_097 | `02:00 UTC` | DATE | presidio | false positive (hallucinated entity) |
| test_097 | `PII` | ORGANIZATION | spacy | false positive (hallucinated entity) |
| test_098 | `NewPass` | ORGANIZATION | gliner | wrong entity type |
| test_099 | `world` | LOCATION | gliner | false positive (hallucinated entity) |
| test_102 | `Barclays London` | ORGANIZATION | spacy | partial span / incorrect boundary |
| test_105 | `Infosys Support Team` | ORGANIZATION | gliner | partial span / incorrect boundary |

---

## False Negatives

Total FNs: **11**

| Prompt ID | Expected Text | Expected Type | System Predicted Instead | Error Classification |
|-----------|--------------|----------------|--------------------------|---------------------|
| test_031 | `EMP-7712` | CUSTOMER_ID | `ID EMP-7712` (USER_ID) | partial span / incorrect boundary |
| test_053 | `123456789012` | BANK_ACCOUNT | `123456789012` (PASSWORD) | wrong entity type (span correct) |
| test_055 | `742 Evergreen Terrace` | ADDRESS | `Evergreen Terrace` (ORGANIZATION) | partial span / incorrect boundary |
| test_056 | `EMP-5501` | CUSTOMER_ID | `EMP-5501` (PERSON) | wrong entity type (span correct) |
| test_061 | `Pass#2024!` | PASSWORD | `Pass#2024` (PASSWORD) | partial span / incorrect boundary |
| test_079 | `Db@Pass2024Secret!` | PASSWORD | `Db` (USER_ID) | partial span / incorrect boundary |
| test_089 | `New York` | LOCATION | `45 Park Avenue, New York, NY 10022` (ADDRESS) | partial span / incorrect boundary |
| test_098 | `NewPass#456` | PASSWORD | `NewPass` (ORGANIZATION) | partial span / incorrect boundary |
| test_102 | `Barclays` | ORGANIZATION | `Barclays London` (ORGANIZATION) | partial span / incorrect boundary |
| test_102 | `London` | LOCATION | `Barclays London` (ORGANIZATION) | partial span / incorrect boundary |
| test_105 | `Infosys` | ORGANIZATION | `Infosys Support Team` (ORGANIZATION) | partial span / incorrect boundary |

---

## Error Summary by Type

### Most Over-Detected Types (FP)

| Entity Type | FP Count |
|-------------|----------|
| ORGANIZATION | 19 |
| DATE | 14 |
| LOCATION | 8 |
| USER_ID | 5 |
| PERSON | 5 |
| PASSWORD | 3 |

### Most Frequently Missed Types (FN)

| Entity Type | FN Count |
|-------------|----------|
| PASSWORD | 3 |
| CUSTOMER_ID | 2 |
| LOCATION | 2 |
| ORGANIZATION | 2 |
| BANK_ACCOUNT | 1 |
| ADDRESS | 1 |

---

## Unsupported Ground-Truth Entity Types

The following GT entity types have no matching PromptShield EntityType.
They are excluded from TP/FP/FN scoring.

| GT Label | Count |
|----------|-------|
