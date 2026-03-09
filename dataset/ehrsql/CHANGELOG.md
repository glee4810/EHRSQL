# EHRSQL Dataset Changelog

## v.1.5.0 (2026-03-09)

This release improves annotation quality across all 6 dataset files (mimic\_iii + eicu, train/valid/test). All 22,505 SQL queries execute successfully. No schema, task definition, or annotation structure changes.

### Summary

| Category | Count | Description |
|---|---|---|
| SQL: ORDER BY → subquery | 1,840 | Fix non-deterministic tie-breaking |
| SQL: IS NOT NULL guards | 148 | Fix NULL propagation on discharge columns |
| SQL: COUNT DISTINCT | 97 | Fix counting prescriptions vs distinct drugs |
| Question: naturalness rewrite | 24,405 | Grammar, capitalization, time expressions |
| Question: formatting & grammar | ~165 | Date format, singular/plural, pronouns |
| Question: surgery → procedure | 110 | Align with schema terminology |

---

### SQL fixes

#### 1. ORDER BY LIMIT 1 → correlated subquery (1,840 queries)

Queries selecting a value at the first/last time point used `ORDER BY time DESC LIMIT 1`, which is non-deterministic when multiple records share the same timestamp. Replaced with a correlated subquery that returns all matching rows at the extreme timestamp. 12.9% of fixes changed actual query results.

```sql
-- Before
SELECT v.valuenum FROM chartevents v WHERE ... ORDER BY v.charttime DESC LIMIT 1

-- After
SELECT v.valuenum FROM chartevents v WHERE ...
  AND v.charttime = (SELECT DISTINCT v2.charttime FROM chartevents v2
    WHERE ... ORDER BY v2.charttime DESC LIMIT 1)
```

#### 2. IS NOT NULL guards (148 queries)

Queries referencing nullable discharge time columns (`admissions.dischtime`, `patient.hospitaldischargetime`) without NULL guards. Added `col IS NOT NULL AND` to affected WHERE clauses.

#### 3. COUNT(\*) → COUNT(DISTINCT drug) (97 queries)

Questions asking ``how many drugs'' used `COUNT(*)`, which counts prescription rows rather than distinct drugs. Fixed to `COUNT(DISTINCT prescriptions.drug)` / `COUNT(DISTINCT medication.drugname)`.

---

### Question text improvements

#### 1. Naturalness rewrite (24,405 questions)

All questions rewritten by GPT-4o-mini to fix:
- Capitalization (24,336 lowercase starts)
- Punctuation (3,656 trailing periods → question marks)
- Subject-verb agreement (``have patient X been'' → ``has patient X been'')
- Time expression diversity (13 patterns: ``prior to'', ``within the past'', ``more than X ago'', etc.)

Each rewrite verified by a second LLM call. Failed verifications re-rewritten (up to 2 rounds). Final LLM semantic verification (22,505 items) found and fixed 19 time direction errors.

#### 2. Additional grammar & formatting (~165 questions)

| Fix | Count | Example |
|---|---|---|
| Date format | 134 | ``this month/28'' → ``the 28th of this month'' |
| Singular → plural | 19 | ``top three most common diagnosis'' → ``diagnoses'' |
| Subject-verb | 8 | ``What is the five most'' → ``What are the five most'' |
| Pronoun | 8 | ``measured its heart rate'' → ``measured their heart rate'' |
| Tense | 1 | ``did patient received'' → ``did patient receive'' |
| Broken sentence | 1 | Garbled LLM output → restored from template |

#### 3. surgery → procedure (110 questions)

Replaced ``surgery'' with ``procedure'' in question text to align with the schema and SQL.

---

### Verification

All changes verified with zero tolerance:

- **SQL execution**: 22,505/22,505 queries execute successfully (`current_time='2105-12-31 23:59:00'`)
- **Time direction**: 0 conflicts between question phrasing and SQL comparison operators
- **Patient ID**: 0 missing or altered patient IDs
- **Year range**: All years within 2100--2105
- **Formatting**: All questions capitalized, end with ``?'', no double spaces
