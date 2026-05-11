# Resume Parser

Local-first engineering-resume parser. Takes PDF / DOCX resumes and emits a recruiter-ready CSV with grounded, non-hallucinated extractions.

Built around the **model-as-entity-extractor** pattern: the LLM identifies employment, education, skills, and contact info with verbatim evidence quotes; pure-Python logic derives every CSV column (current vs previous, dedup, ordering, year math, degree filtering, multi-value alignment). No business-rule judgment is delegated to the model.

The architectural inspiration is the Sakhi (Gemma 4) pipeline's six-layer anti-hallucination split for Hindi conversational transcripts.

## What it produces

12-column CSV matching the standard recruiter template:

| Column | Type | Notes |
|---|---|---|
| Name | str | top of resume |
| Phone | str | normalized to `+91-XXXXXXXXXX` for Indian mobiles |
| Email | str | regex-validated |
| Current Company | str | most recent employer (fills even when role ended) |
| Current Role | str | title at the most recent employer |
| Experience | int | total years; prefers a stated number, falls back to merged date intervals |
| Key Skills | str | comma-separated, deduped, capped at 15 items |
| Current CTC | str | optional; null unless explicitly stated |
| Expected CTC | str | optional; null unless explicitly stated |
| Previous Companies | str | pipe-separated, most recent first, deduped (no current overlap) |
| College Name | str | pipe-separated, most recent first |
| Degree | str | pipe-separated, index-aligned with College Name |

## Quick start

```powershell
# Prerequisites: Python 3.11+, Ollama installed, a model pulled
ollama pull gemma4:e4b-it-q4_K_M

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run on a folder of resumes
python cli.py samples\ -o out\parsed.csv

# Or specific files / globs
python cli.py samples\resume.pdf
python cli.py samples\*.pdf samples\*.docx

# Resume from a crash (JSONL log is append-only; CSV regenerated each run)
python cli.py samples\ --jsonl out\log.jsonl -o out\parsed.csv
```

## Architecture

```
PDF / DOCX
   │
   ▼
loaders.py — column-aware pymupdf primary, pdfplumber fallback
            normalizes smart quotes, ligatures, bullet glyphs
   │
   ▼
sections.py — heading-synonym match (word boundaries only)
              splits into: contact, summary, experience, education, skills, ...
   │
   ▼
extractor.py — two Ollama calls (format=json, temperature=0.1):
   ├─ contact pass  →  {name, phone, email}
   └─ entities pass →  {employment_entries, education_entries, skills_raw}
                       each entry has a verbatim `evidence` field
   │
   ▼
derive.py — pure Python; the only place business rules live:
   ├─ verify each entry's `evidence` appears in source (60% token overlap)
   ├─ sort employment by end_date desc
   ├─ top = current_company / current_role
   ├─ rest = previous_companies (dedup case-insensitive, exclude current)
   ├─ merge employment intervals → experience_years (with stated-regex override)
   ├─ filter education: drop 10th/12th/CBSE/HSC, keep degree-shaped entries
   ├─ align college[i] ↔ degree[i] by construction (single iteration)
   └─ dedup + cap skills
   │
   ▼
validation.py — thin final cleanup:
   ├─ phone normalization (Indian-mobile +91- prefix)
   ├─ email regex
   ├─ fake-default strip ("John Doe", "example@example.com")
   └─ name grounding (verbatim check on contact name only)
   │
   ▼
csv_writer.py — append-only JSONL + CSV regen from JSONL
                resumable across crashes; --force to re-process
```

## Project layout

```
cli.py                          # entrypoint
configs/
  schema.yaml                   # 12-column CSV target (authoritative)
  section_synonyms.yaml         # heading aliases for the section splitter
src/
  loaders.py                    # PDF (column-aware) + DOCX → normalized text
  sections.py                   # heading-shaped line detection + synonym match
  extractor.py                  # two Ollama calls (contact + entities)
  derive.py                     # ALL business-rule logic in pure Python
  validation.py                 # final field-level cleanup
  csv_writer.py                 # JSONL log + CSV regen
samples/                        # user-supplied resumes (gitignored)
out/                            # generated CSV + JSONL (gitignored)
```

## Why gemma4:e4b-it-q4_K_M and not a larger / hosted model

Tested qwen2.5:7b-instruct first — it kept hallucinating client names from bullets as employers, refused to fill `current_company` for ended internships, and misapplied a "multi-value uses pipe" rule by jamming pipes into `key_skills`. gemma4:e4b respected the entity contract on every iteration.

Considered hosted Claude / GPT-4 — rejected for this product because:
- Candidate PII shouldn't leave the box for a recruiter tool
- Same resume on two runs must produce the same row (audit trail)
- Bigger models still don't know "PGDM is a degree, CBSE Board is a school exam" without a constraint anyway

The Sakhi-pattern scaffolding (Python decisions on grounded entities) does the work that scaling the model wouldn't.

## Adding / changing fields

The 12 columns are defined in `configs/schema.yaml` and the matching derivation lives in `src/derive.py:derive_csv_fields`. To add a field:

1. Add the spec in `configs/schema.yaml` with `column`, `type`, `grounded`, etc.
2. Have the extractor pull the entity that contains it (edit `ENTITIES_USER_TEMPLATE` in `extractor.py`).
3. Add a deriver in `derive.py` that turns the entity field into the CSV value.
4. If it has a regex or grounding rule, wire it in `validation.py`.

The CSV writer pulls column labels and order from `schema.yaml` automatically.

## Known minor issues

- A trailing location can leak into `Current Role` (e.g., `"Senior Software Engineer, Bangalore"`). Fixable with a city-token strip in derive.py.
- Certification metadata (`Coursera`, `MIT`) sometimes lands in `Key Skills` when the resume's skills/certifications sections are adjacent. Tightenable via the entities prompt or a Python filter against a known cert-provider list.
- Experience years can drift ±1 against external sources (Naukri filename) when the candidate's resume summary states a different number than the dates would compute to. The pipeline prefers what's on the resume; a recruiter-side cross-check is the appropriate fix.

## License

TBD.
