"""
Single-resume pipeline orchestrator (Sakhi-style).

  load_resume(path)
       → text
  split_sections(text)
       → sections dict (used to focus the entities prompt)
  extract_all(text, model, sections)
       → {contact, employment_entries, education_entries, skills_raw}
  derive_csv_fields(entities, source_text)
       → flat dict matching schema.yaml fields, all decisions made in Python
  validate(row, source_text, schema)
       → final field-level cleanup (regex, grounding, fake-default strip)
"""
from __future__ import annotations

from pathlib import Path

from .loaders import load_resume
from .sections import split_sections
from .extractor import extract_all, load_schema
from .derive import derive_csv_fields
from .validation import validate


def process_resume(path: str | Path, model: str) -> dict:
    """End-to-end pipeline for a single resume. Returns a dict keyed by
    schema field names, ready for CSV writing. Source path is included as
    `_source_file` for traceability."""
    p = Path(path)
    print(f"\n[pipeline] {p.name}")

    text = load_resume(p)
    schema = load_schema()
    if not text.strip():
        print(f"[pipeline] WARNING: empty text extracted from {p.name}")
        empty = {k: None for k in schema["fields"]}
        empty["_source_file"] = str(p)
        return empty

    sections = split_sections(text)
    print(f"[pipeline] sections: {list(sections.keys())}")

    extracted = extract_all(text, model, sections=sections)
    print(
        f"[pipeline] entities: "
        f"{len(extracted.get('employment_entries', []))} employment, "
        f"{len(extracted.get('education_entries', []))} education, "
        f"{len(extracted.get('skills_raw', []))} skills"
    )

    row = derive_csv_fields(extracted, source_text=text)
    row = validate(row, source_text=text, schema=schema)
    row["_source_file"] = str(p)
    return row
