"""
Single-resume pipeline orchestrator for the senior / experienced-candidate mode.

  load_resume(path)                       # shared loader
       -> text
  split_sections(text)                    # shared splitter
       -> sections dict (focuses the per-employer chunking on the
          experience section)
  extract_all_senior(text, model, sections)
       -> {contact, employment_entries (chunked, with clientsServed),
           education_entries, skills_raw}
  derive_senior_record(extracted, source_text)
       -> structured record with a real employment[] array

Mirrors student_pipeline.py. Routed to by the hybrid extractor when the
seniority router classifies a resume as senior — see hybrid_extract.py.
"""
from __future__ import annotations

from pathlib import Path

from .loaders import load_resume
from .sections import split_sections
from .senior_extractor import extract_all_senior
from .senior_derive import derive_senior_record


def process_resume_senior(path: str | Path, model: str) -> dict:
    """End-to-end pipeline for a single senior / experienced resume."""
    p = Path(path)
    print(f"\n[senior_pipeline] {p.name}")

    text = load_resume(p)
    if not text.strip():
        print(
            f"[senior_pipeline] CRITICAL: 0 chars extracted from {p.name} — "
            f"likely a scanned / image-only PDF. OCR required."
        )
        return _empty_record(p)

    sections = split_sections(text)
    print(f"[senior_pipeline] sections: {list(sections.keys())}")

    extracted = extract_all_senior(text, model, sections=sections)
    print(
        f"[senior_pipeline] entities: "
        f"{len(extracted.get('employment_entries', []))} employment, "
        f"{len(extracted.get('education_entries', []))} education, "
        f"{len(extracted.get('skills_raw', []))} skills"
    )

    record = derive_senior_record(extracted, source_text=text)
    record["_source_file"] = str(p)
    return record


def _empty_record(p: Path) -> dict:
    return {
        "name": None, "phone": None, "email": None,
        "employment": [], "currentCompany": None, "currentRole": None,
        "experienceYears": None, "college": None, "degree": None,
        "skills": [], "_recordType": "senior", "_source_file": str(p),
    }
