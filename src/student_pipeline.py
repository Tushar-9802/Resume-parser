"""
Single-resume pipeline orchestrator for the student / fresher mode.

  load_resume(path)              # shared loader
       -> text
  split_sections(text)           # shared splitter
       -> sections dict (used to focus the entities prompt)
  extract_all_student(text, model, sections)
       -> {identity, education_entries, internship_entries, project_entries,
           skills_raw, tools_raw, certifications, languages_known}
  derive_student_record(extracted, source_text)
       -> Pulse-shaped dict (per-project array preserved)

Output is the Pulse student-record dict, ready for JSONL append. Source path
is attached as _source_file for traceability + resume support.
"""
from __future__ import annotations

from pathlib import Path

from .loaders import load_resume, extract_pdf_hyperlinks
from .sections import split_sections
from .student_extractor import extract_all_student
from .student_derive import derive_student_record


def process_resume_student(path: str | Path, model: str) -> dict:
    """End-to-end pipeline for a single fresher resume."""
    p = Path(path)
    print(f"\n[student_pipeline] {p.name}")

    text = load_resume(p)
    if not text.strip():
        # Most likely a scanned / image-only PDF (no text layer). Worth a
        # loud signal so the caller can route it to OCR rather than silently
        # producing an empty record.
        print(
            f"[student_pipeline] CRITICAL: 0 chars extracted from {p.name} — "
            f"likely a scanned / image-only PDF. OCR required (not yet supported)."
        )
        return _empty_record(p)

    sections = split_sections(text)
    print(f"[student_pipeline] sections: {list(sections.keys())}")

    # PDF link annotations — separate from the text stream. Used downstream to
    # resolve linkedinUrl / portfolioUrl / per-project links from the visible
    # "GitHub" / "LinkedIn" labels.
    hyperlinks = extract_pdf_hyperlinks(p)
    if hyperlinks:
        print(f"[student_pipeline] hyperlinks: {len(hyperlinks)} URL annotations")

    extracted = extract_all_student(text, model, sections=sections)
    print(
        f"[student_pipeline] entities: "
        f"{len(extracted.get('education_entries', []))} education, "
        f"{len(extracted.get('internship_entries', []))} internships, "
        f"{len(extracted.get('project_entries', []))} projects, "
        f"{len(extracted.get('skills_raw', []))} skills, "
        f"{len(extracted.get('tools_raw', []))} tools"
    )

    record = derive_student_record(extracted, source_text=text, hyperlinks=hyperlinks)
    record["_source_file"] = str(p)
    return record


def _empty_record(p: Path) -> dict:
    return {
        "name": None, "phone": None, "email": None,
        "linkedinUrl": None, "portfolioUrl": None,
        "dateOfBirth": None, "city": None, "state": None,
        "degree": None, "branch": None, "cgpa": None, "cgpaScale": None,
        "graduationYear": None,
        "skills": [], "toolsUsed": [], "certifications": [], "languagesKnown": [],
        "internshipDetails": None, "projects": [],
        "targetRoles": None, "preferredLocations": [],
        "_source_file": str(p),
    }
