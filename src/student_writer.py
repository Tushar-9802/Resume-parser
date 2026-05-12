"""
Student-pipeline writer.

JSONL is the canonical output (one Pulse student record per line) because the
schema has per-project arrays that don't flatten cleanly into CSV.

A companion summary CSV is also emitted for quick eyeballing — one row per
candidate with the flat scalar fields plus project/internship counts.

The resumable-pipeline plumbing (load_seen / append_row) is identical to the
recruiter writer, so we re-export rather than duplicate.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

# Re-export so the CLI can import resume-state helpers from one place
from .csv_writer import load_seen, append_row  # noqa: F401


# Summary-CSV columns. Order is what reads naturally left-to-right when
# eyeballing a spreadsheet of candidates.
_SUMMARY_COLUMNS = [
    "Name", "Phone", "Email",
    "Degree", "Branch", "CGPA", "CGPA Scale", "Graduation Year",
    "City", "State",
    "# Projects", "# Internships",
    "Top Skills", "Top Tools",
    "Target Roles",
    "Source File",
]


def write_summary_csv(jsonl_path: str | Path, csv_path: str | Path, max_skill_preview: int = 6) -> int:
    """Rebuild a flat-view summary CSV from the JSONL log.

    Each row is one candidate. Multi-value lists are previewed (top-N) rather
    than fully serialized — the JSONL is the source of truth for the full
    arrays. Returns rows written."""
    jp = Path(jsonl_path)
    rows: list[dict] = []
    if jp.exists():
        with open(jp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    cp = Path(csv_path)
    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(_SUMMARY_COLUMNS)
        for r in rows:
            writer.writerow([
                _s(r.get("name")),
                _s(r.get("phone")),
                _s(r.get("email")),
                _s(r.get("degree")),
                _s(r.get("branch")),
                _s(r.get("cgpa")),
                _s(r.get("cgpaScale")),
                _s(r.get("graduationYear")),
                _s(r.get("city")),
                _s(r.get("state")),
                len(r.get("projects") or []),
                _count_internships(r.get("internshipDetails")),
                ", ".join((r.get("skills") or [])[:max_skill_preview]),
                ", ".join((r.get("toolsUsed") or [])[:max_skill_preview]),
                _s(r.get("targetRoles")),
                _s(r.get("_source_file")),
            ])
    return len(rows)


def _s(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return str(int(val)) if val == int(val) else str(val)
    return str(val)


def _count_internships(blob: str | None) -> int:
    """Internship blocks are separated by blank lines in the flattened blob.
    Count blocks that have a header line (one with ' - ' between role/company)."""
    if not blob:
        return 0
    blocks = [b for b in blob.split("\n\n") if b.strip()]
    return len(blocks)


# Companion CSV: one row per project per candidate. Spreadsheet-friendly view
# of the projects[] array — each candidate fans out to N rows.
_PROJECTS_COLUMNS = [
    "Candidate", "Source File",
    "Project #", "Title",
    "Description", "Skills Used", "Tech Stack Count",
    "Start", "End", "In Progress",
    "Evidence",
]


def write_projects_csv(jsonl_path: str | Path, csv_path: str | Path) -> int:
    """Rebuild a one-row-per-project CSV from the JSONL log. Returns rows."""
    jp = Path(jsonl_path)
    cp = Path(csv_path)
    cp.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(cp, "w", encoding="utf-8-sig", newline="") as out:
        writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(_PROJECTS_COLUMNS)
        if not jp.exists():
            return 0
        with open(jp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                projects = r.get("projects") or []
                if not projects:
                    continue
                candidate = _s(r.get("name"))
                source = _s(r.get("_source_file"))
                for idx, p in enumerate(projects, 1):
                    writer.writerow([
                        candidate,
                        source,
                        idx,
                        _s(p.get("title")),
                        _s(p.get("description")),
                        ", ".join(p.get("skillsUsed") or []),
                        len(p.get("skillsUsed") or []),
                        _format_project_date(p.get("startMonth"), p.get("startYear")),
                        _format_project_date(p.get("endMonth"), p.get("endYear")),
                        "yes" if p.get("isInProgress") else "no",
                        _s(p.get("evidence")),
                    ])
                    n += 1
    return n


def _format_project_date(month, year) -> str:
    """Render (month, year) -> 'YYYY-MM' / 'YYYY' / '' for the CSV cell."""
    if year is None:
        return ""
    if month is None:
        return str(year)
    return f"{year}-{int(month):02d}"
