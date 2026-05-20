"""
Deterministic derivation of the senior-candidate record from raw entities.

Sakhi-pattern split: senior_extractor emits employers + nested client
engagements with verbatim evidence; every business rule (grounding, ordering,
current-vs-previous, duration arithmetic, experience years) is decided here in
Python. No model judgment leaks past this layer.

The output is a structured record — crucially `employment[]` is a real array
with nested `clientsServed[]`, NOT a collapsed string. This is the shape a
multi-employer / consultancy CV needs and the student pipeline cannot produce.

Reuses the recruiter pipeline's deterministic helpers (date parsing, evidence
grounding, employment sort, experience-years interval merge, education filter)
so senior and recruiter outputs agree on every shared rule.
"""
from __future__ import annotations

import re

from .derive import (
    TODAY_YEAR,
    TODAY_MONTH,
    parse_date,
    _norm_for_grounding,
    _evidence_grounded,
    sort_employment,
    derive_experience_years,
    filter_and_align_education,
    normalize_skills,
)

_PRESENT = {"present", "current", "ongoing", "till date", "now"}

# Phrases in an entry's evidence that mark an ongoing role — used to recover an
# end date the model left empty (e.g. "...as a Contractor since Nov'23 till date").
_PRESENT_EVIDENCE_RE = re.compile(
    r"\b(present|current|till\s*date|to\s*date|till\s*now|ongoing)\b", re.IGNORECASE
)


def _month_index(iso: str | None) -> int:
    """'YYYY-MM' -> sortable month integer. -1 when absent/unparseable."""
    if not iso:
        return -1
    m = re.match(r"(\d{4})-(\d{2})", iso)
    return int(m.group(1)) * 12 + int(m.group(2)) if m else -1


def _employment_sort_key(emp: dict) -> tuple[int, int]:
    """Sort key — most recent first. An ongoing role sorts above any dated
    role; among ongoing roles the one with the latest start wins (that is
    the genuine current job — fixes a 2015-start role outranking a 2023 one)."""
    end_idx = 10 ** 7 if emp.get("isOngoing") else _month_index(emp.get("endDate"))
    return (end_idx, _month_index(emp.get("startDate")))


def _normalize_open_ended(entries: list[dict]) -> list[dict]:
    """Recover an empty `end` for ongoing roles from the entry's evidence. The
    employer-listing pass sometimes drops 'till date'; without an end the role
    missorts to the bottom and the wrong company becomes 'current'."""
    out: list[dict] = []
    for e in entries:
        e = dict(e)
        end = e.get("end")
        if not (end and parse_date(end)):
            if _PRESENT_EVIDENCE_RE.search(e.get("evidence") or ""):
                e["end"] = "Present"
        out.append(e)
    return out


def _iso(d: tuple[int, int] | None) -> str | None:
    """(year, month) -> 'YYYY-MM'."""
    if not d:
        return None
    return f"{d[0]:04d}-{d[1]:02d}"


def _is_present(end_raw) -> bool:
    return bool(end_raw) and str(end_raw).strip().lower() in _PRESENT or \
        bool(end_raw) and str(end_raw).strip().lower().startswith("present")


def _duration_months(start_raw, end_raw) -> int | None:
    """Whole months between two verbatim date strings. Open-ended 'Present'
    counts to today. None when either endpoint won't parse."""
    s = parse_date(start_raw)
    if not s:
        return None
    e = parse_date(end_raw) or ((TODAY_YEAR, TODAY_MONTH) if _is_present(end_raw) else None)
    if not e:
        return None
    months = (e[0] * 12 + e[1]) - (s[0] * 12 + s[1])
    return months if 0 <= months <= 60 * 12 else None


def _derive_clients(clients: list[dict], source_norm: str) -> list[dict]:
    """Shape + ground each clientsServed entry. Ungrounded entries are dropped
    with a log line (anti-hallucination — same rule as employment entries)."""
    out: list[dict] = []
    for c in clients:
        if not isinstance(c, dict) or not c.get("client"):
            continue
        if not _evidence_grounded(c, source_norm):
            print(f"[senior_derive] client dropped (ungrounded): {c.get('client')!r}")
            continue
        start_raw, end_raw = c.get("start"), c.get("end")
        out.append({
            "client": (c.get("client") or "").strip(),
            "role": (c.get("role") or None),
            "location": c.get("location") or None,
            "project": c.get("project") or None,
            "startDate": _iso(parse_date(start_raw)),
            "endDate": "Present" if _is_present(end_raw) else _iso(parse_date(end_raw)),
            "durationMonths": _duration_months(start_raw, end_raw),
            "details": [d for d in (c.get("details") or []) if isinstance(d, str) and d.strip()],
            "evidence": c.get("evidence") or None,
        })
    return out


def _derive_employment(entries: list[dict], source_text: str) -> list[dict]:
    """Build the structured employment[] array, most-recent-first, grounded."""
    # sort_employment grounds each entry's evidence against the source + drops
    # ungrounded ones; the final order is re-derived below on the shaped dicts.
    sorted_emp = sort_employment(entries, source_text=source_text)
    source_norm = _norm_for_grounding(source_text)

    employment: list[dict] = []
    for e in sorted_emp:
        start_raw, end_raw = e.get("start"), e.get("end")
        is_ongoing = _is_present(end_raw)
        # Entry keys match the gold schema: role / isOngoing / details. An
        # ongoing role carries endDate=null (the isOngoing flag is the signal).
        employment.append({
            "company": (e.get("company") or "").strip() or None,
            "role": (e.get("title") or "").strip() or None,
            "location": e.get("location") or None,
            "startDate": _iso(parse_date(start_raw)),
            "endDate": None if is_ongoing else _iso(parse_date(end_raw)),
            "isOngoing": is_ongoing,
            "durationMonths": _duration_months(start_raw, end_raw),
            "details": [
                r for r in (e.get("responsibilities") or []) if isinstance(r, str) and r.strip()
            ],
            "clientsServed": _derive_clients(e.get("clientsServed") or [], source_norm),
            "evidence": e.get("evidence") or None,
        })
    employment.sort(key=_employment_sort_key, reverse=True)
    return employment


# ── Engagement -> employer mapping ──────────────────────────────────────────
# Consultancy CVs list employers as one-line summaries and put the real per-role
# detail in a separate "Projects" section as dated client-engagement blocks.
# senior_extractor extracts those blocks; here Python attaches each to its
# employer — first by company-name match (the 'Client/Vendor' header), then by
# date-interval overlap with the employer's tenure. No LLM judgment involved.

_CORP_SUFFIX_RE = re.compile(
    r"\b(?:pvt|private|ltd|limited|inc|incorporated|llc|llp|corp|corporation|"
    r"technologies|technology|solutions|services|consulting|software|systems|"
    r"global|india|usa)\b",
    re.IGNORECASE,
)


def _company_norm(s: str) -> str:
    """Lowercase, drop punctuation + corporate suffixes, collapse whitespace."""
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    s = _CORP_SUFFIX_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _company_match(a: str, b: str) -> bool:
    """True when two company names refer to the same organisation — substring
    match after suffix stripping, length-guarded against trivial collisions."""
    na, nb = _company_norm(a), _company_norm(b)
    if len(na) < 3 or len(nb) < 3:
        return False
    return na == nb or na in nb or nb in na


def _raw_interval(start_raw, end_raw) -> tuple[int, int]:
    """Verbatim date strings -> (start_month_index, end_month_index); -1 unknown."""
    s = parse_date(start_raw)
    e = parse_date(end_raw)
    if e is None and _is_present(end_raw):
        e = (TODAY_YEAR, TODAY_MONTH)
    return (s[0] * 12 + s[1] if s else -1, e[0] * 12 + e[1] if e else -1)


def _employer_interval(emp: dict) -> tuple[int, int]:
    s = _month_index(emp.get("startDate"))
    end = emp.get("endDate")
    e = TODAY_YEAR * 12 + TODAY_MONTH if end == "Present" else _month_index(end)
    return (s, e)


def _overlap_months(iv1: tuple[int, int], iv2: tuple[int, int]) -> int:
    """Months of overlap between two month-index intervals. 0 if either is
    unknown or they are disjoint."""
    if -1 in iv1 or -1 in iv2:
        return 0
    return max(0, min(iv1[1], iv2[1]) - max(iv1[0], iv2[0]))


def _engagement_to_client(eng: dict, employer_company: str | None) -> dict:
    """Shape one engagement as a clientsServed entry. The client is the header
    company that is NOT the employer (a 'Realogy/LTIMindtree' block under
    employer LTIMindtree has client Realogy)."""
    companies = eng.get("companies") or []
    clients = [c for c in companies if not _company_match(c, employer_company or "")]
    client_name = (clients[0] if clients else (companies[0] if companies else None))
    start_raw, end_raw = eng.get("start"), eng.get("end")
    return {
        "client": (client_name or "").strip() or None,
        "role": eng.get("role") or None,
        "location": eng.get("location") or None,
        "project": eng.get("project") or None,
        "startDate": _iso(parse_date(start_raw)),
        "endDate": "Present" if _is_present(end_raw) else _iso(parse_date(end_raw)),
        "durationMonths": _duration_months(start_raw, end_raw),
        "details": [d for d in (eng.get("responsibilities") or []) if isinstance(d, str) and d.strip()],
        "evidence": eng.get("evidence") or None,
    }


def _engagement_as_project(eng: dict) -> dict:
    """An engagement that maps to no employer — kept as a standalone project so
    nothing is dropped (lossless)."""
    companies = eng.get("companies") or []
    return {
        "title": eng.get("project") or (companies[0] if companies else None),
        "companies": companies,
        "role": eng.get("role") or None,
        "location": eng.get("location") or None,
        "startDate": _iso(parse_date(eng.get("start"))),
        "endDate": "Present" if _is_present(eng.get("end")) else _iso(parse_date(eng.get("end"))),
        "details": [d for d in (eng.get("responsibilities") or []) if isinstance(d, str) and d.strip()],
        "evidence": eng.get("evidence") or None,
    }


def _attach_engagements(employment: list[dict], engagements: list[dict],
                        source_norm: str) -> list[dict]:
    """Attach each engagement to its employer's clientsServed[]. Returns the
    engagements that mapped to no employer (kept as standalone projects)."""
    unmapped: list[dict] = []
    for eng in engagements:
        if not _evidence_grounded(eng, source_norm):
            print(f"[senior_derive] engagement dropped (ungrounded): {eng.get('evidence')!r}")
            continue
        companies = eng.get("companies") or []

        # 1. company-name match against a known employer.
        employer = None
        for emp in employment:
            if any(_company_match(c, emp.get("company") or "") for c in companies):
                employer = emp
                break

        # 2. fall back to date-interval overlap with employer tenures.
        if employer is None:
            eng_iv = _raw_interval(eng.get("start"), eng.get("end"))
            best, best_ov = None, 0
            for emp in employment:
                ov = _overlap_months(_employer_interval(emp), eng_iv)
                if ov > best_ov:
                    best, best_ov = emp, ov
            employer = best

        if employer is None:
            unmapped.append(_engagement_as_project(eng))
            continue
        employer["clientsServed"].append(_engagement_to_client(eng, employer.get("company")))
    return unmapped


def derive_senior_record(extracted: dict, source_text: str) -> dict:
    """Compose the senior-candidate record from extracted entities + source.

    Output keys overlap the hybrid merge contract: name/phone/email,
    employment[] (structured), currentCompany/currentRole, experienceYears,
    college/degree, skills."""
    contact = extracted.get("contact") or {}
    raw_employment = _normalize_open_ended(extracted.get("employment_entries") or [])
    education = extracted.get("education_entries") or []
    skills = extracted.get("skills_raw") or []

    employment = _derive_employment(raw_employment, source_text)

    # Attach projects-section client engagements onto their employers. Returns
    # engagements that mapped to no employer — kept as standalone projects[].
    source_norm = _norm_for_grounding(source_text)
    projects = _attach_engagements(
        employment, extracted.get("engagements") or [], source_norm
    )

    # current company/role come straight off the sorted employment[] (top entry).
    current_company = employment[0]["company"] if employment else None
    current_role = employment[0]["role"] if employment else None
    # experience years: deterministic interval merge over the grounded entries.
    sorted_emp = sort_employment(raw_employment, source_text=source_text)
    experience_years = derive_experience_years(sorted_emp, source_text=source_text)

    college_name, degree = filter_and_align_education(education, source_text=source_text)
    key_skills = normalize_skills(skills)

    n_clients = sum(len(e["clientsServed"]) for e in employment)
    print(
        f"[senior_derive] {len(employment)} employment entries, "
        f"{n_clients} client engagements, {len(projects)} unmapped project(s), "
        f"experienceYears={experience_years}"
    )

    return {
        "name": contact.get("name"),
        "phone": contact.get("phone"),
        "email": contact.get("email"),
        "employment": employment,
        "projects": projects,
        "currentCompany": current_company,
        "currentRole": current_role,
        "experienceYears": experience_years,
        "college": college_name or None,
        "degree": degree or None,
        "skills": [s.strip() for s in key_skills.split(",")] if key_skills else [],
        "_recordType": "senior",
    }
