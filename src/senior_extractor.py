"""
Entity extraction for the senior / experienced-candidate pipeline (Sakhi-style).

A senior resume (multi-employer, multi-page, often consultancy CVs with nested
client engagements) breaks the student pipeline two ways:
  1. The student pipeline collapses N employers into one `internshipDetails`
     string — total structural loss.
  2. A whole-resume one-shot LLM call drops entries on long CVs: the model is
     asked to emit 7 employers x nested clients in one JSON response and runs
     out of output budget / loses the tail.

This module fixes (2) with **per-employer-block chunking** so page count stops
mattering:

  Stage A  — ONE cheap call lists every employer (company/title/dates + the
             verbatim header line). Bounded output: just a list, no bullets.
  Python   — slices the experience section into per-employer text blocks at the
             header-line offsets Stage A reported. Deterministic.
  Stage B  — ONE focused call per block extracts that employer's responsibility
             bullets + nested clientsServed[]. Bounded input AND output.

A 2-page senior and a 20-page senior take the same path; nothing truncates,
nothing drops. The LLM only ever extracts entities; Python decides how many
calls and assembles the result — the Sakhi contract.

Reuses the recruiter pipeline's Ollama wrapper and the contact/education/skills
passes so model setup and timing logs match exactly.
"""
from __future__ import annotations

import re

from .extractor import _run_ollama as run_ollama
from .extractor import extract_contact, extract_entities


# Senior resumes overflow the 8192-token default; 16384 fits ~50K-char CVs
# safely (KV-cache cost is negligible vs the model weight). Per-block Stage B
# calls are far smaller — this ceiling is the safety net for Stage A.
SENIOR_NUM_CTX = 16384


# ── Stage A: employer listing ───────────────────────────────────────────────

EMPLOYERS_SYSTEM_PROMPT = (
    "You list the EMPLOYERS from a resume's work-experience section. An employer "
    "is an organization the candidate was directly employed by — it appears on a "
    "role/title line such as 'Senior Engineer - CompanyName, City' or "
    "'CompanyName | Role | 2019-2022'.\n\n"
    "CRITICAL: Companies named INSIDE bullet points are clients or projects, NOT "
    "employers. Do not list them here. If the same employer appears with two "
    "titles (an internal promotion), list it TWICE — one row per role.\n\n"
    "ONGOING ROLES: if a role has no end date — phrased 'till date', 'present', "
    "'current', or 'since <date>' with nothing after — set \"end\" to 'Present'. "
    "Never leave \"end\" empty for an ongoing role.\n\n"
    "Return JSON only. Do NOT invent employers."
)

EMPLOYERS_USER_TEMPLATE = """List every employer in this work-experience section.

Return JSON with EXACTLY this shape:

{{
  "employers": [
    {{
      "company": "Employer organization name",
      "title": "Job title held at that employer",
      "start": "verbatim start date string (e.g. 'Jul 2020', \\"Mar'24\\", '2018')",
      "end": "verbatim end date string OR 'Present'",
      "header_line": "the VERBATIM line from the resume that names this company/title/date — copied exactly, character for character"
    }}
  ]
}}

The "header_line" must be an exact copy of a line in the text below — it is used
to locate the entry. Empty list [] if there is no work experience.

WORK-EXPERIENCE SECTION:
------
{experience_text}
------

Return JSON only."""


# ── Stage B: per-employer detail ────────────────────────────────────────────

EMPLOYER_DETAIL_SYSTEM_PROMPT = (
    "You extract the details of ONE employment entry from a resume fragment. "
    "The fragment covers a single employer. Extract the responsibility bullets "
    "and any CLIENT engagements nested under this role.\n\n"
    "A clientsServed entry is a client company the candidate served WHILE "
    "employed by this employer (common on consultancy CVs). It is named inside "
    "the role's bullets — not a separate employer.\n\n"
    "Every value must be verbatim from the fragment. Do NOT invent. Return JSON only."
)

EMPLOYER_DETAIL_USER_TEMPLATE = """This fragment is one employment entry for:
  company: {company}
  title:   {title}

Return JSON with EXACTLY this shape:

{{
  "responsibilities": ["verbatim responsibility/achievement bullet", "..."],
  "clientsServed": [
    {{
      "client": "client company name",
      "role": "role/designation on that client engagement, null if absent",
      "start": "verbatim start date OR null",
      "end": "verbatim end date OR 'Present' OR null",
      "details": ["verbatim bullet about this client engagement", "..."],
      "evidence": "verbatim line naming this client"
    }}
  ],
  "evidence": "the verbatim header line of this employment entry"
}}

Empty list [] when a section is absent. clientsServed is usually [] — only
consultancy / staffing CVs have it.

EMPLOYMENT FRAGMENT:
------
{block_text}
------

Return JSON only."""


# ── Stage C: project / client engagement blocks ─────────────────────────────
# Consultancy CVs put per-role detail under a "Projects" heading as dated
# engagement blocks, separate from the one-line employer summaries. Stage C
# extracts those blocks; Python (senior_derive) maps each to its employer by
# company-name match + date-interval overlap, then nests it as clientsServed.

ENGAGEMENTS_SYSTEM_PROMPT = (
    "You list the PROJECT / CLIENT ENGAGEMENT blocks in a resume section. Each "
    "block describes one project or client engagement and begins with a header "
    "line that names the company/client and a date range.\n\n"
    "Return JSON only. Do NOT invent blocks."
)

ENGAGEMENTS_USER_TEMPLATE = """List every project / client engagement block in this section.

Return JSON with EXACTLY this shape:

{{
  "engagements": [
    {{
      "header_line": "the VERBATIM header line that names the company/client — copied exactly",
      "start": "verbatim start date string OR null",
      "end": "verbatim end date string OR 'Present' OR null"
    }}
  ]
}}

The "header_line" must be an exact copy of a line in the text below. Empty
list [] if there are no engagement blocks.

SECTION:
------
{projects_text}
------

Return JSON only."""

ENGAGEMENT_DETAIL_SYSTEM_PROMPT = (
    "You extract ONE project / client engagement from a resume fragment. Every "
    "value must be verbatim from the fragment. Do NOT invent. Return JSON only."
)

ENGAGEMENT_DETAIL_USER_TEMPLATE = """Extract this one project / client engagement.

Return JSON with EXACTLY this shape:

{{
  "companies": ["every company/organisation named in the header line — if the header is 'A/B' or 'A / B', return BOTH as separate items"],
  "location": "location string OR null",
  "role": "role/designation on this engagement OR null",
  "project": "project name OR null",
  "responsibilities": ["verbatim responsibility/achievement bullet", "..."],
  "evidence": "the verbatim header line of this engagement"
}}

Empty list [] when a part is absent.

ENGAGEMENT FRAGMENT:
------
{block_text}
------

Return JSON only."""


# ── Block slicing ───────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _find_offset(haystack: str, needle: str) -> int:
    """Locate a header line in the experience text. Tries exact, then
    whitespace-normalised, then a prefix of the needle. Returns -1 on miss."""
    if not needle:
        return -1
    idx = haystack.find(needle)
    if idx != -1:
        return idx
    # Whitespace-normalised search — PDF text often reflows the header line.
    h_norm = re.sub(r"\s+", " ", haystack)
    n_norm = re.sub(r"\s+", " ", needle).strip()
    if n_norm:
        j = h_norm.lower().find(n_norm.lower())
        if j != -1:
            # Map normalised offset back approximately by token count.
            return len(haystack) and _approx_back_map(haystack, h_norm, j)
    # Fall back to the first 25 chars of the needle.
    head = needle.strip()[:25]
    if len(head) >= 8:
        k = haystack.lower().find(head.lower())
        if k != -1:
            return k
    return -1


def _approx_back_map(orig: str, norm_text: str, norm_idx: int) -> int:
    """Map an offset in whitespace-collapsed text back to the original.
    Walks both strings in lockstep; good enough for block boundaries."""
    oi = ni = 0
    while ni < norm_idx and oi < len(orig):
        if orig[oi].isspace():
            # A run of original whitespace collapsed to one space in norm_text.
            while oi < len(orig) and orig[oi].isspace():
                oi += 1
            ni += 1
        else:
            oi += 1
            ni += 1
    return oi


def _slice_blocks(experience_text: str, employers: list[dict]) -> list[tuple[dict, str]]:
    """Slice the experience section into one text block per employer, cut at the
    header-line offsets. Employers whose header can't be located keep the whole
    section as their block (rare; Stage B still works, just less bounded)."""
    located: list[tuple[int, dict]] = []
    for emp in employers:
        off = _find_offset(experience_text, emp.get("header_line") or "")
        if off == -1:
            off = _find_offset(experience_text, emp.get("company") or "")
        located.append((off, emp))

    # Keep input order for unlocatable entries; sort the rest by offset.
    found = sorted([x for x in located if x[0] != -1], key=lambda x: x[0])
    missing = [x for x in located if x[0] == -1]

    blocks: list[tuple[dict, str]] = []
    for i, (off, emp) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else len(experience_text)
        blocks.append((emp, experience_text[off:end].strip()))
    for _off, emp in missing:
        blocks.append((emp, experience_text.strip()))
    return blocks


# ── Public ──────────────────────────────────────────────────────────────────

def list_employers(experience_text: str, model: str) -> list[dict]:
    """Stage A — one cheap call. Returns [{company,title,start,end,header_line}]."""
    if not experience_text.strip():
        return []
    user = EMPLOYERS_USER_TEMPLATE.format(experience_text=experience_text)
    parsed, elapsed = run_ollama(
        model, EMPLOYERS_SYSTEM_PROMPT, user, num_ctx=SENIOR_NUM_CTX, num_predict=2048
    )
    print(f"[senior_extractor] employer-list pass: {elapsed:.1f}s")
    if not parsed:
        return []
    employers = parsed.get("employers") or []
    return [e for e in employers if isinstance(e, dict) and e.get("company")]


def extract_employer_detail(emp: dict, block_text: str, model: str) -> dict:
    """Stage B — one focused call per employer block. Returns the employer
    entry enriched with responsibilities + clientsServed."""
    user = EMPLOYER_DETAIL_USER_TEMPLATE.format(
        company=emp.get("company") or "",
        title=emp.get("title") or "",
        block_text=block_text,
    )
    parsed, elapsed = run_ollama(
        model, EMPLOYER_DETAIL_SYSTEM_PROMPT, user,
        num_ctx=SENIOR_NUM_CTX, num_predict=3072,
    )
    print(f"[senior_extractor]   detail [{emp.get('company')}]: {elapsed:.1f}s")
    detail = parsed or {}
    return {
        "company": emp.get("company"),
        "title": emp.get("title"),
        "start": emp.get("start"),
        "end": emp.get("end"),
        "evidence": detail.get("evidence") or emp.get("header_line"),
        "responsibilities": detail.get("responsibilities") or [],
        "clientsServed": [
            c for c in (detail.get("clientsServed") or []) if isinstance(c, dict) and c.get("client")
        ],
    }


def extract_employment(experience_text: str, model: str) -> list[dict]:
    """Chunked employment extraction: list employers, slice the experience
    section into per-employer blocks, detail each block.

    Both calls are bounded: Stage A output is just a list, Stage B input is one
    sliced block and its output is one employer (num_predict-capped). A 2-page
    and a 20-page senior take the same path. When a resume lists its employers
    as a summary block with the per-role detail elsewhere, the sliced block is
    short and Stage B simply extracts what little is in it — it is NOT widened
    to the whole section (that fed the model an unbounded prompt and triggered
    a 30-minute runaway generation)."""
    employers = list_employers(experience_text, model)
    if not employers:
        return []
    blocks = _slice_blocks(experience_text, employers)
    print(f"[senior_extractor] {len(blocks)} employer block(s) to detail")
    return [extract_employer_detail(emp, block, model) for emp, block in blocks]


def list_engagements(projects_text: str, model: str) -> list[dict]:
    """Stage C1 — one cheap call. Returns [{header_line, start, end}]."""
    if not projects_text.strip():
        return []
    user = ENGAGEMENTS_USER_TEMPLATE.format(projects_text=projects_text)
    parsed, elapsed = run_ollama(
        model, ENGAGEMENTS_SYSTEM_PROMPT, user, num_ctx=SENIOR_NUM_CTX, num_predict=2048
    )
    print(f"[senior_extractor] engagement-list pass: {elapsed:.1f}s")
    if not parsed:
        return []
    engs = parsed.get("engagements") or []
    return [e for e in engs if isinstance(e, dict) and e.get("header_line")]


def extract_engagement_detail(eng: dict, block_text: str, model: str) -> dict:
    """Stage C2 — one focused call per engagement block."""
    user = ENGAGEMENT_DETAIL_USER_TEMPLATE.format(block_text=block_text)
    parsed, elapsed = run_ollama(
        model, ENGAGEMENT_DETAIL_SYSTEM_PROMPT, user,
        num_ctx=SENIOR_NUM_CTX, num_predict=3072,
    )
    print(f"[senior_extractor]   engagement [{(eng.get('header_line') or '')[:40]}]: {elapsed:.1f}s")
    d = parsed or {}
    return {
        "companies": [c for c in (d.get("companies") or []) if isinstance(c, str) and c.strip()],
        "start": eng.get("start"),
        "end": eng.get("end"),
        "location": d.get("location") or None,
        "role": d.get("role") or None,
        "project": d.get("project") or None,
        "responsibilities": [r for r in (d.get("responsibilities") or []) if isinstance(r, str) and r.strip()],
        "evidence": d.get("evidence") or eng.get("header_line"),
    }


def extract_engagements(projects_text: str, model: str) -> list[dict]:
    """Chunked engagement extraction over the projects section — same
    list-then-slice-then-detail pattern as employment. Output and per-call
    input are bounded; senior_derive maps each engagement to an employer."""
    engs = list_engagements(projects_text, model)
    if not engs:
        return []
    blocks = _slice_blocks(projects_text, engs)
    print(f"[senior_extractor] {len(blocks)} engagement block(s) to detail")
    return [extract_engagement_detail(eng, block, model) for eng, block in blocks]


def extract_all_senior(text: str, model: str, sections: dict[str, str] | None = None) -> dict:
    """Run the senior extraction passes and merge into one dict.

    Returns:
        {
          "contact": {name, phone, email},
          "employment_entries": [ {company, title, start, end, evidence,
                                    responsibilities, clientsServed[]} ],
          "engagements": [ {companies[], start, end, location, role, project,
                            responsibilities[], evidence} ],
          "education_entries": [...],
          "skills_raw": [...]
        }

    `engagements` are the projects-section client/project blocks; senior_derive
    maps each to its employer and nests it as clientsServed.
    """
    contact = extract_contact(text, model)

    experience_text = ""
    projects_text = ""
    if sections:
        experience_text = sections.get("experience") or ""
        projects_text = sections.get("projects") or ""
    if not experience_text.strip():
        # No detected experience section — fall back to the whole resume so a
        # senior with an unconventional heading still gets employment extracted.
        experience_text = text
    employment_entries = extract_employment(experience_text, model)

    # Consultancy / senior CVs scatter per-role detail and client engagements
    # under a "Projects" heading separate from the employer summaries. Extract
    # those blocks; senior_derive maps them onto employers.
    engagements = extract_engagements(projects_text, model)

    # Education + skills reuse the recruiter entities pass; its employment
    # output is discarded (chunked employment above is authoritative).
    entities = extract_entities(text, model, sections=sections)

    return {
        "contact": contact,
        "employment_entries": employment_entries,
        "engagements": engagements,
        "education_entries": entities.get("education_entries") or [],
        "skills_raw": entities.get("skills_raw") or [],
    }
