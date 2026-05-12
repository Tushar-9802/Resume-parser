"""
Entity extraction for the fresher / student pipeline (Sakhi-style).

Two LLM passes per resume:
  1. IDENTITY  — name, phone, social/portfolio URL labels, DOB, city, state,
                 target roles, preferred locations. Read from the resume header
                 + summary block.
  2. ENTITIES  — education, internships (with verbatim achievement bullets),
                 projects (title + tech-stack + losslessly-condensed description +
                 verbatim evidence), skills, tools, certifications, languages.

The Sakhi contract: the LLM emits entities with verbatim `evidence` quotes.
Business rules (degree/branch enum mapping, skills-vs-tools split, link URL
attachment from PDF annotations, lossless-condensation verification) all live
in student_derive.py — never in this prompt.

Reuses the existing Ollama call wrapper from src.extractor so the model setup
and timing logs match the recruiter pipeline exactly.
"""
from __future__ import annotations

import json
import time


# Reuse the Ollama wrapper from the recruiter pipeline — same model setup,
# same timing log format. Pulled across with a clean alias so this file
# doesn't carry a private-name import.
from .extractor import _run_ollama as run_ollama


# ── Identity pass ───────────────────────────────────────────────────────────

IDENTITY_SYSTEM_PROMPT = (
    "You extract identity, contact, and career-preference fields from the top "
    "of a resume. Use null when the field is absent. Do NOT invent values. "
    "Return JSON only."
)

IDENTITY_USER_TEMPLATE = """Extract the candidate's identity and career-preference fields from the resume header.

Return JSON with EXACTLY these keys (null when absent):
{{
  "name": "candidate full name, as written at the top",
  "phone": "primary phone number, preserve formatting (+91-9876543210)",
  "email": "primary email address",
  "linkedin_label": "the literal text shown for the LinkedIn link, e.g. 'LinkedIn' or 'linkedin.com/in/foo'. The real URL is recovered from PDF annotations downstream — just give the label as it appears.",
  "portfolio_label": "the literal text shown for any portfolio/personal-site link, e.g. 'Portfolio', 'tushar.dev', 'GitHub'. Same recovery rule.",
  "date_of_birth": "DOB if printed on the resume. Accept any format the candidate used verbatim: '15 Aug 2002', '25/12/1992', '2002-08-15', 'D.O.B: 25/12/1992'. Strip only the 'DOB:' / 'D.O.B :' label prefix. Null if absent.",
  "city": "current city if stated, e.g. 'Ghaziabad'",
  "state": "sub-national state/province only, e.g. 'Uttar Pradesh' or 'UP' or 'Karnataka'. NOT a country name like 'India' or 'USA' — if the only location label is a country, return null for state.",
  "target_roles": "what role(s) the candidate is targeting (from a Career Objective / Summary line like 'Product Management Intern' or 'ML Engineer'). One short string.",
  "preferred_locations": ["list of locations the candidate prefers to work in, if stated; usually absent"]
}}

RESUME HEADER + SUMMARY:
------
{header}
------

Return JSON only.
"""


def extract_identity(text: str, model: str) -> dict:
    """First-pass extraction: identity + contact + career preferences.

    Reads the first ~50 lines (header + summary block). Freshers often have a
    longer top — summary, objective, and contact pipes — so this is bigger
    than the recruiter pipeline's 30-line cap."""
    header = "\n".join(text.splitlines()[:50])
    user = IDENTITY_USER_TEMPLATE.format(header=header)
    parsed, elapsed = run_ollama(model, IDENTITY_SYSTEM_PROMPT, user, num_ctx=2048)
    print(f"[student_extractor] identity pass: {elapsed:.1f}s")
    if not parsed:
        return _empty_identity()
    return {
        "name": parsed.get("name"),
        "phone": parsed.get("phone"),
        "email": parsed.get("email"),
        "linkedin_label": parsed.get("linkedin_label"),
        "portfolio_label": parsed.get("portfolio_label"),
        "date_of_birth": parsed.get("date_of_birth"),
        "city": parsed.get("city"),
        "state": parsed.get("state"),
        "target_roles": parsed.get("target_roles"),
        "preferred_locations": parsed.get("preferred_locations") or [],
    }


def _empty_identity() -> dict:
    return {
        "name": None, "phone": None, "email": None,
        "linkedin_label": None, "portfolio_label": None,
        "date_of_birth": None, "city": None, "state": None,
        "target_roles": None, "preferred_locations": [],
    }


# ── Entities pass ───────────────────────────────────────────────────────────

ENTITIES_SYSTEM_PROMPT = (
    "You extract structured entities from a fresher / student resume. The "
    "candidate is typically in college or recently graduated. Use null / empty "
    "list for absent fields. Do NOT invent data.\n\n"
    "OUTPUT RULES:\n"
    "1. EDUCATION — one entry per degree. Keep school-level entries (10th/12th) "
    "if present; downstream code filters them. If the degree includes a "
    "specialization tail (e.g. 'B.Tech in CSE - Data Science' or 'B.Tech "
    "Computer Science (AIML)'), populate 'specialization' with just the "
    "specialization name. Otherwise leave it null. "
    "IMPORTANT: if the resume has NO formal EDUCATION section, look for "
    "degree info in the resume HEADER / SUBTITLE area. Lines like 'Fourth "
    "Year Undergraduate, Mechanical Engineering' or 'Senior, Computer "
    "Science Major' or '2nd Year Postgraduate, ME' are degree statements — "
    "extract them into education_entries with the institution name nearby. "
    "Do NOT leave education_entries empty just because there's no 'EDUCATION' "
    "heading.\n"
    "2. INTERNSHIPS — every internship / industry role / part-time job. The "
    "'company' is the ORGANIZATION the candidate worked for (usually on a "
    "title line like 'Product Intern - Bosscoder Academy'). Bullet points are "
    "achievement strings — keep them verbatim, do not paraphrase, do not "
    "summarize. The downstream pipeline concatenates them into the final "
    "internshipDetails text.\n"
    "3. PROJECTS — every project the candidate worked on. CRITICAL for "
    "freshers (this is most of what they have to show). For each project:\n"
    "   - 'title': the project name as written.\n"
    "   - 'tech_stack_text': verbatim 'Tech: ...' line if present, else null.\n"
    "   - 'description_condensed': one tight paragraph (<=500 chars) that "
    "LOSSLESSLY preserves every metric (percentages, sample counts, model "
    "names, scale numbers), every named technology, every scope statement, "
    "and every distinct contribution. Drop only verbose framing words. Do not "
    "merge two distinct claims into one. If you cannot fit losslessly in 500 "
    "chars, prefer length over loss.\n"
    "   - 'skills_used': 3-8 skills the candidate used in THIS project. Pick "
    "from skills/tools that are either named in the project bullets, or "
    "reasonable inferences from the project type (e.g., a 'PRD authoring' "
    "bullet implies PRDs as a skill; a 'Pandas funnel tracker' implies "
    "Pandas + Python + Data Analysis). For technical projects with a 'Tech:' "
    "line, copy items from that line. For non-technical projects (PM, "
    "research, design), pull from the candidate's overall skills list. Do "
    "NOT invent technologies the candidate doesn't mention anywhere.\n"
    "   - 'evidence': verbatim bullets from the resume that produced the "
    "description. The downstream verifier compares evidence against "
    "description_condensed to catch dropped facts.\n"
    "   - 'start' / 'end': date strings if printed alongside the project "
    "(e.g., 'Nov 2025', 'Present'), else null.\n"
    "4. SKILLS vs TOOLS — extract ONLY from the SKILLS / TECHNICAL SKILLS / "
    "CORE COMPETENCIES section. Do NOT pull skills from project bodies, "
    "internship bullets, or the summary — those project-specific skills are "
    "handled separately in each project's 'skills_used' field. The top-level "
    "skills/tools lists describe the candidate's general capability set.\n"
    "   Split by KIND, not by resume sub-label:\n"
    "   - skills_raw: methodologies, concepts, programming languages, "
    "frameworks, libraries (anything you can learn or practice).\n"
    "   - tools_raw: specific branded software products / SaaS / IDEs / "
    "platforms (anything you log into or install as a product).\n"
    "   CRITICAL: do NOT add tools just because they are typical for the "
    "role. Only emit a tool if its NAME appears in the resume's skills "
    "section. If the skills section lists no tools, tools_raw is [].\n"
    "   When in doubt between skill vs tool, prefer skills_raw.\n"
    "5. CERTIFICATIONS — list of certification strings. Include the issuer if "
    "stated (e.g., 'Salesforce Certified AI Associate', 'Product Management - "
    "GeeksforGeeks'). Do NOT include dates inline; put them in 'evidence'.\n"
    "6. LANGUAGES_KNOWN — spoken/written languages (English, Hindi, etc.). "
    "Not programming languages.\n"
    "7. Return JSON only. No commentary, no markdown fences.\n"
)


ENTITIES_USER_TEMPLATE = """Extract structured entities from this fresher / student resume.

Return JSON with EXACTLY this shape:

{{
  "education_entries": [
    {{
      "institution": "School/college/university name",
      "raw_degree": "Full degree string as written (e.g. 'B.Tech in Computer Science and Engineering')",
      "specialization": "Sub-branch specialization if stated, e.g. 'Data Science' for 'B.Tech in CSE - Data Science', or 'AIML' for 'B.Tech CSE (AIML)', or 'VLSI Design' for 'M.Tech ECE - VLSI'. Null if the degree has no specialization tail.",
      "cgpa_text": "CGPA / aggregate / percentage as written (e.g. '8.5/10', 'Aggregate: 71%'), or null",
      "graduation_year_text": "verbatim year-or-date-range as written (e.g. '2024', 'Nov 2022 - Present')",
      "evidence": "verbatim line(s) from the resume proving this entry"
    }}
  ],
  "internship_entries": [
    {{
      "company": "Employer / organization name",
      "role": "Job title (e.g. 'Product Intern', 'Data Science Intern')",
      "start": "verbatim start date",
      "end": "verbatim end date OR 'Present'",
      "location": "city if stated, else null",
      "achievements": [
        "verbatim bullet point 1",
        "verbatim bullet point 2"
      ],
      "evidence": "the title/company/date line"
    }}
  ],
  "project_entries": [
    {{
      "title": "Project name",
      "tech_stack_text": "verbatim 'Tech: ...' line if present, else null",
      "start": "date string if printed for this project, else null",
      "end": "date string if printed for this project, else null",
      "description_condensed": "one losslessly-condensed paragraph (see system rules)",
      "skills_used": ["skill or tool used in this specific project", "..."],
      "evidence": "verbatim bullet(s) from the resume for this project"
    }}
  ],
  "skills_raw": ["skill 1", "skill 2"],
  "tools_raw": ["tool 1", "tool 2"],
  "certifications": ["cert string 1", "cert string 2"],
  "languages_known": ["English", "Hindi"]
}}

Empty list [] when absent.

{section_blocks}RESUME (full text for context):
------
{full_text}
------

Return JSON only.
"""


def _build_section_blocks(sections: dict | None) -> str:
    """Surface relevant resume sections explicitly when the section splitter
    identified them. Mirrors src/extractor._build_section_blocks but with the
    fresher-relevant sections (projects, certifications, languages)."""
    if not sections:
        return ""

    parts: list[str] = []
    for label, key in [
        ("EDUCATION SECTION (authoritative for education_entries)", "education"),
        ("EXPERIENCE / INTERNSHIPS SECTION (authoritative for internship_entries)", "experience"),
        ("PROJECTS SECTION (authoritative for project_entries)", "projects"),
        ("SKILLS SECTION (authoritative for skills_raw + tools_raw)", "skills"),
        ("CERTIFICATIONS SECTION", "certifications"),
        ("LANGUAGES SECTION", "languages"),
    ]:
        body = sections.get(key)
        if body:
            parts.append(f"=== {label} ===\n{body}\n")
    if not parts:
        return ""
    return "\n".join(parts) + "\n"


def extract_entities(text: str, model: str, sections: dict | None = None) -> dict:
    """Second-pass extraction: everything except identity.

    Returns the entity dict shaped exactly like the prompt's JSON spec. Empty
    lists when nothing was found (never null at the top level — keeps the
    derive layer simple)."""
    section_blocks = _build_section_blocks(sections)
    user = ENTITIES_USER_TEMPLATE.format(section_blocks=section_blocks, full_text=text)
    parsed, elapsed = run_ollama(model, ENTITIES_SYSTEM_PROMPT, user, num_ctx=8192)
    print(f"[student_extractor] entities pass: {elapsed:.1f}s")
    if not parsed:
        return _empty_entities()
    return {
        "education_entries": parsed.get("education_entries") or [],
        "internship_entries": parsed.get("internship_entries") or [],
        "project_entries": parsed.get("project_entries") or [],
        "skills_raw": parsed.get("skills_raw") or [],
        "tools_raw": parsed.get("tools_raw") or [],
        "certifications": parsed.get("certifications") or [],
        "languages_known": parsed.get("languages_known") or [],
    }


def _empty_entities() -> dict:
    return {
        "education_entries": [],
        "internship_entries": [],
        "project_entries": [],
        "skills_raw": [],
        "tools_raw": [],
        "certifications": [],
        "languages_known": [],
    }


# ── Combined entrypoint ─────────────────────────────────────────────────────

def extract_all_student(text: str, model: str, sections: dict | None = None) -> dict:
    """Run both passes and merge into a single dict.

    Returns:
        {
          "identity": {name, phone, email, linkedin_label, portfolio_label,
                       date_of_birth, city, state, target_roles,
                       preferred_locations},
          "education_entries": [...],
          "internship_entries": [...],
          "project_entries": [...],
          "skills_raw": [...],
          "tools_raw": [...],
          "certifications": [...],
          "languages_known": [...],
        }
    """
    identity = extract_identity(text, model)
    entities = extract_entities(text, model, sections=sections)
    return {"identity": identity, **entities}
