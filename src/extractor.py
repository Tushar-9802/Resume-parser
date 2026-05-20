"""
Entity extraction via Ollama, Sakhi-style.

The model's job is bounded: emit RAW ENTITIES with verbatim evidence quotes.
It does NOT decide current-vs-previous, dedup, ordering, or any business
rule. All of those become deterministic Python in src/derive.py.

Two LLM calls per resume:
  1. CONTACT — name, phone, email (from first 30 lines of the resume).
  2. ENTITIES — employment_entries, education_entries, skills_raw.

Both use format="json", temperature=0.1 — same configuration that landed
empirically for the Sakhi project.

The "evidence" field on each entity is the verbatim text fragment from the
resume that establishes it. Downstream we use evidence to (a) verify the
entity exists in source text and (b) audit which line produced which row.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import yaml


SCHEMA_PATH = Path(__file__).parent.parent / "configs" / "schema.yaml"
_CACHED_SCHEMA: dict | None = None


def load_schema() -> dict:
    global _CACHED_SCHEMA
    if _CACHED_SCHEMA is None:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            _CACHED_SCHEMA = yaml.safe_load(f)
    return _CACHED_SCHEMA


# ── Prompts ─────────────────────────────────────────────────────────────────

CONTACT_SYSTEM_PROMPT = (
    "You extract contact information from the top of a resume. Use null when "
    "the field is absent. Do NOT invent names, phone numbers, or email addresses. "
    "Return JSON only."
)

CONTACT_USER_TEMPLATE = (
    "Extract the candidate's contact info from this resume header.\n\n"
    "Return JSON with EXACTLY these keys (null when absent):\n"
    "{{\n"
    '  "name": "candidate full name, as written at the top",\n'
    '  "phone": "primary phone number, preserve formatting (+91-9876543210)",\n'
    '  "email": "primary email address"\n'
    "}}\n\n"
    "RESUME HEADER:\n"
    "------\n"
    "{header}\n"
    "------\n\n"
    "Return JSON only."
)


ENTITIES_SYSTEM_PROMPT = (
    "You extract employment, education, and skills from a resume into structured "
    "JSON. Use null/empty list for absent fields. Do NOT invent data.\n\n"
    "RULES:\n"
    "1. Each EMPLOYMENT entry is a job/internship/role. The employer is the "
    "ORGANIZATION YOU WORKED FOR — usually on a job-title line like "
    "'Senior Engineer – CompanyName, City' or 'CompanyName | Role'. Companies "
    "mentioned INSIDE bullet points are clients/projects, NOT employers — do "
    "NOT add them as separate employment entries.\n"
    "2. If the same employer appears across multiple roles (e.g., promoted "
    "internally), emit ONE entry PER ROLE — same company, different titles, "
    "different date ranges. Do not merge them.\n"
    "3. Each EDUCATION entry is a single degree/qualification. Include any "
    "school-level entries you see (10th, 12th, CBSE, etc.) — downstream code "
    "will filter them. Just extract what's there.\n"
    "4. For dates: copy them VERBATIM from the resume as 'start' and 'end' "
    "strings (e.g., 'Jul 2025', 'March 2024', 'Aug 2018', \"Mar'24\", '2018'). "
    "If the role is ongoing, end = 'Present'. Do NOT compute years yourself.\n"
    "5. The 'evidence' field on each entry MUST be a verbatim quote from the "
    "resume — typically the title/company/date line. This is how downstream "
    "validation confirms the entry exists in the source.\n"
    "6. skills_raw is a flat list of skill strings from the SKILLS section. "
    "Do NOT include section labels like 'Technical Skills:' or 'Languages:'. "
    "Do NOT include sentences or descriptions — just the skill names.\n"
    "7. Return JSON only. No commentary, no markdown fences."
)


ENTITIES_USER_TEMPLATE = """Extract structured entities from this resume.

Return JSON with EXACTLY this shape:

{{
  "employment_entries": [
    {{
      "company": "Employer name",
      "title": "Job title at that employer",
      "start": "verbatim start date string",
      "end": "verbatim end date string OR 'Present'",
      "evidence": "verbatim line(s) from the resume proving this entry"
    }}
  ],
  "education_entries": [
    {{
      "institution": "School/college/university name",
      "degree": "Degree or qualification name",
      "end_year": 2024,
      "evidence": "verbatim line(s) from the resume proving this entry"
    }}
  ],
  "skills_raw": ["skill 1", "skill 2", ...]
}}

Empty list [] when absent. null for unknown end_year.

{section_blocks}RESUME (full text for context):
------
{full_text}
------

Return JSON only."""


def _build_section_blocks(sections: dict[str, str] | None) -> str:
    """If we have detected sections, surface the relevant ones explicitly.
    The model uses these as the authoritative source; the full text is just
    for context. This is the same pattern Sakhi uses to focus the prompt."""
    if not sections:
        return ""

    parts: list[str] = []
    for label, key in [
        ("EXPERIENCE SECTION (authoritative for employment_entries)", "experience"),
        ("EDUCATION SECTION (authoritative for education_entries)", "education"),
        ("SKILLS SECTION (authoritative for skills_raw)", "skills"),
    ]:
        body = sections.get(key)
        if body:
            parts.append(f"=== {label} ===\n{body}\n")
    if not parts:
        return ""
    return "\n".join(parts) + "\n"


# ── Ollama call ─────────────────────────────────────────────────────────────

def _run_ollama(model: str, system: str, user: str, *, num_ctx: int = 8192,
                num_predict: int = -1) -> tuple[dict | None, float]:
    """One Ollama call with format=json. Returns (parsed_dict, elapsed_seconds).

    num_predict caps output tokens (-1 = unlimited). A cap is a cheap safety
    net: a confused prompt can send a small model into an unbounded JSON
    generation loop (observed: a 292K-char unterminated response that ran for
    ~30 min). Bounded-output passes should pass an explicit cap."""
    import ollama

    t0 = time.time()
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format="json",
        options={"temperature": 0.1, "num_ctx": num_ctx, "num_gpu": 999,
                 "num_predict": num_predict},
        keep_alive="10m",
    )
    elapsed = time.time() - t0
    raw = resp.message.content
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[extractor] JSON parse failed despite format=json: {e}")
        print(f"[extractor] raw (first 400 chars): {raw[:400]}")
        return None, elapsed
    return parsed, elapsed


# ── Public ──────────────────────────────────────────────────────────────────

def extract_contact(text: str, model: str) -> dict:
    """Pull name/phone/email from the resume header. Trim to first 30 lines —
    contact info lives at the top, and a tighter context window prevents the
    model from grabbing emails out of project URLs deeper in the resume."""
    header = "\n".join(text.splitlines()[:30])
    user = CONTACT_USER_TEMPLATE.format(header=header)
    parsed, elapsed = _run_ollama(model, CONTACT_SYSTEM_PROMPT, user, num_ctx=2048)
    print(f"[extractor] contact pass: {elapsed:.1f}s")
    if not parsed:
        return {"name": None, "phone": None, "email": None}
    return {
        "name": parsed.get("name"),
        "phone": parsed.get("phone"),
        "email": parsed.get("email"),
    }


def extract_entities(text: str, model: str, sections: dict[str, str] | None = None) -> dict:
    """Pull employment_entries / education_entries / skills_raw from the resume.

    Returns a dict shaped exactly like the JSON the model emits:
        {employment_entries: [...], education_entries: [...], skills_raw: [...]}

    Empty lists when nothing was found (never null at the top level — keeps
    downstream code simple)."""
    section_blocks = _build_section_blocks(sections)
    user = ENTITIES_USER_TEMPLATE.format(section_blocks=section_blocks, full_text=text)
    # 8192 silently truncates the tail of long multi-page CVs (Ollama drops
    # overflow tokens) — the model then returns empty arrays. 16384 fits
    # ~50K-char resumes; KV-cache cost is negligible vs the model weight.
    parsed, elapsed = _run_ollama(model, ENTITIES_SYSTEM_PROMPT, user, num_ctx=16384)
    print(f"[extractor] entities pass: {elapsed:.1f}s")
    if not parsed:
        return {"employment_entries": [], "education_entries": [], "skills_raw": []}
    return {
        "employment_entries": parsed.get("employment_entries") or [],
        "education_entries": parsed.get("education_entries") or [],
        "skills_raw": parsed.get("skills_raw") or [],
    }


def extract_all(text: str, model: str, sections: dict[str, str] | None = None) -> dict:
    """Run both passes and merge into a single dict.

    Returns:
        {
          "contact": {name, phone, email},
          "employment_entries": [...],
          "education_entries": [...],
          "skills_raw": [...]
        }
    """
    contact = extract_contact(text, model)
    entities = extract_entities(text, model, sections=sections)
    return {"contact": contact, **entities}
