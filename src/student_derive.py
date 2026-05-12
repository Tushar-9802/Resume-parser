"""
Deterministic derivation of the Pulse student record from extracted entities.

Sakhi pattern: the LLM emits entities (with verbatim evidence). Python decides
every business rule.

Decisions made here:
  * Degree enum mapping  (B.Tech in CSE  ->  degree=btech, branch=cse)
  * CGPA + scale parsing (8.5/10 vs 71%)
  * Graduation-year parsing (single year vs date range with 'Present')
  * Skills vs toolsUsed split correction via a known-tool allowlist
  * Per-project tech-stack expansion -> skillsUsed list
  * Internship flattening to prose (per-internship blocks)
  * Lossless-condensation check: every metric in evidence must appear in
    description_condensed (warn, don't drop, on miss)
  * Per-project date parsing -> startMonth/Year, endMonth/Year, isInProgress

Out of scope for v1:
  * Hyperlink URL resolution from PDF annotations (linkedinUrl / portfolioUrl /
    per-project link arrays stay empty). Polish-pass once core is proven.
"""
from __future__ import annotations

import re
from datetime import date

# Reuse the date parser AND the evidence-grounding helpers from the recruiter
# pipeline — same Mar'24 / Jul 2025 / Present-token handling; same 60%
# token-overlap threshold for verbatim grounding.
from .derive import (
    parse_date,
    TODAY_YEAR,
    _norm_for_grounding,
    _evidence_grounded,
)
# Phone normalization (+91-XXXXXXXXXX) — same regex set as the recruiter pipeline.
from .validation import normalize_phone


# ── Degree enum mapping ─────────────────────────────────────────────────────

# Order matters: longer/more-specific patterns first so "M.Tech" doesn't match
# the "M" in "MBA". Each entry is (compiled regex, enum value).
_DEGREE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:m\.?\s*tech|master\s+of\s+technology)\b", re.IGNORECASE), "mtech"),
    (re.compile(r"\b(?:b\.?\s*tech|bachelor\s+of\s+technology)\b", re.IGNORECASE), "btech"),
    (re.compile(r"\b(?:m\.?\s*c\.?\s*a)\b", re.IGNORECASE), "mca"),
    (re.compile(r"\b(?:m\.?\s*b\.?\s*a|master\s+of\s+business)\b", re.IGNORECASE), "mba"),
    (re.compile(r"\b(?:m\.?\s*sc|master\s+of\s+science)\b", re.IGNORECASE), "other"),
    (re.compile(r"\b(?:b\.?\s*sc|bachelor\s+of\s+science)\b", re.IGNORECASE), "bsc"),
    (re.compile(r"\b(?:b\.?\s*e\.?|bachelor\s+of\s+engineering)\b", re.IGNORECASE), "be"),
    (re.compile(r"\bm\.?\s*e\.?\b", re.IGNORECASE), "other"),
    # Post-Graduate Diplomas — PGDM, PGDFM (forestry mgmt), PGDBA (business
    # admin), PGDIM (industrial mgmt), PGP, etc. Pulse enum has no separate
    # PGD bucket; map to "other" so the candidate still gets a non-null
    # degree rather than being silently dropped.
    (re.compile(r"\bpgd[a-z]{0,3}\b|\bpgp\b", re.IGNORECASE), "other"),
    (re.compile(r"\bdiploma\b|\bpolytechnic\b", re.IGNORECASE), "diploma"),
    # Informal Indian academic phrasings — common in IIT/NIT student resumes
    # that lack a formal Education section. "Fourth Year Undergraduate,
    # Mechanical Engineering" -> btech (4-yr UG engg in India == B.Tech).
    # "Second Year Postgraduate" alone is ambiguous (ME vs MBA), so we only
    # fall back to btech for the undergrad phrasings here.
    (re.compile(
        r"\b(?:final|fourth|third|fifth|first|second|1st|2nd|3rd|4th|5th)?\s*year\s+(?:undergraduate|undergrad|u\.?\s*g\.?)\b",
        re.IGNORECASE,
    ), "btech"),
]


def map_degree(raw_degree: str | None) -> str | None:
    """Return one of: be | btech | mtech | mba | mca | bsc | diploma | other.

    Returns None if the input is empty / unrecognized as a degree at all. Use
    `other` only when we recognized it as a degree but it doesn't fit the enum
    (M.Sc, M.E., Ph.D., LL.B, etc.)."""
    if not raw_degree:
        return None
    for pat, enum_val in _DEGREE_PATTERNS:
        if pat.search(raw_degree):
            return enum_val
    # If we got here we couldn't recognize a degree at all -> downstream will
    # treat as missing rather than 'other' so the parsing failure is visible.
    return None


# ── Branch enum mapping ─────────────────────────────────────────────────────

# Order matters again. Compound disciplines (ECE) need to win over their
# component words. "AI/ML" must come before any keyword that contains "ai".
_BRANCH_PATTERNS: list[tuple[re.Pattern, str]] = [
    # AI/ML/Data Science cluster — single enum
    (re.compile(r"\b(?:artificial\s+intelligence|machine\s+learning|a\.?\s*i\.?\s*/?\s*m\.?\s*l\.?|data\s+science|ai\s*&\s*ml|ai-ml|aiml)\b", re.IGNORECASE), "ai_ml"),
    # Cybersecurity
    (re.compile(r"\b(?:cyber\s*security|cybersecurity|cyber\s+sec|information\s+security|infosec)\b", re.IGNORECASE), "cybersecurity"),
    # ECE — accept many spellings before "electrical" can grab the leading 'E'
    (re.compile(r"\b(?:electronics?\s+(?:and|&)\s+communication|e\.?\s*c\.?\s*e|electronics?\s+communication|e\s*&\s*c)\b", re.IGNORECASE), "ece"),
    # CSE
    (re.compile(r"\b(?:computer\s+science(?:\s+and\s+engineering)?|c\.?\s*s\.?\s*e|comp\.?\s*sci|computer\s+engineering)\b", re.IGNORECASE), "cse"),
    # Mechatronics — before mechanical so "mecha-" prefix wins
    (re.compile(r"\bmechatronic", re.IGNORECASE), "mechatronics"),
    # Mechanical
    (re.compile(r"\b(?:mechanical(?:\s+engineering)?|mech\.?)\b", re.IGNORECASE), "mechanical"),
    # Civil
    (re.compile(r"\bcivil(?:\s+engineering)?\b", re.IGNORECASE), "civil"),
    # Aerospace
    (re.compile(r"\b(?:aerospace|aeronautical)\b", re.IGNORECASE), "aerospace"),
    # Automotive
    (re.compile(r"\b(?:automotive|automobile)\b", re.IGNORECASE), "automotive"),
    # Instrumentation
    (re.compile(r"\binstrumentation\b", re.IGNORECASE), "instrumentation"),
    # Chemical
    (re.compile(r"\bchemical(?:\s+engineering)?\b", re.IGNORECASE), "chemical"),
    # Electrical — last among engineering disciplines so ECE wins above
    (re.compile(r"\b(?:electrical(?:\s+engineering)?|e\.?\s*e\.?)\b", re.IGNORECASE), "electrical"),
]


# Splits 'CSE - Data Science' / 'CSE (AIML)' / 'CSE with focus in Cybersecurity'
# so that map_branch can prefer the BEFORE-separator part (the department).
# Without this, 'CSE - Data Science' was returning branch='ai_ml' because the
# Data-Science specialization keyword beat the CSE department keyword.
_BRANCH_SPLIT_SEPARATOR = re.compile(
    r"\s*[-–—(]\s*|\s+(?:with|specialisation|specialization|focus|major)\s+in\s+",
    re.IGNORECASE,
)


def map_branch(raw_degree: str | None) -> str | None:
    """Return one of the branch enum values, or None if no branch recognized.

    The branch is encoded inside the raw_degree string in most resumes
    (e.g. 'B.Tech in Computer Science and Engineering').

    For compound degrees like 'B.Tech in CSE - Data Science' the branch is the
    DEPARTMENT (CSE), not the specialization (Data Science). We split on the
    first specialization separator and match the BEFORE part first; only fall
    back to the full string when no department keyword matches there
    (e.g. a pure 'B.Tech in Data Science' degree).

    Returns None instead of guessing — leaving it null is more honest than a
    wrong default."""
    if not raw_degree:
        return None
    parts = _BRANCH_SPLIT_SEPARATOR.split(raw_degree, maxsplit=1)
    primary = parts[0] if parts else raw_degree
    for pat, enum_val in _BRANCH_PATTERNS:
        if pat.search(primary):
            return enum_val
    # No department keyword in the BEFORE-part; the degree is likely a pure
    # specialization name (e.g. 'B.Tech in Data Science'). Scan the full
    # string so ai_ml / cybersecurity / aerospace can match.
    for pat, enum_val in _BRANCH_PATTERNS:
        if pat.search(raw_degree):
            return enum_val
    return None


# ── Branch specialization ───────────────────────────────────────────────────

# Patterns that introduce a specialization tail on a degree string.
# Match the SEPARATOR — what follows up to a sane stop char is the spec.
# Examples handled:
#   "B.Tech in CSE - Data Science"            → 'Data Science'
#   "B.Tech, CSE (AIML)"                      → 'AIML'
#   "B.Tech in Computer Science (Data Science)" → 'Data Science'
#   "B.Tech in CSE with specialization in AI" → 'AI'
#   "M.Tech, ECE - VLSI Design"               → 'VLSI Design'
_SPEC_SEPARATOR_PATTERNS = [
    re.compile(r"[-–—]\s*([\w][\w\s&/+.-]*?)(?:\s*[,()]|$)", re.IGNORECASE),
    re.compile(r"\(([^()]+?)\)", re.IGNORECASE),
    re.compile(r"\b(?:specialisation|specialization)\s+in\s+([\w][\w\s&/+.-]*?)(?:\s*[,()]|$)", re.IGNORECASE),
    re.compile(r"\bwith\s+(?:specialisation|specialization|focus|major)\s+(?:in\s+)?([\w][\w\s&/+.-]*?)(?:\s*[,()]|$)", re.IGNORECASE),
]

# Words that look like a specialization match but are actually the
# branch/degree itself or generic filler. Drop these.
_SPEC_NOISE = {
    "engineering", "engg", "engr",
    "computer science", "computer science and engineering",
    "mechanical engineering", "electrical engineering",
    "civil engineering", "chemical engineering",
    "honours", "honors", "hons",
    "regular", "full time", "part time", "distance",
}


def _extract_specialization(llm_field: str | None, raw_degree: str | None, branch_enum: str | None) -> str | None:
    """Return the sub-branch specialization (e.g. 'Data Science' for a CSE
    candidate, 'VLSI' for ECE), or None when there isn't one.

    Strategy:
      1. Trust the LLM's 'specialization' field if it gave one (and it's not
         just the branch name repeated).
      2. Else apply regex patterns to raw_degree to find a tail after a
         separator (dash, parens, 'specialization in', 'with focus in')."""
    if llm_field and isinstance(llm_field, str):
        cleaned = llm_field.strip(" -,()").strip()
        if cleaned and cleaned.lower() not in _SPEC_NOISE:
            # Reject if it's just the branch name echoed back (e.g. LLM said
            # specialization='Computer Science' for a CSE degree)
            if branch_enum and _norm_skill(cleaned) == branch_enum:
                pass
            else:
                return cleaned

    if not raw_degree:
        return None

    for pat in _SPEC_SEPARATOR_PATTERNS:
        for m in pat.finditer(raw_degree):
            candidate = m.group(1).strip(" -,()").strip()
            if not candidate:
                continue
            if candidate.lower() in _SPEC_NOISE:
                continue
            # Reject branch-name echoes
            if branch_enum and _norm_skill(candidate) == branch_enum:
                continue
            # Reject pure year ranges like '2022-2026'
            if re.fullmatch(r"\d{4}\s*[-–—]?\s*\d{0,4}", candidate):
                continue
            return candidate
    return None


# ── CGPA parsing ────────────────────────────────────────────────────────────

# Match the common shapes:
#   "8.5/10", "8.5 / 10", "CGPA: 8.5", "CGPA 8.5/10"
#   "71%", "Aggregate: 71%", "71.5 %"
_CGPA_OVER_10 = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*10\b")
_CGPA_BARE = re.compile(r"(?:cgpa|gpa|sgpa|aggregate)\s*[:\-]?\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE)
_PERCENTAGE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def parse_cgpa(cgpa_text: str | None) -> tuple[float | None, str | None]:
    """Return (numeric_value, scale).

    scale is one of: '10' (CGPA on a 10-point scale), 'percentage', or None.
    Value is the numeric reading or None if unparseable.
    """
    if not cgpa_text:
        return (None, None)
    s = cgpa_text.strip()

    # Explicit /10 first — wins over % even if % also appears
    m = _CGPA_OVER_10.search(s)
    if m:
        try:
            return (float(m.group(1)), "10")
        except ValueError:
            pass

    # Percentage
    m = _PERCENTAGE.search(s)
    if m:
        try:
            v = float(m.group(1))
            if 0 <= v <= 100:
                return (v, "percentage")
        except ValueError:
            pass

    # Bare "CGPA: 8.5" — assume /10 if value <= 10
    m = _CGPA_BARE.search(s)
    if m:
        try:
            v = float(m.group(1))
            if 0 <= v <= 10:
                return (v, "10")
            if 0 <= v <= 100:
                return (v, "percentage")
        except ValueError:
            pass

    # Just a number on its own
    m = re.search(r"\b(\d+(?:\.\d+)?)\b", s)
    if m:
        try:
            v = float(m.group(1))
            if 0 <= v <= 10:
                return (v, "10")
            if 0 <= v <= 100:
                return (v, "percentage")
        except ValueError:
            pass

    return (None, None)


# ── Graduation year ─────────────────────────────────────────────────────────

# Default program durations (years from start to graduation) — used when the
# resume has a 'Present' end date and we need to project the graduation year.
_DEFAULT_DURATIONS = {
    "btech": 4, "be": 4, "bsc": 3, "bba": 3,
    "mtech": 2, "mba": 2, "mca": 2,
    "diploma": 3, "other": 4,
}


def parse_graduation_year(grad_text: str | None, degree_enum: str | None = None) -> int | None:
    """Return the year the candidate graduates (or graduated).

    Strategies, in order:
      1. The text is a single 4-digit year: return it directly.
      2. The text is a range 'START - END':
         - END is 'Present'/'Current' -> project from START using the default
           program length for the degree.
         - END is a parseable date -> return its year.
         - Neither -> return START year (best effort).
      3. The text is a single Month-Year -> return its year.
    """
    if not grad_text:
        return None
    s = grad_text.strip()

    # Pure 4-digit year
    m = re.fullmatch(r"(19|20)\d{2}", s)
    if m:
        return int(s)

    # Range like 'Nov 2022 - Present' / 'Jul 2020 - May 2024' / '2018 - 2022'
    # Accept en-dash, em-dash, hyphen, or 'to'
    range_match = re.match(r"^(.+?)\s*(?:[-–—]|to)\s*(.+?)$", s, re.IGNORECASE)
    if range_match:
        start_raw, end_raw = range_match.group(1).strip(), range_match.group(2).strip()
        end_parsed = parse_date(end_raw)
        if end_parsed:
            # parse_date treats 'Present' as today, which projects to current
            # year. That's wrong for in-progress degrees; recover by adding
            # default duration onto the start year instead.
            if re.search(r"\bpresent|current|now|ongoing\b", end_raw, re.IGNORECASE):
                start_parsed = parse_date(start_raw)
                if start_parsed:
                    duration = _DEFAULT_DURATIONS.get(degree_enum or "other", 4)
                    return start_parsed[0] + duration
                return end_parsed[0]
            return end_parsed[0]
        # Fall back to start year
        start_parsed = parse_date(start_raw)
        if start_parsed:
            return start_parsed[0]

    # Single date like 'May 2024'
    parsed = parse_date(s)
    if parsed:
        return parsed[0]

    return None


# ── Skills vs Tools split correction ────────────────────────────────────────

# Software / SaaS products that are tools no matter how the LLM classified
# them. Lowercase-keyed. The match is case-insensitive whole-token.
_KNOWN_TOOLS = {
    # PM / design / collab
    "figma", "jira", "notion", "trello", "asana", "slack",
    "mixpanel", "amplitude", "google analytics", "ga4", "hotjar",
    "power bi", "powerbi", "tableau", "looker", "metabase", "google sheets",
    "vs code", "vscode", "cursor", "cursor ai", "github copilot", "copilot",
    "github", "gitlab", "bitbucket",
    "salesforce inspector", "dataloader", "salesforce cli", "copado",
    "webflow", "framer",
    "streamlit", "gradio",
    "jupyter", "jupyter notebook", "colab", "google colab",
    "docker", "kubernetes", "k8s",
    "postman", "insomnia",
    "huggingface hub", "kaggle",
    "huggingface",
    "ms excel", "excel", "ms word", "ms powerpoint", "powerpoint",
    "ms office", "google docs",
    "git", "intellij", "pycharm", "android studio", "xcode",
    "aws cli", "gcloud", "az cli", "terraform",
    "zoho sign", "docusign", "box.com",
    "quickbooks", "mailchimp", "hubspot", "zapier",
    "drchrono", "givecloud", "conga", "pardot", "account engagement",
    # Engineering CAE / CAD / simulation / lab software — common on mech/aero/civil resumes
    "openfoam", "ansys", "ansys fluent", "ansys workbench", "ansys mechanical",
    "abaqus", "comsol", "comsol multiphysics", "comsoll multiphysics",
    "tecplot", "paraview", "originpro", "origin pro",
    "matlab", "simulink", "labview",
    "solidworks", "autocad", "catia", "fusion 360", "creo", "nx", "siemens nx",
    "inventor", "revit", "rhino", "blender",
    "salome", "gmsh", "freecad",
    "lammps", "gromacs", "vasp",
    "minitab", "stata", "spss", "sas", "rstudio", "r studio",
    "copeland pss", "ref tools",
    # Cloud / data platforms
    "snowflake", "databricks", "bigquery", "redshift",
    "s3", "ec2", "lambda", "azure", "gcp",
}


def _norm_skill(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def split_skills_and_tools(skills_raw: list[str], tools_raw: list[str]) -> tuple[list[str], list[str]]:
    """Apply the known-tool allowlist correction on top of the LLM's split.

    The model usually gets the split right but sometimes puts branded software
    (Figma, Notion) in skills. This pass moves any allowlist hits over to
    tools, dedupes, and preserves order within each bucket."""
    skills_out: list[str] = []
    tools_out: list[str] = []
    seen_skills: set[str] = set()
    seen_tools: set[str] = set()

    def _push_tool(item: str):
        key = _norm_skill(item)
        if key and key not in seen_tools:
            seen_tools.add(key)
            tools_out.append(item.strip())

    def _push_skill(item: str):
        key = _norm_skill(item)
        # Mutual exclusion: anything already classified as a tool wins. This
        # prevents 'Cursor AI' from showing up in both lists when the LLM
        # over-reports.
        if key and key not in seen_skills and key not in seen_tools:
            seen_skills.add(key)
            skills_out.append(item.strip())

    # First: anything the LLM called a tool is a tool
    for t in tools_raw or []:
        if not isinstance(t, str):
            continue
        _push_tool(t)

    # Then: things the LLM called a skill — move to tools if they're in the
    # known-tools allowlist, else keep as skills
    for s in skills_raw or []:
        if not isinstance(s, str):
            continue
        if _norm_skill(s) in _KNOWN_TOOLS:
            _push_tool(s)
        else:
            _push_skill(s)

    return (skills_out, tools_out)


# ── Per-project derive: dates + skillsUsed + condensation audit ─────────────

# Match the label at the start of a tech-stack line — any of "Tech:",
# "Tech Stack:", "Tools used:", "Technologies:", "Stack:".
_TECH_LINE_PREFIX = re.compile(
    r"^\s*(?:tech(?:nolog(?:y|ies))?(?:\s*stack)?|tools?(?:\s*used)?|stack)\s*[:\-]\s*",
    re.IGNORECASE,
)
# Numbers / percentages / metric tokens we want to verify survive condensation
_METRIC_TOKEN = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:%|x|years?|hours?|weeks?|days?|samples?|records?|GB|MB|KB|GPU)?\b")


def _parse_tech_stack(tech_text: str | None) -> list[str]:
    """Split a 'Tech: PyTorch, Mistral-7B, PEFT, bitsandbytes' line into items.

    We respect parentheses — 'PyTorch (nightly/cu128)' is one item, not split
    on the inner '/'. Commas at depth=0 split; commas inside () stay."""
    if not tech_text:
        return []
    s = _TECH_LINE_PREFIX.sub("", tech_text.strip())
    items: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            piece = "".join(buf).strip()
            if piece:
                items.append(piece)
            buf = []
        else:
            buf.append(ch)
    if buf:
        piece = "".join(buf).strip()
        if piece:
            items.append(piece)
    # Filter out fragments that are clearly not skills (separators, empty).
    # Length-1 items like 'R' / 'C' are valid language names — keep them.
    return [it for it in items if it and not re.fullmatch(r"[\|\-•\s]+", it)]


def _condensation_lost_metrics(evidence: str | None, description: str | None) -> list[str]:
    """Return the list of metric tokens that appear in evidence but NOT in
    description_condensed. Used to warn (not error) when the LLM dropped facts.

    Skips:
      - Bare 1-2 digit numbers without units (too noisy: "5", "20").
      - 4-digit year tokens in the 1900-2099 range (dates are stored in
        separate fields, so they shouldn't appear in the description anyway).
    """
    if not evidence or not description:
        return []
    ev_metrics = set()
    for m in _METRIC_TOKEN.finditer(evidence):
        tok = m.group(0).strip()
        if re.fullmatch(r"\d{1,2}", tok):
            continue
        if re.fullmatch(r"(?:19|20)\d{2}", tok):
            continue
        ev_metrics.add(tok.lower())
    if not ev_metrics:
        return []
    desc_lower = description.lower()
    lost = [tok for tok in sorted(ev_metrics) if tok not in desc_lower]
    return lost


def _project_dates(entry: dict) -> dict:
    """Parse project start/end into Month/Year ints plus isInProgress flag.

    Returns the four-field shape Pulse's student_projects table expects.
    Missing dates round-trip as None — many fresher projects have no dates."""
    start = parse_date(entry.get("start"))
    end_raw = entry.get("end")
    end = parse_date(end_raw) if end_raw else None

    is_in_progress = False
    if end_raw and re.search(r"\b(?:present|current|ongoing|now)\b", str(end_raw), re.IGNORECASE):
        is_in_progress = True
        end = None  # store as null; isInProgress carries the signal

    out = {
        "startMonth": start[1] if start else None,
        "startYear": start[0] if start else None,
        "endMonth": end[1] if end else None,
        "endYear": end[0] if end else None,
        "isInProgress": is_in_progress,
    }
    return out


def derive_projects(entries: list[dict], known_skills: list[str], known_tools: list[str]) -> list[dict]:
    """Turn project_entries into Pulse student_projects rows.

    skillsUsed is derived in priority order:
      1. tech_stack_text parsed (verbatim resume claim) — strongest signal.
      2. LLM-attributed 'skills_used' field from the extractor (the model
         judging which of the candidate's overall skills were used here).
         Filtered against the candidate kit so the model can't invent.
      3. Fallback: intersect description against the candidate's kit.
    """
    candidate_kit = {_norm_skill(s) for s in (known_skills + known_tools) if s}
    candidate_kit_originals = {_norm_skill(s): s for s in (known_skills + known_tools) if s}
    out: list[dict] = []

    for e in entries or []:
        title = (e.get("title") or "").strip()
        if not title:
            continue
        description = (e.get("description_condensed") or "").strip()
        evidence = (e.get("evidence") or "").strip()

        # 1. tech_stack — best signal, copy verbatim
        skills_used = _parse_tech_stack(e.get("tech_stack_text"))

        # 2. LLM-attributed per-project skills (filtered against the candidate
        # kit so the model can't introduce tech the candidate never listed)
        if not skills_used:
            llm_attributed = e.get("skills_used") or []
            for raw in llm_attributed:
                if not isinstance(raw, str):
                    continue
                key = _norm_skill(raw)
                if not key:
                    continue
                # Accept if it matches a known skill/tool the candidate listed
                if key in candidate_kit:
                    skills_used.append(candidate_kit_originals[key])
                # Or accept verbatim if it's substantive (the candidate may
                # legitimately use a tech in a project that didn't make their
                # general skills list — common with specific methodologies)
                elif len(raw) >= 3:
                    skills_used.append(raw.strip())
            # Dedupe preserving order
            seen_norm: set[str] = set()
            deduped: list[str] = []
            for s in skills_used:
                k = _norm_skill(s)
                if k in seen_norm:
                    continue
                seen_norm.add(k)
                deduped.append(s)
            skills_used = deduped

        # 3. Fallback: description-intersect-candidate-kit
        if not skills_used and description and candidate_kit:
            desc_norm = description.lower()
            skills_used = [
                s for s in (known_skills + known_tools)
                if _norm_skill(s) in candidate_kit and _norm_skill(s) in desc_norm
            ]

        lost_metrics = _condensation_lost_metrics(evidence, description)
        if lost_metrics:
            print(f"[student_derive] WARN project {title!r} dropped metrics in condensation: {lost_metrics}")

        row = {
            "title": title,
            "description": description or None,
            "skillsUsed": skills_used,
            "links": [],  # populated by polish-pass once PDF link extraction lands
            "evidence": evidence or None,
            **_project_dates(e),
        }
        out.append(row)

    return out


# ── State sanity filter + abbreviation expansion ────────────────────────────

# Common country / region names that get mis-extracted into the state field.
# Resume contact lines often read 'City, India' — the model sometimes picks
# 'India' as state. Drop these so the field is null rather than wrong.
_COUNTRY_BLOCKLIST = {
    "india", "bharat",
    "usa", "us", "united states", "united states of america",
    "uk", "united kingdom", "england", "britain",
    "canada", "australia", "new zealand",
    "uae", "united arab emirates", "singapore",
}

# Indian state two-letter codes + common abbreviations -> canonical name.
# Pulse downstream resolves stateId by canonical name; passing a code like
# 'UP' fails the lookup, so expand here.
_INDIAN_STATE_CODES = {
    "ap": "Andhra Pradesh",
    "ar": "Arunachal Pradesh",
    "as": "Assam",
    "br": "Bihar",
    "cg": "Chhattisgarh",
    "ch": "Chhattisgarh",
    "ga": "Goa",
    "gj": "Gujarat",
    "hr": "Haryana",
    "hp": "Himachal Pradesh",
    "jk": "Jammu and Kashmir",
    "jh": "Jharkhand",
    "ka": "Karnataka",
    "kl": "Kerala",
    "mp": "Madhya Pradesh",
    "mh": "Maharashtra",
    "mn": "Manipur",
    "ml": "Meghalaya",
    "mz": "Mizoram",
    "nl": "Nagaland",
    "od": "Odisha",
    "or": "Odisha",
    "pb": "Punjab",
    "rj": "Rajasthan",
    "raj": "Rajasthan",
    "sk": "Sikkim",
    "tn": "Tamil Nadu",
    "tg": "Telangana",
    "ts": "Telangana",
    "tr": "Tripura",
    "uk": "Uttarakhand",      # collides with the country code; resolved by city/country context upstream
    "ut": "Uttarakhand",
    "up": "Uttar Pradesh",
    "u.p.": "Uttar Pradesh",
    "u p": "Uttar Pradesh",
    "wb": "West Bengal",
    "dl": "Delhi",
    "ch.": "Chandigarh",
    "ld": "Lakshadweep",
    "py": "Puducherry",
    "an": "Andaman and Nicobar Islands",
    "dn": "Dadra and Nagar Haveli",
}


def _clean_state(state: str | None) -> str | None:
    """Normalize a state field.

    - Strip whitespace + trailing punctuation.
    - Drop common country names ('India', 'USA', ...).
    - Expand Indian state codes ('UP' -> 'Uttar Pradesh', 'TS' -> 'Telangana').
    """
    if not state:
        return None
    # strip() handles ASCII whitespace, .strip(",.") removes trailing comma/period,
    # and a final strip() handles whitespace exposed after the punctuation strip.
    s = state.strip().strip(",.").strip()
    if not s:
        return None
    low = s.lower()
    if low in _COUNTRY_BLOCKLIST:
        return None
    if low in _INDIAN_STATE_CODES:
        return _INDIAN_STATE_CODES[low]
    # Also expand inline trailing periods ('U.P.' -> 'Uttar Pradesh')
    low_no_dots = low.replace(".", "").replace(" ", "")
    if low_no_dots in _INDIAN_STATE_CODES:
        return _INDIAN_STATE_CODES[low_no_dots]
    return s


# ── Internship flattening to prose ──────────────────────────────────────────

def flatten_internships(entries: list[dict]) -> str | None:
    """Concatenate internship entries into the single text blob Pulse stores.

    Format per internship (separated by blank lines):
        Role - Company, Location (Start - End)
        - achievement 1
        - achievement 2

    Returns None if no internships (so the DB column stays NULL rather than
    empty string)."""
    if not entries:
        return None

    blocks: list[str] = []
    for e in entries:
        role = (e.get("role") or "").strip()
        company = (e.get("company") or "").strip()
        loc = (e.get("location") or "").strip()
        start = (e.get("start") or "").strip()
        end = (e.get("end") or "").strip()
        achievements = e.get("achievements") or []

        if not (role or company):
            continue

        head_parts: list[str] = []
        if role and company:
            head_parts.append(f"{role} - {company}")
        elif role:
            head_parts.append(role)
        elif company:
            head_parts.append(company)
        if loc:
            head_parts.append(loc)
        head = ", ".join(head_parts)
        if start or end:
            head += f" ({start} - {end})".replace("( - )", "").replace("( - ", "(").replace(" - )", ")")

        lines = [head]
        for a in achievements:
            if not isinstance(a, str) or not a.strip():
                continue
            lines.append(f"- {a.strip()}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) if blocks else None


# ── Education picking ───────────────────────────────────────────────────────

# Which education entry is "the" one we report on the Pulse student row?
# Rule: highest-level degree (Master > Bachelor > Diploma) AND most recent
# graduation year wins. If a candidate has both PG and UG, PG is reported.
# Keys here are the enum values produced by map_degree() — substring matching
# against the raw degree string ('B.Tech in CSE') fails because of periods
# and spacing, so we route through map_degree first.
_DEGREE_RANK = {
    "mtech": 4, "mba": 4, "mca": 4,
    "btech": 3, "be": 3, "bsc": 3,
    "diploma": 2,
    "other": 1,
}


def _degree_rank_from_raw(raw_degree: str | None) -> int:
    if not raw_degree:
        return 0
    enum_val = map_degree(raw_degree)
    if enum_val is None:
        return 0
    return _DEGREE_RANK.get(enum_val, 0)


def pick_primary_education(entries: list[dict]) -> dict | None:
    """Choose the entry to report on the flat Pulse student row.

    Sort by (degree_rank desc, graduation_year desc), pick top. School-level
    entries (10th/12th) lose to anything else by rank=0. Other education
    entries are preserved in the output dict's `_all_education` field for
    downstream systems that want the full list."""
    if not entries:
        return None
    scored = []
    for e in entries:
        rank = _degree_rank_from_raw(e.get("raw_degree"))
        grad = parse_graduation_year(e.get("graduation_year_text"))
        scored.append((rank, grad or 0, e))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    top_rank = scored[0][0]
    if top_rank == 0:
        # No real degree at all — caller decides whether to skip the candidate
        return None
    return scored[0][2]


# ── Hyperlink resolution ────────────────────────────────────────────────────

# Domain patterns we recognize. The first match wins, so order matters when
# domains overlap (linkedin.com/in/foo could match a 'social' bucket if we had
# one, but we don't — we just want linkedin → linkedinUrl).
_LINKEDIN_DOMAIN = re.compile(r"^https?://(?:[\w-]+\.)*linkedin\.(?:com|in)/", re.IGNORECASE)
_GITHUB_DOMAIN = re.compile(r"^https?://(?:[\w-]+\.)*github\.(?:com|io)/", re.IGNORECASE)
_HF_DOMAIN = re.compile(r"^https?://huggingface\.co/", re.IGNORECASE)
_KAGGLE_DOMAIN = re.compile(r"^https?://(?:www\.)?kaggle\.com/", re.IGNORECASE)
# Anchor words that signal a portfolio/personal site even without a known domain.
_PORTFOLIO_ANCHORS = {"portfolio", "website", "personal site", "blog", "site"}


def _link_label(link: dict) -> str:
    """The visible anchor text inside the link rect, lowercased + stripped."""
    return (link.get("anchor_text") or "").strip().lower()


def _project_signature(title: str) -> str:
    """Reduce a project title to a lowercase signature used for fuzzy matching
    against PDF link context text. Drops punctuation, keeps alphanum tokens."""
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _attribute_link_to_project(link: dict, projects: list[dict]) -> int | None:
    """Return the index of the project whose title appears in the link's
    surrounding context text, or None if no project matches.

    Match rule (conservative): at least 60% of the project title's significant
    tokens (length >= 3) appear in the link's context_text, AND at least 2
    distinct tokens hit. Both bars together stop spurious matches from header /
    publications sections leaking into the first project below them."""
    ctx = (link.get("context_text") or "").lower()
    if not ctx:
        return None
    best_idx: int | None = None
    best_overlap = 0.0
    for i, p in enumerate(projects):
        title = (p.get("title") or "").strip()
        if not title:
            continue
        sig = _project_signature(title)
        tokens = [t for t in sig.split() if len(t) >= 3]
        if not tokens:
            continue
        hits = sum(1 for t in tokens if t in ctx)
        if hits < 2:
            continue
        overlap = hits / len(tokens)
        if overlap > best_overlap and overlap >= 0.6:
            best_overlap = overlap
            best_idx = i
    return best_idx


def resolve_hyperlinks(
    hyperlinks: list[dict] | None,
    projects: list[dict],
    identity: dict,
) -> tuple[str | None, str | None, list[dict]]:
    """Classify a PDF's hyperlinks into identity URLs + per-project link lists.

    Returns: (linkedinUrl, portfolioUrl, projects_with_links)
      - linkedinUrl: first LinkedIn URL found, else None.
      - portfolioUrl: first URL whose anchor matches a portfolio/personal-site
        signal, or first GitHub URL when the candidate didn't show a separate
        portfolio anchor.
      - projects_with_links: each input project gets its links[] list populated
        with the URLs whose context text matched the project title.
    """
    projects = [dict(p) for p in projects]  # shallow copy; we mutate links

    if not hyperlinks:
        return (None, None, projects)

    linkedin_url: str | None = None
    portfolio_url: str | None = None
    # GitHub URLs that didn't attribute to any project — candidates for
    # portfolio fallback.
    unassigned_github: list[str] = []

    for link in hyperlinks:
        url = link.get("url") or ""
        if not url:
            continue
        label = _link_label(link)

        # LinkedIn — identity-only
        if _LINKEDIN_DOMAIN.match(url):
            if linkedin_url is None:
                linkedin_url = url
            continue

        # Explicit portfolio anchor
        if any(anchor in label for anchor in _PORTFOLIO_ANCHORS):
            if portfolio_url is None:
                portfolio_url = url
            continue

        # GitHub / HF / Kaggle — first try project attribution
        if _GITHUB_DOMAIN.match(url) or _HF_DOMAIN.match(url) or _KAGGLE_DOMAIN.match(url):
            idx = _attribute_link_to_project(link, projects)
            if idx is not None:
                # Attribute to project
                proj_links = projects[idx].setdefault("links", [])
                proj_links.append({
                    "url": url,
                    "label": _link_label(link) or _infer_label_from_url(url),
                })
            else:
                if _GITHUB_DOMAIN.match(url):
                    unassigned_github.append(url)
            continue

        # Other domains — if labeled like a portfolio fallback already handled
        # above. Otherwise drop (could be email mailto:, social, etc.).

    # Portfolio fallback: first unassigned GitHub URL becomes portfolio if the
    # candidate didn't show an explicit portfolio link.
    if portfolio_url is None and unassigned_github:
        portfolio_url = unassigned_github[0]

    return (linkedin_url, portfolio_url, projects)


def _infer_label_from_url(url: str) -> str:
    if _GITHUB_DOMAIN.match(url):
        return "GitHub"
    if _HF_DOMAIN.match(url):
        return "HuggingFace"
    if _KAGGLE_DOMAIN.match(url):
        return "Kaggle"
    if _LINKEDIN_DOMAIN.match(url):
        return "LinkedIn"
    return ""


# ── Top-level derive function ───────────────────────────────────────────────

def _ground_entries(entries: list[dict], source_norm: str | None, kind: str) -> list[dict]:
    """Drop entries whose 'evidence' quote isn't grounded in the source.

    Sakhi-style anti-hallucination: every entity must include a verbatim quote
    from the source resume; we verify either substring match OR >=60% token
    overlap. Ungrounded entries are dropped + logged so the failure is visible
    rather than silently producing fake records.

    `kind` is used in the log line so we can see which type was dropped."""
    if source_norm is None or not entries:
        return entries
    out: list[dict] = []
    for e in entries:
        if _evidence_grounded(e, source_norm):
            out.append(e)
            continue
        ident = (
            e.get("title")
            or e.get("company")
            or e.get("institution")
            or "(unnamed)"
        )
        print(f"[student_derive] {kind} entry dropped (ungrounded evidence): {ident!r}")
    return out


def derive_student_record(
    extracted: dict,
    source_text: str | None = None,
    hyperlinks: list[dict] | None = None,
) -> dict:
    """Compose the full Pulse student record from extracted entities.

    Returns a dict shaped for JSONL write. Fields match the Pulse student
    schema names (camelCase) so downstream consumers don't have to remap.

    When source_text is supplied, every entry's 'evidence' quote is grounded
    against it — entries whose evidence doesn't appear in source are dropped
    (Sakhi-style hallucination guard).

    When hyperlinks (from extract_pdf_hyperlinks) is supplied, linkedinUrl /
    portfolioUrl / per-project links[] are populated."""
    identity = extracted.get("identity") or {}
    education = extracted.get("education_entries") or []
    internships = extracted.get("internship_entries") or []
    projects = extracted.get("project_entries") or []
    skills_raw = extracted.get("skills_raw") or []
    tools_raw = extracted.get("tools_raw") or []
    certifications = extracted.get("certifications") or []
    languages = extracted.get("languages_known") or []

    # Evidence grounding (drops hallucinated entries)
    source_norm = _norm_for_grounding(source_text) if source_text else None
    education = _ground_entries(education, source_norm, "education")
    internships = _ground_entries(internships, source_norm, "internship")
    projects = _ground_entries(projects, source_norm, "project")

    # Skills / tools split correction
    skills, tools_used = split_skills_and_tools(skills_raw, tools_raw)

    # Primary education -> degree/branch/cgpa/graduationYear/specialization
    primary_edu = pick_primary_education(education)
    if primary_edu:
        raw_degree = primary_edu.get("raw_degree")
        degree_enum = map_degree(raw_degree)
        branch_enum = map_branch(raw_degree)
        branch_specialization = _extract_specialization(
            llm_field=primary_edu.get("specialization"),
            raw_degree=raw_degree,
            branch_enum=branch_enum,
        )
        cgpa_value, cgpa_scale = parse_cgpa(primary_edu.get("cgpa_text"))
        graduation_year = parse_graduation_year(
            primary_edu.get("graduation_year_text"),
            degree_enum=degree_enum,
        )
    else:
        degree_enum = branch_enum = branch_specialization = None
        cgpa_value = cgpa_scale = graduation_year = None

    # Projects (with per-project skillsUsed + lossless audit)
    derived_projects = derive_projects(projects, known_skills=skills, known_tools=tools_used)

    # Hyperlink resolution — domain-classify identity URLs, attribute per-project
    # GitHub/HF/Kaggle URLs to their owning project via context-text matching.
    linkedin_url, portfolio_url, derived_projects = resolve_hyperlinks(
        hyperlinks, derived_projects, identity,
    )

    # Internships flattened to prose
    internship_details = flatten_internships(internships)

    return {
        # Identity & contact
        "name": identity.get("name"),
        "phone": normalize_phone(identity.get("phone")),
        "email": identity.get("email"),
        "linkedinUrl": linkedin_url,
        "portfolioUrl": portfolio_url,
        "dateOfBirth": identity.get("date_of_birth"),
        "city": identity.get("city"),
        "state": _clean_state(identity.get("state")),

        # Academic
        "degree": degree_enum,
        "branch": branch_enum,
        "branchSpecialization": branch_specialization,
        "cgpa": cgpa_value,
        "cgpaScale": cgpa_scale,
        "graduationYear": graduation_year,

        # Skills & languages
        "skills": skills,
        "toolsUsed": tools_used,
        "certifications": certifications,
        "languagesKnown": languages,

        # Experience & projects
        "internshipDetails": internship_details,
        "projects": derived_projects,

        # Career preferences
        "targetRoles": identity.get("target_roles"),
        "preferredLocations": identity.get("preferred_locations") or [],

        # Debug / audit fields (not part of Pulse schema, but useful for
        # ground-truthing during iteration)
        "_all_education": education,
        "_linkedin_label": identity.get("linkedin_label"),
        "_portfolio_label": identity.get("portfolio_label"),
    }
