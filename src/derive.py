"""
Deterministic derivation of CSV row from the model's raw entity output.

This is the Sakhi-pattern split: the model emits entities (with verbatim
evidence), Python decides every business rule. No model judgment leaks past
this layer.

Inputs (from extractor.extract_all):
  contact: {name, phone, email}
  employment_entries: [{company, title, start, end, evidence}, ...]
  education_entries:  [{institution, degree, end_year, evidence}, ...]
  skills_raw:         [str, ...]

Output: dict matching schema.yaml field keys, ready for CSV write.
"""
from __future__ import annotations

import re
from datetime import date


# Today's date — used as 'Present' when computing experience years.
# Update if the calendar moves significantly (or wire it to date.today()
# at call time; leaving as a constant for deterministic tests).
TODAY_YEAR = 2026
TODAY_MONTH = 5


# ── Date parsing ────────────────────────────────────────────────────────────

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4,
    "june": 6, "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
}

_PRESENT_TOKENS = {
    "present", "current", "till date", "till now", "now",
    "ongoing", "currently", "to date",
}


def parse_date(s) -> tuple[int, int] | None:
    """Parse a date string from a resume into (year, month).

    Handles:
      'Jul 2025', 'July 2025', 'JUL 2025'
      "Mar'24", "March'2024"
      'Aug 2018', '2024' (just year, treated as January)
      'Present', 'Current', 'Till Date', etc. → today
      '07/2025', '7/25'
      ISO 'YYYY-MM' or 'YYYY-MM-DD'

    Returns None for anything we can't recognize."""
    if s is None:
        return None
    s = str(s).strip().lower().rstrip(".")
    if not s:
        return None

    # Present-style tokens
    if s in _PRESENT_TOKENS or s.startswith("present"):
        return (TODAY_YEAR, TODAY_MONTH)

    # ISO-ish: 2024-05-01, 2024-05, 2024/05
    m = re.match(r"^(\d{4})[-/](\d{1,2})(?:[-/]\d{1,2})?$", s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            return (y, mo)

    # MM/YYYY or M/YY
    m = re.match(r"^(\d{1,2})[/](\d{2,4})$", s)
    if m:
        mo, y = int(m.group(1)), int(m.group(2))
        if y < 100:
            y = 2000 + y if y < 50 else 1900 + y
        if 1 <= mo <= 12:
            return (y, mo)

    # MonthName YYYY  or  Mon'YY
    m = re.match(r"^([a-z]+)\s*[\s'`]?\s*(\d{2,4})$", s)
    if m:
        month_word, year_str = m.group(1), m.group(2)
        mo = _MONTHS.get(month_word)
        if mo is None:
            return None
        y = int(year_str)
        if y < 100:
            y = 2000 + y if y < 50 else 1900 + y
        return (y, mo)

    # Just a year
    m = re.match(r"^(19|20)\d{2}$", s)
    if m:
        return (int(s), 1)

    return None


def _to_month_index(year: int, month: int) -> int:
    return year * 12 + month


# ── Employment sorting + decomposition ──────────────────────────────────────

def _entry_end_index(entry: dict) -> int:
    """Sortable key — the entry's end month as an integer. Entries with no
    parseable end go to the bottom."""
    end = parse_date(entry.get("end"))
    if end is None:
        return -1
    return _to_month_index(*end)


def _norm_for_grounding(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip() if s else ""


def _evidence_grounded(entry: dict, source_norm: str, min_word_overlap: float = 0.6) -> bool:
    """Sakhi-style grounding: the entry's 'evidence' field should appear in
    the source text. We use a relaxed token-overlap check (60% of evidence
    tokens must appear in source) rather than strict substring — the model
    often joins multi-line PDF text into a single evidence quote, which makes
    strict substring fail even though the entry is real."""
    ev = entry.get("evidence") or ""
    if not ev:
        return False
    ev_norm = _norm_for_grounding(ev)
    # Exact substring is the strong signal
    if ev_norm in source_norm:
        return True
    # Token-overlap fallback for multi-line evidence that got reflowed
    ev_tokens = [t for t in re.split(r"\W+", ev_norm) if len(t) >= 2]
    if not ev_tokens:
        return False
    hits = sum(1 for t in ev_tokens if t in source_norm)
    return (hits / len(ev_tokens)) >= min_word_overlap


def sort_employment(entries: list[dict], source_text: str | None = None) -> list[dict]:
    """Sort employment entries by end date descending (most recent first).
    If source_text is given, drop entries whose evidence quote isn't grounded
    in the source — same anti-hallucination principle Sakhi uses on danger
    sign utterance evidence."""
    if source_text is not None:
        source_norm = _norm_for_grounding(source_text)
        grounded = []
        for e in entries:
            if _evidence_grounded(e, source_norm):
                grounded.append(e)
            else:
                print(f"[derive] employment entry dropped (ungrounded evidence): "
                      f"company={e.get('company')!r}, title={e.get('title')!r}")
        entries = grounded
    return sorted(entries, key=_entry_end_index, reverse=True)


def derive_current(sorted_entries: list[dict]) -> tuple[str | None, str | None]:
    """Return (current_company, current_role) from the most recent entry,
    or (None, None) if there are no entries."""
    if not sorted_entries:
        return (None, None)
    top = sorted_entries[0]
    return (top.get("company") or None, top.get("title") or None)


def derive_previous_companies(sorted_entries: list[dict], current_company: str | None) -> str:
    """Pipe-separated list of every other unique employer, preserving the
    sort order from sort_employment (most recent first).

    Dedupe is case-insensitive and whitespace-normalized so 'Cloudy Coders'
    appearing twice (as it would for an internal promotion) collapses to one."""
    if not sorted_entries:
        return ""

    cur_norm = _norm_company(current_company) if current_company else None
    seen: set[str] = set()
    if cur_norm:
        seen.add(cur_norm)

    out: list[str] = []
    for entry in sorted_entries:
        company = entry.get("company")
        if not company:
            continue
        key = _norm_company(company)
        if key in seen:
            continue
        seen.add(key)
        out.append(company.strip())

    return " | ".join(out)


def _norm_company(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()) if s else ""


# ── Experience years (Sakhi-style: deterministic, no model number used) ─────

def derive_experience_years(
    sorted_entries: list[dict],
    source_text: str | None = None,
) -> int | None:
    """Compute total years from merged employment intervals.

    Workflow:
      1. If `source_text` contains an explicit '7 years experience' /
         'Total Experience: 4 years 1 month' / '(49 MONTHS)' / 'with around X
         years', PREFER that (resume's own stated number).
      2. Otherwise, parse start/end on every entry → month-index intervals →
         merge overlapping → sum → round to whole years.

    Returns None when there are no entries and no stated number — leaves the
    cell empty in the CSV."""
    # 1. Stated-years takes priority — same regex set we had before
    if source_text:
        stated = _extract_stated_years(source_text)
        if stated is not None:
            return stated

    # 2. Compute from intervals
    if not sorted_entries:
        return None

    intervals: list[tuple[int, int]] = []
    today_idx = _to_month_index(TODAY_YEAR, TODAY_MONTH)

    for entry in sorted_entries:
        start = parse_date(entry.get("start"))
        end = parse_date(entry.get("end"))
        if not start:
            continue
        if not end:
            # Treat unspecified end as today (likely "Present" that didn't
            # round-trip through the model cleanly)
            end = (TODAY_YEAR, TODAY_MONTH)
        s_idx = _to_month_index(*start)
        e_idx = _to_month_index(*end)
        # Sanity: future start, inverted, or >50 year spans → skip
        if s_idx > today_idx or e_idx < s_idx or (e_idx - s_idx) > 50 * 12:
            continue
        intervals.append((s_idx, e_idx))

    if not intervals:
        return None

    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))

    total_months = sum(e - s for s, e in merged)
    years = round(total_months / 12)
    return max(0, min(60, years))


_MODIFIERS = r"(?:around|approximately|over|nearly|more\s+than|~)\s+"
_STATED_PATTERNS = [
    # 'WORK EXPERIENCE (49 MONTHS)' / '(36 months)' — special: divide by 12
    (re.compile(r"experience\s*\(\s*(\d{1,3})\s*months?\s*\)", re.IGNORECASE), True),
    # 'Total Experience: 4 years 1 month' / 'Total Experience - 4 yrs'
    (re.compile(r"total\s+experience\s*[:\-]?\s*(\d{1,2})\s*(?:years?|yrs?)", re.IGNORECASE), False),
    # '7yrs of experience' / '7 yrs experience' / '7 years professional experience'
    (re.compile(r"\b(\d{1,2})\s*\+?\s*(?:years?|yrs?|y)\s+(?:of\s+)?(?:professional\s+|work\s+|industry\s+)?experience\b", re.IGNORECASE), False),
    # 'with [around/over/approximately] 7 years'
    (re.compile(rf"\bwith\s+(?:{_MODIFIERS})?(\d{{1,2}})\s*\+?\s*(?:years?|yrs?)\b", re.IGNORECASE), False),
]


def _extract_stated_years(text: str) -> int | None:
    head_len = max(1500, len(text) // 3)
    regions = [text[:head_len], text]
    for region in regions:
        for pat, is_months in _STATED_PATTERNS:
            m = pat.search(region)
            if not m:
                continue
            try:
                n = int(m.group(1))
            except (ValueError, IndexError):
                continue
            if is_months:
                if 0 <= n <= 600:
                    return round(n / 12)
                continue
            if 0 <= n <= 60:
                return n
    return None


# ── Education filtering + alignment ─────────────────────────────────────────

# Degree-name patterns we accept. Two lists:
#   _VALID_DEGREE_SUBSTRINGS — long, unambiguous; case-insensitive substring match
#   _VALID_DEGREE_TOKENS     — short bare abbreviations; word-boundary regex
#                              (prevents 'BE' from matching 'BEach', 'MA' matching
#                              'manager', etc.)
_VALID_DEGREE_SUBSTRINGS = [
    "ph.d", "doctorate", "doctoral",
    "m.tech", "mtech", "m tech", "master of technology",
    "b.tech", "btech", "b tech", "bachelor of technology",
    "m.sc", "b.sc", "m.com", "b.com", "b.e.", "m.e.",
    "bachelor of engineering", "master of engineering",
    "bba", "mba", "bca", "mca",
    "pgdm", "pgdfm", "pgdba", "pgp",
    "diploma", "polytechnic",
    "bachelor", "master",
    "ll.b", "ll.m",
]

_VALID_DEGREE_TOKENS = re.compile(
    r"\b(?:phd|be|me|bs|ms|ba|ma|bcom|mcom|bsc|msc|llb|llm)\b",
    re.IGNORECASE,
)

# Strings that indicate school-level education we want to DROP.
_SCHOOL_LEVEL_MARKERS = [
    "10th", "12th",
    "class x", "class xii", "class 10", "class 12",
    "high school", "higher secondary", "senior secondary",
    "cbse board", "icse", "hsc", "ssc",
    "+2", "intermediate",
    "secondary school certificate",
]


def _looks_like_school_level(institution: str, degree: str) -> bool:
    """Return True if this education entry is school-level (10th/12th)."""
    blob = f"{institution} {degree}".lower()
    return any(marker in blob for marker in _SCHOOL_LEVEL_MARKERS)


def _looks_like_real_degree(degree: str) -> bool:
    if not degree:
        return False
    d = degree.lower()
    if any(kw in d for kw in _VALID_DEGREE_SUBSTRINGS):
        return True
    if _VALID_DEGREE_TOKENS.search(d):
        return True
    return False


def filter_and_align_education(entries: list[dict], source_text: str | None = None) -> tuple[str, str]:
    """From raw education entries, return (college_name, degree) as pipe-
    separated strings, most recent first, with college[i] aligned to degree[i].

    Filter rules:
      - Drop entries that look like 10th/12th/CBSE/HSC/etc.
      - Drop entries with no recognizable degree string.
      - Same iteration produces both lists → alignment guaranteed by construction."""
    if not entries:
        return ("", "")

    # Evidence grounding — drop entries whose evidence isn't in the source
    if source_text is not None:
        source_norm = _norm_for_grounding(source_text)
        grounded = []
        for e in entries:
            if _evidence_grounded(e, source_norm):
                grounded.append(e)
            else:
                print(f"[derive] education entry dropped (ungrounded evidence): "
                      f"institution={e.get('institution')!r}, degree={e.get('degree')!r}")
        entries = grounded

    # Sort by end_year desc, with unknown years floated to the bottom
    def _year_key(e):
        y = e.get("end_year")
        try:
            return -int(y) if y is not None else 1
        except (ValueError, TypeError):
            return 1

    sorted_entries = sorted(entries, key=_year_key)

    institutions: list[str] = []
    degrees: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()

    for e in sorted_entries:
        inst = (e.get("institution") or "").strip()
        deg = (e.get("degree") or "").strip()
        if not inst or not deg:
            continue
        if _looks_like_school_level(inst, deg):
            continue
        if not _looks_like_real_degree(deg):
            continue
        key = (inst.lower(), deg.lower())
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        institutions.append(inst)
        degrees.append(deg)

    return (" | ".join(institutions), " | ".join(degrees))


# ── Skills ──────────────────────────────────────────────────────────────────

# Sub-section labels the model sometimes emits along with skills
_SKILL_LABEL_PREFIXES = re.compile(
    r"^\s*(?:technical\s+skills?|programming|frameworks?|tools?|languages?|"
    r"data\s*[\&/]\s*deployment|software|technologies|product|analytics|design)"
    r"\s*[:\-]\s*",
    re.IGNORECASE,
)


def normalize_skills(skills_raw: list, max_items: int = 15) -> str:
    """Comma-separated, case-insensitive deduped, label-prefix-stripped.
    Caps at max_items so the cell stays readable in a spreadsheet."""
    if not skills_raw:
        return ""

    cleaned: list[str] = []
    seen: set[str] = set()
    for s in skills_raw:
        if not isinstance(s, str):
            continue
        # Strip leading sub-section label like "Frameworks: PyTorch"
        s = _SKILL_LABEL_PREFIXES.sub("", s).strip()
        # Drop trailing punctuation
        s = s.rstrip(".,;:")
        if len(s) < 2:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)
        if len(cleaned) >= max_items:
            break

    return ", ".join(cleaned)


# ── Top-level derive function ───────────────────────────────────────────────

def derive_csv_fields(extracted: dict, source_text: str) -> dict:
    """Compose the full CSV row from extracted entities + source text.

    Returns a dict keyed by schema.yaml field names — exactly what csv_writer
    expects."""
    contact = extracted.get("contact") or {}
    employment = extracted.get("employment_entries") or []
    education = extracted.get("education_entries") or []
    skills = extracted.get("skills_raw") or []

    sorted_emp = sort_employment(employment, source_text=source_text)
    current_company, current_role = derive_current(sorted_emp)
    previous_companies = derive_previous_companies(sorted_emp, current_company)
    experience_years = derive_experience_years(sorted_emp, source_text=source_text)

    college_name, degree = filter_and_align_education(education, source_text=source_text)
    key_skills = normalize_skills(skills)

    return {
        "name": contact.get("name"),
        "phone": contact.get("phone"),
        "email": contact.get("email"),
        "current_company": current_company,
        "current_role": current_role,
        "experience_years": experience_years,
        "key_skills": key_skills or None,
        "current_ctc": None,        # not extracted; keep schema slot
        "expected_ctc": None,       # not extracted; keep schema slot
        "previous_companies": previous_companies or None,
        "college_name": college_name or None,
        "degree": degree or None,
    }
