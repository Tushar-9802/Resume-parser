"""
Semantic section splitter.

Resumes use wildly different headings for the same concept — "Work Experience"
vs "Professional Experience" vs "Employment History" vs "Career". This module
loads a synonym map from configs/section_synonyms.yaml and chunks the resume
into a dict of canonical_section -> text_block.

CRITICAL: heading match uses word boundaries (\b), longest-synonym-first, and
restricts matches to lines that LOOK like headings (short, isolated, optionally
all-caps or title-cased). Naive substring matching on body text would split the
resume in the middle of a bullet that happens to mention "Skills:".
"""
from __future__ import annotations

import re
from pathlib import Path
import yaml


SYNONYMS_PATH = Path(__file__).parent.parent / "configs" / "section_synonyms.yaml"


# ── Heading shape filter ────────────────────────────────────────────────────
# A line is heading-shaped if it is:
#   - short (<= 60 chars)
#   - has no terminal sentence punctuation (no trailing . ? ! ; )
#   - has no more than 6 words
# This filters out body lines that happen to contain section keywords.
def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) > 60:
        return False
    if s.rstrip().endswith(('.', '!', '?', ';')):
        return False
    # Strip trailing colon for the word count check
    s_clean = s.rstrip(':').strip()
    if len(s_clean.split()) > 6:
        return False
    return True


_CACHED_SYNONYMS: dict[str, list[str]] | None = None
_CACHED_PATTERNS: list[tuple[str, re.Pattern]] | None = None


def _load_synonyms() -> dict[str, list[str]]:
    global _CACHED_SYNONYMS
    if _CACHED_SYNONYMS is None:
        with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
            _CACHED_SYNONYMS = yaml.safe_load(f)
    return _CACHED_SYNONYMS


def _build_patterns() -> list[tuple[str, re.Pattern]]:
    """Build (canonical_section, compiled_pattern) pairs, sorted longest-first
    so 'work experience' is checked before 'experience' (avoids the substring trap)."""
    global _CACHED_PATTERNS
    if _CACHED_PATTERNS is not None:
        return _CACHED_PATTERNS

    synonyms = _load_synonyms()
    pairs: list[tuple[str, str]] = []
    for canonical, aliases in synonyms.items():
        for alias in aliases:
            pairs.append((canonical, alias))

    # Sort by alias length descending — longest match wins
    pairs.sort(key=lambda x: len(x[1]), reverse=True)

    compiled: list[tuple[str, re.Pattern]] = []
    for canonical, alias in pairs:
        # Whole-line match, optionally with leading/trailing colon, case-insensitive.
        # Word boundaries on both sides prevent substring traps.
        pattern = re.compile(
            rf"^\s*{re.escape(alias)}\s*:?\s*$",
            re.IGNORECASE,
        )
        compiled.append((canonical, pattern))

    _CACHED_PATTERNS = compiled
    return compiled


def _match_heading(line: str) -> str | None:
    """Return canonical section name if line is a recognized heading, else None."""
    if not _looks_like_heading(line):
        return None
    for canonical, pattern in _build_patterns():
        if pattern.match(line):
            return canonical
    return None


# ── PUBLIC ──────────────────────────────────────────────────────────────────

PREAMBLE = "preamble"  # everything before the first recognized section


def split_sections(text: str) -> dict[str, str]:
    """Chunk resume text into canonical sections. The block before the first
    recognized heading is stored under 'preamble' — contact info (name, email,
    phone) typically lives there.

    Returns a dict of {canonical_section: text_block}. Missing sections are
    simply absent from the dict — the caller decides what to do.
    """
    if not text:
        return {}

    lines = text.splitlines()
    sections: dict[str, list[str]] = {PREAMBLE: []}
    current = PREAMBLE
    unknown_headings: list[str] = []

    for ln in lines:
        canonical = _match_heading(ln)
        if canonical:
            current = canonical
            sections.setdefault(current, [])
            continue
        sections[current].append(ln)

    # If a heading-shaped line didn't match any synonym, log it — never silently
    # default-fallthrough (per CLAUDE.md heuristic-classifier note).
    for ln in lines:
        if _looks_like_heading(ln) and _match_heading(ln) is None:
            stripped = ln.strip().rstrip(':').strip()
            # Skip obvious non-headings: emails, URLs, numbers, mixed case
            if (
                stripped
                and not re.search(r"[@/\d]", stripped)
                and stripped not in unknown_headings
                and 2 <= len(stripped.split()) <= 4
            ):
                unknown_headings.append(stripped)

    if unknown_headings:
        print(f"[sections] unknown headings (not in synonym map): {unknown_headings[:10]}")

    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


def get_section(sections: dict[str, str], name: str, fallback: list[str] = None) -> str:
    """Lookup helper. Falls back across a list of alternate section names if
    the primary is missing (e.g., experience → summary for fallback context).
    Returns empty string if nothing matches."""
    if name in sections:
        return sections[name]
    for alt in fallback or []:
        if alt in sections:
            return sections[alt]
    return ""
