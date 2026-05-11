"""
Final-cleanup validation layer.

Most validation logic moved upstream to src/derive.py (which builds the row
from raw entities — alignment, dedup, year computation, current-vs-previous
are all guaranteed by construction there). This module only does the
field-level cleanup that needs the source_text to verify:

  1. Phone normalization     — "+7985883353" → "+91-7985883353"
  2. Email regex             — drop malformed values to null
  3. Fake-default strip      — model defaults like "John Doe" / "example@email.com"
  4. Verbatim grounding      — company/college values must appear in source
                                text (case + whitespace normalized)
"""
from __future__ import annotations

import re
from typing import Any


# ── Email + phone regex ─────────────────────────────────────────────────────

EMAIL_RX = re.compile(r"^[\w\.\+\-]+@[\w\-]+\.[\w\.\-]+$")
PHONE_OK_RX = re.compile(r"^[\+\d\(][\d\s\-\(\)\.]{6,}\d$")

# Indian mobile detection — PDF column extraction sometimes drops "+91"
_INDIAN_MOBILE_TEN = re.compile(r"^(\+)?([6-9]\d{9})$")


def normalize_phone(val: str | None) -> str | None:
    """Patch up Indian-mobile country code if the PDF dropped it.

    '+7985883353'  → '+91-7985883353'   (10 digits after '+' → assume Indian)
    '+91 9876 543210' → '+91-9876543210' (canonicalize separator)
    """
    if not val or not isinstance(val, str):
        return val
    digits_only = re.sub(r"[^\d\+]", "", val)
    m = _INDIAN_MOBILE_TEN.match(digits_only)
    if m:
        return f"+91-{m.group(2)}"
    m2 = re.match(r"^\+?91[\s\-]?([6-9]\d{9})$", digits_only)
    if m2:
        return f"+91-{m2.group(1)}"
    return val.strip()


# ── Fake-default values to strip ────────────────────────────────────────────

# Fields that derive.py already evidence-grounded — skip strict value
# grounding for these (entity's `evidence` quote was already verified upstream).
_DERIVE_GROUNDED_FIELDS = {
    "current_company", "current_role",
    "previous_companies", "college_name", "degree",
}


FAKE_VALUES = {
    "name": {"john doe", "jane doe", "candidate name", "your name", "n/a", "na", "none"},
    "email": {
        "example@example.com", "name@email.com", "candidate@email.com",
        "email@example.com",
    },
    "phone": {"+1 234 567 8900", "1234567890", "000-000-0000", "+91 0000000000"},
    "current_company": {"company name", "current employer", "n/a", "tbd"},
    "current_role": {"job title", "your role", "current role", "n/a"},
}


# ── Grounding ───────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _check_grounded(val: str, source_norm: str) -> str | None:
    """Verbatim-grounding: value (case+whitespace normalized) must appear in
    source. For pipe-separated multi-value strings, check each piece — drop
    only the pieces that don't ground, keep the rest."""
    if not val:
        return None
    if "|" in val:
        pieces = [p.strip() for p in val.split("|") if p.strip()]
        kept = [p for p in pieces if _norm(p) in source_norm]
        dropped = [p for p in pieces if _norm(p) not in source_norm]
        for d in dropped:
            print(f"[validate] ungrounded piece dropped: {d!r}")
        return " | ".join(kept) if kept else None
    if _norm(val) in source_norm:
        return val
    print(f"[validate] ungrounded value dropped: {val!r}")
    return None


# ── Public ──────────────────────────────────────────────────────────────────

def validate(row: dict, source_text: str, schema: dict) -> dict:
    """Apply final field-level cleanup. Does not mutate the input.

    Most cross-field invariants (current ∉ previous, college[i]↔degree[i],
    experience_years sane) are guaranteed upstream by derive.py — this layer
    just enforces field-individual rules."""
    out = dict(row)
    source_norm = _norm(source_text)

    for field_name, field_def in schema["fields"].items():
        val = out.get(field_name)

        # Empty-string → None
        if isinstance(val, str) and not val.strip():
            val = None

        # Strip known fake defaults
        if isinstance(val, str):
            fakes = FAKE_VALUES.get(field_name)
            if fakes and val.strip().lower() in fakes:
                print(f"[validate] strip fake default for {field_name}: {val!r}")
                val = None

        # Pattern validation (email)
        if field_def.get("pattern") == "email" and isinstance(val, str):
            if not EMAIL_RX.match(val.strip()):
                print(f"[validate] invalid email dropped: {val!r}")
                val = None

        # Range checks (experience_years)
        rng = field_def.get("range")
        if rng and val is not None:
            try:
                n = float(val)
                if not (rng[0] <= n <= rng[1]):
                    print(f"[validate] out-of-range {field_name}={val} not in {rng}")
                    val = None
            except (ValueError, TypeError):
                pass

        # Verbatim grounding — only for fields not already evidence-grounded
        # upstream in derive.py. derive.py validates the entity's `evidence`
        # quote in the source; once that's confirmed, the entry's parsed
        # fields (company, title, institution, degree) are trusted as
        # legitimate paraphrases.
        if (
            field_def.get("grounded")
            and isinstance(val, str)
            and field_name not in _DERIVE_GROUNDED_FIELDS
        ):
            val = _check_grounded(val, source_norm)

        out[field_name] = val

    # Phone normalization (after the above pipeline so it runs on a non-None value)
    if isinstance(out.get("phone"), str):
        normalized = normalize_phone(out["phone"])
        if normalized != out["phone"]:
            print(f"[validate] phone normalized: {out['phone']!r} -> {normalized!r}")
            out["phone"] = normalized

    return out
