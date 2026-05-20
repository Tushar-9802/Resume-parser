"""Hybrid extractor — rules-v2 as Layer 1, grounded LLM (resume_parser) as Layer 2.

Decision logic:
  Layer 1 (rules-v2) always runs first — fast, deterministic, no hallucination.
  Layer 2 (LLM) runs ONLY IF the resume needs rescue:
    (a) Tier 4 employment / Tier 5 projects / certifications / languages — fields
        rules-v2 cannot structure at all.
    (b) Tier 1-3 fields rules-v2 returned null AND we expect content
        (heuristic — see _needs_rescue).

  Merge: rules wins on every field it filled (deterministic > model). LLM fills
  the rest. Each field is tagged with _layer for traceability.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from . import rules_extract as rules
from .student_pipeline import process_resume_student
from .senior_pipeline import process_resume_senior
from .derive import _extract_stated_years
from .sections import split_sections


DEFAULT_MODEL = "gemma4:e4b-it-q4_K_M"

# The 9 typed skill buckets rules-v2 emits.
_SKILL_BUCKETS = (
    'programmingLanguages', 'frameworks', 'libraries', 'databases',
    'cloudPlatforms', 'devopsTools', 'aiTools', 'softwareApplications',
    'domainSkills',
)


# Fields that rules-v2 CANNOT structure — always rescue from Layer 2.
LAYER2_ONLY_FIELDS = {
    'employment', 'projects', 'certifications', 'languagesKnown', 'internships',
    'achievements', 'targetRoles', 'preferredLocations', 'experienceYears',
    'careerVision', 'summary', 'currentCtc', 'expectedCtc', 'dateOfBirth',
}

# Fields rules-v2 attempts. Trigger rescue if any of these is null AND content
# is plausibly in the PDF (rough heuristic — anchor tokens present).
TIER123_RESCUE_TRIGGERS = {
    'name': lambda j, t: '@' in t,  # if there's an email, there should be a name
    'phone': lambda j, t: bool(__import__('re').search(r'\b[6-9]\d{9}\b', t)),
    'city': lambda j, t: 'india' in t[:2000].lower() or any(s in t[:3000].lower() for s in ('mumbai','delhi','bangalore','chennai','hyderabad','kolkata','pune')),
    'degree': lambda j, t: any(d in t.lower() for d in ('b.tech','m.tech','bca','mca','b.e.','bachelor','master')),
}


def _needs_layer2(layer1_out: dict, full_text: str) -> tuple[bool, list[str]]:
    """Return (should_call_layer2, fields_to_rescue_or_add)."""
    rescue: list[str] = []
    # Tier 4-5 always need Layer 2 unless Layer 1 marked OCR-only failure.
    if layer1_out.get('_textLayerExists') or layer1_out.get('_extractionMethod', '').endswith('+ocr'):
        rescue.extend(sorted(LAYER2_ONLY_FIELDS))
    # Trigger-based rescue for Tier 1-3.
    for field, trigger in TIER123_RESCUE_TRIGGERS.items():
        if layer1_out.get(field) is None:
            try:
                if trigger(layer1_out, full_text):
                    rescue.append(field)
            except Exception:
                pass
    return (len(rescue) > 0, rescue)


def _read_text(pdf_path: Path) -> str:
    """Cheap text-only read for the trigger heuristics."""
    import fitz
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(p.get_text("text") for p in doc)
        doc.close()
        return text
    except Exception:
        return ''


# ── Seniority router (deterministic — Python decides, never the LLM) ─────────
# Routing which Layer-2 pipeline to run is a business-rule decision: it must be
# reproducible, auditable, and free. The signals below need no comprehension —
# page count, a literal "X years experience" string, and counting employment
# date-ranges are all deterministic. An LLM router would add a call, be
# non-deterministic, and hide its reasoning. So: Python.

_DATE_RANGE_RE = __import__('re').compile(
    r'\b(?:19|20)\d{2}\b\s*(?:[-–—]|to|\bto\b)\s*'
    r'(?:(?:19|20)\d{2}|present|current|till\s*date|now|ongoing)',
    __import__('re').IGNORECASE,
)


def _count_employment_dateranges(full_text: str) -> int:
    """Count date-ranges that belong to employment — a proxy for employer count.

    Scoped to the experience section when one is detected. When heading
    detection misses (dense 1-page senior CVs, unconventional headings), fall
    back to a whole-text count minus the education section's ranges (degrees
    carry year-ranges too) — no cap, so a dense multi-employer CV still scores."""
    try:
        sections = split_sections(full_text)
    except Exception:
        sections = {}
    exp = sections.get('experience') or ''
    if exp.strip():
        return len(_DATE_RANGE_RE.findall(exp))
    # No experience section detected. Count date-ranges in the full text minus
    # every section that is NOT employment — education, projects, positions,
    # achievements, certifications all carry date-ranges that are not jobs. A
    # campus fresher with 3 dated projects must not be counted as senior here.
    total = len(_DATE_RANGE_RE.findall(full_text))
    non_emp = 0
    for key in ('education', 'projects', 'positions', 'achievements',
                'certifications', 'publications', 'coursework'):
        body = sections.get(key) or ''
        if body.strip():
            non_emp += len(_DATE_RANGE_RE.findall(body))
    return max(0, total - non_emp)


def _classify_seniority(layer1: dict, full_text: str) -> tuple[str, str]:
    """Return (route, reason). route in {'senior', 'student'}.

    Senior when the resume clearly belongs to an experienced multi-employer
    candidate; otherwise student (the common Pulse case). Conservative toward
    'student' — misrouting a senior to the student pipeline collapses
    employment[]; misrouting a fresher to senior drops projects[]. Both are
    lossy, so route 'senior' only on a clear signal: a literal experience
    statement, or 3+ employment date-ranges (freshers rarely have 3+ jobs)."""
    page_count = layer1.get('_pdfPageCount') or 1
    stated = _extract_stated_years(full_text)
    emp_ranges = _count_employment_dateranges(full_text)

    signals: list[str] = []
    if stated is not None and stated > 2:
        signals.append(f"resume states ~{stated}y experience")
    if emp_ranges >= 3:
        signals.append(f"{emp_ranges} employment date-ranges")

    if signals:
        return 'senior', f"{'; '.join(signals)} ({page_count}p)"
    return 'student', (
        f"{page_count}p, {emp_ranges} employment date-range(s), "
        f"stated experience={stated}"
    )


# Pulse → gold field-name mapping. Pulse uses a flatter schema than gold; we map
# what's safe, leave the rest as nested Pulse blobs under metadata.
PULSE_TO_GOLD = {
    'name': 'name', 'phone': 'phone', 'email': 'email',
    'linkedinUrl': 'linkedinUrl', 'portfolioUrl': 'portfolioUrl',
    'city': 'city', 'state': 'state',
    'degree': 'degree', 'branch': 'branch',
    'cgpa': 'cgpa', 'cgpaScale': 'cgpaScale',
    'graduationYear': 'graduationYear',
    'dateOfBirth': 'dateOfBirth',
    'certifications': 'certifications',
    'languagesKnown': 'languagesKnown',
    'targetRoles': 'targetRoles', 'preferredLocations': 'preferredLocations',
    'projects': 'projects',
    'internships': 'internships',  # structured array (student_pipeline reshape)
}

# Senior-pipeline field -> gold field. The senior record emits a structured
# employment[] array (with nested clientsServed[]) plus experience fields the
# student schema has no slot for. rules-v2 still wins name/phone/email/degree/
# college (it filled them first); employment/experienceYears are Layer-2-only.
SENIOR_TO_GOLD = {
    'name': 'name', 'phone': 'phone', 'email': 'email',
    'degree': 'degree', 'college': 'college',
    'employment': 'employment',
    'projects': 'projects',
    'experienceYears': 'experienceYears',
    'currentCompany': 'currentCompany',
    'currentRole': 'currentRole',
}


def _merge(layer1: dict, layer2: dict | None, rescued_fields: list[str],
           layer2_map: dict[str, str]) -> dict:
    """Merge Layer 1 (rules) with Layer 2 (LLM rescue).
    Rule: Layer 1 always wins for any field it filled (non-null, non-empty).
    Layer 2 fills null fields + Layer-2-only fields. Each merged field tagged.
    `layer2_map` is the field-name mapping for the routed Layer-2 pipeline
    (PULSE_TO_GOLD for student, SENIOR_TO_GOLD for senior)."""
    out = dict(layer1)
    field_provenance: dict[str, str] = {}
    for f, v in layer1.items():
        if f.startswith('_'):
            continue
        is_filled = v is not None and (not isinstance(v, list) or len(v) > 0)
        if is_filled:
            field_provenance[f] = 'rules_v2'

    if layer2:
        for pulse_field, gold_field in layer2_map.items():
            l2_val = layer2.get(pulse_field)
            if l2_val is None or (isinstance(l2_val, list) and len(l2_val) == 0):
                continue
            l1_val = out.get(gold_field)
            l1_filled = l1_val is not None and (not isinstance(l1_val, list) or len(l1_val) > 0)
            if not l1_filled:
                out[gold_field] = l2_val
                field_provenance[gold_field] = 'llm_layer2'

    # Unified skills list. rules-v2 fills 9 typed buckets from a CS-leaning
    # dictionary — sparse on mechanical / civil / chemical resumes. The Layer-2
    # LLM also extracts a flat skill list that catches the domain terms the
    # dictionary misses (FEA, CFD, GD&T...). Union both so none are dropped;
    # the typed buckets stay as-is for JD-matching.
    if layer2:
        bucket_skills = [s for b in _SKILL_BUCKETS for s in (out.get(b) or [])]
        llm_skills = (layer2.get('skills') or []) + (layer2.get('toolsUsed') or [])
        unified, seen = [], set()
        for s in bucket_skills + llm_skills:
            if isinstance(s, str) and s.strip() and s.strip().lower() not in seen:
                seen.add(s.strip().lower())
                unified.append(s.strip())
        if unified:
            out['skills'] = unified
            field_provenance['skills'] = 'rules_v2+llm_layer2'

    out['_provenance'] = field_provenance
    out['_rescuedFields'] = rescued_fields
    out['_extractionMethod'] = layer1.get('_extractionMethod', 'rules_only_v2') + ('+layer2' if layer2 else '')
    return out


def extract(pdf_path: Path, *, model: str = DEFAULT_MODEL, force_layer2: bool = False) -> dict:
    t_start = time.time()
    # Layer 1: rules-v2.
    t1 = time.time()
    layer1 = rules.extract(pdf_path)
    t_l1 = time.time() - t1

    # Decide on Layer 2.
    full_text = _read_text(pdf_path)
    if force_layer2:
        need_l2, rescue_fields = True, sorted(LAYER2_ONLY_FIELDS)
    else:
        need_l2, rescue_fields = _needs_layer2(layer1, full_text)

    # Route which Layer-2 pipeline handles this resume (deterministic).
    route, route_reason = _classify_seniority(layer1, full_text)
    layer2_map = SENIOR_TO_GOLD if route == 'senior' else PULSE_TO_GOLD

    layer2: dict | None = None
    t_l2 = 0.0
    if need_l2:
        print(f"  [router] {route}  ({route_reason})")
        t2 = time.time()
        try:
            if route == 'senior':
                layer2 = process_resume_senior(pdf_path, model)
            else:
                layer2 = process_resume_student(pdf_path, model)
        except Exception as e:
            print(f"  [layer2] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        t_l2 = time.time() - t2

    merged = _merge(layer1, layer2, rescue_fields, layer2_map)
    merged['_routing'] = {'decision': route, 'reason': route_reason}
    merged['_timing'] = {
        'layer1_s': round(t_l1, 2),
        'layer2_s': round(t_l2, 2),
        'total_s': round(time.time() - t_start, 2),
    }
    return merged


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: hybrid_extract.py <pdf_dir> <out_dir> [--model MODEL] [--force-layer2]", file=sys.stderr)
        return 2
    pdf_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = DEFAULT_MODEL
    force_l2 = False
    args = sys.argv[3:]
    while args:
        a = args.pop(0)
        if a == '--model' and args:
            model = args.pop(0)
        elif a == '--force-layer2':
            force_l2 = True

    pdfs = sorted(pdf_dir.glob('*.pdf'))
    print(f"[hybrid] {len(pdfs)} resumes; model={model}; force_layer2={force_l2}")
    n = 0
    total_l1 = total_l2 = 0.0
    n_l2 = 0
    for pdf in pdfs:
        try:
            data = extract(pdf, model=model, force_layer2=force_l2)
        except Exception as e:
            data = {'_sourceFile': pdf.name, '_error': f'{type(e).__name__}: {e}'}
        (out_dir / (pdf.stem + '.json')).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        timing = data.get('_timing', {})
        if timing.get('layer2_s', 0) > 0:
            n_l2 += 1
            total_l2 += timing['layer2_s']
        total_l1 += timing.get('layer1_s', 0)
        n += 1
        print(f"  [{n}/{len(pdfs)}] {pdf.name}  L1={timing.get('layer1_s',0)}s L2={timing.get('layer2_s',0)}s")
    print(f"\nWrote {n} files. Layer-2 invoked on {n_l2}/{n}. Total L1={total_l1:.1f}s, L2={total_l2:.1f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
