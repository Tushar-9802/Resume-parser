"""
CSV + JSONL writer.

Resumable-pipeline pattern (per CLAUDE.md):
  - JSONL is the source of truth (append-only, flush per row, survives crash)
  - CSV is regenerated from JSONL at end of run (or rebuilt anytime)
  - A `seen` set keyed on _source_file lets re-runs skip already-processed files

If only CSV is wanted, JSONL becomes a sidecar that the caller can ignore.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .extractor import load_schema


# ── Resume state ────────────────────────────────────────────────────────────

def load_seen(jsonl_path: str | Path) -> set[str]:
    """Return the set of source-file paths already processed in this JSONL."""
    p = Path(jsonl_path)
    if not p.exists():
        return set()
    seen: set[str] = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = row.get("_source_file")
            if src:
                seen.add(src)
    return seen


# ── Append-only write ──────────────────────────────────────────────────────

def append_row(jsonl_path: str | Path, row: dict) -> None:
    """Append one row to JSONL with immediate flush (survives crash)."""
    p = Path(jsonl_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


# ── JSONL → CSV ─────────────────────────────────────────────────────────────

def write_csv(jsonl_path: str | Path, csv_path: str | Path) -> int:
    """Rebuild CSV from the full JSONL log. Column order = schema.yaml order,
    using each field's 'column' label as the header. Returns rows written."""
    schema = load_schema()
    field_names = list(schema["fields"].keys())
    column_labels = [schema["fields"][f].get("column", f) for f in field_names]

    rows: list[dict] = []
    jp = Path(jsonl_path)
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
        writer.writerow(column_labels + ["Source File"])
        for row in rows:
            csv_row = [_format_cell(row.get(fn)) for fn in field_names]
            csv_row.append(row.get("_source_file", ""))
            writer.writerow(csv_row)

    return len(rows)


def _format_cell(val) -> str:
    """None → empty cell. Anything else → str()."""
    if val is None:
        return ""
    if isinstance(val, float):
        # 7.0 → "7" for integer-valued floats (cleaner CSV)
        return str(int(val)) if val == int(val) else str(val)
    return str(val)
