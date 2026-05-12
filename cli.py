"""
Resume parser CLI — supports two modes:

  RECRUITER MODE (default): produces the 12-col Cognavi CSV for experienced
  candidates. Single 'current company / previous companies / experience years'
  view, optimized for recruiters scanning long histories.

  STUDENT MODE (--fresher): produces a Pulse-shaped JSONL record per fresher
  candidate, with per-project arrays (title/description/skillsUsed/dates),
  flattened internship details, degree/branch enums, target roles.

Usage:
    python cli.py path/to/resume.pdf                       # recruiter mode, single file
    python cli.py resumes/*.pdf resumes/*.docx             # glob
    python cli.py resumes/ -o output/parsed.csv            # whole folder
    python cli.py resumes/ --model gemma4:e4b-it-q4_K_M    # pick model
    python cli.py resumes/ --jsonl out/log.jsonl           # resumable
    python cli.py resumes/ --force                         # ignore JSONL cache
    python cli.py resumes/ --fresher                       # student mode (Pulse schema)

Default model: gemma4:e4b-it-q4_K_M via Ollama (local). qwen2.5:7b-instruct
was A/B tested early and rejected — it hallucinated client names as
employers and refused to fill current_company for ended internships.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows — resumes can contain bullets (○, •),
# em-dashes, ligatures, etc. that the default cp1252 console encoding can't
# print, which crashes the whole process mid-run.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from src.pipeline import process_resume
from src.csv_writer import load_seen, append_row, write_csv
from src.student_pipeline import process_resume_student
from src.student_writer import write_summary_csv, write_projects_csv


SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md"}


def collect_inputs(paths: list[str]) -> list[Path]:
    """Expand args (files, dirs, globs) into a sorted list of supported files."""
    out: list[Path] = []
    seen: set[Path] = set()
    for arg in paths:
        p = Path(arg)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTS:
                    if child not in seen:
                        out.append(child)
                        seen.add(child)
        elif p.is_file():
            if p.suffix.lower() in SUPPORTED_EXTS and p not in seen:
                out.append(p)
                seen.add(p)
        else:
            # Try as a glob relative to CWD
            for match in sorted(Path(".").glob(arg)):
                if match.is_file() and match.suffix.lower() in SUPPORTED_EXTS and match not in seen:
                    out.append(match)
                    seen.add(match)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Parse resumes (PDF/DOCX) via Ollama. Recruiter mode by default; --fresher for student mode.")
    ap.add_argument("inputs", nargs="+", help="Files, directories, or globs")
    ap.add_argument("-o", "--output", default=None, help="Output CSV path (default depends on mode)")
    ap.add_argument("--jsonl", default=None, help="Resumable JSONL log path (default depends on mode)")
    ap.add_argument("--model", default="gemma4:e4b-it-q4_K_M", help="Ollama model tag")
    ap.add_argument("--force", action="store_true", help="Re-process files already in JSONL")
    ap.add_argument("--fresher", action="store_true", help="Student / fresher mode (Pulse schema). Default output paths point to out/student_extract/.")
    args = ap.parse_args()

    # Mode-specific defaults — picked here so a `--fresher` invocation with no
    # explicit --output writes to the student dir, not the recruiter one.
    if args.fresher:
        output = args.output or "out/student_extract/summary.csv"
        jsonl = args.jsonl or "out/student_extract/parsed.jsonl"
        process_fn = process_resume_student
    else:
        output = args.output or "out/parsed.csv"
        jsonl = args.jsonl or "out/parsed.jsonl"
        process_fn = process_resume

    inputs = collect_inputs(args.inputs)
    if not inputs:
        print("[cli] no supported files found in inputs", file=sys.stderr)
        return 1

    seen = set() if args.force else load_seen(jsonl)
    todo = [p for p in inputs if str(p) not in seen]
    skipped = len(inputs) - len(todo)
    if skipped:
        print(f"[cli] skipping {skipped} already-processed files (use --force to redo)")

    print(f"[cli] mode: {'student' if args.fresher else 'recruiter'}")
    print(f"[cli] model: {args.model}")
    print(f"[cli] processing {len(todo)} resume(s)")

    n_ok = 0
    n_err = 0
    for i, path in enumerate(todo, 1):
        print(f"\n--- [{i}/{len(todo)}] {path.name} ---")
        try:
            row = process_fn(path, args.model)
            append_row(jsonl, row)
            n_ok += 1
        except Exception as e:
            print(f"[cli] ERROR processing {path.name}: {e}")
            traceback.print_exc()
            n_err += 1

    print(f"\n[cli] done: {n_ok} ok, {n_err} errors")
    if args.fresher:
        written = write_summary_csv(jsonl, output)
        print(f"[cli] wrote {written} candidate summary rows to {output}")
        # Companion projects CSV — one row per project per candidate. Sits
        # next to the summary CSV under the same out dir.
        projects_csv = str(Path(output).with_name("projects.csv"))
        n_projects = write_projects_csv(jsonl, projects_csv)
        print(f"[cli] wrote {n_projects} project rows to {projects_csv}")
        print(f"[cli] full per-candidate records in {jsonl}")
    else:
        written = write_csv(jsonl, output)
        print(f"[cli] wrote {written} rows to {output}")
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
