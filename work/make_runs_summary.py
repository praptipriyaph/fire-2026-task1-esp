"""Derive a data-free comparison summary of the two submitted runs.

The submission files themselves cannot be published: their `fact`,
`explanation` keys and `reasoning_traces` all carry verbatim text from the
FIRE test set, which must be obtained from the PROSLEX authors. This script
reduces them to counts and predicted labels only, so the run-comparison
figures in the paper can be recomputed from the repository.

Emitted per case: a sequential label (cases stay in submission-file order),
the sections each run predicted, how many evidence sentences each run
attached, and each reasoning trace's length in words. No case text, no case
identifiers.

Usage:
    python make_runs_summary.py
"""

import json
from pathlib import Path

WORK = Path(__file__).resolve().parent
RUN1 = WORK / "PraptipriyaPhukon_Task1_Run1.jsonl"
RUN2 = WORK / "PraptipriyaPhukon_Task1_Run2.jsonl"
OUT = WORK / "runs_comparison_summary.json"


def load(path: Path) -> list:
    return [json.loads(l) for l in path.open() if l.strip()]


def sections(row: dict) -> list:
    """Union of sections across the row's explanation entries, sorted."""
    out = set()
    for secs in (row.get("explanation") or {}).values():
        out.update(secs if isinstance(secs, list) else [secs])
    return sorted(out)


def n_sentences(row: dict) -> int:
    return len(row.get("explanation") or {})


def n_words(row: dict) -> int:
    return len(str(row.get("reasoning_traces", "")).split())


def main() -> int:
    r1, r2 = load(RUN1), load(RUN2)
    if len(r1) != len(r2):
        raise SystemExit(f"run length mismatch: {len(r1)} vs {len(r2)}")

    cases = []
    for i, (a, b) in enumerate(zip(r1, r2), start=1):
        if a["id"] != b["id"]:
            raise SystemExit(f"case order mismatch at position {i}")
        cases.append({
            "case": f"case_{i:02d}",
            "run1_sections": sections(a),
            "run2_sections": sections(b),
            "run1_evidence_sentences": n_sentences(a),
            "run2_evidence_sentences": n_sentences(b),
            "run1_trace_words": n_words(a),
            "run2_trace_words": n_words(b),
        })

    n = len(cases)
    def mean(key):
        return round(sum(c[key] for c in cases) / n, 4)

    summary = {
        "n_cases": n,
        "identical_section_sets": sum(
            1 for c in cases if c["run1_sections"] == c["run2_sections"]),
        "mean_sections_per_case": {
            "run1": round(sum(len(c["run1_sections"]) for c in cases) / n, 4),
            "run2": round(sum(len(c["run2_sections"]) for c in cases) / n, 4),
        },
        "cases_predicting_302": {
            "run1": sum(1 for c in cases
                        if any("302" in s for s in c["run1_sections"])),
            "run2": sum(1 for c in cases
                        if any("302" in s for s in c["run2_sections"])),
        },
        "mean_evidence_sentences_per_case": {
            "run1": mean("run1_evidence_sentences"),
            "run2": mean("run2_evidence_sentences"),
        },
        "mean_trace_words_per_case": {
            "run1": mean("run1_trace_words"),
            "run2": mean("run2_trace_words"),
        },
    }

    OUT.write_text(json.dumps({
        "note": "Derived from the two submitted run files, which cannot be "
                "published because they embed verbatim FIRE test-set text. "
                "Counts and predicted labels only; no case text or "
                "identifiers. Cases are in submission-file order.",
        "runs": {
            "run1": "525-case model, CV-median thresholds, refined templates, "
                    "two evidence sentences per section",
            "run2": "ensemble of the 420-case and 525-case models, "
                    "development-split thresholds, one sentence per section",
        },
        "summary": summary,
        "per_case": cases,
    }, indent=2) + "\n")

    print(f"wrote {OUT}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
