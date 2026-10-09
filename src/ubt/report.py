"""Generation reporting: generation_report.json + per_table_stats.csv."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from collections import Counter


def repo_git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def louis_version() -> str:
    try:
        import louis  # noqa: PLC0415
        return louis.version()
    except Exception:
        return "unknown"


def write_report(
    out_dir: str,
    cfg: dict,
    counts: dict,
    rejections: Counter,
    per_table_accepted: Counter,
    per_table_rejected: dict[str, Counter],
    extra: dict | None = None,
) -> dict:
    report = {
        "louis_version": louis_version(),
        "liblouis_pin": cfg["liblouis_pin"],
        "ubt_git_sha": repo_git_sha(),
        "source_spec": {
            "train": "flores200 dev (tail) + wiki_txt_A (head)",
            "eval": "flores200 devtest (tail) + wiki_txt_B (eval)",
            "head_langs": cfg["head_langs"],
        },
        "counts": counts,
        "rejection_reasons": dict(rejections.most_common()),
        **(extra or {}),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "generation_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "per_table_stats.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        reasons = sorted({r for c in per_table_rejected.values() for r in c})
        w.writerow(["table", "accepted"] + [f"rej_{r}" for r in reasons])
        all_tables = sorted(set(per_table_accepted) | set(per_table_rejected))
        for t in all_tables:
            rej = per_table_rejected.get(t, Counter())
            w.writerow([t, per_table_accepted.get(t, 0)] + [rej.get(r, 0) for r in reasons])
    return report
