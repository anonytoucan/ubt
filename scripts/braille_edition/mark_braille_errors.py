#!/usr/bin/env python3
"""Project round-trip decoding errors onto individual braille cells.

Inputs: the cell map from make_braille_edition.py, the decoding units and the
eval output. Output: per-cell marks for render_braille_tikz.py:
  err   a source character of the cell was decoded wrongly
  unc   its unit failed certification, but the cell decoded correctly
Certification is re-checked by re-encoding the output through f.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ubt.translate import ForwardTranslator  # noqa: E402

import rapidfuzz  # noqa: E402

TABLE = "en-ueb-g2.ctb"
# distinctions UEB does not encode: no decoder can recover them from cells
NORM = {"—": "-", "–": "-", "“": '"', "”": '"', "‘": "'", "’": "'", "…": "..."}


def norm(s: str) -> str:
    for a, b in NORM.items():
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def bad_source_offsets(ref: str, hyp: str) -> set[int]:
    """Indices in `ref` that the hypothesis got wrong."""
    ops = rapidfuzz.distance.Levenshtein.editops(norm(ref), norm(hyp))
    # map normalised offsets back proportionally; exact when lengths match
    nref = norm(ref)
    scale = (len(ref) / len(nref)) if nref else 1.0
    out = set()
    for op in ops:
        i = int(op.src_pos * scale)
        out.update(range(max(0, i - 1), min(len(ref), i + 2)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--results", required=True,
                    help="glob for eval json shards")
    ap.add_argument("--arm", default="bon_lambda5_rows")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    doc = json.loads(Path(args.cells).read_text(encoding="utf-8"))
    units = [json.loads(l) for l in open(args.units, encoding="utf-8")]
    by_id = {u["id"]: u for u in units}

    rows = {}
    for f in sorted(glob.glob(args.results)):
        d = json.load(open(f, encoding="utf-8"))
        for r in d.get(args.arm, []):
            rows[r["id"]] = r

    tr = ForwardTranslator()
    bad_src: set[int] = set()
    unc_src: set[int] = set()
    n_unit = n_cert = n_exact = 0
    for uid, u in by_id.items():
        r = rows.get(uid)
        if r is None:
            continue
        n_unit += 1
        hyp = r.get("hyp_text") or ""
        enc = tr.translate(TABLE, hyp, check_undefined=False)
        certified = bool(enc.braille) and enc.braille == u["braille"]
        exact = norm(hyp) == norm(u["text"])
        n_cert += certified
        n_exact += exact
        base = u["src_start"]
        if not exact:
            for off in bad_source_offsets(u["text"], hyp):
                bad_src.add(base + off)
        if not certified:
            unc_src.update(range(u["src_start"], u["src_end"]))

    marks: dict[str, str] = {}
    n_err = n_unc = 0
    for li, ln in enumerate(doc["lines"]):
        for ci, srci in enumerate(ln["src"]):
            if srci in bad_src:
                marks[f"{li}:{ci}"] = "err"
                n_err += 1
            elif srci in unc_src:
                marks[f"{li}:{ci}"] = "unc"
                n_unc += 1

    # the figure's claim: no error cell outside an uncertified span
    err_outside = sum(1 for li, ln in enumerate(doc["lines"])
                      for ci, srci in enumerate(ln["src"])
                      if srci in bad_src and srci not in unc_src)

    Path(args.out).write_text(json.dumps({
        "arm": args.arm,
        "summary": {
            "n_units": n_unit, "n_certified": n_cert, "n_exact": n_exact,
            "n_cells": doc["n_cells"], "n_err": n_err, "n_unc": n_unc,
            "err_cells_outside_uncertified": err_outside,
        },
        "cell_marks": marks,
    }, ensure_ascii=False), encoding="utf-8")

    print(f"units {n_unit}  certified {n_cert}  exact {n_exact}")
    print(f"cells {doc['n_cells']:,}  err {n_err}  uncertified {n_unc}")
    print(f"error cells outside an uncertified span: {err_outside}"
          + ("   <- certified errors, the failure mode the paper excludes"
             if err_outside else "   (none, as claimed)"))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
