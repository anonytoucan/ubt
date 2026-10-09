#!/usr/bin/env python3
"""Build the braille edition of the paper.

  <out>.brf        Braille Ready Format, 40 cells x 25 lines, for an embosser
  <out>.cells.json Unicode cells per line, each with its source character
                   index, so a decode can be projected back onto cells
  <out>.text.txt   the transcribed plain text

The cell -> source map is LibLouis's own position array, the same
correspondence bCER uses.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path[:0] = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "*", "")))

import louis  # noqa: E402

from extract_paper_sentences import strip_latex  # noqa: E402

TABLE = "en-ueb-g2.ctb"
CELLS_PER_LINE = 40
LINES_PER_PAGE = 25


def translate_with_map(text: str) -> tuple[str, str, list[int]]:
    """Return (ascii_braille, unicode_braille, inpos); inpos[i] is the index in
    `text` of the character that produced cell i."""
    # translate() returns the position arrays; translateString() does not.
    uni, inpos, _outpos, _cursor = louis.translate(
        ["unicode.dis", TABLE], text, mode=0)
    asc, _, _, _ = louis.translate([TABLE], text, mode=0)
    if len(asc) != len(uni):
        # Truncate rather than mis-pair cells.
        n = min(len(asc), len(uni))
        asc, uni, inpos = asc[:n], uni[:n], inpos[:n]
    return asc, uni, list(inpos[: len(uni)])


def wrap_cells(uni: str, asc: str, inpos: list[int], width: int):
    """Break the cell stream into lines at spaces, never mid-word."""
    lines = []
    i, n = 0, len(uni)
    while i < n:
        j = min(i + width, n)
        if j < n and uni[j] != "⠀":
            k = uni.rfind("⠀", i, j)
            if k > i:
                j = k
        seg = list(range(i, j))
        # strip the leading space of the next line
        while j < n and uni[j] == "⠀":
            j += 1
        while seg and uni[seg[-1]] == "⠀":
            seg.pop()
        if seg:
            lines.append({
                "uni": "".join(uni[t] for t in seg),
                "ascii": "".join(asc[t] for t in seg),
                "src": [inpos[t] for t in seg],
                "cell0": seg[0],
            })
        i = j if j > i else i + 1
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", required=True)
    ap.add_argument("--out", required=True, help="path prefix")
    ap.add_argument("--width", type=int, default=CELLS_PER_LINE)
    ap.add_argument("--page-lines", type=int, default=LINES_PER_PAGE)
    args = ap.parse_args()

    root = Path(args.paper_dir)
    main_tex = (root / "main.tex").read_text(encoding="utf-8")
    inputs = re.findall(r"^\s*\\input\{_sections/([^}]+)\}", main_tex, re.M)

    chunks = []
    for stem in inputs:
        p = root / "_sections" / (stem if stem.endswith(".tex") else stem + ".tex")
        if not p.exists():
            continue
        prose = strip_latex(p.read_text(encoding="utf-8"))
        # sentinels mark maths / cross-references: drop them, keep the prose
        prose = re.sub(r"[\x01\x02]", " ", prose)
        prose = re.sub(r"\s+", " ", prose).strip()
        if prose:
            chunks.append((p.stem, prose))

    text = "\n\n".join(c[1] for c in chunks)
    asc, uni, inpos = translate_with_map(text)
    lines = wrap_cells(uni, asc, inpos, args.width)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- BRF: ASCII braille, form-feed between pages ---
    brf = []
    for pi in range(0, len(lines), args.page_lines):
        page = lines[pi: pi + args.page_lines]
        brf.append("\r\n".join(l["ascii"] for l in page))
    Path(str(out) + ".brf").write_text("\r\n\f\r\n".join(brf) + "\r\n",
                                       encoding="ascii", errors="replace")

    Path(str(out) + ".text.txt").write_text(text, encoding="utf-8")
    Path(str(out) + ".cells.json").write_text(json.dumps({
        "table": TABLE, "liblouis": louis.version(),
        "width": args.width, "page_lines": args.page_lines,
        "n_cells": len(uni), "n_lines": len(lines),
        "n_pages": (len(lines) + args.page_lines - 1) // args.page_lines,
        "sections": [c[0] for c in chunks],
        "text_len": len(text),
        "lines": lines,
    }, ensure_ascii=False), encoding="utf-8")

    print(f"text  {len(text):,} chars over {len(chunks)} sections")
    print(f"cells {len(uni):,}  lines {len(lines):,}  "
          f"pages {(len(lines)+args.page_lines-1)//args.page_lines}")
    print(f"wrote {out}.brf / .cells.json / .text.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
