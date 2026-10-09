#!/usr/bin/env python3
"""Render the braille edition as TikZ, with round-trip errors highlighted.

Yellow: the cell's decoding unit failed verification. Red: the cell's own source characters were decoded wrongly.
Every red cell should lie inside a yellow span; one outside would be a certified error.
The inkprint of each run of cells sharing a source position is set once below it.
Colours and TikZ styles come from the including preamble: dotink dotpale hlerr hlunc inkink inkerr doton dotoff
cellerr cellunc.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# dot -> (column, row) in the 2x3 cell, rows counted downward
DOT_POS = {1: (0, 0), 2: (0, 1), 3: (0, 2),
           4: (1, 0), 5: (1, 1), 6: (1, 2),
           7: (0, 3), 8: (1, 3)}

# Library of Congress geometry, in mm
DOT_DX = DOT_DY = 2.34      # dot spacing within a cell
CELL_DX = 6.20              # cell pitch
LINE_DY = 13.0              # line pitch (extra room for the inkprint)


def dots_of(ch: str) -> list[int]:
    v = ord(ch) - 0x2800
    return [i + 1 for i in range(8) if v >> i & 1]


def esc(t: str) -> str:
    """Escape a source fragment for use inside a TikZ node."""
    for a, b in (("\\", "/"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
                 ("—", "---"), ("–", "--"),
                 ("“", "``"), ("”", "''"),
                 ("‘", "`"), ("’", "'")):
        t = t.replace(a, b)
    return t


def render_page(lines, marks, text=None, r=0.75) -> str:
    """One page of cells as a tikzpicture body, in real braille proportions."""
    out = []
    for li, ln in enumerate(lines):
        y0 = -li * LINE_DY
        for ci, ch in enumerate(ln["uni"]):
            x0 = ci * CELL_DX
            m = marks.get((ln["line_index"], ci))
            if m in ("err", "unc"):
                style = "cellerr" if m == "err" else "cellunc"
                out.append(
                    f"\\fill[{style}]({x0-1.5},{y0+1.9}) rectangle "
                    f"({x0+DOT_DX+1.5},{y0-2*DOT_DY-1.9});")
            ds = dots_of(ch)
            for d in range(1, 7):
                cx, cy = DOT_POS[d]
                X, Y = x0 + cx * DOT_DX, y0 - cy * DOT_DY
                if d in ds:
                    out.append(f"\\fill[doton]({X},{Y}) circle({r});")
                else:
                    out.append(f"\\fill[dotoff]({X},{Y}) circle({r*0.34});")
        if text is None:
            continue
        src = ln["src"]
        i = 0
        while i < len(src):
            j = i
            while j + 1 < len(src) and src[j + 1] == src[i]:
                j += 1
            s0 = src[i]
            s1 = len(text)
            for k in range(j + 1, len(src)):
                if src[k] > s0:
                    s1 = src[k]
                    break
            else:
                nxt = ln.get("next_src")
                if nxt is not None and nxt > s0:
                    s1 = nxt
            frag = text[s0:s1].strip("\n")
            if frag.strip():
                xc = (i * CELL_DX + j * CELL_DX + DOT_DX) / 2.0
                any_err = any(marks.get((ln["line_index"], c)) == "err"
                              for c in range(i, j + 1))
                col = "inkerr" if any_err else "inkink"
                out.append(
                    f"\\node[font=\\tiny,text={col},inner sep=0pt,anchor=north]"
                    f" at ({xc},{y0-2*DOT_DY-2.6}) {{\\texttt{{{esc(frag)}}}}};")
            i = j + 1
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="paper_braille.cells.json")
    ap.add_argument("--marks", default=None, help="paper_braille_marks.json")
    ap.add_argument("--text", default=None,
                    help="paper_braille.text.txt; prints inkprint under cells")
    ap.add_argument("--out", required=True, help=".tex to write")
    ap.add_argument("--pages", default="1", help="e.g. 1,4-5 or 1-24")
    ap.add_argument("--height", type=float, default=0.86,
                    help="fraction of \\textheight a full page occupies")
    args = ap.parse_args()

    doc = json.loads(Path(args.cells).read_text(encoding="utf-8"))
    lines = doc["lines"]
    for i, ln in enumerate(lines):
        ln["line_index"] = i
        ln["next_src"] = lines[i + 1]["src"][0] if i + 1 < len(lines) else None
    text = Path(args.text).read_text(encoding="utf-8") if args.text else None

    marks, stats = {}, {}
    if args.marks:
        md = json.loads(Path(args.marks).read_text(encoding="utf-8"))
        for k, v in md["cell_marks"].items():
            li, ci = k.split(":")
            marks[(int(li), int(ci))] = v
        stats = md.get("summary", {})

    want = set()
    for part in args.pages.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            want.update(range(int(a), int(b) + 1))
        elif part:
            want.add(int(part))

    per = doc["page_lines"]
    body = []
    for pno in sorted(want):
        sel = lines[(pno - 1) * per: pno * per]
        if not sel:
            continue
        # scale every page by the same factor: a short final page ends early
        frac = args.height * len(sel) / per
        body.append(
            "\\begin{center}\n"
            f"\\resizebox{{!}}{{{frac:.4f}\\textheight}}{{%\n"
            "\\begin{tikzpicture}[x=1mm,y=1mm]\n"
            + render_page(sel, marks, text)
            + "\n\\end{tikzpicture}}\n"
            + f"\\\\[4pt]{{\\footnotesize Braille edition, page {pno} "
              f"of {doc['n_pages']}}}\n"
            "\\end{center}")

    Path(args.out).write_text(
        "% Colours and tikz styles are defined in the main preamble.\n"
        + "\n\\clearpage\n".join(body) + "\n", encoding="utf-8")
    print(f"wrote {args.out}  pages={sorted(want)}")
    if stats:
        print(f"  marked cells: err={stats.get('n_err',0)} "
              f"unc={stats.get('n_unc',0)} of {doc['n_cells']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
