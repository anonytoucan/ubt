#!/usr/bin/env python3
"""Transcribe prose sentences of the paper's main sections to UEB grade 2 as eval records for the round-trip check.

Only sections that main.tex inputs are read. LaTeX is stripped, sentences with math, digits, cross-references or markup
are dropped, and a sentence is kept only if f (en-ueb-g2.ctb, same LibLouis build) is defined on it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ubt.translate import ForwardTranslator  # noqa: E402

MATH = "\x01"   # sentinel: sentence contained mathematics
XREF = "\x02"   # sentinel: sentence contained a cross-reference

TABLE = "en-ueb-g2.ctb"
MIN_CHARS = 60
MAX_CHARS = 320

# macros replaced by their prose argument
KEEP_ARG = ("emph", "textbf", "textit", "texttt", "text", "brc")   # \brc{1,2}: a drawn cell, kept as its dots
# macros dropped with their arguments
DROP_CALL = ("label", "citep", "citet", "cite", "input", "vspace",
             "hspace", "footnote", "TODO", "mainsec", "maintab")


def strip_latex(src: str) -> str:
    """Reduce a .tex section body to plain prose, conservatively."""
    s = src
    s = re.sub(r"(?<!\\)%.*", "", s)                       # comments
    s = re.sub(r"\\begin\{(table|table\*|figure|figure\*|tabular|tabularx|"
               r"tabular\*|equation|align|aligned|proposition|assumption|"
               r"proof|remark|corollary|itemize|enumerate)\}.*?"
               r"\\end\{\1\}", " ", s, flags=re.S)          # float/math blocks
    s = re.sub(r"\\\[.*?\\\]", MATH, s, flags=re.S)         # display math
    s = re.sub(r"\$([A-Za-z])\$", r"\1", s)                  # a one-letter variable stays as its letter (best-of-$n$)
    s = re.sub(r"\$[^$]*\$", MATH, s)                        # inline math
    s = re.sub(r"\\S(?![a-zA-Z])", XREF, s)                   # section sign
    s = re.sub(r"\\(ref|eqref|autoref|Cref|cref)\{[^}]*\}", XREF, s)
    s = re.sub(r"\\suppsec\{[^}]*\}\{([^}]*)\}", XREF, s)
    for m in DROP_CALL:
        s = re.sub(r"\\" + m + r"\*?(\[[^\]]*\]){0,2}\{[^{}]*\}", " ", s)   # up to two optional args: \citep[e.g.,][]{k}
    for m in KEEP_ARG:
        s = re.sub(r"\\" + m + r"\*?\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(section|subsection|subsubsection|paragraph)\*?\{[^{}]*\}",
               " ", s)
    s = re.sub(r"\\[a-zA-Z@]+\*?", " ", s)                  # leftover macros
    s = s.replace("~", " ").replace("``", '"').replace("''", '"')
    s = s.replace("---", "\u2014").replace("--", "\u2013")   # LaTeX dashes as rendered
    s = re.sub(r"[{}$\\]", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;:!?)])", r"\1", s)      # citation removal leaves " ,"
    s = re.sub(r"([(])\s+", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


ABBREV = ("Supp", "Prop", "Props", "Eq", "Eqs", "Fig", "Figs", "Tab", "Ref",
          "Sec", "App", "cf", "e.g", "i.e", "vs", "et al", "approx", "No")


def split_sentences(text: str) -> list[str]:
    """Split on sentence-final punctuation, protecting known abbreviations."""
    guard = {}
    for i, a in enumerate(ABBREV):
        tok = f"\x00{i}\x00"
        guard[tok] = a + "."
        text = re.sub(r"\b" + re.escape(a) + r"\.", tok, text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"(])", text)
    out = []
    for p in parts:
        for tok, orig in guard.items():
            p = p.replace(tok, orig)
        p = p.strip()
        if p:
            out.append(p)
    return out


def is_prose(sent: str) -> tuple[bool, str]:
    """Reject anything that is not clean running prose."""
    if not (MIN_CHARS <= len(sent) <= MAX_CHARS):
        return False, "length"
    if not sent.endswith((".", "!", "?")):
        return False, "unterminated"
    if not sent[0].isupper() and sent[0] != '"':
        return False, "no-capital"
    if MATH in sent:
        return False, "mathematics"
    if XREF in sent:
        return False, "cross-reference"
    if re.search(r"[\\{}$^_|<>~@#]", sent):
        return False, "markup-residue"
    if re.search(r"\d+\s*[%×/]|\d\.\d|\brightarrow\b|\bleq\b|\bgeq\b", sent):
        return False, "math-or-figures"
    # any digit: numbers follow a different rule family and the set is prose-only
    if re.search(r"\d", sent):
        return False, "digits"
    if re.search(r"\b(Table|Figure|Supp|Prop|Props|Eq|Eqs|Assumption|Rung|"
                 r"Sec|App)\b", sent):
        return False, "cross-reference"
    if sent.count('"') % 2:
        return False, "unbalanced-quote"
    return True, ""


def transcribe(tr, text: str) -> str | None:
    """Forward-transcribe with f; None when f is undefined on the input, as the generator rejects it."""
    res = tr.translate(TABLE, text, check_undefined=True)
    return res.braille if res.ok else None


def bucket(n_cells: int) -> str:
    if n_cells < 50:
        return "B1"
    if n_cells < 100:
        return "B2"
    if n_cells < 200:
        return "B3"
    return "B4"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", required=True,
                    help="directory holding _sections/ and main.tex")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="paper3")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    root = Path(args.paper_dir)
    main_tex = (root / "main.tex").read_text(encoding="utf-8")
    inputs = re.findall(r"^\s*\\input\{_sections/([^}]+)\}", main_tex, re.M)
    if not inputs:
        print("no \\input{_sections/...} found in main.tex", file=sys.stderr)
        return 1

    tr = ForwardTranslator()
    recs, rejects = [], {}
    for stem in inputs:
        path = root / "_sections" / (stem if stem.endswith(".tex") else stem + ".tex")
        if not path.exists():
            print(f"  skip (missing): {path.name}", file=sys.stderr)
            continue
        prose = strip_latex(path.read_text(encoding="utf-8"))
        for sent in split_sentences(prose):
            ok, why = is_prose(sent)
            if not ok:
                rejects[why] = rejects.get(why, 0) + 1
                continue
            cells = transcribe(tr, sent)
            if cells is None:
                rejects["untranscribable"] = rejects.get("untranscribable", 0) + 1
                continue
            n_cells = sum(1 for c in cells if c != " ")
            recs.append({
                "id": f"{args.prefix}-{len(recs):05d}",
                "text": sent,
                "braille": cells,
                "tables": [TABLE],
                "tag": "core_per_code",
                "cells": n_cells,
                "doc_type": "single",
                "k": 1,
                "langs": ["en"],
                "len_bucket": bucket(n_cells),
                "n_sentences": 1,
                "regime": "single",
                "segments": [{"table": TABLE, "text": sent}],
                "switch_density": 0.0,
                "confusable_set": [],
                "source": path.name,
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    by_src = Counter(r["source"] for r in recs)
    print(f"kept {len(recs)} sentences -> {out}")
    for k, v in sorted(by_src.items()):
        print(f"    {k:28s} {v}")
    print("  rejected:", dict(sorted(rejects.items(), key=lambda kv: -kv[1])))

    if args.report:
        Path(args.report).write_text(json.dumps(
            {"n": len(recs), "by_source": dict(by_src), "rejected": rejects,
             "table": TABLE, "liblouis": tr.louis_version()},
            indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
