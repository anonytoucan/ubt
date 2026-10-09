#!/usr/bin/env python3
"""Pair published (real) braille with liblouis forward braille, verse by verse, for 18 KJV books.

Input (--brf-dir): per book, a Braille Ready Format file of EBAE grade-2 cells and the inkprint verses as JSON. The
inline verse markers (`#<chapter>3<verse>`, `3` is the braille-ASCII colon) make the alignment exact. Each verse is
written with its forward translation under several tables.

  python3 scripts/real_corpora/bible_pairs.py --out work/bible_pairs/real_pairs.jsonl

Braille ASCII maps to cells through the 64-entry `en-us-brf.dis`; the 8-dot `text_nabcc.dis` would add dot 7 to some
cells.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

BRF_DIS = "en-us-brf.dis"
MARKER = re.compile(r"#([A-J]+)3([A-J]+)")
_DIGIT = {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5",
          "F": "6", "G": "7", "H": "8", "I": "9", "J": "0"}
DEFAULT_TABLES = ["en-us-g2.ctb", "en-ueb-g2.ctb", "en-us-g1.ctb"]


def find_dis(name: str = BRF_DIS) -> str:
    """Locate the display table on LOUIS_TABLEPATH (pinned dirs first)."""
    cands = [d for d in os.environ.get("LOUIS_TABLEPATH", "").split(",") if d]
    cands += ["/usr/local/share/liblouis/tables", "/usr/share/liblouis/tables"]
    for d in cands:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    raise RuntimeError(f"{name} not found on LOUIS_TABLEPATH")


def brf_maps(path: str | None = None) -> tuple[dict[str, str], dict[str, str]]:
    """(ascii -> cell, cell -> ascii) from the display table; 64 entries, so the inverse is well defined."""
    to_cell: dict[str, str] = {}
    with open(path or find_dis(), encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3 or parts[0] != "display":
                continue
            ch, dots = parts[1], parts[2]
            if ch == r"\s":
                ch = " "
            elif ch == r"\\":
                ch = "\\"
            if len(ch) != 1:
                continue
            val = 0
            if dots != "0":
                for d in dots:
                    if d.isdigit() and d != "0":
                        val |= 1 << (int(d) - 1)
            to_cell[ch] = chr(0x2800 + val)
    if len(to_cell) != 64:
        raise RuntimeError(f"expected 64 display entries, got {len(to_cell)}")
    from_cell = {v: k for k, v in to_cell.items()}
    if len(from_cell) != 64:
        raise RuntimeError("display table is not bijective")
    return to_cell, from_cell


def ascii_to_cells(s: str, to_cell: dict[str, str]) -> str:
    return "".join(to_cell.get(c.upper(), "�") for c in s)


def cells_to_ascii(s: str, from_cell: dict[str, str]) -> str:
    return "".join(from_cell.get(c, "?") for c in s)


def parse_brf(raw: str) -> tuple[dict[tuple[int, int], str], list[str]]:
    """Split a BRF verse stream on its `#c3v` markers into (verses, notes).

    Verses with a repeated or out-of-order marker are dropped and noted, so no cells are paired with the wrong text.
    """
    notes: list[str] = []
    toks = MARKER.split(raw)
    keys, bodies = [], []
    for i in range(1, len(toks), 3):
        c, v = int("".join(_DIGIT[x] for x in toks[i])), int("".join(_DIGIT[x] for x in toks[i + 1]))
        keys.append((c, v))
        bodies.append(toks[i + 2].strip())
    bad: set[tuple[int, int]] = set()
    seen: dict[tuple[int, int], int] = {}
    for idx, k in enumerate(keys):
        if k in seen:
            bad.add(k)
            notes.append(f"duplicate marker {k[0]}:{k[1]}")
        seen[k] = idx
        if idx and keys[idx - 1] >= k:
            bad.add(k)
            bad.add(keys[idx - 1])
            notes.append(f"non-ascending marker {keys[idx-1]} -> {k}")
    verses = {k: b for k, b in zip(keys, bodies) if k not in bad}
    return verses, notes


def load_text(path: str) -> dict[tuple[int, int], str]:
    j = json.load(open(path, encoding="utf-8"))
    return {(int(ch["chapter"]), int(ve["verse"])): ve["text"]
            for ch in j["chapters"] for ve in ch["verses"]}


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brf-dir", default="out/brf_bible")
    ap.add_argument("--tables", default=",".join(DEFAULT_TABLES))
    ap.add_argument("--out", default="work/bible_pairs/real_pairs.jsonl")
    ap.add_argument("--report", default="work/bible_pairs/build_report.json")
    args = ap.parse_args()

    from ubt.translate import ForwardTranslator  # noqa: PLC0415
    import louis  # noqa: PLC0415

    tables = [t for t in args.tables.split(",") if t]
    tr = ForwardTranslator()
    to_cell, from_cell = brf_maps()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    report: dict = {"liblouis": louis.version(), "dis": find_dis(),
                    "tables": tables, "books": {}, "notes": []}
    n_out = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for bp in sorted(glob.glob(os.path.join(args.brf_dir, "*.brf"))):
            book = os.path.basename(bp)[:-4]
            jp = os.path.join(args.brf_dir, book.capitalize() + ".json")
            if not os.path.isfile(jp):
                report["notes"].append(f"no inkprint json for {book}")
                continue
            verses, notes = parse_brf(open(bp, encoding="latin-1").read())
            text = load_text(jp)
            report["notes"] += [f"{book}: {n}" for n in notes]
            common = sorted(set(verses) & set(text))
            bstat = {"n_brf": len(verses), "n_text": len(text),
                     "n_paired": len(common),
                     "text_only": [f"{c}:{v}" for c, v in sorted(set(text) - set(verses))],
                     "brf_only": [f"{c}:{v}" for c, v in sorted(set(verses) - set(text))]}
            for c, v in common:
                real_ascii = verses[(c, v)]
                real = ascii_to_cells(real_ascii, to_cell)
                t = unicodedata.normalize("NFC", text[(c, v)].strip())
                rec = {"id": f"{book}-{c}-{v}", "book": book, "chapter": c,
                       "verse": v, "text": t, "braille_real": real,
                       "brf_ascii": real_ascii, "n_cells": len(real), "lou": {}}
                for tb in tables:
                    r = tr.translate(tb, t, check_undefined=False)
                    hyp = r.braille.strip() if r.ok else ""
                    rec["lou"][tb] = {
                        "ok": bool(r.ok), "braille": hyp,
                        "exact": bool(r.ok and hyp == real),
                        "edit": levenshtein(real, hyp) if r.ok else None,
                    }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
            report["books"][book] = bstat
    report["n_pairs"] = n_out
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(f"wrote {n_out} pairs -> {args.out}")
    for n in report["notes"]:
        print("note:", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
