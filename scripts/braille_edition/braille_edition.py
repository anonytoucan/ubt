#!/usr/bin/env python3
"""Braille edition of the paper: the main sections' prose in UEB grade 2, read back by the final model.

  build --paper-dir P --out D
      Strips the prose of mathematics, references and markup and transcribes it with the data's own engine
      (en-ueb-g2.ctb only). Writes the layout D/edition.{brf,cells.json,text.txt} (40 cells x 25 lines, per-cell source
      offsets from liblouis inPos), the decoding units D/units.jsonl (sentence-end cuts of at most --max-chars
      characters, each character in exactly one unit) and D/rows.jsonl (units plus the prose sentences on which f is
      defined, in the generator's format, code en-ueb-g2.ctb).
  (gen: scripts/eval/bon_gen.py gen --prefill-code --n 50 --slice D/rows.jsonl, code given)
  score --out D [--rerun-glob G]
      Verified selection (lambda = 2) over each pool. certified: the selection re-transcribes to the unit's cells;
      exact: equal after normalising what UEB does not encode. Cells are marked err (source character decoded wrongly)
      or unc (in an uncertified unit) -> D/marks.json (render_braille_tikz.py input) and D/summary.json.
      --rerun-glob: pools of a second run for the sentences left unverified.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path[:0] = sorted(glob.glob(os.path.join(REPO, "scripts", "*", "")))
from ubt.paths import DATASETS, WORK  # noqa: E402
TABLE = "en-ueb-g2.ctb"
DISPLAY = "unicode.dis"
DATA_U1 = f"{DATASETS}/data_u1_v2"
TOKENIZER = f"{WORK}/models/qwen25_cell_tokenizer"
SENT_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")


def engine():
    """The data's forward engine f for en-ueb-g2: (ForwardTranslator, louis module)."""
    from ubt.u1 import engine as EN  # noqa: PLC0415
    man = json.load(open(os.path.join(DATA_U1, "manifest.json")))
    spec = EN.EngineSpec.from_dict(man["engine_spec"])
    cwd = os.getcwd()
    os.chdir(spec.cwd_dir)
    tr = EN.make_translator(spec)
    os.chdir(cwd)
    return tr, tr._louis


def main_sections(root: str) -> list[str]:
    main_tex = open(os.path.join(root, "main.tex"), encoding="utf-8").read()
    body = main_tex.split("\\appendix")[0] if "\\appendix" in main_tex else main_tex
    stems = re.findall(r"^\s*\\input\{_sections/([^}]+)\}", body, re.M)
    return [s for s in stems if not s.startswith("appendix")]


REV_KEEP = ("added", "gadded", "padded", "gcell", "pcell", "okay")     # content shown as is
REV_NEW = ("replaced", "greplaced", "preplaced")                       # {new}{old}: new shown, old hidden
REV_DROP = ("deleted", "gdeleted", "pdeleted", "gTODO", "gval", "cut", "TODO")   # hidden text, notes, placeholders
_REV = re.compile(r"\\(" + "|".join(sorted(REV_KEEP + REV_NEW + REV_DROP, key=len, reverse=True)) + r")(?![A-Za-z])")


def _group(s: str, i: int) -> tuple[str, int]:
    """Content of the balanced group opening at s[i] == '{', and the index after its closing brace."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "\\" and j + 1 < len(s):
            j += 2
            continue
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1: j], j + 1
        j += 1
    return s[i + 1:], len(s)


def pdf_source(src: str) -> str:
    """Section source as the PDF shows it: comments, cutblocks and hidden revision text removed, since strip_latex
    would keep the hidden text."""
    s = re.sub(r"(?<!\\)%.*", "", src)
    s = re.sub(r"\\begin\{cutblock\}.*?\\end\{cutblock\}", " ", s, flags=re.S)
    out, i = [], 0
    while (m := _REV.search(s, i)) is not None:
        out.append(s[i:m.start()])
        j = m.end()
        while j < len(s) and s[j] in " \t":
            j += 1
        if j >= len(s) or s[j] != "{":
            out.append(s[m.start():m.end()])
            i = m.end()
            continue
        a1, j = _group(s, j)
        if m.group(1) in REV_NEW:
            k = j
            while k < len(s) and s[k] in " \t\n":
                k += 1
            if k < len(s) and s[k] == "{":
                _old, j = _group(s, k)
        out.append(pdf_source(a1) if m.group(1) not in REV_DROP else " ")
        i = j
    out.append(s[i:])
    return "".join(out)


def edition_text(root: str) -> tuple[str, list[str]]:
    from extract_paper_sentences import strip_latex  # noqa: PLC0415
    chunks = []
    for stem in main_sections(root):
        p = os.path.join(root, "_sections", stem if stem.endswith(".tex") else stem + ".tex")
        if not os.path.exists(p):
            continue
        prose = strip_latex(pdf_source(open(p, encoding="utf-8").read()))
        prose = re.sub(r"[\x01\x02]", " ", prose)
        prose = re.sub(r"\s+", " ", prose).strip()
        if prose:
            chunks.append((os.path.basename(p)[:-4], prose))
    return "\n\n".join(c[1] for c in chunks), [c[0] for c in chunks]


def cut_units(text: str, max_chars: int) -> list[tuple[int, int]]:
    """Sentence-end cuts; pieces under 40 characters join their neighbour, pieces over max_chars split at '; ' / ', '."""
    spans = []
    for para in re.finditer(r"[^\n]+", text):
        a0 = para.start()
        cuts = [a0] + [a0 + m.end() for m in SENT_END.finditer(para.group())] + [para.end()]
        for s, e in zip(cuts, cuts[1:]):
            while text[s:e].strip() and len(text[s:e].strip()) > max_chars:
                seg = text[s:e]
                mid = len(seg) // 2
                k = max((i for i in [seg.rfind("; ", 0, mid + 80), seg.rfind(", ", 0, mid + 80)] if i > 40), default=-1)
                k = k + 2 if k > 0 else max_chars
                spans.append((s, s + k))
                s += k
            if text[s:e].strip():
                spans.append((s, e))
    merged = []
    for s, e in spans:
        if merged and len(text[s:e].strip()) < 40 and "\n" not in text[merged[-1][0]:e]:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    out = []
    for s, e in merged:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return out


def cmd_build(a) -> int:
    from transformers import AutoTokenizer  # noqa: PLC0415
    tr, lou = engine()
    import make_braille_edition as MBE  # noqa: PLC0415   (after engine(), so its `import louis` is the pinned build)
    import build_real_corpora as BR  # noqa: PLC0415
    from extract_paper_sentences import is_prose, split_sentences, strip_latex, transcribe  # noqa: PLC0415
    os.makedirs(a.out, exist_ok=True)
    text, sections = edition_text(a.paper_dir)
    uni, inpos, _o, _c = lou.translate([DISPLAY, TABLE], text, mode=0)
    asc, _i, _o, _c = lou.translate([TABLE], text, mode=0)
    if len(asc) != len(uni):
        n = min(len(asc), len(uni))
        asc, uni, inpos = asc[:n], uni[:n], inpos[:n]
    lines = MBE.wrap_cells(uni, asc, list(inpos[: len(uni)]), a.width)
    brf = ["\r\n".join(ln["ascii"] for ln in lines[i: i + a.page_lines]) for i in range(0, len(lines), a.page_lines)]
    open(os.path.join(a.out, "edition.brf"), "w", encoding="ascii", errors="replace").write("\r\n\f\r\n".join(brf) + "\r\n")
    open(os.path.join(a.out, "edition.text.txt"), "w", encoding="utf-8").write(text)
    n_pages = (len(lines) + a.page_lines - 1) // a.page_lines
    json.dump({"table": TABLE, "liblouis": lou.version(), "width": a.width, "page_lines": a.page_lines,
               "n_cells": len(uni), "n_lines": len(lines), "n_pages": n_pages, "sections": sections,
               "text_len": len(text), "lines": lines}, open(os.path.join(a.out, "edition.cells.json"), "w"),
              ensure_ascii=False)
    units = []
    for s, e in cut_units(text, a.max_chars):
        t = text[s:e]
        b = lou.translate([DISPLAY, TABLE], t, mode=0)[0]
        if b:
            units.append({"id": f"be-unit-{len(units):04d}", "text": t, "braille": b, "src_start": s, "src_end": e})
    sents = []
    for stem in main_sections(a.paper_dir):
        p = os.path.join(a.paper_dir, "_sections", stem if stem.endswith(".tex") else stem + ".tex")
        if not os.path.exists(p):
            continue
        for sent in split_sentences(strip_latex(pdf_source(open(p, encoding="utf-8").read()))):
            ok, _why = is_prose(sent)
            cells = transcribe(tr, sent) if ok else None
            if cells:
                sents.append({"id": f"be-sent-{len(sents):04d}", "text": sent, "braille": cells})
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    rows_u, _ = BR.make_rows([{k: u[k] for k in ("id", "text", "braille")} for u in units], TABLE, "edition_unit",
                             [u["braille"] for u in units], tok)
    rows_s, _ = BR.make_rows(sents, TABLE, "edition_sentence", [s_["braille"] for s_ in sents], tok)
    with open(os.path.join(a.out, "units.jsonl"), "w", encoding="utf-8") as fh:
        for u in units:
            fh.write(json.dumps(u, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "rows.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows_u + rows_s:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = {"text_chars": len(text), "cells": len(uni), "lines": len(lines), "pages": n_pages, "sections": sections,
            "units": len(units), "unit_chars_mean": round(sum(len(u["text"]) for u in units) / max(1, len(units)), 1),
            "sentences": len(sents), "liblouis": lou.version()}
    json.dump(info, open(os.path.join(a.out, "build.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))
    return 0


def cmd_score(a) -> int:
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415
    from mark_braille_errors import bad_source_offsets, norm  # noqa: PLC0415
    from ubt.wandb_utils import parse_target  # noqa: PLC0415
    _tr, lou = engine()
    rows = {json.loads(line)["id"]: json.loads(line) for line in open(os.path.join(a.out, "rows.jsonl"), encoding="utf-8")}
    units = {json.loads(line)["id"]: json.loads(line) for line in open(os.path.join(a.out, "units.jsonl"), encoding="utf-8")}

    def pools(pattern):
        g = {}
        for p in sorted(glob.glob(pattern)):
            for line in open(p, encoding="utf-8"):
                x = json.loads(line)
                g[x["id"]] = x["cands"]
        return g
    cache: dict[str, str] = {}

    def f(t: str) -> str:
        if t not in cache:
            cache[t] = lou.translate([DISPLAY, TABLE], t, mode=0)[0] if t else ""
        return cache[t]

    def pick(rid, cands):
        B = rows[rid]["braille"]
        hyps = ["\n".join(x for _, x in parse_target(c["text"])) for c in cands]
        fs = [f(h) for h in hyps]
        ok = [i for i, x in enumerate(fs) if x == B]
        if ok:
            i = max(ok, key=lambda j: cands[j]["lp"])
        else:
            i = max(range(len(cands)), key=lambda j: cands[j]["lp"] - 2.0 * Levenshtein.distance(fs[j], B))
        return hyps[i], bool(ok), hyps[0], fs[0] == B

    gen = pools(os.path.join(a.out, "gen.*.jsonl"))
    res = {}
    for rid, cands in gen.items():
        v, cert, g, g_cert = pick(rid, cands)
        ref = rows[rid]["text"]
        res[rid] = {"hyp": v, "certified": cert, "exact": norm(v) == norm(ref), "greedy": g, "greedy_certified": g_cert,
                    "e": Levenshtein.distance(norm(ref), norm(v)), "e_greedy": Levenshtein.distance(norm(ref), norm(g)),
                    "n": len(norm(ref))}
    with open(os.path.join(a.out, "rerun_unit_rows.jsonl"), "w", encoding="utf-8") as fh:   # the unverified units
        for rid in units:
            if rid in res and not res[rid]["certified"]:
                fh.write(json.dumps(rows[rid], ensure_ascii=False) + "\n")
    unit_rerun = None
    if getattr(a, "rerun_units_glob", None):
        g3 = pools(a.rerun_units_glob)
        left = [u for u in units if u in res and not res[u]["certified"] and u in g3]
        won = 0
        for rid in left:
            v, cert, _g, _gc = pick(rid, g3[rid])
            if cert:
                ref = rows[rid]["text"]
                res[rid].update({"hyp": v, "certified": True, "exact": norm(v) == norm(ref),
                                 "e": Levenshtein.distance(norm(ref), norm(v))})
                won += 1
        unit_rerun = {"units": len(left), "n": len(next(iter(g3.values()))) if g3 else 0, "now_verified": won}
    doc = json.load(open(os.path.join(a.out, "edition.cells.json"), encoding="utf-8"))
    bad, unc = set(), set()
    U = [u for u in units if u in res]
    for uid in U:
        u, r = units[uid], res[uid]
        if not r["exact"]:
            bad.update(u["src_start"] + o for o in bad_source_offsets(u["text"], r["hyp"]))
        if not r["certified"]:
            unc.update(range(u["src_start"], u["src_end"]))
    marks, n_err, n_unc, outside = {}, 0, 0, 0
    for li, ln in enumerate(doc["lines"]):
        for ci, s in enumerate(ln["src"]):
            if s in bad:
                marks[f"{li}:{ci}"] = "err"
                n_err += 1
                outside += s not in unc
            elif s in unc:
                marks[f"{li}:{ci}"] = "unc"
                n_unc += 1
    su = {"units": len(U), "certified": sum(res[u]["certified"] for u in U), "exact": sum(res[u]["exact"] for u in U),
          "certified_and_exact": sum(res[u]["certified"] and res[u]["exact"] for u in U),
          "certified_not_exact": sum(res[u]["certified"] and not res[u]["exact"] for u in U),
          "cells": doc["n_cells"], "err_cells": n_err, "err_cells_pct": round(100 * n_err / max(1, doc["n_cells"]), 2),
          "unc_cells": n_unc, "err_cells_outside_uncertified": outside,
          "n_err": n_err, "n_unc": n_unc}                  # the keys render_braille_tikz.py prints
    if unit_rerun is not None:
        su["rerun"] = unit_rerun
    json.dump({"arm": "verified best-of-n (lambda=2), code given", "summary": su, "cell_marks": marks},
              open(os.path.join(a.out, "marks.json"), "w"), ensure_ascii=False)
    S = [r for r in rows if r.startswith("be-sent-") and r in res]
    ss = {"sentences": len(S), "verified": sum(res[s]["certified"] for s in S), "exact": sum(res[s]["exact"] for s in S),
          "verified_and_exact": sum(res[s]["certified"] and res[s]["exact"] for s in S),
          "cer_verified_selection": round(100 * sum(res[s]["e"] for s in S) / max(1, sum(res[s]["n"] for s in S)), 2),
          "cer_greedy": round(100 * sum(res[s]["e_greedy"] for s in S) / max(1, sum(res[s]["n"] for s in S)), 2)}
    with open(os.path.join(a.out, "rerun_rows.jsonl"), "w", encoding="utf-8") as fh:   # the unverified sentences
        for s_ in S:
            if not res[s_]["certified"]:
                fh.write(json.dumps(rows[s_], ensure_ascii=False) + "\n")
    if a.rerun_glob:
        g2 = pools(a.rerun_glob)
        left = [s for s in S if not res[s]["certified"] and s in g2]
        ss["rerun"] = {"sentences": len(left), "n": len(next(iter(g2.values()))) if g2 else 0,
                       "now_verified": sum(pick(s, g2[s])[1] for s in left)}
    with open(os.path.join(a.out, "unit_results.jsonl"), "w", encoding="utf-8") as fh:   # per unit, for the appendix text
        for uid in U:
            fh.write(json.dumps({"id": uid, "text": units[uid]["text"], "hyp": res[uid]["hyp"], "certified": res[uid]["certified"],
                                 "exact": res[uid]["exact"]}, ensure_ascii=False) + "\n")
    out = {"edition": json.load(open(os.path.join(a.out, "build.json"))), "units": su, "sentence_check": ss}
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--paper-dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--width", type=int, default=40)
    b.add_argument("--page-lines", type=int, default=25)
    b.add_argument("--max-chars", type=int, default=320)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--rerun-glob", default=None)
    s.add_argument("--rerun-units-glob", default=None, help="pools of a second run for the unverified units")
    a = ap.parse_args()
    return {"build": cmd_build, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
