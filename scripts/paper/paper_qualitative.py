"""Paper Table 4 (tables/qualitative.tex): two real core documents read by every system, from the final pools.

Per document: the input cells (drawn with \\qcell), the reference, and one row per system (LibLouis with the gold code,
GPT-5.5 few-shot, ours greedy and verified best-of-20) giving its code, the reading with wrong characters in \\qerr,
whether it re-transcribes to the input cells, and its CER in the scoring normal form.

  python scripts/paper/paper_qualitative.py --ids u1-Ecore-0017343,u1-Ecore-0017469 --out <paper>/tables/qualitative.tex \
      [--bon <best-of-20 pool dir with per_doc.jsonl and gen_main.*.jsonl>]
  (rewrites only the example rows between \\midrule after the header and \\bottomrule)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path[:0] = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "*", "")))
from ubt.paths import DATASETS, WORK  # noqa: E402
TEX_EVAL = f"{WORK}/data/main_eval/eval.jsonl"
BON = f"{WORK}/bon_sft/run"
LIBLOUIS = f"{WORK}/liblouis_bwd/per_doc.jsonl"
FRONTIER = f"{WORK}/frontier"
CELLS_PER_LINE = 45


def tex(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in s)


def marked(ref: str, hyp: str, label: str) -> str:
    """hyp with the characters that differ from ref (normal form alignment) wrapped in \\qerr."""
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415

    from ubt.metrics.textnorm import norm_line  # noqa: PLC0415
    a, b = norm_line(ref, label), norm_line(hyp, label)
    bad = [False] * len(b)
    for op in Levenshtein.editops(a, b):
        if op.tag in ("replace", "insert"):
            bad[op.dest_pos] = True
        elif op.tag == "delete" and b:
            bad[min(op.dest_pos, len(b) - 1)] = True      # mark the character next to a deletion
    if len(b) != len(hyp):                                  # normal form changed the length: no marks
        return tex(hyp)
    out, run = [], []
    for ch, e in zip(hyp, bad):
        if e:
            run.append(ch)
            continue
        if run:
            out.append(r"\qerr{" + tex("".join(run)) + "}")
            run = []
        out.append(tex(ch))
    if run:
        out.append(r"\qerr{" + tex("".join(run)) + "}")
    return "".join(out)


def cells_tex(cells: str) -> list[str]:
    """Cell lines of at most CELLS_PER_LINE, balanced (72 cells -> 36 + 36, not 45 + 27)."""
    k = -(-len(cells) // CELLS_PER_LINE)
    per = -(-len(cells) // k)
    out = []
    for i in range(0, len(cells), per):
        row = []
        for ch in cells[i:i + per]:
            bits = ord(ch) - 0x2800
            row.append(r"\qcell{" + ",".join(str(j + 1) for j in range(8) if bits >> j & 1) + "}")
        out.append("".join(row))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bon", default=BON)
    a = ap.parse_args()
    ids = a.ids.split(",")
    import frontier_fewshot as F  # noqa: PLC0415
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415

    from ubt.metrics.textnorm import norm_line  # noqa: PLC0415
    from ubt.u1.routed_louis import RoutedLouis  # noqa: PLC0415
    from ubt.wandb_utils import parse_target  # noqa: PLC0415
    ev = {}
    for line in open(TEX_EVAL, encoding="utf-8"):
        r = json.loads(line)
        if r["id"] in ids:
            ev[r["id"]] = r
    pd, gen = {}, {}
    for line in open(os.path.join(a.bon, "per_doc.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        if d["slice"] == "main" and d["id"] in ids:
            pd[d["id"]] = d
    for p in glob.glob(os.path.join(a.bon, "gen_main.*.jsonl")):
        for line in open(p, encoding="utf-8"):
            x = json.loads(line)
            if x["id"] in ids:
                gen[x["id"]] = x
    ll = {}
    for line in open(LIBLOUIS, encoding="utf-8"):
        x = json.loads(line)
        if x["id"] in ids:
            ll[x["id"]] = (parse_target(x["hyp"]) or [("", "")])[0][1]
    gpt = {}
    for p in glob.glob(os.path.join(FRONTIER, "batch_output.part*.jsonl")):
        for line in open(p, encoding="utf-8"):
            x = json.loads(line)
            if x["custom_id"] in ids:
                t = F._text_of((x.get("response") or {}).get("body") or {})
                t = re.sub(r"^\s*(text\s*:|the text is\s*:?|transcription\s*:)\s*", "", t, flags=re.I).strip().strip('"').strip()
                gpt[x["custom_id"]] = t.replace("\n", " ")
    rows = {}
    todo = []
    for i in ids:
        r = ev[i]
        gold = r["tables"][0]
        ref = parse_target(r["completion"])[0][1]
        cells = r["prompt"].split("<|braille|>\n", 1)[1].split("\n<|text|>", 1)[0]
        sys_rows = [("LibLouis", gold, ll[i], gold), ("GPT-5.5", None, gpt.get(i), gold)]   # None: not in GPT's 1k
        for name, pol in (("ours, greedy", "G"), ("ours, verified", "V")):
            seg = (parse_target(gen[i]["cands"][pd[i]["pol"][pol]["ci"]]["text"]) or [("?", "")])[0]
            sys_rows.append((name, seg[0], seg[1], seg[0]))
        rows[i] = (gold, ref, cells, sys_rows)
        todo += [(code, text) for _n, _c, text, code in sys_rows if text is not None]
    L = RoutedLouis(f"{DATASETS}/data_u1_v2", timeout_s=300.0)
    fw = dict(zip(todo, L.translate_many(todo)))
    L.close()
    body = []
    for k, i in enumerate(ids, 1):
        gold, ref, cells, sys_rows = rows[i]
        stem = re.sub(r"\.(ctb|utb|tbl)$", "", gold)
        if k > 1:
            body.append(r"\midrule")
        body.append(r"\qtitle{Example " + str(k) + "}{" + tex(stem) + "}{code inferred}")
        lines = cells_tex(cells)
        body.append(r"\qinputraw{" + lines[0] + "}")
        for extra in lines[1:]:
            body.append(r"\rowcolor{qbandC}\multicolumn{5}{@{\hspace{2pt}}l@{}}{\hspace{24pt}" + extra + r"}\\[1pt]")
        body.append(r"\qref{" + tex(ref) + "}")
        for name, code, text, chk in sys_rows:
            if text is None:
                body.append(r"\qrow{" + name + r"}{\qnocode}{\textit{not in the 1k subsample}}{--}{--}")
                continue
            rn, hn = norm_line(ref, gold), norm_line(text, code or gold)
            cer = 100 * Levenshtein.distance(rn, hn) / max(1, len(rn))
            ok = fw.get((chk, text)) == cells
            wrong = hn != rn
            verdict = r"\qsilent" if ok and wrong else (r"\qok" if ok else r"\qbad")
            ctex = r"\qnocode" if code is None else r"\qcode{" + tex(re.sub(r"\.(ctb|utb|tbl)$", "", code)) + "}"
            cmd = r"\qours" if name.startswith("ours") else r"\qrow"
            body.append(f"{cmd}{{{name}}}{{{ctex}}}{{{marked(ref, text, code or gold)}}}{{{verdict}}}{{{cer:.1f}}}")
        print(f"{i} {gold} cells={len(cells)}")
        for name, code, text, chk in sys_rows:
            print(f"   {name:15s} {str(code):20s} consistent={fw.get((chk, text)) == cells if text is not None else '-'}  {text}")
    s = open(a.out, encoding="utf-8").read()
    head = "System & Code & \\multicolumn{1}{l}{Reading} & \\multicolumn{1}{l}{Re-transcribes?} & CER (\\%) \\\\\n\\midrule\n"
    i0 = s.index(head) + len(head)
    i1 = s.index("\\bottomrule", i0)
    s = s[:i0] + "\n".join(body) + "\n" + s[i1:]
    open(a.out, "w", encoding="utf-8").write(s)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
