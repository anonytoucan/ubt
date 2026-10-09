"""Figure 5 rows for multi-code documents (one code per segment), as tables/qualitative.tex macros.

Rows: input cells and reference, one segment per line, then LibLouis (gold codes given), GPT-5.5 few-shot (judged under
the gold codes) and UBT greedy / verified best-of-20 (code inferred). Verdicts per document, as in Figure 2:
inconsistent (a segment does not re-transcribe to its cells, or the segment count differs), wrong code (all segments
re-transcribe, under other codes), consistent misreading (re-transcribes under the gold codes, different text), correct.
CER is over the joined segments in the scoring normal form; re-transcription uses the scorer's engine.

  python scripts/paper/paper_qualitative_multi.py survey --gpt work/frontier_fig5     # verdicts of the candidates
  python scripts/paper/paper_qualitative_multi.py rows --gpt ... --ids <id> <id> --first 1          # LaTeX rows
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path[:0] = sorted(glob.glob(os.path.join(REPO, "scripts", "*", "")))
from ubt.paths import DATASETS, WORK  # noqa: E402
R = WORK
BON = f"{R}/evals/rlvr/bon"
stem = lambda c: re.sub(r"\.(ctb|utb|tbl)$", "", c or "")  # noqa: E731


def load(ids, gpt_dir):
    import frontier_fewshot as F  # noqa: PLC0415
    import paper_qualitative as Q  # noqa: PLC0415
    from ubt.wandb_utils import parse_target  # noqa: PLC0415
    ev = {r["id"]: r for r in map(json.loads, open(Q.TEX_EVAL, encoding="utf-8")) if r["id"] in ids}
    pd = {d["id"]: d for d in map(json.loads, open(f"{BON}/per_doc.jsonl", encoding="utf-8"))
          if d["slice"] == "main" and d["id"] in ids}
    gen = {}
    for p in glob.glob(f"{BON}/gen_main.*.jsonl"):
        for line in open(p, encoding="utf-8"):
            x = json.loads(line)
            if x["id"] in ids:
                gen[x["id"]] = x
    ll = {x["id"]: parse_target(x["hyp"]) for x in map(json.loads, open(Q.LIBLOUIS, encoding="utf-8")) if x["id"] in ids}
    gpt = {}
    for p in glob.glob(os.path.join(gpt_dir, "batch_output.part*.jsonl")) if gpt_dir else []:
        for line in open(p, encoding="utf-8"):
            x = json.loads(line)
            if x["custom_id"] in ids:
                t = F._text_of((x.get("response") or {}).get("body") or {})
                t = re.sub(r"^\s*(text\s*:|the text is\s*:?|transcription\s*:)\s*", "", t, flags=re.I).strip().strip('"').strip()
                gpt[x["custom_id"]] = [ln.strip() for ln in t.split("\n") if ln.strip()]
    docs = {}
    for i in ids:
        r = ev[i]
        ref = parse_target(r["completion"])
        cells = r["prompt"].split("<|braille|>\n", 1)[1].split("\n<|text|>", 1)[0].split("\n")
        gold = [c for c, _ in ref]
        systems = [("LibLouis", ll.get(i))]
        g = gpt.get(i)
        systems.append(("GPT-5.5", None if g is None else [(None, x) for x in g]))
        for name, pol in (("ours, greedy", "G"), ("ours, best-of-$n$", "V")):
            systems.append((name, parse_target(gen[i]["cands"][pd[i]["pol"][pol]["ci"]]["text"]) or []))
        docs[i] = {"ref": ref, "cells": cells, "gold": gold, "systems": systems, "k": r["k"]}
    return docs


def judge(docs):
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415

    from ubt.metrics.textnorm import norm_join  # noqa: PLC0415
    from ubt.u1.routed_louis import RoutedLouis  # noqa: PLC0415
    todo = []
    for d in docs.values():
        for _n, segs in d["systems"]:
            if segs and len(segs) == len(d["cells"]):
                todo += [(c or g, t) for (c, t), g in zip(segs, d["gold"])]
    L = RoutedLouis(f"{DATASETS}/data_u1_v2", timeout_s=300.0)
    fw = dict(zip(todo, L.translate_many(todo)))
    L.close()
    for d in docs.values():
        rn = norm_join(d["ref"])
        d["verdicts"] = []
        for name, segs in d["systems"]:
            if segs is None:
                d["verdicts"].append((name, None, None))
                continue
            codes = [c or g for (c, _t), g in zip(segs, d["gold"])] if len(segs) == len(d["cells"]) else None
            ok = codes is not None and all(fw.get((c, t)) == b for c, (_x, t), b in zip(codes, segs, d["cells"]))
            hn = norm_join([(c or d["gold"][0], t) for c, t in segs])
            cer = 100 * Levenshtein.distance(rn, hn) / max(1, len(rn))
            pred = [c for c, _ in segs]
            if not ok:
                kind = "inconsistent"
            elif any(c is not None and stem(c) != stem(g) for c, g in zip(pred, d["gold"])):
                kind = "wrong code"
            else:
                kind = "correct" if hn == rn else "consistent misreading"
            d["verdicts"].append((name, kind, cer))
    return docs


def cmd_survey(a):
    ids = json.load(open(a.ids_file)) if a.ids_file else a.ids
    docs = judge(load(set(ids), a.gpt))
    for i in ids:
        d = docs[i]
        print(i, "+".join(stem(c) for c in d["gold"]), "cells", sum(len(c) for c in d["cells"]), "|",
              "; ".join(f"{n}: {k}{'' if c is None else f' {c:.1f}'}" for n, k, c in d["verdicts"]))


def cmd_rows(a):
    import paper_qualitative as Q  # noqa: PLC0415
    docs = judge(load(set(a.ids), a.gpt))
    V = {"inconsistent": r"\qbad", "consistent misreading": r"\qsilent", "correct": r"\qok"}
    out = []
    for n, i in enumerate(a.ids, a.first):
        d = docs[i]
        if n > a.first:
            out.append(r"\noalign{\vskip 6pt}")
        out.append(r"\qtitle{Example " + str(n) + "}{" + Q.tex(" + ".join(stem(c) for c in d["gold"])) + "}{code inferred}")
        out.append(r"\qinputraw{" + r"\newline ".join("".join(Q.cells_tex(c)) for c in d["cells"]) + "}")
        out.append(r"\qref{" + r"\newline ".join(Q.tex(t) for _c, t in d["ref"]) + "}")
        for (name, segs), (_n, kind, cer) in zip(d["systems"], d["verdicts"]):
            if segs is None:
                out.append(r"\qskip{" + name + "}{not run}")
                continue
            if kind == "wrong code":
                verdict = r"\qwrong{" + Q.tex(" + ".join(stem(c) for c, _ in segs)) + "}"
            else:
                verdict = V[kind]
            if len(segs) == len(d["ref"]):
                reading = r"\newline ".join(Q.marked(rt, t, c or rc) for (c, t), (rc, rt) in zip(segs, d["ref"]))
            else:
                reading = Q.marked(" ".join(t for _c, t in d["ref"]), " ".join(t for _c, t in segs), d["gold"][0])
            cmd = r"\qours" if name.startswith("ours") else r"\qrow"
            out.append(f"{cmd}{{{name}}}{{\\qnocode}}{{{reading}}}{{{verdict}}}{{{cer:.1f}}}")
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("survey")
    s.add_argument("--gpt", default=None)
    s.add_argument("--ids", nargs="*", default=[])
    s.add_argument("--ids-file", default=None)
    s.set_defaults(fn=cmd_survey)
    r = sub.add_parser("rows")
    r.add_argument("--gpt", default=None)
    r.add_argument("--ids", nargs="+", required=True)
    r.add_argument("--first", type=int, default=1)
    r.set_defaults(fn=cmd_rows)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
