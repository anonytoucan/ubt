"""Paper §identifiability: can the 2006 and 2024 Korean codes be told apart from the cells alone?
kodisc: 500 texts, each under ko-2024-g2 and ko-2006-g2 (1,000 documents).

  identical  : share of pairs with identical cell strings, split by digit-free / digit-bearing text
  ceiling    : best accuracy of any cells-only rule, identical * 0.5 + (1 - identical); an identical pair gets one right
  model      : strict code identification of greedy and of verified best-of-20

  python scripts/eval/kodisc_eval.py build --out work/kodisc/data
  (gen: scripts/eval/bon_gen.py gen --slice <data>/kodisc.jsonl --out <dir>/gen_kodisc.<i>.jsonl --shard i/4)
  python scripts/eval/kodisc_eval.py score --data work/kodisc/data --dir <dir>
  python scripts/eval/kodisc_eval.py build-plain --out work/kodisc_plain/data   # plain-sentence pairs
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path[:0] = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "*", "")))
from ubt.paths import DATASETS, WORK  # noqa: E402
SRC = f"{DATASETS}/data_u1_v2/eval/kodisc.jsonl"
DIGIT = re.compile(r"[0-9]")
LATIN = re.compile(r"[A-Za-z]")
TEX_EVAL = f"{WORK}/data/main_eval/eval.jsonl"


def cmd_build(a) -> None:
    from transformers import AutoTokenizer  # noqa: PLC0415

    import bon_eval as B  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    recs = [json.loads(line) for line in open(SRC, encoding="utf-8")]
    rows = B._render_rows(recs, tok, lambda r: {"set": "kodisc", "group": r["tables"][0], "pair_id": r["pair_id"],
                                               "has_digit": bool(DIGIT.search(r["text"]))})
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "kodisc.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "rows")


def cmd_build_plain(a) -> None:
    """Pairs from the 500 held-out Korean core reference texts under both codes; the kodisc set is Latin-spliced,
    so it cannot show how often the codes coincide on ordinary sentences."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    import bon_eval as B  # noqa: PLC0415
    from ubt.u1.routed_louis import RoutedLouis  # noqa: PLC0415
    from ubt.wandb_utils import parse_target  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    texts = []
    for line in open(TEX_EVAL, encoding="utf-8"):
        r = json.loads(line)
        if r["set"] == "core" and r["tables"][0].startswith("ko-") and r["tables"][0] != "kok.tbl":
            texts.append(parse_target(r["completion"])[0][1])
    texts = list(dict.fromkeys(texts))
    pair = ("ko-2024-g2.ctb", "ko-2006-g2.ctb")
    L = RoutedLouis(f"{DATASETS}/data_u1_v2", timeout_s=300.0)
    cells = L.translate_many([(t, x) for x in texts for t in pair])
    L.close()
    recs, n = [], 0
    for i, x in enumerate(texts):
        c = cells[2 * i: 2 * i + 2]
        if not all(c):
            continue
        for t, b in zip(pair, c):
            n += 1
            recs.append({"id": f"kodisc-plain-{n:05d}", "doc_type": "single", "k": 1, "regime": "single",
                         "switch_density": "none", "text": x, "braille": b, "tables": [t], "segments": None,
                         "confusable_set": [], "pair_id": f"kp-{i:05d}"})
    rows = B._render_rows(recs, tok, lambda r: {"set": "kodisc", "group": r["tables"][0], "pair_id": r["pair_id"],
                                               "has_digit": bool(DIGIT.search(r["text"])),
                                               "has_latin": bool(LATIN.search(r["text"]))})
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "kodisc.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(texts), "texts ->", len(rows), "rows")


def cmd_score(a) -> None:
    import glob  # noqa: PLC0415

    import bon_gen as E  # noqa: PLC0415
    rows = [json.loads(line) for line in open(os.path.join(a.data, "kodisc.jsonl"), encoding="utf-8")]
    gens = {}
    for p in glob.glob(os.path.join(a.dir, "gen_kodisc.*.jsonl")):
        for line in open(p, encoding="utf-8"):
            x = json.loads(line)
            gens[x["id"]] = x
    rows = [r for r in rows if r["id"] in gens]
    feas = E._feasibility(rows, gens, f"{DATASETS}/data_u1_v2")
    pairs = collections.defaultdict(dict)
    for r in rows:
        pairs[r["pair_id"]][r["tables"][0]] = r
    out = {}
    def _in(p, split):
        r0 = next(iter(p.values()))
        if split == "all":
            return True
        if split == "latin-free":
            return not r0.get("has_latin", True)
        if split.startswith("latin-free "):
            return (not r0.get("has_latin", True)) and r0["has_digit"] == (split.endswith("digit-bearing"))
        return r0["has_digit"] == (split == "digit-bearing")
    splits = ["digit-free", "digit-bearing", "all"] + (["latin-free", "latin-free digit-free", "latin-free digit-bearing"]
                                                      if any("has_latin" in r for r in rows) else [])
    for split in splits:
        ps = [p for p in pairs.values() if len(p) == 2 and _in(p, split)]
        ident = sum(1 for p in ps if E._cells(p["ko-2024-g2.ctb"]["prompt"]) == E._cells(p["ko-2006-g2.ctb"]["prompt"]))
        share = ident / max(1, len(ps))
        res = {"pairs": len(ps), "identical_%": 100 * share, "ceiling_%": 100 * (share * 0.5 + (1 - share))}
        for pol in ("G", "V"):
            ok = n = 0
            for p in ps:
                for t, r in p.items():
                    cands = gens[r["id"]]["cands"]
                    f = [feas[(r["id"], i)] for i in range(len(cands))]
                    if pol == "G":
                        ci = 0
                    else:
                        F = [i for i in range(len(cands)) if f[i][0]]
                        D = [i for i in range(len(cands)) if f[i][1] is not None]
                        ci = (max(F, key=lambda i: cands[i]["lp"]) if F else
                              max(D, key=lambda i: cands[i]["lp"] - 2.0 * f[i][1]) if D else 0)
                    pred = cands[ci]["text"].split("⟩", 1)[0].lstrip("⟨")
                    ok += pred == t
                    n += 1
            res[f"code_acc_{pol}_%"] = 100 * ok / max(1, n)
        out[split] = res
    json.dump(out, open(os.path.join(a.dir, "kodisc_report.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--tokenizer", default=f"{WORK}/models/qwen25_cell_tokenizer")
    bp = sub.add_parser("build-plain")
    bp.add_argument("--out", required=True)
    bp.add_argument("--tokenizer", default=f"{WORK}/models/qwen25_cell_tokenizer")
    s = sub.add_parser("score")
    s.add_argument("--data", required=True)
    s.add_argument("--dir", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "build-plain": cmd_build_plain, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
