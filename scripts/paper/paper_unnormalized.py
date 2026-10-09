"""CER without the scoring normal form, for the codes whose headline values depend on it (paper Table 8).

Codes: the six Chinese codes (S/T fold), Khmer (U+200B) and the Bengali-script codes (NFC). The headline's selected
outputs (greedy, verified best-of-20, oracle-in-20, LibLouis) are scored on the raw strings, micro-averaged.
Key "_chain": verified best-of-20 CER of all core documents with none, (i), (i)+(ii) and all three steps applied.

  python scripts/paper/paper_unnormalized.py [--model sft|topup|rlvr]
      -> work/paper/unnormalized_per_code.json (sft) or unnormalized_per_code.<model>.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from ubt.paths import WORK  # noqa: E402
CODES = {"zh_CHN.tbl", "zh-tw.ctb", "zh-hk.ctb", "zhcn-g1.ctb", "zhcn-g2.ctb", "zhcn-cbs.ctb", "km-g1.utb",
         "bn.tbl", "as.tbl", "mni.tbl"}
TEX_EVAL = f"{WORK}/data/main_eval/eval.jsonl"
R = WORK
POOLS = {"sft": {"inferred": (f"{R}/bon_sft/run", "gen_main", "main"),
                 "given": (f"{R}/bon_sft/run_prefill", "gen_core", "core")},
         "topup": {"inferred": (f"{R}/bon_topup/run", "gen_main", "main"),
                   "given": (f"{R}/evals/topup/prefill", "gen_core", "core")},
         "rlvr": {"inferred": (f"{R}/evals/rlvr/bon", "gen_main", "main"),
                  "given": (f"{R}/evals/rlvr/prefill", "gen_core", "core")}}
LIBLOUIS = f"{WORK}/liblouis_bwd/per_doc.jsonl"
OUT = f"{WORK}/paper/unnormalized_per_code{{}}.json"


def _segs(completion: str) -> list[tuple[str, str]]:
    from ubt.wandb_utils import parse_target  # noqa: PLC0415
    return parse_target(completion) or []


def _text(completion: str) -> str:
    return "\n".join(t for _, t in _segs(completion))


def _steps(segs: list[tuple[str, str]], k: int) -> str:
    """The join after the first k normal-form steps (0 none, 1 NFC, 2 + U+200B removed, 3 + S/T fold)."""
    from ubt.metrics.textnorm import ZWSP, fold_st, is_chinese_label  # noqa: PLC0415
    out = []
    for lab, t in segs:
        s = unicodedata.normalize("NFC", t) if k >= 1 else t
        s = s.replace(ZWSP, "") if k >= 2 else s
        out.append(fold_st(s) if k >= 3 and is_chinese_label(lab) else s)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sft", choices=sorted(POOLS))
    a = ap.parse_args()
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415
    ref, ref_all = {}, {}
    for line in open(TEX_EVAL, encoding="utf-8"):
        r = json.loads(line)
        if r["set"] != "core":
            continue
        ref_all[r["id"]] = _segs(r["completion"])
        if r["tables"][0] in CODES:
            ref[r["id"]] = (r["tables"][0], _text(r["completion"]))
    agg = collections.defaultdict(collections.Counter)
    chain = collections.defaultdict(collections.Counter)
    for cond, (d, stem, sl) in POOLS[a.model].items():
        if not os.path.isfile(os.path.join(d, "per_doc.jsonl")):
            print(f"no {cond} pool at {d}; skipped")
            continue
        picks = {}
        for line in open(os.path.join(d, "per_doc.jsonl"), encoding="utf-8"):
            x = json.loads(line)
            if x["slice"] == sl and x["set"] == "core":
                picks[x["id"]] = {p: x["pol"][p]["ci"] for p in ("G", "V", "O")}
        for p in glob.glob(os.path.join(d, f"{stem}.*.jsonl")):
            for line in open(p, encoding="utf-8"):
                x = json.loads(line)
                if x["id"] not in picks:
                    continue
                hs = _segs(x["cands"][picks[x["id"]]["V"]]["text"])
                for k in range(4):
                    rs = _steps(ref_all[x["id"]], k)
                    chain[(cond, k)]["e"] += Levenshtein.distance(rs, _steps(hs, k))
                    chain[(cond, k)]["n"] += len(rs)
                if x["id"] not in ref:
                    continue
                code, rt = ref[x["id"]]
                for pol, ci in picks[x["id"]].items():
                    s = agg[(code, cond, pol)]
                    s["e"] += Levenshtein.distance(rt, _text(x["cands"][ci]["text"]))
                    s["n"] += len(rt)
    for line in open(LIBLOUIS, encoding="utf-8"):
        x = json.loads(line)
        if x.get("set") == "core" and x["id"] in ref:
            code, rt = ref[x["id"]]
            s = agg[(code, "liblouis", "-")]
            s["e"] += Levenshtein.distance(rt, _text(x["hyp"]))
            s["n"] += len(rt)
    out = collections.defaultdict(dict)
    for (code, cond, pol), s in agg.items():
        out[code][cond if cond == "liblouis" else f"{cond}_{pol}"] = 100 * s["e"] / max(1, s["n"])
    names = ("none", "nfc", "nfc_zwsp", "all")
    out["_chain"] = {f"{cond}_V_{names[k]}": 100 * s["e"] / max(1, s["n"]) for (cond, k), s in sorted(chain.items())}
    path = OUT.format("" if a.model == "sft" else f".{a.model}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1, sort_keys=True)
    for code in sorted(out):
        print(code, {k: round(v, 2) for k, v in sorted(out[code].items())})
    print("->", path)


if __name__ == "__main__":
    main()
