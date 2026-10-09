"""RLVR prompt pool from a data-u1 split (default: train of data_u1_v2).

Output rows {"id","form","k","tables","prompt","gold","braille","confusable_set","n_prompt_tok","n_gold_tok"}, with
prompt and gold rendered exactly as in SFT (ubt.task_format.render). NIKL (S7) is never used: it is test/eval only.
Nemeth rows (S4a) are excluded because their forward function is latex2nemeth, not liblouis. Rows are sampled by
form share (DEFAULT_MIX); rows over --max-prompt-tok or --max-gold-tok are dropped and counted.

  python scripts/train/rlvr_build_prompts.py --data datasets/data_u1_v2 \
      --split train --out work/data/pool_agn.jsonl --n 60000 [--hinted]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from ubt.paths import DATASETS  # noqa: E402
from ubt.task_format import render  # noqa: E402

_MIX_V1 = {"S1": 0.30, "S1b": 0.15, "S2": 0.12, "S3": 0.18, "S3p": 0.04, "S5": 0.02, "S6": 0.10}
DEFAULT_MIX = {k: v / sum(_MIX_V1.values()) for k, v in _MIX_V1.items()}   # the mix above without S7, renormalised


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{DATASETS}/data_u1_v2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--hinted", action="store_true")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--max-prompt-tok", type=int, default=1536)
    ap.add_argument("--max-gold-tok", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--all", action="store_true",
                    help="take every liblouis row of the split (eval sets), no form mix; --n caps")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    from transformers import AutoTokenizer  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    by_form: dict[str, list[dict]] = defaultdict(list)
    with open(os.path.join(args.data, f"{args.split}.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("form") == "S7" or r.get("source") == "nikl":
                continue
            if args.all:
                if r.get("tables") and r.get("form") not in ("S4a", "Emath", "Enoise"):
                    by_form["ALL"].append(r)
            elif r.get("form") in DEFAULT_MIX:
                by_form[r["form"]].append(r)
    out, dropped = [], Counter()
    mix = {"ALL": 1.0} if args.all else DEFAULT_MIX
    for form, share in mix.items():
        pool = by_form.get(form, [])
        rng.shuffle(pool)
        want = int(round(args.n * share))
        got = 0
        for r in pool:
            if got >= want:
                break
            ex = render(r, hinted=args.hinted)
            prompt, gold = ex["prompt"], ex["completion"]
            npt = len(tok(prompt, add_special_tokens=False)["input_ids"])
            ngt = len(tok(gold, add_special_tokens=False)["input_ids"])
            if npt > args.max_prompt_tok or ngt > args.max_gold_tok:
                dropped[form] += 1
                continue
            out.append({"id": r["id"], "form": r.get("form"), "k": r["k"], "tables": r["tables"],
                        "prompt": prompt, "gold": gold, "braille": r["braille"],
                        "confusable_set": r.get("confusable_set") or [], "n_prompt_tok": npt,
                        "n_gold_tok": ngt})
            got += 1
    rng.shuffle(out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"n": len(out), "by_form": Counter(r["form"] for r in out),
                      "dropped_too_long": dropped, "hinted": args.hinted,
                      "tokenizer": args.tokenizer}, ensure_ascii=False))


if __name__ == "__main__":
    main()
