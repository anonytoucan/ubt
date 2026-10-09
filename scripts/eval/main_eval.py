"""Evaluate an intermediate checkpoint on the paper's eval sets next to reference numbers (TARGETS).

The data are not identical to the paper's, so this checks reachability, not replication.
  build  : <dir>/eval.jsonl = the scripts/eval/greedy_eval.py set plus k2_paragraph, in the greedy_eval.py row format.
  report : score one model's generations (ubt.eval_harness.score_hyp, micro CER) against TARGETS.

  python scripts/eval/main_eval.py build --out work/data/main_eval
  python scripts/eval/greedy_eval.py gen --model <merged> --eval work/data/main_eval --out g.i.jsonl --shard i/4
  python scripts/eval/main_eval.py report --eval work/data/main_eval --gen 'g.*.jsonl' --step 2000 --out r.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from ubt.paths import DATASETS, WORK  # noqa: E402

# reference values from the paper (T1/T2 = Table 1/2)
TARGETS = {
    "core_all":   {"cer": 1.94, "bcer": 2.56, "strict": 89.4, "conf": 93.8, "src": "T1 A1 Qwen agnostic greedy"},
    "mixed_k2":   {"cer": 2.18, "src": "T2 inter-code k=2 agnostic"},
    "mixed_k3":   {"cer": 3.12, "src": "T2 k=3"},
    "mixed_k4":   {"cer": 2.14, "src": "T2 k=4"},
    "intra":      {"cer": 1.40, "src": "T2 intra-code switching"},
    "k2p":        {"cer": 5.91, "src": "T2 k=2 paragraph"},
    # family rows are verified best-of-20 references, so greedy is expected to be worse
    "nonCJK":     {"cer": 0.96, "bcer": 1.31, "strict": 92.1, "conf": 95.8, "src": "T1 B non-CJK, V20", "v20": True},
    "ko":         {"cer": 2.14, "bcer": 2.04, "strict": 42.8, "conf": 93.0, "src": "T1 B Korean, V20", "v20": True},
    "th":         {"cer": 4.43, "bcer": 3.69, "strict": 98.7, "conf": 99.0, "src": "T1 B Thai, V20", "v20": True},
    "ja":         {"cer": 1.99, "bcer": 1.83, "strict": 100.0, "conf": 100.0, "src": "T1 B Japanese, V20", "v20": True},
    "zh":         {"cer": 14.22, "bcer": 18.38, "strict": 99.3, "conf": 99.3, "src": "T1 B Chinese, V20", "v20": True},
}


def cmd_build(a) -> None:
    from transformers import AutoTokenizer  # noqa: PLC0415

    from ubt.task_format import render  # noqa: PLC0415
    rows = [json.loads(line) for line in open(os.path.join(a.ablation_eval, "eval.jsonl"), encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    for line in open(os.path.join(a.eval_dir, "k2_paragraph.jsonl"), encoding="utf-8"):
        rec = json.loads(line)
        ex = render(rec, hinted=False)
        n_c = len(tok(ex["completion"], add_special_tokens=False)["input_ids"])
        rows.append({"id": rec["id"], "set": "k2p", "tables": rec["tables"], "k": rec["k"], "regime": rec["regime"],
                     "switch_density": rec["switch_density"], "confusable_set": rec.get("confusable_set") or [],
                     "group": "k2p", "lowres": False, "prompt": ex["prompt"], "completion": ex["completion"],
                     "max_tokens": 2 * n_c + 32,
                     "n_prompt_tok_orig": len(tok(ex["prompt"], add_special_tokens=True)["input_ids"]),
                     "n_completion_tok": n_c})
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "eval.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"n": len(rows), "from": [a.ablation_eval, os.path.join(a.eval_dir, "k2_paragraph.jsonl")],
            "max_prompt_plus_cap": max(r["n_prompt_tok_orig"] + r["max_tokens"] for r in rows)}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


def _groups(rows):
    g = {k: [] for k in TARGETS}
    for r in rows:
        if r["set"] == "core":
            g["core_all"].append(r)
            g[r["group"]].append(r)
        elif r["set"] == "mixed":
            g[f"mixed_k{r['k']}"].append(r)
        elif r["set"] == "intra":
            g["intra"].append(r)
        elif r["set"] == "k2p":
            g["k2p"].append(r)
    return g


def _pin_engine(data_dir: str) -> None:
    """Use the data's pinned liblouis engine for bCER, not the inherited LOUIS_TABLEPATH, and run from its empty cwd
    because liblouis searches the working directory for tables first."""
    from ubt.u1 import engine as EN  # noqa: PLC0415
    spec = EN.EngineSpec.from_dict(json.load(open(os.path.join(data_dir, "manifest.json")))["engine_spec"])
    EN.apply_env(spec)
    os.environ.pop("LOUIS_TABLEPATH", None)
    os.environ["UBT_TABLE_FALLBACK_DIRS"] = spec.stage_dir
    os.chdir(spec.cwd_dir)


def _bcer_segments(r: dict):
    from ubt.wandb_utils import parse_target  # noqa: PLC0415
    gold = parse_target(r["completion"])
    cells = r["prompt"].split("<|braille|>\n", 1)[1].split("\n<|text|>", 1)[0].split("\n")
    if len(gold) == len(cells):
        return [(t, x, b) for (t, x), b in zip(gold, cells)]
    return [(gold[0][0], "\n".join(x for _, x in gold), "\n".join(cells))]   # proportional fallback


def cmd_report(a) -> None:
    from ubt.eval_harness import score_hyp  # noqa: PLC0415
    from ubt.metrics.textnorm import doc_error_cells_both  # noqa: PLC0415
    from ubt.wandb_utils import parse_target  # noqa: PLC0415
    a.eval, a.out = os.path.abspath(a.eval), os.path.abspath(a.out)
    gen_files = sorted(os.path.abspath(p) for p in glob.glob(a.gen))
    ev = [json.loads(line) for line in open(os.path.join(a.eval, "eval.jsonl"), encoding="utf-8")]
    gen = {}
    for p in gen_files:
        for line in open(p, encoding="utf-8"):
            x = json.loads(line)
            gen[x["id"]] = x
    if not a.no_bcer:
        _pin_engine(a.data)
    rows = []
    for r in ev:
        x = gen.get(r["id"])
        if x is None:
            continue
        rec = {"id": r["id"], "k": r["k"], "regime": r["regime"], "switch_density": r["switch_density"],
               "confusable_set": r["confusable_set"]}
        row = score_hyp(rec, {"completion": r["completion"]}, x["hyp"], None)
        row.update(set=r["set"], group=r["group"], finish=x["finish"])
        if not a.no_bcer:
            segs, hs = _bcer_segments(r), parse_target(x["hyp"])
            row["bc_err"], row["bc_err_st"], row["bc_n"], row["bc_fallback"] = doc_error_cells_both(segs, hs)
        rows.append(row)
    out = {"step": a.step, "model": a.model, "n_scored": len(rows), "n_eval": len(ev), "groups": {},
           "metric": "CER/bCER in the scoring normal form (ubt.metrics.textnorm: NFC, U+200B removed, Chinese S/T "
                     "folded); *_st = S/T-sensitive; cer_raw = unnormalised"}
    lines = ["| group | n | CER % (tex) | CER S/T-sens. | bCER % (tex) | strict % (tex) | conf % (tex) | cap hits |",
             "|---|---:|---|---:|---|---|---|---:|"]
    for g, rs in _groups(rows).items():
        if not rs:
            continue
        ref = sum(r["ref_len"] for r in rs)
        m = {"n": len(rs), "cer": 100 * sum(r["edit"] for r in rs) / max(1, ref),
             "cer_st": 100 * sum(r["edit_st"] for r in rs) / max(1, sum(r["ref_len_st"] for r in rs)),
             "cer_raw": 100 * sum(r["edit_raw"] for r in rs) / max(1, sum(r["ref_len_raw"] for r in rs)),
             "strict": 100 * sum(r["strict"] for r in rs) / len(rs), "conf": 100 * sum(r["conf_aware"] for r in rs) / len(rs),
             "cap_hits": sum(r["finish"] == "length" for r in rs)}
        if not a.no_bcer:
            n_cells = max(1, sum(r["bc_n"] for r in rs))
            m.update(bcer=100 * sum(r["bc_err"] for r in rs) / n_cells,
                     bcer_st=100 * sum(r["bc_err_st"] for r in rs) / n_cells,
                     bcer_fallback_docs=sum(r["bc_fallback"] for r in rs))
        t = TARGETS[g]
        m["target"] = t
        m["cer_gap"] = m["cer"] - t["cer"]
        out["groups"][g] = m

        def cell(k):
            if k not in m:
                return "–"
            return f"{m[k]:.2f} ({t[k]:.2f})" if k in t else f"{m[k]:.2f}"
        lines.append(f"| {g}{' (tex=V20)' if t.get('v20') else ''} | {m['n']} | {cell('cer')} | {m['cer_st']:.2f} | "
                     f"{cell('bcer')} | {cell('strict')} | {cell('conf')} | {m['cap_hits']} |")
    out["table_md"] = "\n".join(lines)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"step {a.step} ({len(rows)}/{len(ev)} scored)\n" + out["table_md"])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--ablation-eval", default=f"{WORK}/data/ablation_eval")
    b.add_argument("--eval-dir", default=f"{DATASETS}/data_u1_v2/eval")
    b.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B")
    r = sub.add_parser("report")
    r.add_argument("--eval", required=True)
    r.add_argument("--gen", required=True, help="glob of gen jsonl shards")
    r.add_argument("--step", type=int, required=True)
    r.add_argument("--model", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--data", default=f"{DATASETS}/data_u1_v2", help="its manifest pins the bCER engine")
    r.add_argument("--no-bcer", action="store_true")
    a = ap.parse_args()
    {"build": cmd_build, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
