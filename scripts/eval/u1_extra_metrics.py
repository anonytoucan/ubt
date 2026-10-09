"""Metrics beyond CER for scored eval rows: sim (bge-m3 cosine), judge (Qwen2.5-72B-Instruct-AWQ, expected 1-5),
BLEU / chrF++ (sacrebleu); definitions in src/ubt/metrics/extra.py.

Input: one JSON row per document with `id` and `hyp_text`. References are the joined segment texts ("\\n".join, as
CER uses them), looked up by id in --eval (data-u1 records or rendered rows with `completion`).

  python scripts/eval/u1_extra_metrics.py --rows rows_A.jsonl --eval work/data/ablation_eval/eval.jsonl \
      --metrics sim,judge,bleu --out extra_A.json [--judge-tp 2] [--limit N]
Writes per-document values (<out>.rows.jsonl) and a summary (<out>) overall and per `group`/`set`.
Run sim and judge in separate processes if GPU memory is tight.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))


def _refs(paths: list[str]) -> dict[str, tuple[str, str | None]]:
    from ubt.task_format import record_segments  # noqa: PLC0415
    from ubt.wandb_utils import parse_target  # noqa: PLC0415   (the harness's parser: same ref as CER)
    out = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if "completion" in r and "segments" not in r:          # rendered row (greedy_eval.py build)
                out[r["id"]] = ("\n".join(t for _, t in parse_target(r["completion"])), (r.get("tables") or [None])[0])
            else:                                                  # data-u1 record
                segs = record_segments(r)
                out[r["id"]] = ("\n".join(s["text"] for s in segs), r["tables"][0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--eval", required=True, action="append", help="jsonl(s) holding the references")
    ap.add_argument("--metrics", default="sim,judge,bleu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge-tp", type=int, default=2)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--sim-model", default=None)
    ap.add_argument("--bleu-tokenize", default="13a", help="use 'zh' for Chinese-only sets")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from ubt.metrics import extra as X  # noqa: PLC0415
    refs = _refs(a.eval)
    rows = [json.loads(line) for line in open(a.rows, encoding="utf-8")]
    if a.limit:
        rows = rows[: a.limit]
    miss = [r["id"] for r in rows if r["id"] not in refs]
    if miss:
        sys.exit(f"{len(miss)} rows have no reference in --eval (e.g. {miss[:3]})")
    ref = [refs[r["id"]][0] for r in rows]
    hyp = [r.get("hyp_text") or "" for r in rows]
    lang = [refs[r["id"]][1] for r in rows]
    want = set(a.metrics.split(","))
    per = [{"id": r["id"], **{k: r[k] for k in ("set", "group", "k") if k in r}} for r in rows]
    if "sim" in want:
        s = X.run_sim(ref, hyp, **({"model": a.sim_model} if a.sim_model else {}))
        for p, v in zip(per, s):
            p["sim"] = round(float(v), 6)
    if "judge" in want:
        j = X.run_judge(list(zip(ref, hyp, lang)), tp=a.judge_tp, **({"model": a.judge_model} if a.judge_model else {}))
        for p, v in zip(per, j):
            p["judge"] = None if v is None else round(float(v), 6)
    summ: dict = {"n": len(rows), "rows": a.rows, "metrics": sorted(want)}
    groups: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(per):
        groups["all"].append(i)
        for k in ("set", "group"):
            if k in p:
                groups[f"{k}={p[k]}"].append(i)
    for g, idx in sorted(groups.items()):
        d = {"n": len(idx)}
        for m in ("sim", "judge"):
            vals = [per[i][m] for i in idx if per[i].get(m) is not None]
            if vals:
                d[m] = round(100 * sum(vals) / len(vals), 3) if m == "sim" else round(sum(vals) / len(vals), 4)
                d[f"{m}_n"] = len(vals)
        if "bleu" in want:
            d.update(X.bleu_chrf([ref[i] for i in idx], [hyp[i] for i in idx], tokenize=a.bleu_tokenize))
        summ[g] = d
    with open(a.out + ".rows.jsonl", "w", encoding="utf-8") as fh:
        for p in per:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    json.dump(summ, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ.get("all"), ensure_ascii=False))


if __name__ == "__main__":
    main()
