"""GPT-5.5 few-shot baseline (Table 1) through the OpenAI Batch API.

  sample : 1,000 core documents, round-robin over the codes (seed 13)
  shots  : 4 random single-code documents under 40 cells from the training split, no code label anywhere
  model  : gpt-5.5, reasoning effort medium, Responses API via /v1/batches
  scoring: the output under the gold code (code-ID not scored), CER, bCER, and consistency: forward re-transcription
           under the gold code reproduces the input cells

  export OPENAI_API_KEY=...
  python scripts/baselines/frontier_fewshot.py build  --out work/frontier
  python scripts/baselines/frontier_fewshot.py probe  --out work/frontier      # 2 synchronous requests
  python scripts/baselines/frontier_fewshot.py submit --out work/frontier --part 0/2   # then 1/2
  python scripts/baselines/frontier_fewshot.py fetch  --out work/frontier --part 0/2   # poll; prints cost
  python scripts/baselines/frontier_fewshot.py score  --out work/frontier
Real corpora (Table 4): build --rows <real.jsonl> --group bible --n 0, then score the outputs as a one-candidate pool
with scripts/real_corpora/real_corpora_score.py (scripts/baselines/frontier_real_pool.py).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path[:0] = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "*", "")))
from ubt.paths import DATASETS, WORK  # noqa: E402
TEX_EVAL = f"{WORK}/data/main_eval/eval.jsonl"
TRAIN = f"{DATASETS}/data_u1_v2/train.jsonl"
SYSTEM = ("You transcribe braille (Unicode cells U+2800..U+28FF) back to text. Output ONLY the transcribed text, "
          "no explanation.")


def _cells(prompt: str) -> str:
    return prompt.split("<|braille|>\n", 1)[1].split("\n<|text|>", 1)[0]


def stratified(pool, n, rng):
    by = {}
    for r in pool:
        by.setdefault(r["tables"][0], []).append(r)
    out, keys = [], sorted(by)
    while len(out) < n and any(by.values()):
        for k in keys:
            if by[k] and len(out) < n:
                out.append(by[k].pop(rng.randrange(len(by[k]))))
    return out


def cmd_build(a) -> None:
    rng = random.Random(a.seed)
    pool = [json.loads(line) for line in open(a.rows, encoding="utf-8")]
    pool = [r for r in pool if (r.get("group") == a.group if a.group else r["set"] == "core")]
    sample = stratified(pool, a.n, rng) if a.n else pool
    bank = []
    for line in open(TRAIN, encoding="utf-8"):
        if '"doc_type": "single"' not in line:
            continue
        d = json.loads(line)
        if d["k"] == 1 and d.get("cells", 999) < 40 and "\n" not in d["braille"] and "$$" not in d["text"]:
            bank.append({"braille": d["braille"], "text": d["text"], "table": d["tables"][0]})
    os.makedirs(a.out, exist_ok=True)
    reqs = []
    with open(os.path.join(a.out, "sample.jsonl"), "w", encoding="utf-8") as fs:
        for i, r in enumerate(sample):
            shots = rng.sample(bank, a.shots)
            ex = "".join(f"braille: {s['braille']}\ntext: {s['text']}\n\n" for s in shots)
            user = f"{ex}braille: {_cells(r['prompt'])}\ntext:"
            body = {"model": a.model, "reasoning": {"effort": a.effort}, "max_output_tokens": a.max_output_tokens,
                    "input": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
            reqs.append({"custom_id": r["id"], "method": "POST", "url": "/v1/responses", "body": body})
            fs.write(json.dumps({"id": r["id"], "table": r["tables"][0], "shots": [s["table"] for s in shots]},
                                ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "batch_input.jsonl"), "w", encoding="utf-8") as fh:
        for q in reqs:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    json.dump({"n": len(sample), "codes": len({r["tables"][0] for r in sample}), "shots": a.shots, "model": a.model,
               "effort": a.effort, "shot_bank": len(bank), "seed": a.seed}, open(os.path.join(a.out, "meta.json"), "w"),
              indent=1)
    print(f"{len(reqs)} requests over {len({r['tables'][0] for r in sample})} codes, shot bank {len(bank)}")


def _client():
    from openai import OpenAI  # noqa: PLC0415
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    return OpenAI()


def _text_of(resp: dict) -> str:
    if resp.get("output_text"):
        return resp["output_text"]
    parts = []
    for o in resp.get("output", []) or []:
        for c in o.get("content", []) or []:
            if c.get("type") in ("output_text", "text") and c.get("text"):
                parts.append(c["text"])
    return "".join(parts)


def cmd_probe(a) -> None:
    cl = _client()
    reqs = [json.loads(line) for line in open(os.path.join(a.out, "batch_input.jsonl"), encoding="utf-8")][:2]
    for q in reqs:
        r = cl.responses.create(**q["body"])
        d = r.model_dump()
        u = d.get("usage") or {}
        print(q["custom_id"], "| status", d.get("status"), "| usage", {k: u.get(k) for k in ("input_tokens", "output_tokens")},
              "reasoning", (u.get("output_tokens_details") or {}).get("reasoning_tokens"), "| text:", _text_of(d)[:100])


def _part(a) -> tuple[int, int, str]:
    """--part k/m: the k-th of m contiguous slices of batch_input.jsonl, so spend can be checked between parts.
    The sample is round-robin over the codes, so each slice covers all codes about equally."""
    k, m = (int(x) for x in a.part.split("/"))
    n = sum(1 for _ in open(os.path.join(a.out, "batch_input.jsonl"), encoding="utf-8"))
    return k * n // m, (k + 1) * n // m, f"part{k}of{m}"


def cmd_submit(a) -> None:
    cl = _client()
    lo, hi, tag = _part(a)
    lines = open(os.path.join(a.out, "batch_input.jsonl"), encoding="utf-8").readlines()[lo:hi]
    sub = os.path.join(a.out, f"batch_input.{tag}.jsonl")
    open(sub, "w", encoding="utf-8").writelines(lines)
    f = cl.files.create(file=open(sub, "rb"), purpose="batch")
    b = cl.batches.create(input_file_id=f.id, endpoint="/v1/responses", completion_window="24h",
                          metadata={"job": f"ubt frontier few-shot {os.path.basename(a.out)} {tag}"})
    json.dump({"file_id": f.id, "batch_id": b.id, "requests": [lo, hi], "n": len(lines),
               "submitted_utc": time.strftime("%F %T", time.gmtime())},
              open(os.path.join(a.out, f"batch.{tag}.json"), "w"), indent=1)
    print("batch", b.id, b.status, f"requests {lo}..{hi - 1}")


PRICE = {"input": 2.50, "output": 15.00}          # USD per 1M tokens, gpt-5.5 Batch API


def cmd_fetch(a) -> None:
    cl = _client()
    _lo, _hi, tag = _part(a)
    info = json.load(open(os.path.join(a.out, f"batch.{tag}.json")))
    b = cl.batches.retrieve(info["batch_id"])
    print("batch", b.id, b.status, b.request_counts)
    if b.status != "completed":
        return
    out = cl.files.content(b.output_file_id).text
    open(os.path.join(a.out, f"batch_output.{tag}.jsonl"), "w", encoding="utf-8").write(out)
    if b.error_file_id:
        open(os.path.join(a.out, f"batch_errors.{tag}.jsonl"), "w", encoding="utf-8").write(
            cl.files.content(b.error_file_id).text)
    tin = tout = inc = 0
    for line in out.splitlines():
        body = (json.loads(line).get("response") or {}).get("body") or {}
        u = body.get("usage") or {}
        tin += u.get("input_tokens") or 0
        tout += u.get("output_tokens") or 0
        inc += body.get("status") != "completed"
    cost = tin / 1e6 * PRICE["input"] + tout / 1e6 * PRICE["output"]
    print(f"downloaded {len(out.splitlines())} results; incomplete {inc}; tokens in {tin} out {tout}; "
          f"cost about ${cost:.2f}")


def cmd_score(a) -> None:
    import collections  # noqa: PLC0415

    import main_eval as TC  # noqa: PLC0415
    from ubt.eval_harness import score_hyp  # noqa: PLC0415
    from ubt.metrics.textnorm import _error_cells, _reference_cells  # noqa: PLC0415
    from ubt.u1.routed_louis import RoutedLouis  # noqa: PLC0415
    from ubt.wandb_utils import parse_target  # noqa: PLC0415
    a.out = os.path.abspath(a.out)
    ev = {json.loads(line)["id"]: json.loads(line) for line in open(TEX_EVAL, encoding="utf-8")}
    res, usage = {}, collections.Counter()
    outputs = sorted(glob.glob(os.path.join(a.out, "batch_output.part*.jsonl")))
    submitted = set()
    for p in glob.glob(os.path.join(a.out, "batch_input.part*.jsonl")):
        submitted |= {json.loads(line)["custom_id"] for line in open(p, encoding="utf-8")}
    for line in (ln for p in outputs for ln in open(p, encoding="utf-8")):
        x = json.loads(line)
        body = (x.get("response") or {}).get("body") or {}
        t = _text_of(body)
        t = re.sub(r"^\s*(text\s*:|the text is\s*:?|transcription\s*:)\s*", "", t, flags=re.I).strip().strip('"').strip()
        res[x["custom_id"]] = t.replace("\n", " ")
        u = body.get("usage") or {}
        usage["input"] += u.get("input_tokens") or 0
        usage["output"] += u.get("output_tokens") or 0
        usage["reasoning"] += (u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0
    sample = [json.loads(line) for line in open(os.path.join(a.out, "sample.jsonl"), encoding="utf-8")]
    sample = [s for s in sample if s["id"] in submitted]      # parts not submitted are not scored
    L = RoutedLouis(f"{DATASETS}/data_u1_v2", timeout_s=300.0)
    fw = dict(zip([s["id"] for s in sample], L.translate_many([(s["table"], res.get(s["id"], "")) for s in sample])))
    L.close()
    TC._pin_engine(f"{DATASETS}/data_u1_v2")
    m = collections.Counter()
    kinds = collections.defaultdict(collections.Counter)
    per_doc = []
    for s in sample:
        r = ev[s["id"]]
        hyp = f"⟨{s['table']}⟩" + res.get(s["id"], "")
        row = score_hyp({"id": r["id"], "confusable_set": r["confusable_set"]}, {"completion": r["completion"]}, hyp, None)
        segs = TC._bcer_segments(r)
        cells, starts, n, _ = _reference_cells(segs)
        bc = _error_cells(segs, starts, cells, n, parse_target(hyp), True)
        cons = fw.get(s["id"]) == _cells(r["prompt"])
        kind = "consistent_misreading" if cons and not row["hyp_eq_ref"] else ("correct" if cons else "inconsistent")
        m["e"] += row["edit"]; m["n"] += row["ref_len"]; m["e_st"] += row["edit_st"]; m["n_st"] += row["ref_len_st"]
        m["bc"] += bc; m["cells"] += n; m["docs"] += 1; m["exact"] += row["hyp_eq_ref"]
        m["capped"] += min(row["edit"], row["ref_len"])
        k = kinds[kind]; k["docs"] += 1; k["e"] += row["edit"]; k["n"] += row["ref_len"]; k["bc"] += bc; k["cells"] += n
        per_doc.append({"id": s["id"], "table": s["table"], "e": row["edit"], "n": row["ref_len"], "e_st": row["edit_st"],
                        "n_st": row["ref_len_st"], "bc": bc, "cells": n, "exact": bool(row["hyp_eq_ref"]), "kind": kind,
                        "empty": not res.get(s["id"])})
    out = {"n": m["docs"], "parts": [os.path.basename(p) for p in outputs], "missing_outputs": sum(1 for s in sample if s["id"] not in res),
           "cer": 100 * m["e"] / m["n"], "cer_st": 100 * m["e_st"] / m["n_st"], "cer_capped": 100 * m["capped"] / m["n"],
           "bcer": 100 * m["bc"] / m["cells"], "exact": 100 * m["exact"] / m["docs"], "usage_tokens": dict(usage),
           "kinds": {k: {"share": 100 * v["docs"] / m["docs"], "cer": 100 * v["e"] / max(1, v["n"]),
                         "bcer": 100 * v["bc"] / max(1, v["cells"])} for k, v in kinds.items()}}
    json.dump(out, open(os.path.join(a.out, "score.json"), "w"), indent=1)
    with open(os.path.join(a.out, "per_doc.jsonl"), "w", encoding="utf-8") as fh:
        for d in per_doc:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "probe", "submit", "fetch", "score"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000, help="stratified sample size; 0 = every row")
    ap.add_argument("--rows", default=TEX_EVAL, help="bon_eval rows (default: core evaluation documents)")
    ap.add_argument("--group", default=None, help="keep only this row group, e.g. bible")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-output-tokens", type=int, default=16000)
    ap.add_argument("--part", default="0/1", help="submit/fetch: the k-th of m slices, e.g. 0/2")
    a = ap.parse_args()
    {"build": cmd_build, "probe": cmd_probe, "submit": cmd_submit, "fetch": cmd_fetch, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
