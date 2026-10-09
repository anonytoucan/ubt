"""vLLM sampling with verifiable scoring for RLVR: greedy eval, or G samples for difficulty mining.

  # greedy eval of a merged model (optionally + a LoRA), one shard per GPU:
  CUDA_VISIBLE_DEVICES=0 python scripts/train/rlvr_sample.py --model <dir> [--lora <dir>] \
      --prompts pool.jsonl --out out.jsonl --n 1 --temperature 0 --shard 0/4
  # difficulty mining: G samples at T=1
  ... --n 8 --temperature 1.0

Scored by ubt.rlvr.reward.score_batch, with the RL reward's definitions and the data-u1 engine as f.
Output rows: {id, form, tables, n_prompt_tok, samples:[{text, reward, cer, feasible, exact, conf_ok,
finish, n_tok}], mean_reward, pass_any_exact}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from ubt.paths import DATASETS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=f"{DATASETS}/data_u1_v2",
                    help="data-u1 dir whose manifest engine_spec defines f")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    si, sn = (int(x) for x in args.shard.split("/"))
    rows = [json.loads(line) for line in open(args.prompts, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    rows = rows[si::sn]
    t0 = time.time()

    from vllm import LLM, SamplingParams  # noqa: PLC0415
    kw = dict(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, seed=args.seed, enable_prefix_caching=True)
    if args.lora:
        kw.update(enable_lora=True, max_lora_rank=args.lora_rank, max_loras=1)
    llm = LLM(**kw)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed)
    lreq = None
    if args.lora:
        from vllm.lora.request import LoRARequest  # noqa: PLC0415
        lreq = LoRARequest("rl", 1, args.lora)
    outs = llm.generate([r["prompt"] for r in rows], sp, lora_request=lreq)
    t1 = time.time()
    print(f"[sample] generated {len(rows)}x{args.n} in {t1 - t0:.0f}s", flush=True)

    from ubt.rlvr.louis_pool import LouisPool  # noqa: PLC0415
    from ubt.rlvr.reward import RewardConfig, score_batch  # noqa: PLC0415
    man = json.load(open(os.path.join(args.data, "manifest.json")))
    inv = json.load(open(os.path.join(args.data, "inventory.json")))
    pool = LouisPool(man["engine_spec"], known_tables={t["table_id"] for t in inv["tables"]},
                      overrides=man.get("engine_spec_overrides"))
    cfg = RewardConfig()
    comps, golds, brs, confs, meta = [], [], [], [], []
    for r, o in zip(rows, outs):
        for c in o.outputs:
            comps.append(c.text)
            golds.append(r["gold"])
            brs.append(r["braille"])
            confs.append(r["confusable_set"])
            meta.append((c.finish_reason, len(c.token_ids)))
    scores = []
    B = 4096
    for i in range(0, len(comps), B):
        scores += score_batch(comps[i:i + B], golds[i:i + B], brs[i:i + B], confs[i:i + B], pool, cfg)
    pool.close()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    k = 0
    n_ex = 0
    cer_sum = 0.0
    with open(args.out, "w", encoding="utf-8") as fh:
        for r, o in zip(rows, outs):
            samples = []
            for c in o.outputs:
                s = scores[k]
                fin, ntok = meta[k]
                k += 1
                samples.append({"text": c.text, "reward": s["reward"], "cer": s["cer"],
                                "feasible": s["feasible"], "exact": s["exact"],
                                "conf_ok": s["conf_ok"], "finish": fin, "n_tok": ntok})
            n_ex += samples[0]["exact"]
            cer_sum += min(1.0, samples[0]["cer"])
            fh.write(json.dumps({"id": r["id"], "form": r["form"], "tables": r["tables"],
                                 "n_prompt_tok": r["n_prompt_tok"], "samples": samples,
                                 "mean_reward": sum(x["reward"] for x in samples) / len(samples),
                                 "pass_any_exact": any(x["exact"] for x in samples)},
                                ensure_ascii=False) + "\n")
    print(json.dumps({"n": len(rows), "first_sample_exact": n_ex, "first_sample_mean_cer_clip":
                      round(cer_sum / max(1, len(rows)), 5), "gen_sec": round(t1 - t0, 1),
                      "score_sec": round(time.time() - t1, 1)}), flush=True)


if __name__ == "__main__":
    main()
