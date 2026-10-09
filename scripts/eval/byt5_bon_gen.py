"""ByT5 candidate pools in the bon_gen.py gen format, scored by bon_eval.py score like the Qwen pools.

Per document: greedy + n-1 samples (tau 0.8, top-p 0.95), each with its sequence log-probability. Prompt and
completion are UTF-8 bytes + 3 with EOS, as in train_byt5_u1.py. Uses HF generate since vLLM does not serve
encoder-decoder models; on Blackwell GPUs source scripts/eval/env_vllm_sm120.sh first (torch's Triton kernels need its
ptxas). Max new tokens: 2 x reference bytes + 32, capped at the training target length. Documents are written as they
finish, so an interrupted shard resumes; a batch that runs out of GPU memory is halved and retried.

  CUDA_VISIBLE_DEVICES=<g> python3 scripts/eval/byt5_bon_gen.py --model <ckpt>/final --rows main.jsonl \
      --out gen_main.<i>.jsonl --shard <i>/<k>
"""

from __future__ import annotations

import argparse
import json
import os
import time


def enc(s: str) -> list[int]:
    return [b + 3 for b in s.encode("utf-8")] + [1]


def dec(ids) -> str:
    return bytes(i - 3 for i in ids if 3 <= i < 259).decode("utf-8", errors="ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=148)
    ap.add_argument("--max-seqs", type=int, default=160, help="sequences per sampling batch (documents x (n-1))")
    ap.add_argument("--max-tgt", type=int, default=3072)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch  # noqa: PLC0415
    from transformers import AutoModelForSeq2SeqLM  # noqa: PLC0415
    si, sn = (int(x) for x in a.shard.split("/"))
    rows = [json.loads(line) for line in open(a.rows, encoding="utf-8")][si::sn][: a.limit or None]
    for r in rows:
        r["_src"] = enc(r["prompt"])
        r["_max"] = min(a.max_tgt, 2 * len(r["completion"].encode("utf-8")) + 32)
    keep = []
    if os.path.isfile(a.out):                     # resume an interrupted run
        for line in open(a.out, encoding="utf-8"):
            try:
                keep.append(json.loads(line))
            except json.JSONDecodeError:              # a partly written last line
                break
    have = {x["id"] for x in keep}
    order = sorted((i for i in range(len(rows)) if rows[i]["id"] not in have),
                   key=lambda i: (len(rows[i]["_src"]), rows[i]["_max"]))
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fh = open(a.out, "w", encoding="utf-8")
    for x in keep:
        fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    fh.flush()

    def run(idx, sample: bool, nret: int):
        width = max(len(rows[i]["_src"]) for i in idx)
        x = torch.zeros((len(idx), width), dtype=torch.long)
        m = torch.zeros((len(idx), width), dtype=torch.long)
        for k, i in enumerate(idx):
            s = rows[i]["_src"]
            x[k, :len(s)] = torch.tensor(s)
            m[k, :len(s)] = 1
        kw = dict(do_sample=True, temperature=a.temp, top_p=a.top_p, num_return_sequences=nret) if sample else \
            dict(do_sample=False, num_beams=1)
        with torch.no_grad():
            g = model.generate(input_ids=x.cuda(), attention_mask=m.cuda(), max_new_tokens=max(rows[i]["_max"] for i in idx),
                               output_scores=True, return_dict_in_generate=True, eos_token_id=1, pad_token_id=0, **kw)
            sc = model.compute_transition_scores(g.sequences, g.scores, normalize_logits=True).float().cpu()
        seqs = g.sequences[:, 1:].cpu()                       # drop the decoder start token
        res = []
        for j in range(seqs.shape[0]):
            ids = seqs[j].tolist()
            n_gen = next((k + 1 for k, t in enumerate(ids) if t == 1), len(ids))   # through EOS
            lp = float(sc[j, :n_gen].sum())
            doc = rows[idx[j // nret]]
            finish = "stop" if 1 in ids[:n_gen] else "length"
            res.append((idx[j // nret], {"text": dec(ids[:n_gen]), "lp": lp, "finish": finish, "greedy": not sample}))
            del doc
        return res
    def gen(idx, sample: bool, nret: int):
        """run(), halving the batch (documents, then samples) when it runs out of GPU memory."""
        try:
            return run(idx, sample, nret)
        except torch.cuda.OutOfMemoryError:
            pass                                      # retry outside the handler, so the failed batch is freed
        torch.cuda.empty_cache()
        if len(idx) > 1:
            h = len(idx) // 2
            return gen(idx[:h], sample, nret) + gen(idx[h:], sample, nret)
        if sample and nret > 1:
            return gen(idx, sample, nret // 2) + gen(idx, sample, nret - nret // 2)
        raise RuntimeError(f"out of GPU memory on a single sequence ({rows[idx[0]]['id']})")
    done = len(keep)
    per = max(1, a.max_seqs // max(1, a.n - 1))
    for b0 in range(0, len(order), per):
        idx = order[b0:b0 + per]
        torch.manual_seed(a.seed + b0)
        out = {}
        for i, c in gen(idx, False, 1):
            out.setdefault(i, []).insert(0, c)
        if a.n > 1:
            for i, c in gen(idx, True, a.n - 1):
                out.setdefault(i, []).append(c)
        for i in idx:
            fh.write(json.dumps({"id": rows[i]["id"], "cands": out[i]}, ensure_ascii=False) + "\n")
        fh.flush()
        done += len(idx)
        if (b0 // per) % 20 == 0:
            el = time.time() - t0
            left = len(rows) - done
            print(f"[byt5] {done}/{len(rows)} docs, {el:.0f}s, eta {el / max(1, done - len(keep)) * left:.0f}s", flush=True)
    fh.close()
    print(f"[byt5] {a.rows} shard {a.shard}: {len(rows)} docs x {a.n} ({len(keep)} resumed), {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
