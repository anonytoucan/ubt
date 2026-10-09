"""ByT5 baseline on data-u1 (full seq2seq fine-tune), the paper's ByT5 row.

The encoder reads the prompt and the decoder the completion (ubt.task_format.render); overlong documents are dropped and
counted. Paper setting: google/byt5-base, adamw_torch_fused 3e-4, batch 1x64, max source/target 5,120 / 3,072 bytes.
Under DDP, non-reentrant checkpointing and find_unused_parameters handle T5's tied embedding.

  torchrun --nproc_per_node <G> scripts/train/train_byt5_u1.py --data <data_u1_v2> --out <dir> \
      [--epochs 1] [--report-to wandb]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
IGNORE_INDEX = -100
_TOK = None


def _tok_chunk(args):
    """Tokenize a chunk into int16 id arrays (ByT5 vocab is 384), a quarter of the memory of Python lists."""
    global _TOK
    import numpy as np  # noqa: PLC0415

    from ubt.task_format import render  # noqa: PLC0415
    lines, model, max_src, max_tgt, hinted = args
    if _TOK is None:
        from transformers import AutoTokenizer  # noqa: PLC0415
        _TOK = AutoTokenizer.from_pretrained(model)
    # ByT5 ids are UTF-8 bytes + 3 plus EOS; computing them directly matches ByT5Tokenizer and is far faster.
    # Strings containing an added token's text (such as '</s>') go through the tokenizer, which reads it as one id.
    added = list(_TOK.get_added_vocab()) if type(_TOK).__name__ == "ByT5Tokenizer" else None

    def ids(s):
        if added is None or ("<" in s and any(t in s for t in added)):
            return np.asarray(_TOK(s, add_special_tokens=True)["input_ids"], dtype=np.int16)
        return np.append(np.frombuffer(s.encode("utf-8"), dtype=np.uint8).astype(np.int16) + 3,
                         np.int16(_TOK.eos_token_id))

    out, dropped = [], 0
    for line in lines:
        ex = render(json.loads(line), hinted=hinted)
        src, tgt = ids(ex["prompt"]), ids(ex["completion"])
        if len(src) > max_src or len(tgt) > max_tgt:
            dropped += 1
            continue
        out.append({"input_ids": src, "labels": tgt})
    return out, dropped


def build(path, model, max_src, max_tgt, hinted, limit, cache_root, workers=64):
    st = os.stat(path)
    key = hashlib.sha1(f"u1-byt5-np16|{path}|{st.st_size}|{int(st.st_mtime)}|{limit}|{max_src}|"
                       f"{max_tgt}|{hinted}|{model}".encode()).hexdigest()[:16]
    os.makedirs(os.path.join(cache_root, "tokcache"), exist_ok=True)
    cache = os.path.join(cache_root, "tokcache", f"byt5_{key}.pkl")
    if os.path.isfile(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    import fcntl  # noqa: PLC0415
    with open(cache + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.isfile(cache):
            with open(cache, "rb") as fh:
                return pickle.load(fh)
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()[: limit or None]
        step = max(1, (len(lines) + workers - 1) // workers)
        jobs = [(lines[i:i + step], model, max_src, max_tgt, hinted) for i in range(0, len(lines), step)]
        with mp.get_context("fork").Pool(len(jobs)) as pool:
            res = pool.map(_tok_chunk, jobs)
        out = [e for part, _ in res for e in part]
        dropped = sum(d for _, d in res)
        tmp = cache + f".tmp{os.getpid()}"
        with open(tmp, "wb") as fh:
            pickle.dump((out, dropped), fh)
        os.rename(tmp, cache)
        return out, dropped


class Seq2SeqCollator:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, feats):
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415
        sw = max(len(f["input_ids"]) for f in feats)
        tw = max(len(f["labels"]) for f in feats)
        ids = np.full((len(feats), sw), self.pad, dtype=np.int64)
        att = np.zeros((len(feats), sw), dtype=np.int64)
        lab = np.full((len(feats), tw), IGNORE_INDEX, dtype=np.int64)
        for i, f in enumerate(feats):
            n, m = len(f["input_ids"]), len(f["labels"])
            ids[i, :n] = f["input_ids"]
            att[i, :n] = 1
            lab[i, :m] = f["labels"]
        return {"input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(att),
                "labels": torch.from_numpy(lab)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/byt5-base")
    ap.add_argument("--hinted", action="store_true")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=0, help="0: 3%% of the total steps")
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=64)
    ap.add_argument("--max-src", type=int, default=5120)
    ap.add_argument("--max-tgt", type=int, default=3072)
    ap.add_argument("--max-drop-frac", type=float, default=0.005)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--dev-n", type=int, default=500)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--weights-dtype", default="fp32", choices=["fp32", "bf16"],
                    help="weight dtype; compute is bf16 autocast either way, and ByT5 does not learn with bf16 weights")
    ap.add_argument("--trust-own-checkpoints", action="store_true",
                    help="torch < 2.6: allow loading the state files of our own --resume checkpoint "
                         "(refused by default for CVE-2025-32434)")
    args = ap.parse_args()
    if args.trust_own_checkpoints:
        import torch  # noqa: PLC0415
        import transformers.trainer as _tr  # noqa: PLC0415
        import transformers.utils.import_utils as _iu  # noqa: PLC0415
        _iu.check_torch_load_is_safe = _tr.check_torch_load_is_safe = lambda: None
        _torch_load = torch.load

        def _load_own(*a, **kw):      # RNG state holds numpy arrays, which torch 2.5's weights_only unpickler rejects
            kw["weights_only"] = False
            return _torch_load(*a, **kw)
        torch.load = _load_own
    t0 = time.time()
    rank = int(os.environ.get("RANK", "0"))
    import torch  # noqa: PLC0415
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(args.model)
    cache_root = os.path.dirname(os.path.abspath(args.out))
    train_ds, drop = build(os.path.join(args.data, "train.jsonl"), args.model, args.max_src, args.max_tgt,
                           args.hinted, args.train_limit, cache_root)
    dev_ds, _ = build(os.path.join(args.data, "dev.jsonl"), args.model, args.max_src, args.max_tgt,
                      args.hinted, None, cache_root, workers=16)
    dev_ds = dev_ds[: args.dev_n]
    frac = drop / max(1, drop + len(train_ds))
    if rank == 0:
        print(f"train={len(train_ds)} dropped={drop} ({frac:.3%}) dev={len(dev_ds)} ({time.time() - t0:.0f}s)", flush=True)
    if frac > args.max_drop_frac and args.train_limit is None:
        sys.exit(f"overlong drop {frac:.3%} > {args.max_drop_frac:.3%}")
    wdt = torch.float32 if args.weights_dtype == "fp32" else torch.bfloat16
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model, dtype=wdt)
    except TypeError:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=wdt)
    if rank == 0:
        print(f"weights {next(model.parameters()).dtype}; lm_head tied to shared: "
              f"{model.lm_head.weight.data_ptr() == model.shared.weight.data_ptr()}; encoder embeddings tied to shared: "
              f"{model.encoder.embed_tokens.weight.data_ptr() == model.shared.weight.data_ptr()}", flush=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    os.makedirs(args.out, exist_ok=True)
    if rank == 0:
        json.dump({**vars(args), "train_docs": len(train_ds), "dropped": drop, "drop_frac": frac,
                   "data_engine_id": json.load(open(os.path.join(args.data, "manifest.json"))).get("engine_id")},
                  open(os.path.join(args.out, "run_config.json"), "w"), indent=1)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    total = args.max_steps if args.max_steps > 0 else int(-(-len(train_ds) // (args.per_device_batch * args.grad_accum * world)) * args.epochs)
    targs = TrainingArguments(
        output_dir=args.out, run_name=os.path.basename(args.out), report_to=args.report_to,
        num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_steps=args.warmup_steps or max(1, int(0.03 * total)),
        per_device_train_batch_size=args.per_device_batch, per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum, bf16=True, train_sampling_strategy="group_by_length",
        logging_steps=10, eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps",
        save_steps=args.save_steps, save_total_limit=6, optim="adamw_torch_fused", seed=args.seed,
        dataloader_num_workers=2, remove_unused_columns=False, ddp_find_unused_parameters=True,
        ddp_timeout=7200)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                      data_collator=Seq2SeqCollator(tok.pad_token_id))
    trainer.train(resume_from_checkpoint=args.resume)
    if rank == 0:
        trainer.save_model(os.path.join(args.out, "final"))
        tok.save_pretrained(os.path.join(args.out, "final"))
        # the saved model, which every evaluation loads, must give the same dev loss
        with torch.no_grad():
            back = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(args.out, "final"), dtype=wdt).to(model.device).eval()
            model.eval()
            col = Seq2SeqCollator(tok.pad_token_id)
            la = lb = 0.0
            for k in range(0, min(64, len(dev_ds)), 8):
                b = {kk: v.to(model.device) for kk, v in col(dev_ds[k:k + 8]).items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    la += float(model(**b).loss); lb += float(back(**b).loss)
            print(f"reload check: dev loss in memory {la:.4f} vs reloaded {lb:.4f}", flush=True)


if __name__ == "__main__":
    main()
