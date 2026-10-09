"""SFT on data-u1 (the RLVR starting point).

Task format and masking from ubt.sft_data.build_split: prompt-masked, EOS-terminated completion, overlong
documents dropped, never truncated. bf16 LoRA on all linear layers rather than QLoRA: it fits 32 GB, is faster
and merges cleanly for RLVR. Batches are grouped by length; the data manifest is recorded in run_config.json.

  torchrun --nproc_per_node 4 scripts/train/train_sft_u1.py --data datasets/data_u1_v2 \
      --out work/ckpt/sft --epochs 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))


def completion_only_forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
    """Causal-LM forward that runs lm_head only at label positions, since the prompt is most of each sequence.
    Same loss as HF: shifted CE, summed / num_items_in_batch when the Trainer passes it, else the mean."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    from transformers.modeling_outputs import CausalLMOutputWithPast  # noqa: PLC0415
    num_items = kw.pop("num_items_in_batch", None)
    thr = getattr(self, "_ckpt_threshold", 0)
    if thr and self.training:                 # recompute activations only for big micro-batches
        flag = input_ids.numel() > thr
        for layer in self.model.layers:
            layer.gradient_checkpointing = flag
    out = self.model(input_ids=input_ids, attention_mask=attention_mask)
    h = out.last_hidden_state
    if labels is None:
        return CausalLMOutputWithPast(logits=self.lm_head(h))
    sl = labels[:, 1:]
    m = sl != -100
    cap = getattr(self.config, "final_logit_softcapping", None)
    chunk = getattr(self, "_ce_chunk", 0)
    hs, ys = h[:, :-1][m], sl[m]

    def ce(hc, yc):
        lg = self.lm_head(hc).float()
        if cap:                               # Gemma-2: same softcap as its own forward
            lg = torch.tanh(lg / cap) * cap
        return F.cross_entropy(lg, yc, reduction="sum")
    if chunk and hs.shape[0] > chunk:
        # same summed loss with one chunk's fp32 logits at a time (recomputed in backward); a 256k vocab can OOM
        # 24 GB cards otherwise
        from torch.utils.checkpoint import checkpoint  # noqa: PLC0415
        loss = sum(checkpoint(ce, hs[i:i + chunk], ys[i:i + chunk], use_reentrant=False)
                   for i in range(0, hs.shape[0], chunk))
    else:
        loss = ce(hs, ys)
    loss = loss / (num_items if num_items is not None else m.sum().clamp(min=1))
    return CausalLMOutputWithPast(loss=loss)


BRAILLE_CELLS = [chr(0x2800 + i) for i in range(256)]


def init_braille_rows(model, tok, base_tok) -> list[int]:
    """Init each cell token's embedding row in `tok` to the mean of the cell's original `base_tok` token rows.
    Returns the cell ids."""
    import torch  # noqa: PLC0415
    emb = model.get_input_embeddings().weight
    ids = tok.convert_tokens_to_ids(BRAILLE_CELLS)
    # cells that already were single tokens keep their pretrained id; the mean init leaves those rows unchanged
    if None in ids or len(set(ids)) != 256 or max(ids) >= emb.shape[0]:
        sys.exit(f"--braille-vocab: tokenizer lacks 256 distinct cell tokens inside the embedding ({emb.shape[0]} rows)")
    bad = [c for c, i in zip(BRAILLE_CELLS, ids) if tok(c, add_special_tokens=False)["input_ids"] != [i]]
    if bad:
        sys.exit(f"--braille-vocab: {len(bad)} cells are not a single token, e.g. {bad[:3]}")
    with torch.no_grad():
        for c, i in zip(BRAILLE_CELLS, ids):
            sub = base_tok(c, add_special_tokens=False)["input_ids"]
            emb[i] = emb[sub].float().mean(0).to(emb.dtype)
    return ids


def grouped_lr_trainer(trainer_cls, mult: float):
    """Trainer whose PEFT trainable-token rows get lr x mult (same schedule, own param group)."""
    class GroupedLRTrainer(trainer_cls):
        def create_optimizer(self, model=None):
            opt_model = self.model if model is None else model
            if self.optimizer is None and mult != 1.0:
                named = [(n, p) for n, p in opt_model.named_parameters() if p.requires_grad]
                tok_p = [p for n, p in named if "trainable_tokens_delta" in n]
                if tok_p:
                    decay = set(self.get_decay_parameter_names(opt_model))
                    cls, kw = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
                    rest = [(n, p) for n, p in named if "trainable_tokens_delta" not in n]
                    self.optimizer = cls([
                        {"params": [p for n, p in rest if n in decay], "weight_decay": self.args.weight_decay},
                        {"params": [p for n, p in rest if n not in decay], "weight_decay": 0.0},
                        {"params": tok_p, "weight_decay": 0.0, "lr": kw["lr"] * mult}], **kw)
            return super().create_optimizer(model)
    return GroupedLRTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--hinted", action="store_true")
    ap.add_argument("--init-adapter", default=None, help="LoRA adapter to warm-start from")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--per-device-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=3584)
    ap.add_argument("--max-drop-frac", type=float, default=0.005)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--dev-n", type=int, default=1000)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--attn-impl", default="sdpa", help="eager for Gemma-2 (logit softcapping)")
    ap.add_argument("--qlora", action="store_true", help="4-bit NF4 base for small GPUs or 9B+ models")
    ap.add_argument("--sampler", default="group_by_length", choices=["group_by_length", "batch_rebalance", "random"])
    ap.add_argument("--liger", action="store_true", help="Liger fused RoPE/RMSNorm/SwiGLU (Qwen2 / Gemma2)")
    ap.add_argument("--stop-at-step", type=int, default=0,
                    help="save and stop at this global step; the LR schedule is unchanged")
    ap.add_argument("--stop-at-seconds", type=float, default=0,
                    help="save and stop once training time exceeds this")
    ap.add_argument("--stop-min-step", type=int, default=0,
                    help="--stop-at-seconds fires only at or after this global step")
    ap.add_argument("--tokenizer", default=None, help="tokenizer dir (default: --model)")
    ap.add_argument("--braille-vocab", action="store_true",
                    help="--tokenizer has the 256 cells as tokens; mean-init and train their rows")
    ap.add_argument("--new-token-lr-mult", type=float, default=10.0,
                    help="lr multiplier for the --braille-vocab embedding rows")
    ap.add_argument("--ckpt-threshold", type=int, default=0,
                    help="gradient-checkpoint only micro-batches with more than N tokens (0 = always)")
    ap.add_argument("--ce-chunk", type=int, default=0,
                    help="lm_head + CE in chunks of N label tokens, recomputed in backward (0 = one shot)")
    ap.add_argument("--trust-own-checkpoints", action="store_true",
                    help="torch < 2.6: allow torch.load of our own --resume checkpoint files "
                         "(blocked by default over CVE-2025-32434)")
    ap.add_argument("--isolate-gpus", action="store_true",
                    help="each DDP rank sees only its own GPU, so stray CUDA contexts cannot OOM a resume")
    args = ap.parse_args()
    t0 = time.time()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.isolate_gpus and "LOCAL_RANK" in os.environ:       # must run before torch initialises CUDA
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[local_rank] if visible else str(local_rank)
        os.environ["LOCAL_RANK"] = "0"
        local_rank = 0

    import torch  # noqa: PLC0415
    if args.trust_own_checkpoints:
        import transformers.trainer as _tr  # noqa: PLC0415
        import transformers.utils.import_utils as _iu  # noqa: PLC0415
        _iu.check_torch_load_is_safe = _tr.check_torch_load_is_safe = lambda: None
        _torch_load = torch.load

        def _load_own(*a, **kw):          # RNG state holds numpy arrays (rejected by torch 2.5 weights_only)
            kw["weights_only"] = False
            return _torch_load(*a, **kw)
        torch.load = _load_own
    if torch.cuda.is_available():
        # make implicit "cuda" this rank's GPU: on --resume PEFT loads the adapter onto cuda:0 in every rank,
        # which can OOM GPU 0
        torch.cuda.set_device(local_rank)
        import peft.utils.save_and_load as _peft_sl  # noqa: PLC0415
        _peft_sl.infer_device = lambda: f"cuda:{local_rank}"
    from peft import LoraConfig, PeftModel, get_peft_model  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments  # noqa: PLC0415

    from ubt.sft_data import CausalCollator, build_split  # noqa: PLC0415

    if args.braille_vocab and not args.tokenizer:
        sys.exit("--braille-vocab needs --tokenizer <extended tokenizer dir> (its path keys the token cache)")
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    os.environ.setdefault("UBT_BASE", os.path.dirname(os.path.abspath(args.out)))   # token cache dir
    train_ds, train_drop = build_split(os.path.join(args.data, "train.jsonl"), tok, args.max_len,
                                       args.hinted, limit=args.train_limit, workers=64)
    dev_ds, dev_drop = build_split(os.path.join(args.data, "dev.jsonl"), tok, args.max_len,
                                   args.hinted, limit=None, workers=16)
    dev_ds = dev_ds[: args.dev_n]
    drop_frac = train_drop / max(1, train_drop + len(train_ds))
    if rank == 0:
        print(f"train={len(train_ds)} dropped={train_drop} ({drop_frac:.3%}) dev={len(dev_ds)} "
              f"(tokenize {time.time() - t0:.0f}s)", flush=True)
    if drop_frac > args.max_drop_frac and args.train_limit is None:
        sys.exit(f"overlong drop fraction {drop_frac:.3%} > {args.max_drop_frac:.3%} — stop and report")

    if args.liger:
        from liger_kernel.transformers import apply_liger_kernel_to_gemma2, apply_liger_kernel_to_qwen2  # noqa: PLC0415
        if "gemma" in args.model.lower():
            apply_liger_kernel_to_gemma2(rope=True, rms_norm=True, geglu=True, cross_entropy=False,
                                         fused_linear_cross_entropy=False)
        else:
            apply_liger_kernel_to_qwen2(rope=True, rms_norm=True, swiglu=True, cross_entropy=False,
                                        fused_linear_cross_entropy=False)
    load_kw = dict(attn_implementation=args.attn_impl, device_map={"": local_rank})
    if args.qlora:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, **load_kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, **load_kw)
    if args.braille_vocab and len(tok) > model.get_input_embeddings().weight.shape[0]:
        # no free embedding slots (Gemma-2): grow the tied embedding; the new rows are set below
        model.resize_token_embeddings(len(tok), pad_to_multiple_of=64, mean_resizing=False)
    if args.qlora:
        from peft import prepare_model_for_kbit_training  # noqa: PLC0415
        model = prepare_model_for_kbit_training(model)
        if args.braille_vocab:
            # kbit prep upcasts the embedding to fp32 and PEFT trainable tokens copy it every forward; keep it bf16
            # so it fits 24 GB cards
            model.get_input_embeddings().to(torch.bfloat16)
            if model.get_output_embeddings() is not None:
                model.get_output_embeddings().to(torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id
    new_ids = init_braille_rows(model, tok, AutoTokenizer.from_pretrained(args.model)) if args.braille_vocab else []
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                                 lora_dropout=0.05, target_modules="all-linear",
                                                 task_type="CAUSAL_LM",
                                                 trainable_token_indices={"embed_tokens": new_ids} if new_ids else None))
    import types  # noqa: PLC0415
    base = model.get_base_model()
    base.forward = types.MethodType(completion_only_forward, base)
    base._ckpt_threshold = args.ckpt_threshold
    base._ce_chunk = args.ce_chunk
    if rank == 0:
        model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    os.makedirs(args.out, exist_ok=True)
    if rank == 0:
        man = json.load(open(os.path.join(args.data, "manifest.json")))
        vr = os.path.join(args.data, "verify_report.json")
        cfg = {**vars(args), "train_docs": len(train_ds), "train_dropped": train_drop,
               "drop_frac": round(drop_frac, 6), "data_engine_id": man.get("engine_id"),
               "data_manifest_sha256": hashlib.sha256(open(os.path.join(args.data, "manifest.json"), "rb").read()).hexdigest(),
               "data_verify_pass": json.load(open(vr)).get("pass") if os.path.exists(vr) else None,
               "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                         cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip(),
               "world_size": int(os.environ.get("WORLD_SIZE", "1")), "torch": torch.__version__,
               "tokenizer_len": len(tok), "new_token_ids": [new_ids[0], new_ids[-1]] if new_ids else None}
        json.dump(cfg, open(os.path.join(args.out, "run_config.json"), "w"), indent=1)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    per_step = args.per_device_batch * args.grad_accum * world
    total_steps = args.max_steps if args.max_steps > 0 else int(
        -(-len(train_ds) // per_step) * args.epochs)
    targs = TrainingArguments(
        output_dir=args.out, run_name=os.path.basename(args.out), report_to=args.report_to,
        num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_steps=max(1, int(0.02 * total_steps)), weight_decay=0.0,
        per_device_train_batch_size=args.per_device_batch, per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum, bf16=True,
        train_sampling_strategy=args.sampler,
        logging_steps=10, logging_first_step=True, eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=6,
        optim="adamw_torch_fused", seed=args.seed, dataloader_num_workers=2,
        remove_unused_columns=False, ddp_find_unused_parameters=False, ddp_timeout=7200,
    )
    from transformers import TrainerCallback  # noqa: PLC0415

    class RunControl(TrainerCallback):
        """Adds elapsed_sec to every log row and applies the optional stop points."""
        def on_train_begin(self, a, state, control, **kw):
            self.t0 = time.time()
            self.base = state.global_step

        def on_log(self, a, state, control, logs=None, **kw):
            if logs is not None:
                extra = {"elapsed_sec": round(time.time() - self.t0, 1),
                         "steps_this_launch": state.global_step - self.base}
                logs.update(extra)
                # Trainer.log() stores a copy in log_history before on_log (transformers v5)
                if state.log_history and state.log_history[-1].get("step") == state.global_step:
                    state.log_history[-1].update(extra)

        def on_step_end(self, a, state, control, **kw):
            hit = (args.stop_at_step and state.global_step >= args.stop_at_step) or \
                  (args.stop_at_seconds and time.time() - self.t0 >= args.stop_at_seconds
                   and state.global_step >= args.stop_min_step)
            if hit:
                control.should_save = True
                control.should_evaluate = True
                control.should_training_stop = True

    trainer_cls = grouped_lr_trainer(Trainer, args.new_token_lr_mult) if new_ids else Trainer
    trainer = trainer_cls(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                          data_collator=CausalCollator(tok.pad_token_id), callbacks=[RunControl()])
    trainer.train(resume_from_checkpoint=args.resume)
    print(f"[rank{rank}] peak_mem_GB={torch.cuda.max_memory_allocated() / 2**30:.2f}", flush=True)
    if rank == 0:
        trainer.save_model(os.path.join(args.out, "final"))
        tok.save_pretrained(os.path.join(args.out, "final"))
        print(f"done in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
