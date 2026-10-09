"""RLVR (GRPO/DAPO-style) for braille -> print with verifiable rewards.

Policy: the merged SFT model plus a fresh LoRA; the reference (used only if beta > 0) is the same model with the LoRA
off. Reward: ubt.rlvr.reward (CER, re-encoding feasibility, exact match, code label), logged per component.
Rollouts run on a vLLM server on its own GPU, training on the others.

  # GPU 0: rollouts
  CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model <merged> --max-model-len 3072 --gpu-memory-utilization 0.9
  # GPUs 1-3: training
  CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes 3 scripts/train/train_rlvr.py \
      --model <merged> --train work/data/pool_agn_mined.jsonl --out <run dir> ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from ubt.paths import DATASETS  # noqa: E402

_POOL = None
_CFG = None
_CACHE: dict = {}


def _pool():
    global _POOL
    if _POOL is None:
        from ubt.rlvr.louis_pool import LouisPool  # noqa: PLC0415
        d = os.environ.get("RLVR_DATA", f"{DATASETS}/data_u1_v2")
        man = json.load(open(os.path.join(d, "manifest.json")))
        inv = json.load(open(os.path.join(d, "inventory.json")))
        _POOL = LouisPool(man["engine_spec"], known_tables={t["table_id"] for t in inv["tables"]},
                      overrides=man.get("engine_spec_overrides"))
    return _POOL


def _scores(completions, gold, braille, confusable_set):
    """Score once per batch; the four component reward functions read the same result."""
    from ubt.rlvr.reward import RewardConfig, score_batch  # noqa: PLC0415
    global _CFG
    _CFG = _CFG or RewardConfig()
    key = (tuple(completions), tuple(gold))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = score_batch(list(completions), list(gold), list(braille),
                                  [list(c) for c in confusable_set], _pool(), _CFG)
    return _CACHE[key]


def _text(c):
    # standard (non-chat) format: completions are strings
    return c if isinstance(c, str) else c[0]["content"]


def r_cer(prompts, completions, gold, braille, confusable_set, **kw):
    s = _scores([_text(c) for c in completions], gold, braille, confusable_set)
    return [(-0.5 if not x["parse_ok"] else max(0.0, 1.0 - x["cer"])) for x in s]


def r_feasible(prompts, completions, gold, braille, confusable_set, **kw):
    s = _scores([_text(c) for c in completions], gold, braille, confusable_set)
    return [float(x["feasible"]) for x in s]


def r_exact(prompts, completions, gold, braille, confusable_set, **kw):
    s = _scores([_text(c) for c in completions], gold, braille, confusable_set)
    return [float(x["exact"]) for x in s]


def r_label(prompts, completions, gold, braille, confusable_set, **kw):
    s = _scores([_text(c) for c in completions], gold, braille, confusable_set)
    return [float(x["conf_ok"]) for x in s]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged SFT model dir")
    ap.add_argument("--train", required=True, help="prompt pool jsonl (rlvr_build_prompts/mined)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--per-device-bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--max-prompt-len", type=int, default=1536,
                    help="longer prompts abort the run; nothing is truncated")
    ap.add_argument("--max-completion-len", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--loss-type", default="dapo")
    ap.add_argument("--eps-low", type=float, default=0.2)
    ap.add_argument("--eps-high", type=float, default=0.28)
    ap.add_argument("--weights", default="1.0,0.5,0.25,0.25", help="cer,feasible,exact,label")
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--vllm-host", default="127.0.0.1")
    ap.add_argument("--vllm-port", type=int, default=8000)
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--resume", action="store_true", help="continue from the newest checkpoint-* in --out, if any")
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    from datasets import load_dataset  # noqa: PLC0415
    from peft import LoraConfig  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    from trl import GRPOConfig, GRPOTrainer  # noqa: PLC0415

    ds = load_dataset("json", data_files=args.train, split="train")
    keep = {"prompt", "gold", "braille", "confusable_set", "id", "form"}
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    too_long = sum(len(tok(p, add_special_tokens=False)["input_ids"]) > args.max_prompt_len for p in ds["prompt"])
    if too_long:
        sys.exit(f"{too_long} prompts exceed --max-prompt-len {args.max_prompt_len} (rebuild the pool with this tokenizer)")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    peft_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"],
                          task_type="CAUSAL_LM")
    w = [float(x) for x in args.weights.split(",")]
    cfg = GRPOConfig(
        output_dir=args.out, run_name=args.run_name, seed=args.seed,
        learning_rate=args.lr, lr_scheduler_type="constant_with_warmup", warmup_steps=10,
        per_device_train_batch_size=args.per_device_bs, gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps, bf16=True, gradient_checkpointing=True,
        # TRL >= 1.x never truncates prompts, so the pool is length-checked above
        num_generations=args.num_generations, max_completion_length=args.max_completion_len,
        temperature=args.temperature, top_p=1.0,
        beta=args.beta, loss_type=args.loss_type, epsilon=args.eps_low, epsilon_high=args.eps_high,
        mask_truncated_completions=True, reward_weights=w,
        use_vllm=True, vllm_mode="server", vllm_server_host=args.vllm_host,
        vllm_server_port=args.vllm_port,
        logging_steps=1, save_steps=args.save_steps, save_total_limit=20,
        report_to=args.report_to, log_completions=True, num_completions_to_print=2,
    )
    trainer = GRPOTrainer(model=model, processing_class=tok, args=cfg, train_dataset=ds,
                          reward_funcs=[r_cer, r_feasible, r_exact, r_label], peft_config=peft_cfg)
    last = None
    if args.resume and os.path.isdir(args.out):
        ck = [d for d in os.listdir(args.out) if d.startswith("checkpoint-") and d.split("-")[1].isdigit()]
        last = os.path.join(args.out, max(ck, key=lambda d: int(d.split("-")[1]))) if ck else None
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(os.path.join(args.out, "final"))


if __name__ == "__main__":
    main()
