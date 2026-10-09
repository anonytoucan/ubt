"""Merge a Gemma-2-9B QLoRA checkpoint of scripts/train/train_sft_u1.py (--braille-vocab) into full bf16 weights for vLLM.

Same steps as training: base weights (the ungated unsloth/gemma-2-9b mirror) with the braille tokenizer, embeddings
resized to a multiple of 64 without mean resizing, then the LoRA and the trained cell-token rows merged (input and
output embeddings are tied); the padding rows are cut before saving.

  python scripts/train/gemma_merge.py --adapter <checkpoint dir> --tokenizer <gemma2_cell_tokenizer> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="unsloth/gemma-2-9b")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    import torch  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map=a.device,
                                                 attn_implementation="eager")
    model.resize_token_embeddings(len(tok), pad_to_multiple_of=64, mean_resizing=False)
    cfg = json.load(open(os.path.join(a.adapter, "adapter_config.json")))
    ids = cfg["trainable_token_indices"]["embed_tokens"]
    model = PeftModel.from_pretrained(model, a.adapter)
    delta = next(p for n, p in model.named_parameters() if "trainable_tokens_delta" in n).detach().float().clone()
    model = model.merge_and_unload()
    emb = model.get_input_embeddings().weight
    assert torch.equal(emb[ids].float(), delta.to(emb.dtype).float()), "cell-token rows not merged"
    # cut the random padding rows: this draw differs from training's and, through the tied output layer, can outscore
    # the right token and silently drop characters
    model.resize_token_embeddings(len(tok))
    emb = model.get_input_embeddings().weight
    assert emb.shape[0] == len(tok) == model.config.vocab_size
    assert model.config.tie_word_embeddings and model.lm_head.weight.data_ptr() == emb.data_ptr(), "output layer not tied"
    model.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    json.dump({"adapter": os.path.abspath(a.adapter), "tokenizer": os.path.abspath(a.tokenizer), "base": a.base,
               "vocab_rows": emb.shape[0], "tokenizer_len": len(tok), "cell_token_rows": len(ids)},
              open(os.path.join(a.out, "merge_info.json"), "w"), indent=1)
    print(f"merged -> {a.out}: {emb.shape[0]} rows (tokenizer {len(tok)}), {len(ids)} cell-token rows")


if __name__ == "__main__":
    main()
