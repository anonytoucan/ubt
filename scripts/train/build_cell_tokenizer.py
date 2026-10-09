"""Tokenizer with the 256 braille cells (U+2800-U+28FF) as single tokens, for train_sft_u1.py --braille-vocab.

  python scripts/train/build_cell_tokenizer.py --model Qwen/Qwen2.5-7B --out work/models/qwen25_cell_tokenizer
  python scripts/train/build_cell_tokenizer.py --model google/gemma-2-9b --out work/models/gemma2_cell_tokenizer --keep-single

Default: every cell is added as a new token in code-point order (for Qwen2.5 the new ids fit in the padded embedding,
so it is not resized). --keep-single: cells that already are one token keep their id. Other tokens are unaffected.
"""

from __future__ import annotations

import argparse

from transformers import AddedToken, AutoTokenizer

CELLS = [chr(c) for c in range(0x2800, 0x2900)]


def single_id(tok, cell: str) -> int | None:
    ids = tok(cell, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-single", action="store_true", help="reuse the ids of cells that already are one token")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    n0 = len(tok)
    new = [c for c in CELLS if not (a.keep_single and single_id(tok, c) is not None)]
    tok.add_tokens([AddedToken(c, normalized=False, special=False) for c in new])
    ids = [single_id(tok, c) for c in CELLS]
    bad = [c for c, i in zip(CELLS, ids) if i is None]
    if bad or len(set(ids)) != len(CELLS):
        raise SystemExit(f"{len(bad)} cells are not a single token, e.g. {bad[:3]}")
    tok.save_pretrained(a.out)
    print(f"{a.model}: {len(new)} cells added (vocabulary {n0} -> {len(tok)}), "
          f"{len(CELLS) - len(new)} kept; new ids {min(ids[i] for i, c in enumerate(CELLS) if c in new) if new else '-'}"
          f"-{max(ids[i] for i, c in enumerate(CELLS) if c in new) if new else '-'} -> {a.out}")


if __name__ == "__main__":
    main()
