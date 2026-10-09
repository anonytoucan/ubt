#!/usr/bin/env python3
"""Convention-adaptation data for Vision-Braille: fine-tune on the release's train split, test on the same 1,000 docs.

  python scripts/real_corpora/vision_adapt_build.py [--out work/data/vision_adapt] [--n-train 51200] [--n-dev 500]

51,200 train documents (400 LoRA steps at batch 128) are drawn without replacement; dev = the first 500 validation
rows. Documents are filtered and normalised as the test rows are (build_real_corpora.build_vision) and labelled
zhcn-g1.ctb. Train rows whose text equals a test document's are dropped, since the release repeats some sentences
across splits. Records use the data-u1 single-segment layout that train_sft_u1.py reads.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from ubt.paths import DATASETS, WORK  # noqa: E402
VB = f"{DATASETS}/external/vision_braille_sentence/hf_snapshot/data"
CODE = "zhcn-g1.ctb"
TEST_ROWS = f"{WORK}/real_corpora/vision.jsonl"      # the 1,000 test documents (build_real_corpora.py)


def records(split: str, idx: list[int], stats: collections.Counter, banned: frozenset = frozenset()) -> list[dict]:
    import pyarrow.parquet as pq  # noqa: PLC0415
    table = pq.read_table(os.path.join(VB, f"{split}-00000-of-00001.parquet"))
    rows = table.take(idx).to_pylist() if idx is not None else table.to_pylist()
    out = []
    for i, r in zip(idx, rows):
        text, cells = r["chinese_text"] or "", r["braille_text"] or ""
        if not text or not cells or any(not 0x2800 <= ord(ch) <= 0x28FF for ch in cells):
            stats[f"{split}_dropped_non_braille_or_empty"] += 1
            continue
        t = unicodedata.normalize("NFC", text)
        stats[f"{split}_nfc_changed"] += t != text
        if "\n" in t or "\n" in cells:
            stats[f"{split}_dropped_multiline"] += 1
            continue
        if t in banned:
            stats[f"{split}_dropped_text_in_test"] += 1
            continue
        out.append({"id": f"vision-{split}-{i:06d}", "doc_type": "single", "k": 1, "regime": "single",
                    "switch_density": "none", "len_bucket": "sentence", "cells": len(cells), "text": t,
                    "braille": cells, "tables": [CODE], "langs": ["zh"], "confusable_set": [], "segments": None,
                    "fiber_ambiguous": False, "source": "vision_braille_train" if split == "train" else "vision_braille_dev",
                    "n_sentences": 1, "tag": f"vision:{split}", "table_status": ["current"], "form": "VA",
                    "split": "train" if split == "train" else "dev"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{WORK}/data/vision_adapt")
    ap.add_argument("--n-train", type=int, default=51200)
    ap.add_argument("--n-dev", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260925)
    a = ap.parse_args()
    import pyarrow.parquet as pq  # noqa: PLC0415
    n_train = pq.ParquetFile(os.path.join(VB, "train-00000-of-00001.parquet")).metadata.num_rows
    rng = random.Random(a.seed)
    idx = sorted(rng.sample(range(n_train), min(n_train, a.n_train + a.n_train // 50)))   # 2% spare for drops
    stats = collections.Counter()
    banned = frozenset(json.loads(line)["text"] for line in open(TEST_ROWS, encoding="utf-8"))
    assert len(banned) >= 990, len(banned)
    train = records("train", idx, stats, banned)
    rng.shuffle(train)
    train = train[: a.n_train]
    dev = records("validation", list(range(a.n_dev)), stats, banned)
    os.makedirs(a.out, exist_ok=True)
    for name, rows in (("train", train), ("dev", dev)):
        with open(os.path.join(a.out, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"source": VB, "code": CODE, "train_split_rows": n_train, "train": len(train), "dev": len(dev),
            "seed": a.seed, "stats": dict(stats),
            "median_cells_train": sorted(r["cells"] for r in train)[len(train) // 2]}
    json.dump(meta, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
