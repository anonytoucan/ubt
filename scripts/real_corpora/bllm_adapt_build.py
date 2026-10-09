"""BrailleLLM test set (paper Table 5): in-domain data for (a) adaptation training and (b) few-shot decoding.

The source corpus Data/ACL_Chinese.txt contains the 1,000 test lines, so lines whose text or cells equal a test line's
are removed first. Cells are converted as in scripts/real_corpora/build_real_corpora.py build_braillellm.
- (a) 10,000 train pairs (the size BrailleLLM's specialist uses) + 500 dev as single-code zhcn-cbs.ctb records, plus
  --replay main-corpus docs -> <out>/adapt/{train.jsonl, dev.jsonl, manifest.json} for train_sft_u1.py.
- (b) the 1,000 test rows with k=3 in-domain examples prefilled as earlier segments, so the model writes only the
  target; examples are random short pairs (rand) or the pairs sharing the most cell 4-grams with the target's cells
  (ret, input only) -> <out>/fewshot_{rand,ret}.jsonl.

  python scripts/real_corpora/bllm_adapt_build.py --out work/data/bllm
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import random
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path[:0] = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "*", "")))
from ubt.paths import DATASETS, WORK  # noqa: E402
REPO = f"{DATASETS}/external/braillellm_acl_zh/repo/Data"
TEST_ROWS = f"{WORK}/real_corpora/braillellm.jsonl"
REPLAY_SRC = f"{WORK}/data/fixed_split"
CODE = "zhcn-cbs.ctb"


def _lines(path: str) -> list[str]:
    raw = open(path, "rb").read().decode("utf-8")
    out = [ln.removesuffix("\r") for ln in raw.split("\n")]
    return out[:-1] if out and out[-1] == "" else out


def _record(i: int, text: str, cells: str, split: str) -> dict:
    return {"id": f"bllm-{split}-{i:05d}", "doc_type": "single", "k": 1, "regime": "single", "switch_density": "none",
            "len_bucket": "sentence", "cells": len(cells), "text": text, "braille": cells, "tables": [CODE],
            "langs": ["cmn"], "confusable_set": [], "segments": None, "fiber_ambiguous": False,
            "source": "braillellm_acl_chinese", "n_sentences": 1, "tag": "BLLM", "table_status": ["parallel"],
            "form": "BLLM", "split": split}


def _grams(cells: str, n: int = 4) -> set[str]:
    return {cells[i:i + n] for i in range(max(0, len(cells) - n + 1))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-dev", type=int, default=500)
    ap.add_argument("--replay", type=int, default=10000)
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--shot-max-cells", type=int, default=60)
    ap.add_argument("--seed", type=int, default=147)
    ap.add_argument("--tokenizer", default=f"{WORK}/models/qwen25_cell_tokenizer")
    a = ap.parse_args()
    from transformers import AutoTokenizer  # noqa: PLC0415

    import build_real_corpora as R  # noqa: PLC0415
    from bible_pairs import ascii_to_cells  # noqa: PLC0415
    from ubt.task_format import render  # noqa: PLC0415
    fwd = R.Forward()
    to_cell, dis = R.brf_display(fwd.spec)
    fwd.close()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    test = [json.loads(line) for line in open(TEST_ROWS, encoding="utf-8")]
    test_text = {r["text"] for r in test}
    test_cells = {r["braille"] for r in test}
    stats = collections.Counter()
    pool, seen = [], set()
    for ln in _lines(os.path.join(REPO, "ACL_Chinese.txt")):
        if "\t" not in ln:
            stats["no_tab"] += 1
            continue
        text, brf = ln.split("\t", 1)
        text = unicodedata.normalize("NFC", text)
        cells = ascii_to_cells(brf, to_cell)
        if not text or not cells or "�" in cells or "\n" in text:
            stats["unusable"] += 1
            continue
        if text in test_text or cells in test_cells:
            stats["test_overlap_removed"] += 1
            continue
        if text in seen:
            stats["duplicate_text"] += 1
            continue
        seen.add(text)
        pool.append((text, cells))
    rng = random.Random(a.seed)
    rng.shuffle(pool)
    train, dev = pool[:a.n_train], pool[a.n_train:a.n_train + a.n_dev]
    stats["pool_after_filters"] = len(pool)

    # (a) adaptation data + replay
    d = os.path.join(a.out, "adapt")
    os.makedirs(d, exist_ok=True)
    recs = [_record(i, t, c, "train") for i, (t, c) in enumerate(train)]
    replay = []
    if a.replay:
        with open(os.path.join(REPLAY_SRC, "train.jsonl"), encoding="utf-8") as fh:
            lines = fh.readlines()
        replay = [json.loads(x) for x in random.Random(a.seed).sample(lines, a.replay)]
    mix = recs + replay
    random.Random(a.seed + 1).shuffle(mix)
    with open(os.path.join(d, "train.jsonl"), "w", encoding="utf-8") as fh:
        for r in mix:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(d, "dev.jsonl"), "w", encoding="utf-8") as fh:
        for i, (t, c) in enumerate(dev):
            fh.write(json.dumps(_record(i, t, c, "dev"), ensure_ascii=False) + "\n")
    man = json.load(open(os.path.join(REPLAY_SRC, "manifest.json"), encoding="utf-8"))
    man["bllm_adapt"] = {"script": "scripts/real_corpora/bllm_adapt_build.py", "argv": sys.argv, "seed": a.seed,
                         "n_train_indomain": len(recs), "n_replay": len(replay), "n_dev_indomain": len(dev),
                         "source": os.path.join(REPO, "ACL_Chinese.txt"),
                         "source_sha256": hashlib.sha256(open(os.path.join(REPO, "ACL_Chinese.txt"), "rb").read()).hexdigest(),
                         "display_table": dis, "filters": dict(stats),
                         "test_exclusion": "text or cells equal to any of the 1,000 test lines (ACL_Chinese_Test.txt)"}
    json.dump(man, open(os.path.join(d, "manifest.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # (b) few-shot rows
    short = [(t, c) for t, c in train if len(c) <= a.shot_max_cells]
    grams = [_grams(c) for _t, c in short]
    index = collections.defaultdict(list)
    for j, g in enumerate(grams):
        for x in g:
            index[x].append(j)
    for mode in ("rand", "ret"):
        r2 = random.Random(a.seed + (0 if mode == "rand" else 7))
        out = []
        for r in test:
            if mode == "rand":
                ex = r2.sample(short, a.shots)
            else:
                score = collections.Counter()
                for x in _grams(r["braille"]):
                    for j in index.get(x, ()):
                        score[j] += 1
                ex = [short[j] for j, _s in score.most_common(a.shots)]
                while len(ex) < a.shots:
                    ex.append(r2.choice(short))
            segs = [{"table": CODE, "text": t, "braille": c} for t, c in ex] + \
                   [{"table": CODE, "text": r["text"], "braille": r["braille"]}]
            rec = {"id": r["id"], "doc_type": "single", "k": len(segs), "regime": "single", "switch_density": "none",
                   "text": "\n".join(s["text"] for s in segs), "braille": "\n".join(s["braille"] for s in segs),
                   "tables": [CODE] * len(segs), "segments": segs}
            ex_ = render(rec, hinted=False)
            target = f"⟨{CODE}⟩{r['text']}"
            assert ex_["completion"].endswith(target)
            prefill = ex_["completion"][: -len(r["text"])]
            n_c = len(tok(r["text"], add_special_tokens=False)["input_ids"])
            out.append({"id": r["id"], "set": "real", "group": f"braillellm_fs_{mode}", "tables": [CODE], "k": 1,
                        "regime": "single", "switch_density": "none", "confusable_set": [],
                        "prompt": ex_["prompt"], "completion": target, "prefill": prefill, "fewshot_k": a.shots,
                        "max_tokens": 2 * n_c + 32, "text": r["text"], "braille": r["braille"],
                        "f_consistent": r.get("f_consistent"), "cell_similarity": r.get("cell_similarity")})
        with open(os.path.join(a.out, f"fewshot_{mode}.jsonl"), "w", encoding="utf-8") as fh:
            for x in out:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps({"filters": dict(stats), "train": len(recs), "replay": len(replay), "dev": len(dev),
                      "shot_pool": len(short)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
