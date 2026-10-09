#!/usr/bin/env python3
"""Enforce train/eval sentence-hash disjointness.

Drops from wiki_txt_B/*.txt every sentence whose hash occurs in a train source (wiki_txt_A or FLORES dev, any language),
which also catches cross-language duplicates. FLORES devtest is not edited: colliding hashes go to
data/eval_exclusions.txt and are filtered at load time.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from ubt.paths import DATASETS  # noqa: E402
from ubt.sources import sentence_hash  # noqa: E402

DATA = os.environ.get("UBT_DATA", f"{DATASETS}/data")


def train_hashes() -> set[str]:
    out: set[str] = set()
    for d, suf in ((f"{DATA}/flores200_dataset/dev", ".dev"),
                   (f"{DATA}/wiki_txt_A", ".txt")):
        for f in sorted(os.listdir(d)):
            if f.endswith(suf):
                with open(os.path.join(d, f), encoding="utf-8") as fh:
                    out |= {sentence_hash(ln) for ln in fh if ln.strip()}
    return out


def main() -> None:
    train = train_hashes()
    dropped = {}
    for f in sorted(os.listdir(f"{DATA}/wiki_txt_B")):
        if not f.endswith(".txt"):
            continue
        path = os.path.join(DATA, "wiki_txt_B", f)
        with open(path, encoding="utf-8") as fh:
            lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
        kept = [ln for ln in lines if sentence_hash(ln) not in train]
        if len(kept) != len(lines):
            dropped[f] = len(lines) - len(kept)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(kept) + "\n")

    exclusions: set[str] = set()
    d = f"{DATA}/flores200_dataset/devtest"
    for f in sorted(os.listdir(d)):
        if f.endswith(".devtest"):
            with open(os.path.join(d, f), encoding="utf-8") as fh:
                exclusions |= {h for ln in fh if ln.strip()
                               and (h := sentence_hash(ln)) in train}
    with open(f"{DATA}/eval_exclusions.txt", "w") as fh:
        fh.write("\n".join(sorted(exclusions)) + "\n")

    print(json.dumps({"wiki_B_dropped": dropped,
                      "flores_devtest_excluded_hashes": len(exclusions)}))


if __name__ == "__main__":
    main()
