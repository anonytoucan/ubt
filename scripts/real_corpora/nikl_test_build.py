#!/usr/bin/env python3
"""Sealed NIKL test rows (real-corpora table, NIKL rows) for the final model.

Local only: rows and per-document outputs stay under --out, outside every git repository, and only aggregate numbers
go into the paper. The test is never used for training or selection (licence-bound).

  python scripts/real_corpora/nikl_test_build.py [--out work/nikl_test/run]
      -> <out>/real.jsonl (scripts/real_corpora/real_corpora_score.py input, group "nikl"), <out>/../meta.json

Rows join the data_u1_v2 test pointers with the NIKL pairs and are built like the NIKL dev rows: human braille kept
(targets with other non-braille characters dropped), text NFC, label ko-2024-g2.ctb, f_consistent = forward(text)
equals the human braille. Row format as the other real corpora (build_real_corpora.make_rows).
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path[:0] = sorted(glob.glob(os.path.join(REPO, "scripts", "*", "")))
from ubt.paths import DATASETS, PAPER, WORK  # noqa: E402
POINTER = f"{DATASETS}/data_u1_v2/pointers/nikl_test_pointer.jsonl"
TOKENIZER = f"{WORK}/models/qwen25_cell_tokenizer"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{WORK}/nikl_test/run")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    for repo in (REPO, PAPER):
        assert not out.startswith(os.path.abspath(repo) + os.sep), f"NIKL outputs must stay outside git ({repo})"
    import build_real_corpora as BR  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415
    from ubt.u1.nikl import human_braille, load_nikl  # noqa: PLC0415
    ptr = [json.loads(line) for line in open(POINTER, encoding="utf-8")]
    assert all(p["split"] == "test" and p["table"] == "ko-2024-g2.ctb" for p in ptr)
    pairs = {r["id"]: r for r in load_nikl(REPO)}
    stats = collections.Counter()
    docs = []
    for p in ptr:
        r = pairs.get(p["nikl_id"])
        if r is None:
            stats["missing_pair"] += 1
            continue
        assert r["split"] == "test", p["nikl_id"]
        b = human_braille(r["tgt"])
        if b is None:
            stats["dropped_non_braille_target"] += 1
            continue
        t = BR.nfc(r["src"], stats)
        if "\n" in t or "\n" in b or not t or not b:
            stats["dropped_multiline_or_empty"] += 1
            continue
        docs.append({"id": f"nikl-test-{p['nikl_id']}", "text": t, "braille": b})
    fwd = BR.Forward()
    try:
        f = fwd.many([("ko-2024-g2.ctb", d["text"]) for d in docs])
    finally:
        fwd.close()
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    rows, meta = BR.make_rows(docs, "ko-2024-g2.ctb", "nikl", f, tok)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "real.jsonl"), "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta.update({"pointer": POINTER, "pointer_rows": len(ptr), "rows": len(rows), "drops": dict(stats),
                 "pointer_sha256": hashlib.sha256(open(POINTER, "rb").read()).hexdigest()})
    json.dump(meta, open(os.path.join(os.path.dirname(out), "meta.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(meta, ensure_ascii=False, indent=1)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
