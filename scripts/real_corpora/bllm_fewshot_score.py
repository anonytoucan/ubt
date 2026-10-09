"""Score the BrailleLLM few-shot pools with the real-corpora scorer.

Candidates keep the prefilled example segments, so each is cut to its last segment (the target) and scored against the
single-segment BrailleLLM rows, the same way as the zero-shot pools.

  python scripts/real_corpora/bllm_fewshot_score.py --pools work/bllm/run --mode rand
  -> <pools>/score_<mode>/report.{json,md}  (both condition slots hold the same pool)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from ubt.paths import WORK  # noqa: E402

ROWS = f"{WORK}/real_corpora/run/real.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--mode", required=True, choices=["rand", "ret"])
    a = ap.parse_args()
    out = os.path.join(a.pools, f"score_{a.mode}")
    os.makedirs(out, exist_ok=True)
    rows = [json.loads(line) for line in open(ROWS, encoding="utf-8")]
    rows = [r for r in rows if r["group"] == "braillellm"]
    with open(os.path.join(out, "rows.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    files = sorted(glob.glob(os.path.join(a.pools, f"gen_fs_{a.mode}.*.jsonl")))
    n = 0
    for i, p in enumerate(files):
        dst = os.path.join(out, f"gen_given.{i}.jsonl")
        with open(p, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
            for line in fi:
                x = json.loads(line)
                for c in x["cands"]:
                    c["text"] = c["text"].rsplit("\n", 1)[-1]           # the target segment only
                fo.write(json.dumps(x, ensure_ascii=False) + "\n")
                n += 1
        link = os.path.join(out, f"gen_inferred.{i}.jsonl")
        if not os.path.exists(link):
            os.symlink(os.path.basename(dst), link)
    print(f"{n} documents from {len(files)} shards -> {out}", flush=True)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.exit(subprocess.call([sys.executable, os.path.join(here, "real_corpora_score.py"), "--dir", out,
                              "--rows", os.path.join(out, "rows.jsonl")]))


if __name__ == "__main__":
    main()
