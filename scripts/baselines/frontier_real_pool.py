"""GPT-5.5 few-shot outputs on a real corpus as a one-candidate pool for scripts/real_corpora/real_corpora_score.py.

As for the LibLouis baseline, the output is set under the gold code label (the model emits none) and real.jsonl keeps
only documents with a result; both scorer conditions get the same candidate.

  python scripts/baselines/frontier_real_pool.py --frontier work/frontier_bible \
      --rows work/real_corpora/run/real.jsonl --out work/evals/gpt55/real
  python scripts/real_corpora/real_corpora_score.py --dir work/evals/gpt55/real
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path[:0] = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "*", "")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import frontier_fewshot as F  # noqa: PLC0415
    res = {}
    for p in sorted(glob.glob(os.path.join(a.frontier, "batch_output.part*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            x = json.loads(line)
            t = F._text_of((x.get("response") or {}).get("body") or {})
            t = re.sub(r"^\s*(text\s*:|the text is\s*:?|transcription\s*:)\s*", "", t, flags=re.I).strip().strip('"').strip()
            res[x["custom_id"]] = t.replace("\n", " ")
    rows = [r for r in map(json.loads, open(a.rows, encoding="utf-8")) if r["id"] in res]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "real.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for cond in ("given", "inferred"):
        with open(os.path.join(a.out, f"gen_{cond}.0.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                cand = {"text": f"⟨{r['tables'][0]}⟩{res[r['id']]}", "lp": 0.0, "finish": "stop", "greedy": True}
                fh.write(json.dumps({"id": r["id"], "cands": [cand]}, ensure_ascii=False) + "\n")
        open(os.path.join(a.out, f"gen_{cond}.0.log"), "w").write(f"gpt-5.5 few-shot ({a.frontier})\n")
    print(f"{len(rows)} documents with an output ({sum(not v for v in res.values())} empty) -> {a.out}")


if __name__ == "__main__":
    main()
