"""LibLouis backward baseline (code given) on the real-braille corpora, written as a one-candidate pool.

scripts/real_corpora/real_corpora_score.py scores the pool exactly like the model's. Each document is back-translated under its gold
code with the data's own engine, one table per fresh process (the worker of scripts/baselines/liblouis_backward_eval.py). The
same candidate is written for both scorer conditions; only "given" is meaningful.

  python scripts/baselines/real_liblouis_baseline.py --rows work/real_corpora/run/real.jsonl \
      --out work/evals/liblouis/real
  python scripts/real_corpora/real_corpora_score.py --dir work/evals/liblouis/real
The sealed NIKL test uses its own rows (--rows work/nikl_test/run/real.jsonl) and a directory outside git; its outputs
stay local.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path[:0] = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "*", "")))
from ubt.paths import DATASETS  # noqa: E402
CHUNK = 2000        # lines per job; each job is a fresh process with one table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine-data", default=f"{DATASETS}/data_u1_v2")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    import bon_gen as E  # noqa: PLC0415
    from liblouis_backward_eval import _bt_table  # noqa: PLC0415
    man = json.load(open(os.path.join(a.engine_data, "manifest.json")))
    base, over = man["engine_spec"], man.get("engine_spec_overrides") or {}
    rows = [json.loads(line) for line in open(a.rows, encoding="utf-8")]
    by = collections.defaultdict(list)                     # table -> [(doc idx, cells)]
    for di, r in enumerate(rows):
        cells = E._cells(r["prompt"])
        assert "\n" not in cells, r["id"]                  # every real-corpus document is one line under one code
        by[r["tables"][0]].append((di, cells))
    jobs, where = [], []
    for t, v in by.items():
        for k in range(0, len(v), CHUNK):
            jobs.append((t, over.get(t, base), [c for _, c in v[k:k + CHUNK]]))
            where.append([di for di, _ in v[k:k + CHUNK]])
    text = {}
    with mp.get_context("spawn").Pool(min(a.workers, len(jobs)), maxtasksperchild=1) as pool:
        for j, (_t, out) in enumerate(pool.imap(_bt_table, jobs)):
            for di, x in zip(where[j], out):
                text[di] = x
    os.makedirs(a.out, exist_ok=True)
    fails = sum(text[di] is None for di in range(len(rows)))
    for cond in ("given", "inferred"):
        with open(os.path.join(a.out, f"gen_{cond}.0.jsonl"), "w", encoding="utf-8") as fh:
            for di, r in enumerate(rows):
                cand = {"text": f"⟨{r['tables'][0]}⟩{text[di] or ''}", "lp": 0.0, "finish": "stop", "greedy": True}
                fh.write(json.dumps({"id": r["id"], "cands": [cand]}, ensure_ascii=False) + "\n")
        with open(os.path.join(a.out, f"gen_{cond}.0.log"), "w", encoding="utf-8") as fh:
            fh.write(f"liblouis backward, code given ({man.get('liblouis_version', 'LibLouis 3.38 fork')}; {a.rows})\n")
    if os.path.abspath(a.rows) != os.path.join(os.path.abspath(a.out), "real.jsonl"):
        with open(os.path.join(a.out, "real.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} documents, {len(jobs)} jobs, {fails} back-translation failures -> {a.out}")


if __name__ == "__main__":
    main()
