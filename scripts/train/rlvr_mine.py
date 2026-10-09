"""RLVR difficulty mining: keep the prompts on which the initial policy's samples differ.

Input: the prompt pool and scripts/train/rlvr_sample.py's G samples per prompt. A prompt is dropped when every sample is
exact, feasible and correctly coded (all_perfect) or no sample earns any reward (all_zero). Prompts whose samples tie
for another reason are kept (other_zero_var), since at G=8 in training they can still split. Kept rows are written
unchanged in pool order; stats go to <out>.stats.json.

  python scripts/train/rlvr_mine.py --pool pool.jsonl --samples '<dir>/samples.*.jsonl' --out pool_mined.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict


def verdict(samples: list[dict]) -> str:
    if all(s["exact"] and s["feasible"] and s["conf_ok"] for s in samples):
        return "all_perfect"
    if max(s["reward"] for s in samples) <= 0.0:
        return "all_zero"
    if len({round(s["reward"], 9) for s in samples}) == 1:
        return "other_zero_var"
    return "mixed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--samples", required=True, help="glob of rlvr_sample.py output shards")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(a.samples))
    if not files:
        raise SystemExit(f"no files match {a.samples}")
    got: dict[str, dict] = {}
    for p in files:
        for line in open(p, encoding="utf-8"):
            x = json.loads(line)
            got[x["id"]] = x
    pool = [json.loads(line) for line in open(a.pool, encoding="utf-8")]
    missing = [r["id"] for r in pool if r["id"] not in got]
    if missing:
        raise SystemExit(f"{len(missing)} pool prompts have no samples (first {missing[:3]})")
    G = Counter(len(got[r["id"]]["samples"]) for r in pool)
    by_form: dict[str, Counter] = defaultdict(Counter)
    by_script: dict[str, Counter] = defaultdict(Counter)
    kept, total = [], Counter()
    reward_all, reward_kept = [], []
    for r in pool:
        s = got[r["id"]]["samples"]
        v = verdict(s)
        total[v] += 1
        by_form[r["form"]][v] += 1
        fam = "k>1" if len(r["tables"]) > 1 else r["tables"][0].split("-")[0].split(".")[0]
        by_script[fam][v] += 1
        m = sum(x["reward"] for x in s) / len(s)
        reward_all.append(m)
        if v in ("mixed", "other_zero_var"):
            kept.append(r)
            reward_kept.append(m)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    top = sorted(by_script.items(), key=lambda kv: -sum(kv[1].values()))
    stats = {"pool": os.path.abspath(a.pool), "samples": files, "G": dict(G), "n_pool": len(pool),
             "n_kept": len(kept), "verdicts": dict(total),
             "mean_reward_pool": round(sum(reward_all) / len(reward_all), 4),
             "mean_reward_kept": round(sum(reward_kept) / max(1, len(reward_kept)), 4),
             "by_form": {f: dict(c) for f, c in sorted(by_form.items())},
             "by_code_prefix": {f: dict(c) for f, c in top}}
    json.dump(stats, open(a.out + ".stats.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: stats[k] for k in ("G", "n_pool", "n_kept", "verdicts", "mean_reward_pool",
                                            "mean_reward_kept", "by_form")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
