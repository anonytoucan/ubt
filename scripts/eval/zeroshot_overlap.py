"""Held-out vs trained codes on the single-code zeroshot_eval sentences.

Per held-out code: the share of sentences whose cells some trained code also emits for the same text, and the nearest
trained code by mean cell similarity (1 - normalised cell edit distance).

  python scripts/eval/zeroshot_overlap.py --out work/analysis/zeroshot_cell_overlap.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from ubt.paths import DATASETS  # noqa: E402
DATA = f"{DATASETS}/data_u1_v2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415

    from ubt.u1.routed_louis import RoutedLouis  # noqa: PLC0415
    inv = json.load(open(os.path.join(DATA, "inventory.json")))["tables"]
    trained = sorted(t["table_id"] for t in inv if (t.get("weight") or 0) > 0 and not t.get("holdout"))
    docs = collections.defaultdict(list)
    for line in open(os.path.join(DATA, "eval", "zeroshot_eval.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        if d["doc_type"] == "single":
            docs[d["tables"][0]].append(d)
    items = sorted({(t, d["text"]) for ds in docs.values() for d in ds for t in trained})
    L = RoutedLouis(DATA, timeout_s=300.0)
    enc = dict(zip(items, L.translate_many(items)))
    L.close()
    out = {}
    for code, ds in sorted(docs.items()):
        sims = collections.defaultdict(list)
        ident = 0
        for d in ds:
            hit = False
            for t in trained:
                b = enc.get((t, d["text"]))
                s = 0.0 if not b else 1 - Levenshtein.normalized_distance(b, d["braille"])
                sims[t].append(s)
                hit |= bool(b) and b == d["braille"]
            ident += hit
        mean = {t: sum(v) / len(v) for t, v in sims.items()}
        best = max(mean, key=mean.get)
        per_sent_best = [max(sims[t][i] for t in trained) for i in range(len(ds))]
        out[code] = {"docs": len(ds), "identical_%": 100 * ident / len(ds), "nearest_trained_code": best,
                     "nearest_mean_similarity_%": 100 * mean[best],
                     "mean_best_cell_similarity_%": 100 * sum(per_sent_best) / len(ds),
                     "top3": sorted(((round(100 * v, 1), t) for t, v in mean.items()), reverse=True)[:3]}
        print(code, json.dumps(out[code], ensure_ascii=False))
    json.dump({"trained_codes": len(trained), "codes": out}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
