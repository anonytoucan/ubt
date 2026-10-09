"""95% confidence intervals for the measured rows of Table 1, from the per-document records of tracking_table.py.

Method: bootstrap over documents, 1,000 resamples with replacement, 95% percentile interval. The first five rows share
one stream (default_rng(0), in row order); each later row i uses default_rng([0, i]), so adding a row moves no other
interval. bCER and CER are ratios of sums over the resampled documents, code-ID strict / conf are shares of documents,
sim is the mean bge-m3 cosine x 100 and judge the mean judge score (an empty output scores 1). The table reports the
half-width (hi - lo) / 2.

  python scripts/paper/table1_ci.py [--out work/paper/table1_ci.json]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from ubt.paths import WORK  # noqa: E402

R = WORK
B = 1000
SEED = 0


def pool(path, sl, pol):
    """Per-document records of one selection policy of a scored best-of-20 pool (core documents)."""
    out = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if d["slice"] == sl and d.get("set") == "core":
            x = d["pol"][pol]
            out.append({"id": d["id"], "e": x["e"], "n": x["n"], "bc": x["bc"], "cells": d["n_cells"],
                        "strict": float(bool(x["strict"])), "conf": float(bool(x["conf"]))})
    return out


def flat(path, where=lambda d: True, cells="cells"):
    out = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if where(d):
            out.append({"id": d["id"], "e": d["e"], "n": d["n"], "bc": d["bc"], "cells": d[cells]})
    return out


def extra(model, name, met):
    """Per-document sim (x 100) or judge (empty output -> 1) of rows_<name>.jsonl; None if not measured."""
    d = f"{R}/evals/{model}/extra"
    if not os.path.isfile(os.path.join(d, f"{name}.{met}.json.rows.jsonl")):
        return None
    hyp = {}
    for line in open(os.path.join(d, f"rows_{name}.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        hyp[r["id"]] = r.get("hyp_text") or ""
    v = {}
    for line in open(os.path.join(d, f"{name}.{met}.json.rows.jsonl"), encoding="utf-8"):
        x = json.loads(line)
        if x.get(met) is None:
            continue
        v[x["id"]] = 100.0 * x[met] if met == "sim" else (1.0 if met == "judge" and not hyp[x["id"]].strip() else x[met])
    return v


def boot(docs, rng):
    """Point estimate and percentile interval of every metric present, resampling documents."""
    k = len(docs)
    idx = rng.integers(0, k, size=(B, k))
    res = {}

    def put(name, point, samples):
        lo, hi = np.percentile(samples, [2.5, 97.5])
        res[name] = {"value": float(point), "lo": float(lo), "hi": float(hi), "half": float((hi - lo) / 2), "docs": k}
    arr = lambda f: np.array([d[f] for d in docs], dtype=float)  # noqa: E731
    e, n, bc, cells = arr("e"), arr("n"), arr("bc"), arr("cells")
    put("bcer", 100 * bc.sum() / cells.sum(), 100 * bc[idx].sum(1) / cells[idx].sum(1))
    put("cer", 100 * e.sum() / n.sum(), 100 * e[idx].sum(1) / n[idx].sum(1))
    for f in ("strict", "conf"):
        if f in docs[0]:
            x = arr(f)
            put(f, 100 * x.mean(), 100 * x[idx].mean(1))
    return res


def boot_mean(vals, rng):
    x = np.array(list(vals), dtype=float)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    lo, hi = np.percentile(x[idx].mean(1), [2.5, 97.5])
    return {"value": float(x.mean()), "lo": float(lo), "hi": float(hi), "half": float((hi - lo) / 2), "docs": len(x)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{R}/paper/table1_ci.json")
    a = ap.parse_args()
    rows = {   # Table 1 row -> (per-document records, sim/judge model, rows name); missing rows are skipped
        "gpt55_fewshot": (lambda: flat(f"{R}/frontier/per_doc.jsonl"), "rlvr", "gpt55"),
        "liblouis_given": (lambda: flat(f"{R}/liblouis_bwd/per_doc.jsonl", lambda d: d["set"] == "core"), "rlvr", "liblouis"),
        "ours_bon": (lambda: pool(f"{R}/evals/rlvr/bon/per_doc.jsonl", "main", "V"), "rlvr", "rlvr_V_agn"),
        "ours_bon_oracle_code": (lambda: pool(f"{R}/evals/rlvr/prefill/per_doc.jsonl", "core", "V"), "rlvr", "rlvr_V_given"),
        "ours_greedy": (lambda: pool(f"{R}/evals/rlvr/bon/per_doc.jsonl", "main", "G"), "rlvr", "rlvr_G_agn"),
        "ours_bon_wo_rl": (lambda: pool(f"{R}/evals/topup/bon/per_doc.jsonl", "main", "V"), "topup", "topup_V_agn"),
        "byt5_bon": (lambda: pool(f"{R}/evals/byt5/bon/per_doc.jsonl", "core", "V"), "byt5", "byt5_V_agn"),
        "gemma_bon": (lambda: pool(f"{R}/evals/gemma/bon/per_doc.jsonl", "core", "V"), "gemma", "gemma_V_agn"),   # core slice
        "byt5_ck5000_bon": (lambda: pool(f"{R}/evals/byt5_ck5000/bon/per_doc.jsonl", "core", "V"), "byt5_ck5000", "byt5_ck5000_V_agn"),
        "gemma_ck6000_bon": (lambda: pool(f"{R}/evals/gemma_ck6000/bon/per_doc.jsonl", "core", "V"), "gemma_ck6000", "gemma_ck6000_V_agn"),
        "ours_bon_base_only": (lambda: pool(f"{R}/bon_sft/run/per_doc.jsonl", "main", "V"), "sft", "sft_V_agn"),   # first SFT
    }
    out = {"method": __doc__.split("Method")[1].split("  python")[0].strip(), "B": B, "seed": SEED, "rows": {}}
    shared = np.random.default_rng(SEED)           # first five rows
    for i, (row, (load, model, name)) in enumerate(rows.items()):
        try:
            docs = load()
        except FileNotFoundError:
            continue
        rng = shared if i < 5 else np.random.default_rng([SEED, i])
        r = boot(docs, rng)
        if row == "ours_bon_oracle_code":        # the code is given: no code-ID
            r.pop("strict", None), r.pop("conf", None)
        for met in ("sim", "judge"):
            v = extra(model, name, met)
            if v is not None:
                r[met] = boot_mean(v.values(), rng)
        out["rows"][row] = r
        print(f"{row:22s} " + "  ".join(f"{m} {v['value']:.2f} ±{v['half']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}] n={v['docs']}"
                                           for m, v in r.items()))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
