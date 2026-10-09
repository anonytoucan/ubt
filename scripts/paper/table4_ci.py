"""Point estimates and 95% confidence intervals for the rows of Table 4, with the method of scripts/paper/table1_ci.py.

Bootstrap over documents, 1,000 resamples, 95% percentile interval, half-width reported; row i uses default_rng([0, i]).
UBT infers the code on the Bible and NIKL and is given it on the two Chinese corpora, whose cells no table reproduces.
bCER / CER as in real_corpora_score.py (whitespace removed for BrailleLLM), sim = mean bge-m3 cosine x 100, judge =
mean judge score with an empty output scored 1. NIKL records stay in their local directory; only aggregates leave it.

  python scripts/paper/table4_ci.py [--out work/paper/table4_ci.json]
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
NIKL = f"{R}/nikl_test"
X = f"{R}/evals/rlvr/extra"


def spec(d, cond, grp, fcons=False, nows=False, extra=None):
    """One table row: pool dir, condition, corpus, subset flags and per-document sim / judge / row files.
    The row file gives hyp_text for the judge's empty-output rule."""
    if extra:                                  # extra_rows_build.py layout: <X>/<name>.<metric>.json.rows.jsonl
        files = {m: f"{X}/{extra}.{m}.json.rows.jsonl" for m in ("sim", "judge")}
        files["rows"] = f"{X}/rows_{extra}.jsonl"
    else:                                      # pool directory layout: <d>/<metric>_<cond>.json.rows.jsonl
        files = {m: f"{d}/{m}_{cond}.json.rows.jsonl" for m in ("sim", "judge")}
        files["rows"] = f"{d}/rows_sim_{cond}.jsonl"
    return {"dir": d, "cond": cond, "group": grp, "fcons": fcons, "nows": nows, **files}


ROWS = {
    "bible_liblouis": spec(f"{R}/evals/liblouis/real", "given", "bible"),
    "bible_ubt": spec(f"{R}/evals/rlvr/real", "inferred", "bible", extra="rlvr_real_inferred"),
    "nikl_fcons_ubt": spec(f"{NIKL}/run", "inferred", "nikl", fcons=True),
    "nikl_full_ubt": spec(f"{NIKL}/run", "inferred", "nikl"),
    "braillellm_ubt": spec(f"{R}/evals/rlvr/real", "given", "braillellm", nows=True, extra="rlvr_real_given"),
    "braillellm_ubt_ft": spec(f"{R}/evals/rlvr_bllm_adapt/real", "given", "braillellm", nows=True),
    "vision_ubt": spec(f"{R}/evals/rlvr/real", "given", "vision", extra="rlvr_real_given"),
    "vision_ubt_ft": spec(f"{R}/evals/rlvr_vision_adapt/real", "given", "vision"),
    "nikl_fcons_liblouis": spec(f"{NIKL}/liblouis", "given", "nikl", fcons=True),
    "nikl_full_liblouis": spec(f"{NIKL}/liblouis", "given", "nikl"),
    "braillellm_liblouis": spec(f"{R}/evals/liblouis/real", "given", "braillellm", nows=True),
    "vision_liblouis": spec(f"{R}/evals/liblouis/real", "given", "vision"),
    "bible_gpt55": spec(f"{R}/evals/gpt55/real", "given", "bible"),     # few-shot, 1,000 of the 3,000 verses
}


def per_doc(sp, met):
    """{id: value} of one measured metric (sim x 100; judge with an empty output scored 1), or None."""
    if not os.path.isfile(sp[met]):
        return None
    hyp = {}
    if os.path.isfile(sp["rows"]):
        for line in open(sp["rows"], encoding="utf-8"):
            r = json.loads(line)
            hyp[r["id"]] = r.get("hyp_text") or ""
    v = {}
    for line in open(sp[met], encoding="utf-8"):
        x = json.loads(line)
        if x.get(met) is None:
            continue
        v[x["id"]] = 100.0 * x[met] if met == "sim" else (1.0 if x["id"] in hyp and not hyp[x["id"]].strip() else x[met])
    return v


def interval(point, samples, k):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"value": float(point), "lo": float(lo), "hi": float(hi), "half": float((hi - lo) / 2), "docs": k}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{R}/paper/table4_ci.json")
    a = ap.parse_args()
    out = {"method": "bootstrap over documents, 1,000 resamples, 95% percentile interval, row i: default_rng([0, i])",
           "rows": {}}
    for i, (row, sp) in enumerate(ROWS.items()):
        d, cond, grp, fcons, nows = sp["dir"], sp["cond"], sp["group"], sp["fcons"], sp["nows"]
        p = os.path.join(d, "per_doc.jsonl")
        if not os.path.isfile(p):
            continue
        docs = []
        for line in open(p, encoding="utf-8"):
            x = json.loads(line)
            if x["cond"] != cond or x["group"] != grp or (fcons and not x.get("f_consistent")):
                continue
            s = x["pol"]["V"]
            if nows:
                docs.append((x["id"], s["e_nows"], s["n_nows"], s["bc_nows"], s["cells_nows"]))
            else:
                docs.append((x["id"], s["e"], s["n"], s["bc"], x["n_cells"]))
        ids = [t[0] for t in docs]
        e, n, bc, cells = (np.array([t[j] for t in docs], dtype=float) for j in (1, 2, 3, 4))
        rng = np.random.default_rng([SEED, i])
        idx = rng.integers(0, len(docs), size=(B, len(docs)))
        r = {"bcer": interval(100 * bc.sum() / cells.sum(), 100 * bc[idx].sum(1) / cells[idx].sum(1), len(docs)),
             "cer": interval(100 * e.sum() / n.sum(), 100 * e[idx].sum(1) / n[idx].sum(1), len(docs))}
        for met in ("sim", "judge"):
            v = per_doc(sp, met)
            if v is None:
                continue
            missing = [j for j in ids if j not in v]
            assert not missing, (row, met, len(missing))
            x = np.array([v[j] for j in ids], dtype=float)
            r[met] = interval(x.mean(), x[idx].mean(1), len(ids))
        out["rows"][row] = r
        print(f"{row:22s} " + "  ".join(f"{m} {v['value']:.2f} ±{v['half']:.3f} (n={v['docs']})" for m, v in r.items()))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
