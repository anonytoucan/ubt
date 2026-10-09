"""Best-of-n evaluation of the final checkpoint, scored on the normalized text form.

One candidate pool per document (greedy + 19 samples at tau 0.8, top-p 0.95, from bon_gen.py gen); every rule picks
from the same pool: G greedy; LP max log-probability; V verified, lambda = 2 (feasible -> max log p, else argmax
log p - 2 d_cell); Vinf verified, lambda -> inf (none feasible -> closest candidate); O oracle-in-20 (lowest CER).
Slices: main (main_eval set), noise (dot-flip variants plus their clean sources), zeroshot (held-out codes).

  python scripts/eval/bon_eval.py build --out work/bon_data
  (gen: scripts/eval/bon_gen.py gen --slice <data>/<slice>.jsonl --out <dir>/gen_<slice>.<i>.jsonl --shard i/4)
  python scripts/eval/bon_eval.py score --data <data> --dir <dir>
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path[:0] = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "*", "")))
from ubt.paths import DATASETS, WORK  # noqa: E402
EVAL_DIR = f"{DATASETS}/data_u1_v2/eval"
POLICIES = ("G", "LP", "V", "Vinf", "O")
LAMBDA = 2.0


def _render_rows(recs, tok, extra):
    from ubt.task_format import render  # noqa: PLC0415
    rows = []
    for r in recs:
        ex = render(r, hinted=False)
        n_c = len(tok(ex["completion"], add_special_tokens=False)["input_ids"])
        row = {"id": r["id"], "set": extra(r)["set"], "tables": r["tables"], "k": r["k"], "regime": r["regime"],
               "switch_density": r["switch_density"], "confusable_set": r.get("confusable_set") or [],
               "prompt": ex["prompt"], "completion": ex["completion"], "max_tokens": 2 * n_c + 32}
        row.update(extra(r))
        rows.append(row)
    return rows


def cmd_build(a) -> None:
    from transformers import AutoTokenizer  # noqa: PLC0415

    from ubt.u1.routed_louis import RoutedLouis  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    os.makedirs(a.out, exist_ok=True)
    main = [json.loads(line) for line in open(a.main_eval, encoding="utf-8")]
    noise = [json.loads(line) for line in open(os.path.join(EVAL_DIR, "noise_variants.jsonl"), encoding="utf-8")]
    # matched clean baseline: the same source texts re-transcribed without flips
    srcs = {}
    for r in noise:
        srcs.setdefault((r["tables"][0], r["text"]), r)
    L = RoutedLouis(f"{DATASETS}/data_u1_v2", timeout_s=300.0)
    keys = list(srcs)
    clean = L.translate_many([(t, x) for t, x in keys])
    L.close()
    clean_recs = []
    for (t, x), b in zip(keys, clean):
        if b:
            r = dict(srcs[(t, x)])
            r.update(id=r["id"] + "-clean", braille=b, noise_p=0.0)
            clean_recs.append(r)
    zs = [json.loads(line) for line in open(os.path.join(EVAL_DIR, "zeroshot_eval.jsonl"), encoding="utf-8")]
    out = {"main": main,
           "noise": _render_rows(noise + clean_recs, tok, lambda r: {"set": "noise", "group": f"p{r['noise_p']}"}),
           "zeroshot": _render_rows(zs, tok, lambda r: {"set": "zeroshot", "group": r["doc_type"]})}
    meta = {}
    for name, rows in out.items():
        with open(os.path.join(a.out, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta[name] = {"n": len(rows), "groups": dict(collections.Counter(r.get("group", r["set"]) for r in rows))}
    meta["noise_clean_sources"] = {"unique": len(keys), "re_transcribed": len(clean_recs)}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


def _wer(ref: str, hyp: str) -> tuple[int, int]:
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415
    r, h = ref.split(), hyp.split()
    return Levenshtein.distance(r, h), len(r)


def _select(cands, feas, pol):
    idx = range(len(cands))
    if pol == "G":
        return 0
    if pol == "O":
        return min(idx, key=lambda i: (cands[i]["_cer"], i))
    if pol == "LP":
        return max(idx, key=lambda i: cands[i]["lp"])
    F = [i for i in idx if feas[i][0]]
    if F:
        return max(F, key=lambda i: cands[i]["lp"])
    D = [i for i in idx if feas[i][1] is not None]
    if not D:
        return 0
    if pol == "Vinf":
        return min(D, key=lambda i: (feas[i][1], -cands[i]["lp"]))
    return max(D, key=lambda i: cands[i]["lp"] - LAMBDA * feas[i][1])


def cmd_score(a) -> None:
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415

    import bon_gen as E  # noqa: PLC0415
    import main_eval as TC  # noqa: PLC0415
    from ubt.eval_harness import score_hyp  # noqa: PLC0415
    from ubt.metrics.textnorm import _error_cells, _reference_cells, norm_join  # noqa: PLC0415
    from ubt.wandb_utils import parse_target  # noqa: PLC0415
    a.data, a.dir = os.path.abspath(a.data), os.path.abspath(a.dir)
    slices = {}
    for name in a.slices.split(","):
        rows = [json.loads(line) for line in open(os.path.join(a.data, f"{name}.jsonl"), encoding="utf-8")]
        gens = {}
        for p in sorted(glob.glob(os.path.join(a.dir, f"gen_{name}.*.jsonl"))):
            for line in open(p, encoding="utf-8"):
                x = json.loads(line)
                gens[x["id"]] = x
        rows = [r for r in rows if r["id"] in gens]
        print(f"[score] {name}: {len(rows)} documents with candidates; feasibility ...", flush=True)
        feas = E._feasibility(rows, gens, a.engine_data)
        for r in rows:
            ref = norm_join(parse_target(r["completion"]))
            for c in gens[r["id"]]["cands"]:
                segs = parse_target(c["text"])
                c["_cer"] = (Levenshtein.distance(ref, norm_join(segs)) / max(1, len(ref))) if segs else 1.0
        slices[name] = (rows, gens, feas)
    TC._pin_engine(a.engine_data)                  # bCER alignment engine; pinned after the pool closes (it chdirs)
    docs = {}
    for name, (rows, gens, feas) in slices.items():
        for r in rows:
            cands = gens[r["id"]]["cands"]
            f = [feas[(r["id"], i)] for i in range(len(cands))]
            gold = parse_target(r["completion"])
            lines = E._cells(r["prompt"]).split("\n")
            segs_ref = ([(t, x, b) for (t, x), b in zip(gold, lines)] if len(gold) == len(lines)
                        else [(gold[0][0], "\n".join(x for _, x in gold), "\n".join(lines))])
            cells, starts, n_cells, _fb = _reference_cells(segs_ref)
            ref_n = norm_join(gold)
            d = {"slice": name, "set": r["set"], "group": r.get("group"), "table": r["tables"][0],
                 "k": r["k"], "n_cells": n_cells, "any_feasible": any(x[0] for x in f), "pol": {}}
            for pol in POLICIES:
                ci = _select(cands, f, pol)
                c = cands[ci]
                row = score_hyp({"id": r["id"], "confusable_set": r["confusable_set"]}, {"completion": r["completion"]},
                                c["text"], None)
                hs = parse_target(c["text"])
                we, wn = _wer(ref_n, norm_join(hs) if hs else "")
                feasible = f[ci][0]
                kind = ("inconsistent" if not feasible else "wrong_code" if not row["strict"]
                        else "correct" if row["hyp_eq_ref"] else "consistent_misreading")
                d["pol"][pol] = {"ci": ci, "e": row["edit"], "n": row["ref_len"], "e_st": row["edit_st"],
                                 "n_st": row["ref_len_st"], "strict": row["strict"], "conf": row["conf_aware"],
                                 "exact": row["hyp_eq_ref"], "feasible": feasible, "kind": kind, "we": we, "wn": wn,
                                 "bc": _error_cells(segs_ref, starts, cells, n_cells, hs, True),
                                 "bc_st": _error_cells(segs_ref, starts, cells, n_cells, hs, False),
                                 "finish": c.get("finish")}
            docs[r["id"]] = d
    with open(os.path.join(a.dir, "per_doc.jsonl"), "w", encoding="utf-8") as fh:
        for i, d in docs.items():
            fh.write(json.dumps({"id": i, **d}, ensure_ascii=False) + "\n")

    def agg(ds, pol):
        m = collections.Counter()
        kinds = collections.defaultdict(collections.Counter)
        for d in ds:
            p = d["pol"][pol]
            for k in ("e", "n", "e_st", "n_st", "we", "wn", "bc", "bc_st", "strict", "conf", "exact", "feasible"):
                m[k] += p[k]
            m["cells"] += d["n_cells"]
            m["any_feasible"] += d["any_feasible"]
            kk = kinds[p["kind"]]
            kk["docs"] += 1
            kk["e"] += p["e"]
            kk["n"] += p["n"]
            kk["bc"] += p["bc"]
            kk["cells"] += d["n_cells"]
        n = max(1, len(ds))
        return {"docs": len(ds), "cer": 100 * m["e"] / max(1, m["n"]), "cer_st": 100 * m["e_st"] / max(1, m["n_st"]),
                "bcer": 100 * m["bc"] / max(1, m["cells"]), "bcer_st": 100 * m["bc_st"] / max(1, m["cells"]),
                "wer": 100 * m["we"] / max(1, m["wn"]), "strict": 100 * m["strict"] / n, "conf": 100 * m["conf"] / n,
                "exact": 100 * m["exact"] / n, "consistent": 100 * m["feasible"] / n,
                "any_feasible": 100 * m["any_feasible"] / n,
                "kinds": {k: {"share": 100 * v["docs"] / n, "cer": 100 * v["e"] / max(1, v["n"]),
                              "bcer": 100 * v["bc"] / max(1, v["cells"])} for k, v in kinds.items()}}
    ev = {}
    for name in ("main", "core"):
        if name in slices:
            ev.update({json.loads(line)["id"]: json.loads(line)
                       for line in open(os.path.join(a.data, f"{name}.jsonl"), encoding="utf-8")})
    groups = collections.defaultdict(list)
    for i, d in docs.items():
        if d["slice"] in ("main", "core"):
            s = d["set"]
            if s == "core":
                groups["core_all"].append(d)
                groups["fam:" + ev[i]["group"]].append(d)
                groups["code:" + d["table"]].append(d)
            elif s == "mixed":
                groups[f"mixed_k{d['k']}"].append(d)
            else:
                groups[s].append(d)
        elif d["slice"] == "noise":
            groups["noise:" + d["group"]].append(d)
        else:
            groups["zeroshot:" + d["group"]].append(d)
            if d["group"] == "single":
                groups["zscode:" + d["table"]].append(d)
    report = {g: {pol: agg(ds, pol) for pol in POLICIES} for g, ds in sorted(groups.items())}
    json.dump(report, open(os.path.join(a.dir, "report.json"), "w"), indent=1)
    show = [g for g in report if not g.startswith(("code:", "zscode:"))]
    print("| group | n | " + " | ".join(f"{p} CER/bCER" for p in POLICIES) + " | strict/conf (V) | consistent G→V |")
    print("|---|---:|" + "---:|" * (len(POLICIES) + 2))
    for g in show:
        r = report[g]
        print(f"| {g} | {r['G']['docs']} | " + " | ".join(f"{r[p]['cer']:.2f}/{r[p]['bcer']:.2f}" for p in POLICIES)
              + f" | {r['V']['strict']:.1f}/{r['V']['conf']:.1f} | {r['G']['consistent']:.1f}→{r['V']['consistent']:.1f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--main-eval", default=f"{WORK}/data/main_eval/eval.jsonl")
    b.add_argument("--tokenizer", default=f"{WORK}/models/qwen25_cell_tokenizer")
    s = sub.add_parser("score")
    s.add_argument("--data", required=True)
    s.add_argument("--dir", required=True)
    s.add_argument("--slices", default="main,noise,zeroshot")
    s.add_argument("--engine-data", default=f"{DATASETS}/data_u1_v2")
    a = ap.parse_args()
    {"build": cmd_build, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
