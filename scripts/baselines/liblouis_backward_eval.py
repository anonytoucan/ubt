"""LibLouis backward translation with the code given: the rule-based baseline row of the paper tables.

Each line is back-translated under its gold table with the data's own engine (ko-2024 with its override build), one
fresh process per table because LibLouis output depends on the tables loaded earlier in a process. The output is
scored like any system and re-transcribed forward under the same table to measure consistency with the input cells.

  python scripts/baselines/liblouis_backward_eval.py --data work/bon_data --slices main --out <dir>
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


def _bt_table(args):
    """(table, spec_dict, [cells]) -> (table, [text | None]) with the data generator's engine; make_translator checks
    the loaded liblouis file against the spec."""
    table, spec_dict, cells = args
    from ubt.translate import DISPLAY_TABLE  # noqa: PLC0415
    from ubt.u1 import engine as EN  # noqa: PLC0415
    spec = EN.EngineSpec.from_dict(spec_dict)
    os.chdir(spec.cwd_dir)
    lou = EN.make_translator(spec)._louis
    out = []
    for c in cells:
        try:
            out.append(lou.backTranslateString([DISPLAY_TABLE, table], c))
        except Exception:  # noqa: BLE001
            out.append(None)
    return table, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="bon_eval build dir")
    ap.add_argument("--slices", default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine-data", default=f"{DATASETS}/data_u1_v2")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    a.data, a.out = os.path.abspath(a.data), os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)
    import bon_gen as E  # noqa: PLC0415
    import main_eval as TC  # noqa: PLC0415
    from ubt.eval_harness import score_hyp  # noqa: PLC0415
    from ubt.metrics.textnorm import _error_cells, _reference_cells, norm_join  # noqa: PLC0415
    from ubt.u1.routed_louis import RoutedLouis  # noqa: PLC0415
    from ubt.wandb_utils import parse_target  # noqa: PLC0415
    man = json.load(open(os.path.join(a.engine_data, "manifest.json")))
    base, over = man["engine_spec"], man.get("engine_spec_overrides") or {}
    rows = []
    for s in a.slices.split(","):
        rows += [json.loads(line) for line in open(os.path.join(a.data, f"{s}.jsonl"), encoding="utf-8")]
    by = collections.defaultdict(list)                     # table -> [(doc idx, line idx, cells)]
    for di, r in enumerate(rows):
        gold = parse_target(r["completion"])
        lines = E._cells(r["prompt"]).split("\n")
        for li, ((t, _x), b) in enumerate(zip(gold, lines)):
            by[t].append((di, li, b))
    jobs = [(t, over.get(t, base), [b for _, _, b in v]) for t, v in by.items()]
    texts = collections.defaultdict(dict)
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        for t, out in pool.imap_unordered(_bt_table, jobs):
            for (di, li, _b), x in zip(by[t], out):
                texts[di][li] = x
    hyps = []
    for di, r in enumerate(rows):
        gold = parse_target(r["completion"])
        hyps.append("\n".join(f"⟨{t}⟩{texts[di].get(li) or ''}" for li, (t, _x) in enumerate(gold)))
    # consistency: forward re-transcription under the same table
    L = RoutedLouis(a.engine_data, timeout_s=300.0)
    items, where = [], []
    for di, h in enumerate(hyps):
        for li, (t, x) in enumerate(parse_target(h)):
            items.append((t, x))
            where.append((di, li))
    fw = L.translate_many(items)
    L.close()
    enc = collections.defaultdict(dict)
    for (di, li), b in zip(where, fw):
        enc[di][li] = b
    TC._pin_engine(a.engine_data)
    out_rows = []
    for di, r in enumerate(rows):
        gold = parse_target(r["completion"])
        lines = E._cells(r["prompt"]).split("\n")
        row = score_hyp({"id": r["id"], "confusable_set": r["confusable_set"]}, {"completion": r["completion"]},
                        hyps[di], None)
        segs_ref = ([(t, x, b) for (t, x), b in zip(gold, lines)] if len(gold) == len(lines)
                    else [(gold[0][0], "\n".join(x for _, x in gold), "\n".join(lines))])
        cells, starts, n_cells, _ = _reference_cells(segs_ref)
        hs = parse_target(hyps[di])
        parts = [enc[di].get(li) for li in range(len(gold))]
        consistent = None not in parts and "\n".join(parts) == "\n".join(lines)
        ref_n = norm_join(gold)
        out_rows.append({"id": r["id"], "set": r["set"], "group": r.get("group"), "table": r["tables"][0], "k": r["k"],
                         "hyp": hyps[di], "e": row["edit"], "n": row["ref_len"], "e_st": row["edit_st"],
                         "n_st": row["ref_len_st"], "exact": row["hyp_eq_ref"], "consistent": consistent,
                         "bc": _error_cells(segs_ref, starts, cells, n_cells, hs, True),
                         "bc_st": _error_cells(segs_ref, starts, cells, n_cells, hs, False), "cells": n_cells,
                         "bt_fail": any(texts[di].get(li) is None for li in range(len(gold))),
                         "ref_n_words": len(ref_n.split())})
    with open(os.path.join(a.out, "per_doc.jsonl"), "w", encoding="utf-8") as fh:
        for o in out_rows:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")
    groups = collections.defaultdict(list)
    for o in out_rows:
        if o["set"] == "core":
            groups["core_all"].append(o)
            groups["fam:" + o["group"]].append(o)
            groups["code:" + o["table"]].append(o)
        elif o["set"] == "mixed":
            groups[f"mixed_k{o['k']}"].append(o)
        else:
            groups[o["set"]].append(o)
    rep = {}
    for g, os_ in sorted(groups.items()):
        m = collections.Counter()
        kinds = collections.defaultdict(collections.Counter)
        for o in os_:
            for k in ("e", "n", "e_st", "n_st", "bc", "bc_st", "cells", "exact", "consistent", "bt_fail"):
                m[k] += o[k]
            kind = "inconsistent" if not o["consistent"] else ("correct" if o["exact"] else "consistent_misreading")
            kinds[kind]["docs"] += 1
            kinds[kind]["e"] += o["e"]
            kinds[kind]["n"] += o["n"]
            kinds[kind]["bc"] += o["bc"]
            kinds[kind]["cells"] += o["cells"]
        n = max(1, len(os_))
        rep[g] = {"docs": len(os_), "cer": 100 * m["e"] / max(1, m["n"]), "cer_st": 100 * m["e_st"] / max(1, m["n_st"]),
                  "bcer": 100 * m["bc"] / max(1, m["cells"]), "bcer_st": 100 * m["bc_st"] / max(1, m["cells"]),
                  "exact": 100 * m["exact"] / n, "consistent": 100 * m["consistent"] / n, "bt_fail_docs": m["bt_fail"],
                  "kinds": {k: {"share": 100 * v["docs"] / n, "cer": 100 * v["e"] / max(1, v["n"]),
                                "bcer": 100 * v["bc"] / max(1, v["cells"])} for k, v in kinds.items()}}
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    for g in [g for g in rep if not g.startswith("code:")]:
        r = rep[g]
        print(f"{g:14s} n={r['docs']:6d} CER {r['cer']:6.2f} (S/T {r['cer_st']:6.2f}) bCER {r['bcer']:6.2f} "
              f"exact {r['exact']:5.1f}% consistent {r['consistent']:5.1f}% bt-fail {r['bt_fail_docs']}")


if __name__ == "__main__":
    main()
