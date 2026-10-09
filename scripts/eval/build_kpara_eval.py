"""Paragraph-length multi-code eval documents with k = 3 and 4 codes (Table 2 'paragraphs' rows).

Built like the k = 2 set of data_u1_v2: 4-8 sentences per segment from the eval sources (disjoint from training),
regimes anchor and uniform, 400 documents each, with data_u1_v2's inventory. Segments are transcribed with the data's
own engine and must be braille-only; a segment's confusable set is the sibling tables that give the same cells. Rows
go through scripts/eval/bon_gen.py and scripts/eval/bon_eval.py like the k = 2 rows.

  python scripts/eval/build_kpara_eval.py --out work/data/kpara
      -> <out>/records.jsonl, <out>/main.jsonl (set k3p / k4p), <out>/meta.json
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from ubt.paths import DATASETS, WORK  # noqa: E402
DATA = f"{DATASETS}/data_u1_v2"
SEED = 20260926              # a fresh draw, not the k = 2 build seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cell", type=int, default=400)
    ap.add_argument("--ks", default="3,4")
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "unified_u1.yaml"))
    ap.add_argument("--tokenizer", default=f"{WORK}/models/qwen25_cell_tokenizer")
    a = ap.parse_args()
    import yaml  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    from ubt.plan import Planner  # noqa: PLC0415
    from ubt.tables import TableInfo  # noqa: PLC0415
    from ubt.task_format import render  # noqa: PLC0415
    from ubt.u1.records import make_u1_record  # noqa: PLC0415
    from ubt.u1.routed_louis import RoutedLouis  # noqa: PLC0415
    from ubt.u1.sources import U1Source  # noqa: PLC0415
    from ubt.verify import is_braille_only  # noqa: PLC0415
    cfg = yaml.safe_load(open(a.config))
    cfg_eval = copy.deepcopy(cfg)
    cfg_eval["intra_switch"]["natural_first"] = False          # as build_unified.py for the evaluation planner
    inv = json.load(open(os.path.join(DATA, "inventory.json")))
    man = json.load(open(os.path.join(DATA, "manifest.json")))
    tables = [TableInfo(table_id=t["table_id"], path=t["path"], language=t["language"], base_lang=t["base_lang"],
                        type="literary", grade=t["grade"], contraction=t["contraction"], direction=t["direction"],
                        weight=t["weight"], holdout=t["holdout"], status=t["status"], groups=t["groups"])
              for t in inv["tables"]]
    status = {t.table_id: t.status for t in tables}
    src = U1Source(cfg["data_root"], False, cfg["head_langs"], cfg.get("extra_wiki_first"))
    src.sourced_langs(sorted({t.base_lang for t in tables}))
    planner = Planner(cfg_eval, tables, inv["groups"], src, random.Random(SEED))
    louis = RoutedLouis(DATA, timeout_s=300.0)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    os.makedirs(a.out, exist_ok=True)
    records, rows, rejects = [], [], collections.Counter()
    for k in (int(x) for x in a.ks.split(",")):
        n_doc = 0
        for regime in ("anchor", "uniform"):
            got, tries = 0, 0
            while got < a.per_cell and tries < 40 * a.per_cell:
                batch = []
                while len(batch) < 64 and tries < 40 * a.per_cell:
                    tries += 1
                    p = planner.plan_inter(k, regime=regime, seg_sents=(4, 8))
                    if p is None:
                        rejects["no_source"] += 1
                        continue
                    if len({s["table"] for s in p["segments"]}) < k:
                        rejects["repeated_code"] += 1
                        continue
                    batch.append(p)
                items = [(s["table"], s["text"]) for p in batch for s in p["segments"]]
                items += [(sib, s["text"]) for p in batch for s in p["segments"] for sib in s["siblings"]]
                cells = dict(zip(items, louis.translate_many(items)))
                for p in batch:
                    segs, ok = [], True
                    for s in p["segments"]:
                        b = cells.get((s["table"], s["text"]))
                        if not b or not is_braille_only(b, allow_newline=False):
                            rejects["charset_or_fail"] += 1
                            ok = False
                            break
                        conf = [sib for sib in s["siblings"] if cells.get((sib, s["text"])) == b]
                        segs.append({"table": s["table"], "lang": s["lang"], "text": s["text"], "braille": b,
                                     "confusable_set": conf})
                    if not ok or got >= a.per_cell:
                        continue
                    n_doc += 1
                    rec = make_u1_record(rec_id=f"u1-Ek{k}p-{n_doc:07d}", form=f"Ek{k}p", split="eval",
                                         engine=man.get("engine_id"), doc_type="inter", regime=p["regime"],
                                         switch_density="none", segments=segs, source=p["source"],
                                         n_sentences=p["n_sentences"], tag=f"k{k}_paragraph:{regime}",
                                         status_by_id=status)
                    ex = render(rec, hinted=False)
                    n_c = len(tok(ex["completion"], add_special_tokens=False)["input_ids"])
                    rows.append({"id": rec["id"], "set": f"k{k}p", "tables": rec["tables"], "k": rec["k"],
                                 "regime": rec["regime"], "switch_density": rec["switch_density"],
                                 "confusable_set": rec.get("confusable_set") or [], "group": f"k{k}p", "lowres": False,
                                 "prompt": ex["prompt"], "completion": ex["completion"], "max_tokens": 2 * n_c + 32,
                                 "n_prompt_tok_orig": len(tok(ex["prompt"], add_special_tokens=True)["input_ids"]),
                                 "n_completion_tok": n_c})
                    records.append(rec)
                    got += 1
            print(f"k={k} {regime}: {got} documents ({tries} plans)", flush=True)
    louis.close()
    for name, xs in (("records.jsonl", records), ("main.jsonl", rows)):
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as fh:
            for x in xs:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    meta = {"n": len(rows), "per_set": collections.Counter(r["set"] for r in rows), "rejects": rejects, "seed": SEED,
            "max_prompt_plus_cap": max(r["n_prompt_tok_orig"] + r["max_tokens"] for r in rows),
            "config": a.config, "inventory": os.path.join(DATA, "inventory.json"), "engine": man.get("engine_id")}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
