"""Extra data-u1 training records, built exactly like the data (per-table pinned liblouis, strict noUndefined
acceptance, the builder's train/eval unit-hash rule).

  cjk     : top-up for codes projected to stay weak after SFT: new single-code docs from wiki_A lines unused by that
            table, with the sentence-count mix of the table's S1 rows.
  fewshot : nested k-shot pools (k = 4/16/64/256/all) for the 8 zero-shot codes from FLORES dev; the zero-shot eval
            uses devtest.

  python scripts/data/u1_topup_build.py cjk --out datasets/topup_cjk \
      --target ja-kantenji-ucs2.utb=30000 --target zhcn-g2.ctb=30000 ...
  python scripts/data/u1_topup_build.py fewshot --out datasets/fewshot_heldout
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
import unicodedata

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path[:0] = sorted(glob.glob(os.path.join(REPO, "scripts", "*", "")))
from ubt.paths import DATASETS  # noqa: E402

DATA = f"{DATASETS}/data_u1_v2"
SRC = f"{DATASETS}/data"
LANG_SRC = {"ja": f"{SRC}/wiki_txt_A/ja.txt", "zh": f"{SRC}/wiki_txt_A/cmn.txt", "de": f"{SRC}/wiki_txt_A/de.txt",
            "tr": f"{SRC}/wiki_txt_A/tr.txt"}
HELDOUT = {"hr-g1.tbl": "hrv_Latn", "mt.ctb": "mlt_Latn", "sw-ke-g1.utb": "swh_Latn", "sw-ke-g1-2.ctb": "swh_Latn",
           "sw-ke-g1-3.ctb": "swh_Latn", "sw-ke-g1-4.ctb": "swh_Latn", "sw-ke-g1-5.ctb": "swh_Latn",
           "sw-ke-g2.ctb": "swh_Latn"}
LANG_OF = {"hrv_Latn": "hr", "mlt_Latn": "mt", "swh_Latn": "sw"}
K_SHOTS = (4, 16, 64, 256)

# --- the builder's unit hashing, copied from src/ubt/u1/registry.py so working-tree edits cannot drift ---
_WS = re.compile(r"\s+")
_SUB_SPLIT = re.compile(r"(?<=[.!?…؟।])\s+|(?<=[。！？；;])")
_UNSPACED = re.compile(r"[฀-໿က-႟ក-៿぀-ヿ㐀-鿿豈-﫿]")


def _norm_ws(s: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", s.strip()))


def h64(s: str) -> int:
    return int.from_bytes(hashlib.sha1(_norm_ws(s).encode("utf-8")).digest()[:8], "big")


def text_units(text: str) -> set[int]:
    out = set()
    if not text or not text.strip():
        return out
    out.add(h64(text))
    for line in text.split("\n"):
        if line.strip():
            out.add(h64(line))
            for p in _SUB_SPLIT.split(line):
                p = p.strip()
                if len(p) >= 16 or (len(p) >= 8 and _UNSPACED.search(p)):
                    out.add(h64(p))
    return out


def eval_units() -> set[int]:
    import numpy as np  # noqa: PLC0415
    z = np.load(os.path.join(DATA, "registry.npz"))
    return set(int(x) for x in z["eval_units"].tolist())


def engine_specs() -> dict:
    man = json.load(open(os.path.join(DATA, "manifest.json")))
    return {"default": man["engine_spec"], **(man.get("engine_spec_overrides") or {})}, man["engine_id"]


def transcribe(pairs: list[tuple[str, str]], workers: int, chunk: int = 2000) -> list[str | None]:
    """[(table, text)] -> canonical braille or None (strict pass failed), one fresh process per (table, chunk)."""
    import u1_retranscribe as RT  # noqa: PLC0415
    specs, _ = engine_specs()
    by_tab: dict[str, list[int]] = collections.defaultdict(list)
    for i, (t, _x) in enumerate(pairs):
        by_tab[t].append(i)
    tasks, where = [], []
    for t, idxs in by_tab.items():
        for j in range(0, len(idxs), chunk):
            part = idxs[j:j + chunk]
            where.append(part)
            tasks.append((len(tasks), t, [pairs[i][1] for i in part]))
    out: list[str | None] = [None] * len(pairs)
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(workers, len(tasks)), initializer=RT._init, initargs=(specs,), maxtasksperchild=1) as pool:
        for k, res in pool.imap_unordered(RT._task, tasks):
            for i, (b, ok) in zip(where[k], res):
                out[i] = b if ok else None
    return out


def _cells(b: str) -> int:
    return sum(1 for ch in b if 0x2800 <= ord(ch) <= 0x28FF)


def record(rid, table, lang, text, braille, n_sent, source, tag, form, engine_id) -> dict:
    from ubt.formats import len_bucket  # noqa: PLC0415
    return {"id": rid, "doc_type": "single", "k": 1, "regime": "single", "switch_density": "none",
            "len_bucket": len_bucket(_cells(braille), n_sent), "cells": _cells(braille), "text": text, "braille": braille,
            "tables": [table], "langs": [lang], "confusable_set": [], "segments": None, "fiber_ambiguous": False,
            "source": source, "n_sentences": n_sent, "tag": tag, "table_status": ["current"], "form": form,
            "split": "train", "engine": engine_id}


UNSPACED_LANGS = ("ja", "zh", "cmn", "yue")


def cmd_cjk(a) -> None:
    """Top up any code; --src LANG=PATH adds a pool of sentences, one per line in article order (LANG = base_lang)."""
    t0 = time.time()
    targets = dict((t.split("=")[0], int(t.split("=")[1])) for t in a.target)
    inv = {t["table_id"]: t for t in json.load(open(os.path.join(DATA, "inventory.json")))["tables"]}
    lang = {t: ("ja" if t.startswith("ja") else "zh" if t.startswith("zh") else inv[t]["base_lang"]) for t in targets}
    srcs = dict(LANG_SRC)
    for kv in a.src or []:
        k, v = kv.split("=", 1)
        srcs[k] = v
    ev = eval_units()
    # units already in each table's train/dev rows, so every new doc is new for that table; plus the n_sentences mix
    used = collections.defaultdict(set)
    mix = collections.defaultdict(collections.Counter)
    for split in ("train.jsonl", "dev.jsonl"):
        for line in open(os.path.join(DATA, split), encoding="utf-8"):
            r = json.loads(line)
            ts = [t for t in r["tables"] if t in targets]
            if not ts:
                continue
            for t in ts:
                used[t] |= text_units(r["text"])
            if r["k"] == 1 and r.get("form") in ("S1", "S1b") and r["tables"][0] in targets:
                mix[r["tables"][0]][int(r.get("n_sentences") or 1)] += 1
    specs, engine_id = engine_specs()
    rng = random.Random(a.seed)
    pairs, meta = [], []
    for t, n in targets.items():
        lines = [x.strip() for x in open(srcs[lang[t]], encoding="utf-8") if x.strip()]
        sep = "" if lang[t] in UNSPACED_LANGS else " "
        ns = [k for k, c in mix[t].items() for _ in range(c)] or [1]
        order = list(range(len(lines)))
        rng.shuffle(order)                       # random article positions, consecutive lines inside a doc
        want, got, pos = int(n * a.overdraw), 0, 0
        while got < want and pos < len(order):
            k = rng.choice(ns)
            i0 = order[pos]
            pos += 1
            span = lines[i0:i0 + k]
            if len(span) < k:
                continue
            text = unicodedata.normalize("NFC", sep.join(span))
            u = text_units(text)
            if u & ev or u & used[t]:
                continue
            used[t] |= u                          # no duplicate sentences inside the top-up either
            pairs.append((t, text))
            meta.append((t, k))
            got += 1
        print(f"[cjk] {t}: {got} candidate docs (want {n}, overdraw {a.overdraw}), mix {dict(mix[t].most_common(4))}",
              flush=True)
    brs = transcribe(pairs, a.workers)
    os.makedirs(a.out, exist_ok=True)
    kept = collections.Counter()
    rej = collections.Counter()
    path = os.path.join(a.out, "train.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for (t, text), (_t, k), b in zip(pairs, meta, brs):
            if b is None:
                rej[t] += 1
                continue
            if kept[t] >= targets[t]:
                continue
            kept[t] += 1
            src_tag = os.path.basename(os.path.dirname(srcs[lang[t]])) + ":" + os.path.basename(srcs[lang[t]]).split(".")[0]
            rec = record(f"u1-{a.form}-{t.split('.')[0]}-{kept[t]:06d}", t, lang[t], text, b, k,
                         f"{src_tag}:topup", f"topup_{a.form.lower()}:{t}", a.form, engine_id)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    man = {"kind": f"{a.form} top-up", "data": DATA, "engine_id": engine_id,
           "sources": {lang[t]: srcs[lang[t]] for t in targets},
           "targets": targets, "kept": dict(kept), "strict_rejected": dict(rej), "seed": a.seed,
           "rule": "strict noUndefined acceptance; no unit in data_u1_v2 eval registry; no unit already in the "
                   "table's train/dev rows; n_sentences mix = table's S1/S1b rows",
           "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "sec": round(time.time() - t0)}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(man, indent=1, ensure_ascii=False))


def cmd_fewshot(a) -> None:
    t0 = time.time()
    ev = eval_units()
    specs, engine_id = engine_specs()
    pairs, keys = [], []
    for t, fl in HELDOUT.items():
        sents = [x.strip() for x in open(f"{SRC}/flores200_dataset/dev/{fl}.dev", encoding="utf-8") if x.strip()]
        for i, s in enumerate(sents):
            s = unicodedata.normalize("NFC", s)
            if text_units(s) & ev:
                continue
            pairs.append((t, s))
            keys.append((t, fl, i))
    brs = transcribe(pairs, a.workers)
    os.makedirs(a.out, exist_ok=True)
    pools = collections.defaultdict(list)
    for (t, s), (_t, fl, i), b in zip(pairs, keys, brs):
        if b is not None:
            pools[t].append(record(f"u1-Fshot-{t.split('.')[0]}-{i:04d}", t, LANG_OF[fl], s, b, 1,
                                   f"flores_dev:{fl}", f"fewshot:{t}", "Fshot", engine_id))
    rng = random.Random(a.seed)
    counts = {}
    for t, rows in pools.items():
        rng.shuffle(rows)                          # one order per code, so k-shot sets are nested prefixes
        counts[t] = len(rows)
    files = {}
    for k in list(K_SHOTS) + ["all"]:
        p = os.path.join(a.out, f"k{k}.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            for t, rows in sorted(pools.items()):
                for r in (rows if k == "all" else rows[:k]):
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[f"k{k}"] = {"docs": sum(len(r) if k == "all" else min(k, len(r)) for r in pools.values()),
                          "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    man = {"kind": "few-shot pools for the zero-shot codes", "data": DATA, "engine_id": engine_id,
           "codes": HELDOUT, "per_code_pool": counts, "files": files, "seed": a.seed,
           "rule": "FLORES dev only (zeroshot_eval = devtest); strict acceptance; no unit in the eval registry; "
                   "k-shot files are nested prefixes of one seeded order per code",
           "sec": round(time.time() - t0)}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(man, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cjk")
    c.add_argument("--out", required=True)
    c.add_argument("--target", action="append", required=True, help="TABLE=N docs")
    c.add_argument("--overdraw", type=float, default=1.35, help="candidates per wanted doc (strict rejections)")
    c.add_argument("--workers", type=int, default=64)
    c.add_argument("--seed", type=int, default=20260924)
    c.add_argument("--src", action="append", help="LANG=PATH sentence pool (base_lang of the inventory)")
    c.add_argument("--form", default="topup_cjk", help="form / id prefix of the new records")
    f = sub.add_parser("fewshot")
    f.add_argument("--out", required=True)
    f.add_argument("--workers", type=int, default=16)
    f.add_argument("--seed", type=int, default=20260924)
    a = ap.parse_args()
    {"cjk": cmd_cjk, "fewshot": cmd_fewshot}[a.cmd](a)


if __name__ == "__main__":
    main()
