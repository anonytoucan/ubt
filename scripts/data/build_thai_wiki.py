#!/usr/bin/env python3
"""Thai Wikipedia sentence shards A (train side) and B (eval only) for the Thai scheme adapter.

Thai is not in scripts/data/build_wiki_shards.py and FLORES-200 tha_Thai leaves only 29 unused lines, so this script follows
build_wiki_shards.py (same dump, B first then A, same cleaning and filters, hash-disjoint shards) with two changes:
  1. Thai has no full stop, so a paragraph is cut at spaces into chunks of at most --max-chunk-chars
     (300 chars is about 800 tokens, under the adapter's max_len of 1024).
  2. The article id is kept as the document id; the pairs script splits held-out data by it.

Output in data/thai_wiki/: th_B.jsonl (eval only, never trained on), th_A.jsonl, and stats.json (counts, parquet
sha256s, licence).

    nice -n 10 python scripts/data/build_thai_wiki.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from ubt.paths import REPO  # noqa: E402
from ubt.sources import MIN_LINE_CHARS, sentence_hash  # noqa: E402

UBT_BASE = os.environ.get("UBT_BASE", REPO)
DUMP = "20231101"
REPO = "wikimedia/wikipedia"
CONFIG = f"{DUMP}.th"
LICENCE = "Wikipedia dump 20231101 (wikimedia/wikipedia via HF) — CC BY-SA 4.0 (per upstream)"
TARGET_A, TARGET_B = 300_000, 30_000
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")


def clean_paragraph(para: str) -> str:
    para = _URL_RE.sub(" ", para)
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", para)).strip()


def alpha_ok(s: str) -> bool:
    """build_wiki_shards' debris filter: at least half of the non-space chars are alphabetic.
    Thai vowel and tone marks are not alphabetic, but Thai prose still passes."""
    n_alpha = sum(ch.isalpha() for ch in s)
    return n_alpha >= 0.5 * len(s.replace(" ", ""))


def chunk_thai(para: str, max_chunk: int, min_chars: int = MIN_LINE_CHARS) -> tuple[list[str], dict]:
    """Cut a cleaned paragraph at spaces into chunks of at most max_chunk chars; longer space-free runs are dropped."""
    st = {"para": 1, "chunks": 0, "dropped_short": 0, "dropped_long_run": 0, "dropped_alpha": 0}
    out: list[str] = []
    cur: list[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            s = " ".join(cur)
            if len(s) < min_chars:
                st["dropped_short"] += 1
            elif not alpha_ok(s):
                st["dropped_alpha"] += 1
            else:
                out.append(s)
                st["chunks"] += 1
        cur, cur_len = [], 0

    for tok in para.split(" "):
        if not tok:
            continue
        if len(tok) > max_chunk:
            flush()
            st["dropped_long_run"] += 1
            continue
        add = len(tok) + (1 if cur else 0)
        if cur_len + add > max_chunk:
            flush()
            add = len(tok)
        cur.append(tok)
        cur_len += add
    flush()
    return out, st


def iter_articles(files: list[str]):
    import pyarrow.parquet as pq  # noqa: PLC0415
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=512, columns=["id", "title", "text"]):
            ids, titles, texts = batch.column(0).to_pylist(), batch.column(1).to_pylist(), batch.column(2).to_pylist()
            for i, t, x in zip(ids, titles, texts):
                yield str(i), t, x or ""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(UBT_BASE, "data", "thai_wiki"))
    ap.add_argument("--target-a", type=int, default=TARGET_A)
    ap.add_argument("--target-b", type=int, default=TARGET_B)
    ap.add_argument("--max-chunk-chars", type=int, default=300)
    ap.add_argument("--max-articles", type=int, default=0, help="0 = all")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415
    api = HfApi()
    names = sorted(s.rfilename for s in api.dataset_info(REPO).siblings
                   if s.rfilename.startswith(CONFIG + "/") and s.rfilename.endswith(".parquet"))
    files = [hf_hub_download(REPO, n, repo_type="dataset") for n in names]
    print(f"[thai_wiki] {len(files)} parquet files: {names}", flush=True)

    hashes_b: set[str] = set()
    hashes_a: set[str] = set()
    b_lines: list[dict] = []
    a_lines: list[dict] = []
    n_articles = 0
    b_boundary = None
    chunk_stats = {"para": 0, "chunks": 0, "dropped_short": 0, "dropped_long_run": 0, "dropped_alpha": 0,
                   "para_short": 0}
    dup_a = dup_b = 0
    for aid, title, text in iter_articles(files):
        n_articles += 1
        if args.max_articles and n_articles > args.max_articles:
            break
        sents: list[str] = []
        for para in text.split("\n"):
            para = clean_paragraph(para)
            if len(para) < MIN_LINE_CHARS:
                chunk_stats["para_short"] += 1
                continue
            chunks, st = chunk_thai(para, args.max_chunk_chars)
            for k in st:
                chunk_stats[k] += st[k]
            sents.extend(chunks)
        doc = f"thwiki-{aid}"
        if b_boundary is None:
            for s in sents:
                h = sentence_hash(s)
                if h in hashes_b:
                    dup_b += 1
                    continue
                hashes_b.add(h)
                b_lines.append({"doc": doc, "title": title, "text": s})
            if len(b_lines) >= args.target_b:
                b_boundary = n_articles
        else:
            for s in sents:
                h = sentence_hash(s)
                if h in hashes_b or h in hashes_a:
                    dup_a += 1
                    continue
                hashes_a.add(h)
                a_lines.append({"doc": doc, "title": title, "text": s})
            if len(a_lines) >= args.target_a:
                break
        if n_articles % 20000 == 0:
            print(f"[thai_wiki] articles {n_articles:,}  B {len(b_lines):,}  A {len(a_lines):,}  ({time.time() - t0:.0f}s)", flush=True)

    if b_boundary is None:
        n_b = max(300, len(b_lines) // 10)
        a_lines, b_lines = b_lines[n_b:], b_lines[:n_b]
        hashes_b = {sentence_hash(r["text"]) for r in b_lines}
        hashes_a = {sentence_hash(r["text"]) for r in a_lines}
    assert not (hashes_a & hashes_b), "th: shard A/B hash overlap"
    assert not ({r["doc"] for r in a_lines} & {r["doc"] for r in b_lines}), "th: shard A/B article overlap"

    paths = {}
    for name, rows in (("th_B.jsonl", b_lines), ("th_A.jsonl", a_lines)):
        p = os.path.join(args.out_dir, name)
        with open(p, "w", encoding="utf-8") as fh:
            for i, r in enumerate(rows):
                fh.write(json.dumps({"line": i, **r}, ensure_ascii=False) + "\n")
        paths[name] = {"path": os.path.relpath(p, UBT_BASE), "lines": len(rows),
                       "docs": len({r["doc"] for r in rows}), "sha256": sha256_file(p)}
    lens = sorted(len(r["text"]) for r in a_lines)
    stats = {"lang": "th", "hf_repo": REPO, "hf_config": CONFIG, "licence": LICENCE,
             "parquet": [{"name": n, "sha256": sha256_file(f), "bytes": os.path.getsize(f)}
                         for n, f in zip(names, files)],
             "protocol": "scripts/data/build_wiki_shards.py (B first, A after, hash-disjoint) + Thai space-chunking, article id kept",
             "max_chunk_chars": args.max_chunk_chars, "min_line_chars": MIN_LINE_CHARS,
             "target_a": args.target_a, "target_b": args.target_b,
             "shard_A": len(a_lines), "shard_B": len(b_lines), "b_boundary_article": b_boundary,
             "articles_read": n_articles, "disjoint": True,
             "dup_skipped_a": dup_a, "dup_skipped_b": dup_b, "chunking": chunk_stats,
             "a_chars_mean": round(sum(lens) / max(1, len(lens)), 1),
             "a_chars_p50": lens[len(lens) // 2] if lens else None,
             "a_chars_p95": lens[int(0.95 * len(lens))] if lens else None,
             "a_chars_max": lens[-1] if lens else None,
             "files": paths, "wall_sec": round(time.time() - t0, 1), "argv": sys.argv}
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("parquet", "files", "argv")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
