#!/usr/bin/env python3
"""Build Wikipedia sentence shards A (train) and B (eval) per head language.

One NFC sentence per line, Latin tokens and digits kept, URLs stripped, 20-600 chars, hash-deduped.
B is built first from the start of the stream; A starts after it and skips any sentence hash seen in B, so the
shards are disjoint (asserted).

Usage: python scripts/data/build_wiki_shards.py [--langs en ko ...] [--workers 15]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from ubt.paths import DATASETS  # noqa: E402
from ubt.sources import MAX_LINE_CHARS, MIN_LINE_CHARS, sentence_hash  # noqa: E402

DUMP = "20231101"
TARGET_A = 300_000
TARGET_B = 30_000
# HF config names; gom (Goan Konkani) is the closest corpus for the kok tables.
HF_LANG = {"cmn": "zh", "kok": "gom"}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!?…؟।])\s+|(?<=[。！？])")


def clean_and_split(article_text: str) -> list[str]:
    out: list[str] = []
    for para in article_text.split("\n"):
        para = _URL_RE.sub(" ", para)
        para = _WS_RE.sub(" ", unicodedata.normalize("NFC", para)).strip()
        if len(para) < MIN_LINE_CHARS:
            continue
        for sent in _SENT_SPLIT.split(para):
            sent = sent.strip()
            if not (MIN_LINE_CHARS <= len(sent) <= MAX_LINE_CHARS):
                continue
            n_alpha = sum(ch.isalpha() for ch in sent)
            if n_alpha < 0.5 * len(sent.replace(" ", "")):
                continue  # tables/refs/formula debris
            out.append(sent)
    return out


def build_lang(args: tuple[str, str, int, int]) -> dict:
    lang, out_root, target_a, target_b = args
    from datasets import load_dataset  # noqa: PLC0415

    hf_lang = HF_LANG.get(lang, lang)
    ds = load_dataset("wikimedia/wikipedia", f"{DUMP}.{hf_lang}", split="train",
                      streaming=True)
    hashes_b: set[str] = set()
    hashes_a: set[str] = set()
    b_lines: list[str] = []
    a_lines: list[str] = []
    n_articles = 0
    b_boundary = None  # article index where shard B stopped
    for article in ds:
        n_articles += 1
        sents = clean_and_split(article["text"])
        if b_boundary is None:
            for s in sents:
                h = sentence_hash(s)
                if h not in hashes_b:
                    hashes_b.add(h)
                    b_lines.append(s)
            if len(b_lines) >= target_b:
                b_boundary = n_articles  # A starts at the next article
        else:
            for s in sents:
                h = sentence_hash(s)
                if h not in hashes_b and h not in hashes_a:
                    hashes_a.add(h)
                    a_lines.append(s)
            if len(a_lines) >= target_a:
                break

    if b_boundary is None:
        # wiki exhausted before B filled: keep ~10% (min 300) as B, the rest as A
        n_b = max(300, len(b_lines) // 10)
        a_lines = b_lines[n_b:]
        b_lines = b_lines[:n_b]
        hashes_b = {sentence_hash(s) for s in b_lines}
        hashes_a = {sentence_hash(s) for s in a_lines}

    assert not (hashes_a & hashes_b), f"{lang}: shard A/B hash overlap"
    os.makedirs(os.path.join(out_root, "wiki_txt_B"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "wiki_txt_A"), exist_ok=True)
    with open(os.path.join(out_root, "wiki_txt_B", f"{lang}.txt"), "w") as fh:
        fh.write("\n".join(b_lines) + "\n")
    with open(os.path.join(out_root, "wiki_txt_A", f"{lang}.txt"), "w") as fh:
        fh.write("\n".join(a_lines) + "\n")
    stats = {"lang": lang, "shard_B": len(b_lines), "shard_A": len(a_lines),
             "b_boundary_article": b_boundary, "articles_read": n_articles,
             "disjoint": True}
    print(json.dumps(stats), flush=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=f"{DATASETS}/data")
    ap.add_argument("--langs", nargs="*", default=[
        "en", "cmn", "es", "ar", "hi", "pt", "fr", "ru", "bn", "de", "ko",
        "vi", "it", "tr", "ja"])
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--target-a", type=int, default=TARGET_A)
    ap.add_argument("--target-b", type=int, default=TARGET_B)
    ap.add_argument("--stats-name", default="wiki_shard_stats.json")
    args = ap.parse_args()

    jobs = [(lang, args.data_root, args.target_a, args.target_b)
            for lang in args.langs]
    with mp.Pool(min(args.workers, len(jobs))) as pool:
        stats = pool.map(build_lang, jobs)
    with open(os.path.join(args.data_root, args.stats_name), "w") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    print("ALL_DONE")


if __name__ == "__main__":
    main()
