"""Record schema and difficulty stamps.

Every record carries: k, len_bucket, regime, switch_density, confusable_set,
fiber_ambiguous (default False).
"""

from __future__ import annotations

from ubt.verify import cell_count

LEN_BUCKETS = ("snippet_1_4", "snippet_5_10", "snippet_11_20", "snippet_21_40",
               "snippet_41_80", "sentence", "paragraph")
REGIMES = ("single", "anchor", "uniform", "confusable", "math")
DENSITIES = ("none", "light", "heavy")


def len_bucket(cells: int, n_sentences: int) -> str:
    if cells <= 4:
        return "snippet_1_4"   # out of spec; excluded from 5-10 fills
    if cells <= 10:
        return "snippet_5_10"
    if cells <= 20:
        return "snippet_11_20"
    if cells <= 40:
        return "snippet_21_40"
    if cells <= 80:
        return "snippet_41_80"
    return "paragraph" if n_sentences >= 4 else "sentence"


def make_record(
    rec_id: str,
    doc_type: str,
    regime: str,
    switch_density: str,
    segments: list[dict],
    source: str,
    n_sentences: int,
    extra: dict | None = None,
) -> dict:
    """Assemble one record from verified segments [{table, lang, text, braille, confusable_set}]: one segment for
    single and intra-switch docs, k for inter-mixed docs."""
    if regime not in REGIMES:
        raise ValueError(f"bad regime {regime}")
    if switch_density not in DENSITIES:
        raise ValueError(f"bad switch_density {switch_density}")
    braille = "\n".join(s["braille"] for s in segments)
    text = "\n".join(s["text"] for s in segments)
    cells = cell_count(braille)
    k = len(segments)
    rec = {
        "id": rec_id,
        "doc_type": doc_type,
        "k": k,
        "regime": regime,
        "switch_density": switch_density,
        "len_bucket": len_bucket(cells, n_sentences),
        "cells": cells,
        "text": text,
        "braille": braille,
        "tables": [s["table"] for s in segments],
        "langs": [s["lang"] for s in segments],
        "confusable_set": sorted(set(segments[0]["confusable_set"])) if k == 1
                          else sorted({t for s in segments for t in s["confusable_set"]}),
        "segments": segments if k > 1 else None,
        "fiber_ambiguous": False,
        "source": source,
        "n_sentences": n_sentences,
    }
    if extra:
        rec.update(extra)
    return rec


def dedup_key(segments: list[dict]) -> tuple:
    """Mixed-doc dedup key: tuple of (table, text) per segment."""
    return tuple((s["table"], s["text"]) for s in segments)
