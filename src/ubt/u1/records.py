"""data-u1 record schema: the 18-field core schema plus form, split, engine.

Core: id, doc_type, k, regime, switch_density, len_bucket, cells, text, braille, tables[], langs[],
confusable_set[], segments[], fiber_ambiguous, source, n_sentences, tag, table_status[]. data-u1 adds form
(S-code or eval set), split (train|dev|eval) and engine (engine id; the sha256 map is in manifest.json).

`segments` is null for single-segment rows and holds one entry per span otherwise. Spans are joined with
`seg_join`: "\\n" by default, "" for inline math (carrier + ⠸⠩…⠸⠱ + carrier). `len_bucket` and `cells` are
always computed from the final braille; the verifier recomputes both.
"""
from __future__ import annotations

import re

from ubt.formats import DENSITIES, REGIMES, len_bucket
from ubt.verify import cell_count

CORE_FIELDS = ("id", "doc_type", "k", "regime", "switch_density", "len_bucket",
               "cells", "text", "braille", "tables", "langs", "confusable_set",
               "segments", "fiber_ambiguous", "source", "n_sentences", "tag",
               "table_status")
U1_FIELDS = ("form", "split", "engine")
ALL_FIELDS = CORE_FIELDS + U1_FIELDS

# allowed extra fields per form; any other field is a schema error
FORM_EXTRAS = {
    "S4a": {"latex_canonical", "embedded", "seg_join"},
    "S4b": {"latex_canonical", "embedded", "kmath_input", "kmath_text"},
    "S7": {"f_consistent", "nikl_id", "nikl_doc"},
    "Ekodisc": {"pair_id"},
    "Enoise": {"noise_p", "noise_base_id", "noise_n_dots", "noise_n_flips"},
    "Emath": {"latex_canonical", "embedded", "seg_join", "kmath_input", "kmath_text"},
}
REQUIRED_EXTRAS = {
    "S4a": {"latex_canonical", "embedded"},
    "S4b": {"latex_canonical", "embedded", "kmath_input", "kmath_text"},
    "S7": {"f_consistent", "nikl_id", "nikl_doc"},
    "Ekodisc": {"pair_id"},
    "Enoise": {"noise_p", "noise_base_id", "noise_n_dots", "noise_n_flips"},
    "Emath": {"latex_canonical", "embedded"},
}
_TYPES = {"id": str, "doc_type": str, "k": int, "regime": str, "switch_density": str,
          "len_bucket": str, "cells": int, "text": str, "braille": str, "tables": list,
          "langs": list, "confusable_set": list, "fiber_ambiguous": bool, "source": str,
          "n_sentences": int, "tag": str, "table_status": list, "form": str, "split": str,
          "engine": str}

TRAIN_FORMS = ("S1", "S1b", "S2", "S3", "S3p", "S4a", "S4b", "S5", "S6", "S7")
EVAL_FORMS = ("Ecore", "Esnip", "Emix", "Ek2p", "Eintra", "Ekodisc", "Ezs",
              "Enoise", "Emath")
DOC_TYPES = ("single", "intra_switch", "inter", "math")
SPLITS = ("train", "dev", "eval")
ID_RE = re.compile(r"^u1-(?P<form>[A-Za-z0-9]+)-(?P<seq>\d{7})$")


def make_u1_record(*, rec_id: str, form: str, split: str, engine: str,
                   doc_type: str, regime: str, switch_density: str,
                   segments: list[dict], source: str, n_sentences: int,
                   tag: str, status_by_id: dict[str, str], seg_join: str = "\n",
                   tables: list[str] | None = None, langs: list[str] | None = None,
                   extra: dict | None = None) -> dict:
    """segments: verified [{table, lang, text, braille, confusable_set}]. `tables`/`langs` default to one
    entry per segment; inline-math rows pass the distinct codes (k = number of codes)."""
    if regime not in REGIMES:
        raise ValueError(f"bad regime {regime}")
    if switch_density not in DENSITIES:
        raise ValueError(f"bad switch_density {switch_density}")
    if doc_type not in DOC_TYPES:
        raise ValueError(f"bad doc_type {doc_type}")
    braille = seg_join.join(s["braille"] for s in segments)
    text = seg_join.join(s["text"] for s in segments)
    cells = cell_count(braille)
    tables = tables if tables is not None else [s["table"] for s in segments]
    langs = langs if langs is not None else [s["lang"] for s in segments]
    k = len(tables)
    conf = sorted({t for s in segments for t in (s.get("confusable_set") or [])})
    single = len(segments) == 1
    rec = {
        "id": rec_id, "doc_type": doc_type, "k": k, "regime": regime,
        "switch_density": switch_density,
        "len_bucket": len_bucket(cells, n_sentences), "cells": cells,
        "text": text, "braille": braille, "tables": tables, "langs": langs,
        "confusable_set": conf,
        "segments": None if single else [
            {kk: v for kk, v in s.items() if kk != "siblings"} for s in segments],
        "fiber_ambiguous": False, "source": source, "n_sentences": n_sentences,
        "tag": tag,
        "table_status": [status_by_id.get(t, "current") for t in tables],
        "form": form, "split": split, "engine": engine,
    }
    if not single and seg_join != "\n":
        rec["seg_join"] = seg_join
    if extra:
        rec.update(extra)
    return rec


def dedup_key(tables: list[str], text: str) -> bytes:
    import hashlib  # noqa: PLC0415
    return hashlib.sha1(("\x1f".join(tables) + "\x1e" + text).encode("utf-8")).digest()


def schema_errors(rec: dict) -> list[str]:
    """Structural validity of one record (the verifier's schema check)."""
    errs = []
    for f in ALL_FIELDS:
        if f not in rec:
            errs.append(f"missing:{f}")
    if errs:
        return errs
    m = ID_RE.match(rec["id"])
    if not m:
        errs.append("id_pattern")
    elif m.group("form") != rec["form"]:
        errs.append("id_form_mismatch")
    if rec["split"] not in SPLITS:
        errs.append("split")
    if rec["doc_type"] not in DOC_TYPES:
        errs.append("doc_type")
    if rec["regime"] not in REGIMES:
        errs.append("regime")
    if rec["switch_density"] not in DENSITIES:
        errs.append("switch_density")
    if not isinstance(rec["tables"], list) or not rec["tables"]:
        errs.append("tables")
    if len(rec["langs"]) != len(rec["tables"]):
        errs.append("langs_len")
    if len(rec["table_status"]) != len(rec["tables"]):
        errs.append("table_status_len")
    if rec["k"] != len(rec["tables"]):
        errs.append("k")
    segs = rec["segments"]
    join = rec.get("seg_join", "\n")
    if segs is None:
        if rec["k"] != 1 and rec["doc_type"] != "math":
            errs.append("segments_null_k>1")
    else:
        if len(segs) < 2:
            errs.append("segments_len")
        if join.join(s["text"] for s in segs) != rec["text"]:
            errs.append("segments_text_join")
        if join.join(s["braille"] for s in segs) != rec["braille"]:
            errs.append("segments_braille_join")
        if join == "\n" and [s["table"] for s in segs] != rec["tables"]:
            errs.append("segments_tables")
    if rec["cells"] != cell_count(rec["braille"]):
        errs.append("cells_stale")
    if rec["len_bucket"] != len_bucket(rec["cells"], rec["n_sentences"]):
        errs.append("len_bucket_stale")
    # a span never lists its own table as a confusable (e.g. a retabled kodisc twin)
    if segs is None:
        if rec["tables"][0] in rec["confusable_set"]:
            errs.append("confusable_self")
    elif any(s["table"] in (s.get("confusable_set") or []) for s in segs):
        errs.append("confusable_self")
    if not isinstance(rec["text"], str) or not rec["text"]:
        errs.append("text_empty")
    if not isinstance(rec["braille"], str) or not rec["braille"]:
        errs.append("braille_empty")
    errs += strict_schema_errors(rec)
    return errs


def strict_schema_errors(rec: dict) -> list[str]:
    """Types, allowed/required extra fields, private fields, confusable union and newline structure."""
    errs = []
    form = rec.get("form")
    extra = set(rec) - set(ALL_FIELDS)
    allowed = FORM_EXTRAS.get(form, set()) | ({"seg_join"} if rec.get("segments") else set())
    for kf in sorted(extra - allowed):
        errs.append(f"unexpected_field:{kf}")
    for kf in sorted(REQUIRED_EXTRAS.get(form, set()) - set(rec)):
        errs.append(f"missing_extra:{kf}")
    if any(str(kf).startswith("_") for kf in rec):
        errs.append("private_field")
    for kf, ty in _TYPES.items():
        v = rec.get(kf)
        if not isinstance(v, ty) or (ty is int and isinstance(v, bool)):
            errs.append(f"type:{kf}")
    for kf in ("tables", "langs", "confusable_set", "table_status"):
        if isinstance(rec.get(kf), list) and not all(isinstance(x, str) for x in rec[kf]):
            errs.append(f"type_items:{kf}")
    if isinstance(rec.get("n_sentences"), int) and rec["n_sentences"] < 1:
        errs.append("n_sentences<1")
    if not rec.get("tag"):
        errs.append("tag_empty")
    conf = rec.get("confusable_set") or []
    if conf != sorted(set(conf)):
        errs.append("confusable_not_sorted_unique")
    segs = rec.get("segments")
    br, tx = rec.get("braille") or "", rec.get("text") or ""
    if segs is not None:
        if not isinstance(segs, list) or not all(isinstance(x, dict) for x in segs):
            errs.append("type:segments")
            return errs
        union = sorted({t for sg in segs for t in (sg.get("confusable_set") or [])})
        if union != conf:
            errs.append("confusable_union")
        if rec.get("seg_join", "\n") == "\n":
            if br.count("\n") != len(segs) - 1 or tx.count("\n") != len(segs) - 1:
                errs.append("newline_count")
        elif "\n" in br or "\n" in tx:
            errs.append("newline_in_inline_row")
    else:
        if "\n" in br or "\n" in tx:
            errs.append("newline_in_single_row")
    return errs
