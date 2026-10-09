#!/usr/bin/env python3
"""Evaluation rows for three real braille corpora made outside our pipeline (evaluation only, never trained on).

  python scripts/real_corpora/build_real_corpora.py bible|braillellm|vision|all [--out DIR]
  python scripts/real_corpora/build_real_corpora.py check [--out DIR]          # counts, schema, round trip, samples, rates

Writes <out>/{bible,braillellm,vision}.jsonl, one row per document in the bon_eval row format plus the reference and
its forward consistency:

  {id, set="real", group, tables=[code], k=1, regime="single", switch_density="none", confusable_set=[],
   prompt, completion, max_tokens = 2 x completion tokens + 32, text, braille, f_consistent, cell_similarity}

and <out>/meta.json (per corpus: sources and hashes, selection, probes, rates, dropped documents).

  * prompt / completion = ubt.task_format.render(hinted=False) of a single-segment record; each document is one line.
  * text = the corpus's print side, NFC.
  * f_consistent = (forward(code, text) == corpus cells); cell_similarity = 1 - normalised Levenshtein distance.
    forward = the data_u1_v2 engine, one process per table; labels outside the inventory bypass RoutedLouis.
  * strict acceptance = the data engine's undefined-character gate, run in a fresh single-table process.

Sources (provenance and licence in each directory's SOURCE.md; every raw file read is checked against SHA256SUMS):
  bible       datasets/external/kjv_brf_bible: Internet Archive TheHolyBibleBrailleFiles (18 books, BRF, EBAE grade
              2, inline verse markers) + aruljohn/Bible-kjv print JSON.
  braillellm  datasets/external/braillellm_acl_zh: BrailleLLM Data/ACL_Chinese_Test.txt, `chinese_text TAB
              braille_ascii` lines.
  vision      datasets/external/vision_braille_sentence: HF Violet-yo/Chinese-Braille-Dataset-Full-Tone, test split.

Procedure:
  bible       BRF -> cells with the engine's pinned en-us-brf.dis (not text_nabcc.dis) and verse alignment, both via
              scripts/real_corpora/bible_pairs.py. A book is kept when >= 48 of its first 60 paired verses satisfy
              forward(code, text) == cells (pro rata for shorter books); this reproduces the paper's 14-book pool,
              a random 60-verse sample does not. Test set: the pool shuffled with random.Random(0), first 3,000.
              The label en-us-g2.ctb is not in the data_u1_v2 inventory (there it is en_US.tbl, same cells); for
              code-given prefill or RoutedLouis verification rebuild with --bible-code en_US.tbl.
  braillellm  code zhcn-cbs.ctb; braille ASCII -> cells with en-us-brf.dis, space -> U+2800, checked against the
              SOURCE.md fingerprints. The paper's 999/1000 is strict acceptance; cell-exact forward consistency is
              near 0 because the corpus's word division is not what liblouis cbs emits.
  vision      the first 1,000 test rows in file order. The code is the Chinese table with the highest mean cell
              similarity on the first 60 (zhcn-g1.ctb); no table emits this convention, so f_consistent is near 0.
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import glob
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import unicodedata

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path[:0] = sorted(glob.glob(os.path.join(REPO, "scripts", "*", "")))
from ubt.paths import DATASETS, WORK  # noqa: E402

EXT = f"{DATASETS}/external"
BIBLE_DIR = os.path.join(EXT, "kjv_brf_bible")
BRLM_DIR = os.path.join(EXT, "braillellm_acl_zh")
VB_DIR = os.path.join(EXT, "vision_braille_sentence")
DATA_U1 = f"{DATASETS}/data_u1_v2"
OUT = f"{WORK}/real_corpora"
TOKENIZER = f"{WORK}/models/qwen25_cell_tokenizer"

BRF_DIS = "en-us-brf.dis"
BRF_DIS_SHA256 = "47ff9400f4b8a0206b0c1942e13c32b539d17ec099dcd1097873a1b9f5cb8b3c"
ZH_TABLES = ("zh_CHN.tbl", "zh-tw.ctb", "zh-hk.ctb", "zhcn-g1.ctb", "zhcn-g2.ctb", "zhcn-cbs.ctb")
BIBLE_CODE = "en-us-g2.ctb"
BIBLE_ALIASES = ("en_US.tbl", "en_GB.tbl", "en-ueb-g2.ctb")   # probes only, for alias-aware code accuracy
BRAILLELLM_CODE = "zhcn-cbs.ctb"
BLANK = "\u2800"
GROUPS = ("bible", "braillellm", "vision")
ROW_KEYS = ("id", "set", "group", "tables", "k", "regime", "switch_density", "confusable_set", "prompt",
            "completion", "max_tokens", "text", "braille", "f_consistent", "cell_similarity")
# conversion fingerprints recorded in the sources' SOURCE.md
BRLM_PRINTS = {"cells": 95618, "U+2800": 16279, "U+2824": 1252, "U+281B": 4463, "U+281A": 936}
VB_PRINTS = {"cells": 97228, "U+2800": 13, "U+281B": 4209, "U+281A": 938}


# ----------------------------------------------------------------------------------------------------- helpers
def _json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fixity(src_dir: str, paths: list[str]) -> dict:
    """sha256 of every raw file read (symlinks resolved), checked against the source directory's SHA256SUMS."""
    sums = {}
    with open(os.path.join(src_dir, "SHA256SUMS"), encoding="utf-8") as fh:
        for line in fh:
            parts = line.split(None, 1)
            if len(parts) == 2:
                sums[parts[1].strip().lstrip("*")] = parts[0]
    files = {os.path.relpath(os.path.realpath(p), os.path.realpath(src_dir)): sha256_file(p) for p in paths}
    bad = sorted(r for r, s in files.items() if sums.get(r) != s)
    if bad:
        raise SystemExit(f"{src_dir}: sha256 differs from SHA256SUMS for {bad}")
    return {"files": files, "match_SHA256SUMS": True}


def git_rev(path: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def counts(cells_list: list[str], keys: dict) -> dict:
    c = collections.Counter(ch for s in cells_list for ch in s)
    out = {"cells": sum(map(len, cells_list))}
    for k in keys:
        if k.startswith("U+"):
            out[k] = c[chr(int(k[2:], 16))]
    return out


def one_line(doc: dict, stats: collections.Counter) -> None:
    """render() joins segments with \\n, so a document must be one line: braille newlines -> U+2800, text -> space."""
    b = re.sub(r"\r\n|\r|\n", BLANK, doc["braille"])
    t = re.sub(r"\r\n|\r|\n", " ", doc["text"])
    stats["braille_newlines_replaced"] += b != doc["braille"]
    stats["text_newlines_replaced"] += t != doc["text"]
    doc["braille"], doc["text"] = b, t


def nfc(text: str, stats: collections.Counter) -> str:
    t = unicodedata.normalize("NFC", text)
    stats["text_nfc_changed"] += t != text
    return t


def sim(a: str | None, b: str) -> float:
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415
    return 1.0 - Levenshtein.normalized_distance(a or "", b)


def brf_display(spec: dict) -> tuple[dict, dict]:
    """ASCII -> cell map from the engine's own en-us-brf.dis (sha256 pinned)."""
    from bible_pairs import brf_maps  # noqa: PLC0415
    path = os.path.join(spec["louis_tables"], BRF_DIS)
    sha = sha256_file(path)
    if sha != BRF_DIS_SHA256:
        raise SystemExit(f"{path}: sha256 {sha} != pinned {BRF_DIS_SHA256}")
    to_cell, _ = brf_maps(path)
    return to_cell, {"path": path, "sha256": sha}


# ----------------------------------------------------------------------------------------------------- engine
class Forward:
    """forward(table, text) under the data's engine, one process per table: RoutedLouis for inventory tables, the same
    LouisPool spec without the inventory filter for other labels (RoutedLouis would return None)."""

    def __init__(self, data_dir: str = DATA_U1):
        from ubt.rlvr.louis_pool import LouisPool  # noqa: PLC0415
        from ubt.u1.routed_louis import RoutedLouis  # noqa: PLC0415
        man = _json(os.path.join(data_dir, "manifest.json"))
        self.inventory = {t["table_id"] for t in _json(os.path.join(data_dir, "inventory.json"))["tables"]}
        self.spec = man["engine_spec"]
        self.overrides = dict(man.get("engine_spec_overrides") or {})
        self.info = {"data_dir": data_dir, "engine_id": man.get("engine_id"),
                     "louis_version": self.spec.get("louis_version"), "louis_lib": self.spec.get("louis_lib"),
                     "louis_tables": self.spec.get("louis_tables"),
                     "engine_spec_overrides": sorted((man.get("engine_spec_overrides") or {}).keys()), "routes": {}}
        self.routed = RoutedLouis(data_dir, timeout_s=300.0)
        self.extra = LouisPool(self.spec, known_tables=None, timeout_s=300.0,
                               overrides=man.get("engine_spec_overrides"))

    def many(self, items: list[tuple[str, str]]) -> list[str | None]:
        res: list[str | None] = [None] * len(items)
        by: dict[bool, list[int]] = {True: [], False: []}
        for i, (tab, _t) in enumerate(items):
            inv = tab in self.inventory
            self.info["routes"][tab] = "RoutedLouis" if inv else "LouisPool(engine_spec, no inventory filter)"
            by[inv].append(i)
        for inv, idx in by.items():
            if idx:
                outs = (self.routed.translate_many if inv else self.extra.translate_many)([items[i] for i in idx])
                for i, o in zip(idx, outs):
                    res[i] = o
        return res

    def spec_for(self, table: str) -> dict:
        """The engine spec a table's worker runs under: its override or the manifest spec."""
        return self.overrides.get(table, self.spec)

    def close(self) -> None:
        self.routed.close()
        self.extra.close()


def _strict_worker(table: str, spec_dict: dict, texts: list[str]) -> list[str]:
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    from ubt.rlvr.louis_pool import _norm  # noqa: PLC0415
    from ubt.u1 import engine as EN  # noqa: PLC0415
    spec = EN.EngineSpec.from_dict(spec_dict)
    os.chdir(spec.cwd_dir)
    tr = EN.make_translator(spec)
    return [tr.translate(table, _norm(t), check_undefined=True).reason for t in texts]


def strict_accept(fwd: Forward, table: str, docs: list[dict], scope: str) -> dict:
    """The data engine's undefined-character gate, in a fresh process that loads only `table`."""
    import multiprocessing as mp  # noqa: PLC0415
    with mp.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        reasons = pool.apply(_strict_worker, (table, fwd.spec_for(table), [d["text"] for d in docs]))
    bad = {d["id"]: r for d, r in zip(docs, reasons) if r}
    return {"table": table, "docs": scope, "n": len(docs), "n_accepted": len(docs) - len(bad), "rejected": bad}


def table_comparison(docs: list[dict], outs: dict[str, list]) -> dict:
    """Per table: cell-exact forward agreement, mean cell similarity, and on how many documents it is the nearest."""
    tabs = list(outs)
    sims = {t: [sim(o, d["braille"]) for o, d in zip(outs[t], docs)] for t in tabs}
    nearest, tied = collections.Counter(), collections.Counter()
    for i in range(len(docs)):
        m = max(sims[t][i] for t in tabs)
        win = [t for t in tabs if sims[t][i] == m]
        (nearest if len(win) == 1 else tied).update(win)
    return {t: {"exact": sum(o == d["braille"] for o, d in zip(outs[t], docs)),
                "forward_none": sum(o is None for o in outs[t]),
                "mean_cell_similarity": round(statistics.fmean(sims[t]), 6),
                "nearest_unique": nearest[t], "nearest_tied": tied[t]} for t in tabs}


# ----------------------------------------------------------------------------------------------------- rows
def make_rows(docs: list[dict], code: str, group: str, fwd_outs: list, tok) -> tuple[list[dict], dict]:
    from ubt.task_format import render  # noqa: PLC0415
    rows, n_prompt = [], []
    for d, f in zip(docs, fwd_outs):
        assert "\n" not in d["braille"] and "\n" not in d["text"] and d["text"] and d["braille"], d["id"]
        rec = {"id": d["id"], "text": d["text"], "braille": d["braille"], "tables": [code], "segments": None,
               "doc_type": "single", "k": 1, "regime": "single", "switch_density": "none", "confusable_set": []}
        ex = render(rec, hinted=False)
        n_c = len(tok(ex["completion"], add_special_tokens=False)["input_ids"])
        n_prompt.append(len(tok(ex["prompt"], add_special_tokens=True)["input_ids"]))
        row = {"id": d["id"], "set": "real", "group": group, "tables": [code], "k": 1, "regime": "single",
               "switch_density": "none", "confusable_set": [], "prompt": ex["prompt"],
               "completion": ex["completion"], "max_tokens": 2 * n_c + 32, "text": d["text"],
               "braille": d["braille"], "f_consistent": f is not None and f == d["braille"],
               "cell_similarity": round(sim(f, d["braille"]), 6)}
        assert tuple(row) == ROW_KEYS
        rows.append(row)
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids), f"{group}: duplicate ids"
    n = len(rows)
    tokstats = {"max_completion_tokens": max((r["max_tokens"] - 32) // 2 for r in rows),
                "max_max_tokens": max(r["max_tokens"] for r in rows),
                "max_prompt_tokens(add_special_tokens=True)": max(n_prompt),
                "max_prompt_plus_max_tokens": max(p + r["max_tokens"] for p, r in zip(n_prompt, rows)),
                "median_cells": statistics.median(len(r["braille"]) for r in rows)}
    fc = {"n_true": sum(r["f_consistent"] for r in rows), "n": n,
          "rate": round(sum(r["f_consistent"] for r in rows) / n, 6),
          "mean_cell_similarity": round(statistics.fmean(r["cell_similarity"] for r in rows), 6),
          "forward_none": sum(f is None for f in fwd_outs)}
    return rows, {"f_consistent": fc, "tokens": tokstats}


def write_rows(out: str, group: str, rows: list[dict]) -> str:
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{group}.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


# ----------------------------------------------------------------------------------------------------- corpora
def build_bible(a, fwd: Forward, tok) -> tuple[list[dict], dict]:
    from bible_pairs import MARKER, ascii_to_cells, load_text, parse_brf  # noqa: PLC0415
    to_cell, dis = brf_display(fwd.spec)
    code = a.bible_code
    brf_dir = os.path.join(BIBLE_DIR, "brf_bible_18")
    stats: collections.Counter = collections.Counter()
    books: dict[str, dict] = {}
    verses: list[dict] = []
    dropped: list[dict] = []
    read: list[str] = []
    for bp in sorted(glob.glob(os.path.join(brf_dir, "*.brf"))):
        book = os.path.basename(bp)[:-4]
        jp = os.path.join(brf_dir, book.capitalize() + ".json")
        read += [bp, jp]
        with open(bp, encoding="latin-1") as fh:
            raw = fh.read()
        assert not re.search(r"[\r\n\f]", raw), f"{book}: BRF has line breaks"
        brf, notes = parse_brf(raw)
        txt = load_text(jp)
        for c, v in sorted(set(txt) - set(brf)):
            why = ("print verse without a usable BRF body: the marker parser drops duplicate / non-ascending "
                   "markers (" + "; ".join(notes) + ")" if notes else "print verse without a BRF verse marker")
            if book == "acts" and notes:
                why += "; source defect (SOURCE.md): Acts 20:36 is encoded as #BJ3C = 20:3"
            dropped.append({"id": f"real-bible-{book}-{c}-{v}", "reason": why})
        for c, v in sorted(set(brf) - set(txt)):
            dropped.append({"id": f"real-bible-{book}-{c}-{v}", "reason": "BRF verse without print text"})
        n_book = 0
        for c, v in sorted(set(brf) & set(txt)):
            cells = ascii_to_cells(brf[(c, v)], to_cell)
            vid = f"real-bible-{book}-{c}-{v}"
            if "\ufffd" in cells:
                dropped.append({"id": vid, "reason": "BRF character outside en-us-brf.dis"})
                continue
            d = {"id": vid, "book": book, "text": nfc(txt[(c, v)].strip(), stats), "braille": cells}
            one_line(d, stats)
            verses.append(d)
            n_book += 1
        books[book] = {"n_markers": len(MARKER.findall(raw)), "n_print_verses": len(txt), "n_paired": n_book,
                       "parse_notes": notes}
    outs = dict(zip((d["id"] for d in verses), fwd.many([(code, d["text"]) for d in verses])))

    # book filter: first-N probe (see module docstring)
    per_book, kept = {}, []
    for book in books:
        ps = [d for d in verses if d["book"] == book]
        probe = ps[: a.probe_n]
        agree = sum(outs[d["id"]] == d["braille"] for d in probe)
        need = math.ceil(a.probe_pass * len(probe) / a.probe_n)
        per_book[book] = {**books[book], "probe_n": len(probe), "probe_agree": agree, "probe_need": need,
                          "kept": agree >= need,
                          "exact_all_paired": round(sum(outs[d["id"]] == d["braille"] for d in ps) / len(ps), 6)}
        if agree >= need:
            kept.append(book)
    pool = [d for d in verses if d["book"] in kept]
    test = list(pool)
    random.Random(a.seed).shuffle(test)
    test = test[: a.n_test]

    # probes: inventory aliases of EBAE grade 2 and UEB grade 2, for code accuracy modulo aliasing
    alias = {t: dict(zip((d["id"] for d in verses), fwd.many([(t, d["text"]) for d in verses])))
             for t in BIBLE_ALIASES if t != code}
    probes = {}
    for t in (code, *alias):
        o = outs if t == code else alias[t]
        rep = {s: f"{sum(o[d['id']] == d['braille'] for d in ds)}/{len(ds)}"
               for s, ds in (("all_18_books", verses), ("pool", pool), ("test", test), ("test_first120", test[:120]))}
        ent = {"reproduces_corpus_cells": rep}
        if t != code:
            ent[f"same_cells_as_{code}"] = {s: f"{sum(o[d['id']] == outs[d['id']] for d in ds)}/{len(ds)}"
                                            for s, ds in (("all_18_books", verses), ("test", test),
                                                          ("test_first150", test[:150]))}
        ent["route"] = fwd.info["routes"].get(t)
        probes[t] = ent

    rows, rstats = make_rows(test, code, "bible", [outs[d["id"]] for d in test], tok)
    src = fixity(BIBLE_DIR, read)
    meta = {
        "n": len(rows), "group": "bible", "code": code,
        "code_note": ("en-us-g2.ctb is not in the data_u1_v2 inventory (the model's label vocabulary; RoutedLouis "
                      "answers None for it): EBAE grade 2 is en_US.tbl there (= include en-us-g2.ctb + "
                      "braille-patterns.cti). Forward consistency is computed with en-us-g2.ctb itself through "
                      "the same engine without the inventory filter; en_US.tbl gives the same cells on every verse "
                      "(probes). For code-given prefill / RoutedLouis verification use --bible-code en_US.tbl."
                      if code not in fwd.inventory else "code is in the data_u1_v2 inventory"),
        "source": {"braille": "Internet Archive item TheHolyBibleBrailleFiles (no revision ids; files match IA "
                              "md5/sha1, SOURCE.md)",
                   "print": "github.com/aruljohn/Bible-kjv",
                   "print_git_commit": git_rev(os.path.join(BIBLE_DIR, "inkprint_aruljohn_Bible-kjv")),
                   "read_via": brf_dir, **src},
        "display_table": dis,
        "id_scheme": "real-bible-<book file stem>-<chapter>-<verse>",
        "selection": {"pool": f"books kept by the probe, {len(pool)} pairable verses "
                              f"({sum(books[b]['n_markers'] for b in kept)} verse markers)",
                      "order": "pool in (book file order, chapter, verse), random.Random(seed).shuffle, first n_test",
                      "seed": a.seed, "n_test": a.n_test},
        "book_filter": {"rule": (f"probe = the first {a.probe_n} paired verses of each book (canonical order); a "
                                 f"verse aligns when forward({code}, text) == its BRF cells; keep a book at >= "
                                 f"{a.probe_pass}/{a.probe_n} (pro rata, rounded up, if shorter)"),
                        "kept": kept, "dropped": [b for b in books if b not in kept],
                        "n_books_kept": len(kept), "pool_verse_markers": sum(books[b]["n_markers"] for b in kept),
                        "pool_pairable_verses": len(pool), "per_book": per_book,
                        "paper": "14 books, 9,866 verses, 'dropping Psalms, Isaiah and Jonah' (Job is also dropped)"},
        **rstats,
        "strict_accept": strict_accept(fwd, code, test, "test"),
        "probes": probes,
        "dropped": dropped,
        "normalisation": dict(stats) or {"text_nfc_changed": 0},
        "notes": [("text = aruljohn print verse, NFC, stripped: LORD stays in full capitals, so verses "
                   "with small-caps LORD are f-inconsistent under the code")],
    }
    return rows, meta


def build_braillellm(a, fwd: Forward, tok) -> tuple[list[dict], dict]:
    from bible_pairs import ascii_to_cells  # noqa: PLC0415
    to_cell, dis = brf_display(fwd.spec)
    path = os.path.join(BRLM_DIR, "repo", "Data", "ACL_Chinese_Test.txt")
    with open(path, "rb") as fh:
        raw = fh.read().decode("utf-8")
    lines = [ln.removesuffix("\r") for ln in raw.split("\n")]
    if lines and lines[-1] == "":
        lines.pop()
    stats: collections.Counter = collections.Counter()
    docs, dropped = [], []
    for i, ln in enumerate(lines):
        did = f"real-braillellm-{i:04d}"
        if "\t" not in ln:
            dropped.append({"id": did, "reason": "line without a TAB"})
            continue
        text, brf = ln.split("\t", 1)
        cells = ascii_to_cells(brf, to_cell)
        if "\ufffd" in cells or not text or not cells:
            dropped.append({"id": did, "reason": "empty side or braille character outside en-us-brf.dis"})
            continue
        d = {"id": did, "text": nfc(text, stats), "braille": cells}
        one_line(d, stats)
        docs.append(d)
    prints = counts([d["braille"] for d in docs], BRLM_PRINTS)
    outs = {t: fwd.many([(t, d["text"]) for d in docs]) for t in ZH_TABLES}
    code = BRAILLELLM_CODE
    rows, rstats = make_rows(docs, code, "braillellm", outs[code], tok)
    no_blank = sum(o is not None and o.replace(BLANK, "") == d["braille"].replace(BLANK, "")
                   for o, d in zip(outs[code], docs))
    meta = {
        "n": len(rows), "group": "braillellm", "code": code,
        "source": {"repo": "github.com/Tianyuan-Huang/BrailleLLM",
                   "git_commit": git_rev(os.path.join(BRLM_DIR, "repo")), **fixity(BRLM_DIR, [path])},
        "display_table": dis,
        "id_scheme": "real-braillellm-<0-based line index in ACL_Chinese_Test.txt>",
        "selection": f"all {len(lines)} lines (CRLF), split at the first TAB",
        "conversion": {"method": "one braille-ASCII character -> one cell (en-us-brf.dis), ASCII space -> U+2800",
                       "observed": prints, "expected_SOURCE_md": BRLM_PRINTS, "match": prints == BRLM_PRINTS},
        **rstats,
        "f_consistent_ignoring_blank_cells": f"{no_blank}/{len(docs)}",
        "strict_accept": strict_accept(fwd, code, docs, "all"),
        "table_comparison_all_docs": table_comparison(docs, outs),
        "paper": ("'re-transcribes to our cbs cell scheme on 999/1000 documents' = strict_accept (the cbs table "
                  "leaves no character undefined), NOT cell-exact forward consistency"),
        "dropped": dropped,
        "normalisation": dict(stats) or {"text_nfc_changed": 0},
    }
    return rows, meta


def build_vision(a, fwd: Forward, tok) -> tuple[list[dict], dict]:
    import pyarrow.parquet as pq  # noqa: PLC0415
    path = os.path.join(VB_DIR, "hf_snapshot", "data", "test-00000-of-00001.parquet")
    table = pq.read_table(path)
    n_split = table.num_rows
    stats: collections.Counter = collections.Counter()
    docs, dropped = [], []
    for i, r in enumerate(table.slice(0, a.vision_n).to_pylist()):
        did = f"real-vision-{i:04d}"
        text, cells = r["chinese_text"] or "", r["braille_text"] or ""
        if not text or not cells or any(not 0x2800 <= ord(ch) <= 0x28FF for ch in cells):
            dropped.append({"id": did, "reason": "empty side or non-braille character in braille_text"})
            continue
        d = {"id": did, "text": nfc(text, stats), "braille": cells}
        one_line(d, stats)
        docs.append(d)
    prints = counts([d["braille"] for d in docs], VB_PRINTS)
    outs = {t: fwd.many([(t, d["text"]) for d in docs]) for t in ZH_TABLES}
    probe_docs = docs[: a.vision_probe]
    probe = table_comparison(probe_docs, {t: o[: a.vision_probe] for t, o in outs.items()})
    for t in ZH_TABLES:
        probe[t]["strict_accepted"] = strict_accept(fwd, t, probe_docs, "probe")["n_accepted"]
    code = max(ZH_TABLES, key=lambda t: probe[t]["mean_cell_similarity"])
    rows, rstats = make_rows(docs, code, "vision", outs[code], tok)
    rev_meta = os.path.join(VB_DIR, "hf_snapshot", ".cache", "huggingface", "download", "data",
                            "test-00000-of-00001.parquet.metadata")
    revision = None
    if os.path.isfile(rev_meta):                   # huggingface_hub download metadata: line 1 = commit
        with open(rev_meta, encoding="utf-8") as fh:
            revision = fh.readline().strip()
    meta = {
        "n": len(rows), "group": "vision", "code": code,
        "source": {"hf_dataset": "Violet-yo/Chinese-Braille-Dataset-Full-Tone", "revision": revision,
                   "split": "test", "split_rows": n_split, **fixity(VB_DIR, [path])},
        "id_scheme": "real-vision-<0-based row index in the test split>",
        "selection": f"the first {a.vision_n} rows of the test split in file order (not a seeded sample)",
        "fingerprints": {"observed": prints, "expected_SOURCE_md": VB_PRINTS, "match": prints == VB_PRINTS},
        "code_probe": {"docs": f"the first {len(probe_docs)} selected documents ({probe_docs[0]['id']} .. "
                               f"{probe_docs[-1]['id']})",
                       "rule": "code = the table with the highest mean cell similarity on the probe",
                       "chosen": code, "per_table": probe,
                       "paper": "'our closest table matches 1 of 60 probe documents'"},
        "table_comparison_all_docs": table_comparison(docs, outs),
        **rstats,
        "strict_accept": strict_accept(fwd, code, docs, "all"),
        "dropped": dropped,
        "normalisation": dict(stats) or {"text_nfc_changed": 0},
    }
    return rows, meta


BUILDERS = {"bible": build_bible, "braillellm": build_braillellm, "vision": build_vision}


# ----------------------------------------------------------------------------------------------------- check
def cmd_check(a) -> int:
    from transformers import AutoTokenizer  # noqa: PLC0415

    from ubt.wandb_utils import parse_target  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    meta = _json(os.path.join(a.out, "meta.json"))
    ok = True
    for g in GROUPS:
        path = os.path.join(a.out, f"{g}.jsonl")
        if not os.path.isfile(path):
            print(f"[{g}] missing {path}")
            ok = False
            continue
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        m = meta.get(g, {})
        code = m.get("code")
        errs = collections.Counter()
        for r in rows:
            errs["keys"] += tuple(r) != ROW_KEYS
            errs["fixed_fields"] += (r["set"], r["group"], r["k"], r["regime"], r["switch_density"],
                                     r["confusable_set"], r["tables"]) != ("real", g, 1, "single", "none", [], [code])
            errs["prompt"] += r["prompt"] != "<|braille|>\n" + r["braille"] + "\n<|text|>\n"
            errs["completion"] += parse_target(r["completion"]) != [(code, r["text"])]
            n_c = len(tok(r["completion"], add_special_tokens=False)["input_ids"])
            errs["max_tokens"] += r["max_tokens"] != 2 * n_c + 32
            errs["newline"] += "\n" in r["braille"] or "\n" in r["text"]
        errs["n_vs_meta"] = len(rows) != m.get("n")
        errs["dup_ids"] = len({r["id"] for r in rows}) != len(rows)
        bad = {k: v for k, v in errs.items() if v}
        ok &= not bad
        n_fc = sum(r["f_consistent"] for r in rows)
        print(f"[{g}] n={len(rows)} code={code} f_consistent={n_fc}/{len(rows)} ({100 * n_fc / len(rows):.2f}%) "
              f"mean cell_similarity={statistics.fmean(r['cell_similarity'] for r in rows):.4f} "
              f"errors={bad or 'none'}")
        for r in (rows[0], rows[len(rows) // 2], rows[-1]):
            print(f"   {r['id']}  f={r['f_consistent']} sim={r['cell_similarity']:.4f} max_tokens={r['max_tokens']}")
            print(f"      text   : {r['text'][:70]}{'…' if len(r['text']) > 70 else ''}")
            print(f"      braille: {r['braille'][:70]}{'…' if len(r['braille']) > 70 else ''}")
    if "bible" in meta:
        bf = meta["bible"]["book_filter"]
        print(f"[bible] book filter kept {bf['n_books_kept']} books ({bf['pool_verse_markers']} markers / "
              f"{bf['pool_pairable_verses']} pairable), dropped {bf['dropped']}")
    if "vision" in meta:
        cp = meta["vision"]["code_probe"]
        print(f"[vision] probe {cp['docs']}: chosen {cp['chosen']}; "
              + ", ".join(f"{t} {v['exact']}/{v['mean_cell_similarity']:.3f}" for t, v in cp["per_table"].items()))
    print("CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ----------------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("cmd", choices=[*GROUPS, "all", "check"])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--data-dir", default=DATA_U1, help="data-u1 dir whose manifest engine is forward")
    ap.add_argument("--bible-code", default=BIBLE_CODE, help="Bible code label (inventory name: en_US.tbl)")
    ap.add_argument("--seed", type=int, default=0, help="Bible test shuffle seed")
    ap.add_argument("--n-test", type=int, default=3000, help="Bible test verses")
    ap.add_argument("--probe-n", type=int, default=60, help="Bible book probe: first N verses of each book")
    ap.add_argument("--probe-pass", type=int, default=48, help="Bible book probe: keep at >= PASS of N")
    ap.add_argument("--vision-n", type=int, default=1000, help="Vision-Braille: first N test rows (file order)")
    ap.add_argument("--vision-probe", type=int, default=60, help="Vision-Braille code probe: first N documents")
    a = ap.parse_args()
    if a.cmd == "check":
        return cmd_check(a)

    from transformers import AutoTokenizer  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    fwd = Forward(a.data_dir)
    meta_path = os.path.join(a.out, "meta.json")
    meta = _json(meta_path) if os.path.isfile(meta_path) else {}
    script_sha = sha256_file(os.path.abspath(__file__))
    try:
        for g in (GROUPS if a.cmd == "all" else (a.cmd,)):
            rows, m = BUILDERS[g](a, fwd, tok)
            path = write_rows(a.out, g, rows)
            m["file"] = {"path": path, "sha256": sha256_file(path)}
            m["built"] = {"created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "script_sha256": script_sha,
                          "repo_git_head": git_rev(REPO), "argv": sys.argv[1:]}
            meta[g] = m
            fc = m["f_consistent"]
            print(f"[{g}] {len(rows)} rows -> {path}  code={m['code']}  f_consistent {fc['n_true']}/{fc['n']}",
                  flush=True)
    finally:
        fwd.close()
    meta["builder"] = {"script": os.path.relpath(os.path.abspath(__file__), REPO), "script_sha256": script_sha,
                       "repo_git_head": git_rev(REPO), "argv": sys.argv[1:],
                       "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "tokenizer": a.tokenizer,
                       "tokenizer_json_sha256": sha256_file(os.path.join(a.tokenizer, "tokenizer.json")),
                       "engine": fwd.info, "row_keys": list(ROW_KEYS)}
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print(f"meta -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
