"""data-u1: canonical re-transcription with one liblouis table per process.

liblouis output for a table can depend on which other tables the process compiled before, so the
mixed-table builder and verifier pools are not a trustworthy f. Here f(table, text) is the translation
in a process that loaded only `table` (normal pass, then the strict shadow pass for acceptance), and
every row that differs is rewritten. --ko-tables-dir stages a newer ko-2024 table; --only-group
script:Hangul limits the run to rows with a Hangul-group segment.

Usage (run from the code snapshot the dataset was built with):
  python scripts/data/u1_retranscribe.py --data <in> --out <new dir> [--workers 160]
        [--ko-tables-dir <dir> --ko-table-file ko-2020-g2.ctb] [--only-group script:Hangul]
        [--check]      # report only, write nothing

Per row: each liblouis segment gets the canonical braille and a strict-pass failure drops the row;
confusable_set = siblings with the same canonical braille; document fields are rebuilt as in
make_u1_record. Nemeth segments are untouched; S7 (NIKL) keeps the human braille and only f_consistent
is recomputed; Enoise rows are re-noised deterministically when their Ecore base changed. Text, ids,
splits and the registry never change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

from ubt.formats import len_bucket  # noqa: E402
from ubt.noise import flip_dots  # noqa: E402
from ubt.u1 import engine as EN  # noqa: E402
from ubt.verify import cell_count, is_braille_only  # noqa: E402
from ubt.u1.sources import _bengali_fix  # noqa: E402
import unicodedata  # noqa: E402


def _norm(text: str) -> str:
    """The u1 re-encoding normal form: NFC + Bengali nukta fix."""
    return _bengali_fix(unicodedata.normalize("NFC", text))

EN_DISPLAY = "unicode.dis"
PSEUDO = {EN.NEMETH_LABEL, getattr(EN, "KMATH_LABEL", "ko-math-2024")}
_SPEC: dict | None = None          # {"default": spec_dict, <table>: spec_dict, ...}


def _task(args: tuple[int, str, list[str]]) -> tuple[int, list[tuple[str | None, bool]]]:
    """Load only `table` in a fresh process; normal pass for all texts, then the strict pass.
    Returns (task index, [(braille or None, strict_ok)])."""
    idx, table, texts = args
    specs = _SPEC if "default" in (_SPEC or {}) else {"default": _SPEC}
    spec = EN.EngineSpec.from_dict(specs.get(table, specs["default"]))
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.chdir(spec.cwd_dir)
    tr = EN.make_translator(spec)
    lou = tr._louis
    tabs = [EN_DISPLAY, table]
    normal = []
    for t in texts:
        try:
            normal.append(lou.translateString(tabs, _norm(t), mode=0))
        except Exception:  # noqa: BLE001
            normal.append(None)
    out = []
    strict_tabs = [EN_DISPLAY, tr.strict_table(table)]
    for t, b in zip(texts, normal):
        if b is None:
            out.append((None, False))
            continue
        try:
            s = lou.translateString(strict_tabs, _norm(t), mode=tr._no_undefined)
        except Exception:  # noqa: BLE001
            s = None
        out.append((b, s == b and is_braille_only(b, allow_newline=False)))
    return idx, out


def _init(spec_dict: dict) -> None:
    """spec_dict: one EngineSpec dict, or {"default": ..., <table>: ...} (per-table engines)."""
    global _SPEC
    _SPEC = spec_dict


def load_jsonl(p: str) -> list[dict]:
    with open(p, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def row_segments(r: dict) -> list[dict]:
    if r.get("segments"):
        return r["segments"]
    return [{"table": r["tables"][0], "lang": (r.get("langs") or [None])[0], "text": r["text"],
             "braille": r["braille"], "confusable_set": list(r.get("confusable_set") or [])}]


def sha256_path(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=160)
    ap.add_argument("--ko-tables-dir", default=None)
    ap.add_argument("--ko-table-file", default="ko-2020-g2.ctb")
    ap.add_argument("--only-group", default=None, help="e.g. script:Hangul")
    ap.add_argument("--table-engine", action="append", default=[],
                    help="TABLE=PREFIX: translate TABLE with the liblouis build at PREFIX; repeatable")
    ap.add_argument("--louis-prefix", default=None,
                    help="use the liblouis build under this prefix (lib/, share/liblouis/tables)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--chunk", type=int, default=20000)
    args = ap.parse_args()
    t0 = time.time()
    src = os.path.abspath(args.data)
    if not args.check:
        if not args.out:
            sys.exit("--out required unless --check")
        out = os.path.abspath(args.out)
        if os.path.exists(out) and os.listdir(out):
            sys.exit(f"--out {out} is not empty")
    man = json.load(open(os.path.join(src, "manifest.json")))
    inv = json.load(open(os.path.join(src, "inventory.json")))
    groups = inv["groups"]
    tgroups = {t["table_id"]: t.get("groups", []) for t in inv["tables"]}

    def siblings(t: str) -> list[str]:
        s = set()
        for g in tgroups.get(t, []):
            s.update(groups.get(g, []))
        s.discard(t)
        return sorted(s)

    only = set(groups[args.only_group]) if args.only_group else None

    # ---- engine: same build; new ko-2024 staged under <out>/engine when requested
    old = EN.EngineSpec.from_dict(man["engine_spec"])
    new_engine = None
    if (args.ko_tables_dir or args.louis_prefix) and not args.check:
        eng = os.path.join(out, "engine")
        lt, ll = old.louis_tables, old.louis_lib
        if args.louis_prefix:
            lt = os.path.join(os.path.abspath(args.louis_prefix), "share", "liblouis", "tables")
            ll = os.path.join(os.path.abspath(args.louis_prefix), "lib", "liblouis.so")
        kd = os.path.abspath(args.ko_tables_dir) if args.ko_tables_dir else old.ko2024_src_dir
        kf = args.ko_table_file if args.ko_tables_dir else old.ko2024_src_file
        spec = EN.EngineSpec(
            louis_tables=lt, louis_lib=ll, louis_version=old.louis_version,
            ko2024_src_dir=kd, ko2024_src_file=kf,
            stage_dir=os.path.join(eng, "ko2024"), cwd_dir=os.path.join(eng, "cwd"),
            nemeth_jar=old.nemeth_jar, ko2024_certified=False, java=old.java,
            strict_dir=os.path.join(eng, "strict"))
        os.makedirs(out, exist_ok=True)
        staging = EN.stage_ko2024(spec)
        cert = EN.check_certificate(spec, staging)
        spec.ko2024_certified = cert["certified"]
        shadow = EN.build_strict_shadow(spec)
        new_engine = {"staging": staging, "cert": cert, "shadow": shadow}
    else:
        spec = old
    overrides: dict[str, EN.EngineSpec] = {}
    if not args.table_engine and man.get("engine_spec_overrides") and (args.check or not new_engine):
        # a dataset amended with per-table engines: re-check with the same engines
        overrides = {t: EN.EngineSpec.from_dict(d) for t, d in man["engine_spec_overrides"].items()}
    if args.table_engine:
        if args.check:
            sys.exit("--table-engine needs a write run (it stages strict shadows under --out)")
        for item in args.table_engine:
            tab, prefix = item.split("=", 1)
            prefix = os.path.abspath(prefix)
            tag = "".join(c if c.isalnum() else "_" for c in tab)
            eng_o = os.path.join(out, "engine", f"override_{tag}")
            so = EN.EngineSpec(
                louis_tables=os.path.join(prefix, "share", "liblouis", "tables"),
                louis_lib=os.path.join(prefix, "lib", "liblouis.so"),
                louis_version=spec.louis_version, ko2024_src_dir=spec.ko2024_src_dir,
                ko2024_src_file=spec.ko2024_src_file, stage_dir=spec.stage_dir,
                cwd_dir=os.path.join(eng_o, "cwd"), nemeth_jar=spec.nemeth_jar,
                ko2024_certified=spec.ko2024_certified, java=spec.java,
                strict_dir=os.path.join(eng_o, "strict"))
            os.makedirs(so.cwd_dir, exist_ok=True)
            EN.build_strict_shadow(so)
            overrides[tab] = so
        print(f"[iso] per-table engines: { {t: o.louis_lib for t, o in overrides.items()} }", flush=True)
    pool_specs = {"default": spec.to_dict(), **{t: o.to_dict() for t, o in overrides.items()}}

    # ---- load rows
    files = [f for f in man["files"] if f.endswith(".jsonl")]
    data = {f: load_jsonl(os.path.join(src, f)) for f in files}
    print(f"[iso] loaded {sum(len(v) for v in data.values())} rows ({time.time() - t0:.0f}s)", flush=True)

    def in_scope(r: dict) -> bool:
        if not r.get("tables") or r.get("form") == "Enoise":
            return False
        segs = row_segments(r)
        if only is not None and not any(s["table"] in only for s in segs):
            return False
        return any(s["table"] not in PSEUDO for s in segs)

    need: dict[str, set[str]] = defaultdict(set)
    for f, rows in data.items():
        for r in rows:
            if not in_scope(r):
                continue
            for s in row_segments(r):
                t = s["table"]
                if t in PSEUDO:
                    continue
                need[t].add(s["text"])
                if r.get("form") != "S7":
                    for sib in siblings(t):
                        need[sib].add(s["text"])
    tasks = []
    for t, texts in need.items():
        tl = sorted(texts)
        for i in range(0, len(tl), args.chunk):
            tasks.append((t, tl[i:i + args.chunk]))
    tasks.sort(key=lambda x: -len(x[1]))
    tasks = [(i, t, tl) for i, (t, tl) in enumerate(tasks)]
    print(f"[iso] {sum(len(v) for v in need.values())} (table,text) pairs over {len(need)} tables, "
          f"{len(tasks)} isolated tasks ({time.time() - t0:.0f}s)", flush=True)
    canon: dict[tuple[str, str], tuple[str | None, bool]] = {}
    ctx = mp.get_context("spawn")    # no inherited liblouis state, one table per process
    done = 0
    with ctx.Pool(args.workers, initializer=_init, initargs=(pool_specs,),
                  maxtasksperchild=1) as pool:
        for idx, res in pool.imap_unordered(_task, tasks, chunksize=1):
            _i, table, texts = tasks[idx]
            for t, v in zip(texts, res):
                canon[(table, t)] = v
            done += 1
            if done % 200 == 0:
                print(f"[iso] {done}/{len(tasks)} tasks ({time.time() - t0:.0f}s)", flush=True)
    print(f"[iso] canonical translations done ({time.time() - t0:.0f}s)", flush=True)

    # ---- rewrite rows
    st = {"changed": Counter(), "dropped": Counter(), "conf_changed": Counter(),
          "changed_tables": Counter(), "examples": []}
    new_core: dict[str, str] = {}
    dropped: set[str] = set()
    nikl = [0, 0]
    out_rows: dict[str, list[dict]] = {}
    for f in files:
        rows_out = []
        for r in data[f]:
            if in_scope(r):
                segs = row_segments(r)
                single = not r.get("segments")
                if r.get("form") == "S7":
                    b, _ok = canon[(r["tables"][0], r["text"])]
                    r = {**r, "f_consistent": b is not None and b == r["braille"]}
                    nikl[0] += r["f_consistent"]
                    nikl[1] += 1
                    rows_out.append(r)
                    continue
                new_segs, drop, changed = [], None, False
                for s in segs:
                    t = s["table"]
                    if t in PSEUDO:
                        new_segs.append(s)
                        continue
                    b, ok = canon[(t, s["text"])]
                    if b is None or not ok:
                        drop = f"{t}:{'exception' if b is None else 'strict'}"
                        break
                    conf = sorted(x for x in siblings(t) if canon.get((x, s["text"]), (None,))[0] == b)
                    if b != s["braille"]:
                        changed = True
                        st["changed_tables"][t] += 1
                        if len(st["examples"]) < 30:
                            st["examples"].append({"id": r["id"], "table": t, "old": s["braille"][:120], "new": b[:120]})
                    if conf != sorted(s.get("confusable_set") or []):
                        st["conf_changed"][r.get("form")] += 1
                    new_segs.append({**s, "braille": b, "confusable_set": conf})
                if drop:
                    st["dropped"][f"{r.get('form')}:{drop.split(':')[1]}"] += 1
                    dropped.add(r["id"])
                    continue
                join = r.get("seg_join", "\n")
                braille = join.join(s["braille"] for s in new_segs)
                cells = cell_count(braille)
                r = {**r, "braille": braille, "cells": cells,
                     "len_bucket": len_bucket(cells, r["n_sentences"]),
                     "confusable_set": sorted({c for s in new_segs for c in (s.get("confusable_set") or [])}),
                     "segments": None if single else new_segs}
                if changed:
                    st["changed"][r.get("form")] += 1
                    if r.get("form") == "Ecore":
                        new_core[r["id"]] = braille
            rows_out.append(r)
        out_rows[f] = rows_out
    # noise pass
    for f in files:
        fixed = []
        for r in out_rows[f]:
            if r.get("form") == "Enoise":
                b = r.get("noise_base_id")
                if b in dropped:
                    st["dropped"]["Enoise:base_dropped"] += 1
                    continue
                if b in new_core:
                    base = new_core[b]
                    rng = random.Random(int(hashlib.sha256((r["id"] + "iso").encode()).hexdigest()[:16], 16))
                    for _ in range(10000):
                        nb = flip_dots(base, r["noise_p"], rng, n_dots=r.get("noise_n_dots", 8))
                        if nb != base:
                            break
                    flips = sum(bin((ord(a) - 0x2800) ^ (ord(c) - 0x2800)).count("1")
                                for a, c in zip(base, nb) if a != c)
                    r = {**r, "braille": nb, "noise_n_flips": flips}
                    st["changed"]["Enoise"] += 1
            fixed.append(r)
        out_rows[f] = fixed

    report = {"changed_rows": dict(st["changed"]), "changed_by_table": dict(st["changed_tables"].most_common()),
              "confusable_changed_rows": dict(st["conf_changed"]), "dropped": dict(st["dropped"]),
              "n_dropped": len(dropped), "nikl_f_consistent": nikl, "examples": st["examples"],
              "n_pairs": sum(len(v) for v in need.values()), "wall_sec": round(time.time() - t0, 1)}
    if args.check:
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return

    # ---- write
    table_ids = sorted({t for rows in out_rows.values() for r in rows for t in (r.get("tables") or [])} - PSEUDO)
    extra = {"kmath": EN.kmath_fingerprint()} if hasattr(EN, "kmath_fingerprint") else None
    os.makedirs(out, exist_ok=True)
    if not new_engine:     # same engine: carry the staged engine dir over (ids are path-free)
        shutil.copytree(os.path.join(src, "engine"), os.path.join(out, "engine"), dirs_exist_ok=True)
        spec = EN.EngineSpec.from_dict({**man["engine_spec"],
                                        "stage_dir": os.path.join(out, "engine", "ko2024"),
                                        "cwd_dir": os.path.join(out, "engine", "cwd"),
                                        "strict_dir": os.path.join(out, "engine", "strict")})
    if overrides:
        extra = dict(extra or {})
        extra["table_engine_overrides"] = {
            t: {"engine_id": EN.fingerprint(o, [t])[0], "louis_lib": o.louis_lib,
                "louis_lib_sha256": EN.sha256_file(os.path.realpath(o.louis_lib))}
            for t, o in overrides.items()}
    eid, emap = EN.fingerprint(spec, table_ids, extra=extra)
    for f, rows in out_rows.items():
        p = os.path.join(out, f)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            for r in rows:
                if "engine" in r:
                    r["engine"] = eid
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name in ("inventory.json", "registry.npz"):
        if os.path.exists(os.path.join(src, name)):
            shutil.copy2(os.path.join(src, name), os.path.join(out, name))
    new = dict(man)
    new["engine_id"], new["engines"], new["engine_spec"] = eid, {eid: emap}, spec.to_dict()
    if overrides:
        new["engine_spec_overrides"] = {t: o.to_dict() for t, o in overrides.items()}
    else:
        new.pop("engine_spec_overrides", None)
    if new_engine:
        new["ko2024"], new["ko2024_certificate"], new["strict_shadow"] = (
            new_engine["staging"], new_engine["cert"], new_engine["shadow"])
    for f in files:
        p = os.path.join(out, f)
        new["files"][f] = {"n_docs": len(out_rows[f]), "bytes": os.path.getsize(p), "sha256": sha256_path(p)}
    new["n_train"] = new["files"].get("train.jsonl", {}).get("n_docs", new.get("n_train"))
    new["n_dev"] = new["files"].get("dev.jsonl", {}).get("n_docs", new.get("n_dev"))
    new["amendments"] = list(man.get("amendments", [])) + [{
        "kind": "isolated-retranscription" + ("+ko2024" if args.ko_tables_dir else "")
                + (f"+louis:{args.louis_prefix}" if args.louis_prefix else ""),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv,
        "source_dataset": src, "source_manifest_sha256": sha256_path(os.path.join(src, "manifest.json")),
        "old_engine_id": man["engine_id"], "new_engine_id": eid, **report}]
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(new, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
