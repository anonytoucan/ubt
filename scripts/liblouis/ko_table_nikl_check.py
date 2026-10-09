"""Check a Korean table (default ko-2020-g2) against the whole NIKL print-braille parallel corpus.

Usage:
    python scripts/liblouis/ko_table_nikl_check.py --tables-dir work/liblouis-src/tables \
        --out work/ko_table_regress/baseline [--workers 200] [--split all|dev|test]

- input: data_nikl/raw/nikl/NIKL_KB_20{22,23,24}_v1.0/**.json (2023: JSON only; its CSVs are copies)
- forward translation with unicode.dis; the blank cell U+2800 counts as a space
- verdicts: exact_raw (source as is) and exact_ws (whitespace collapsed and stripped on both sides)
- split: sha256(document id) % 5 == 0 is test (20%); table repairs look at dev only
- output: <out>/rows.jsonl.gz (all pairs), <out>/summary.json
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from ubt.paths import LIBLOUIS  # noqa: E402

NIKL_ROOT = "data_nikl/raw/nikl"
WS = re.compile(r"\s+")

_LOUIS = None
_TABLES = None


def load_pairs(root: str = NIKL_ROOT) -> list[dict]:
    files = sorted(glob.glob(os.path.join(root, "NIKL_KB_2022_v1.0", "*.json"))
                   + glob.glob(os.path.join(root, "NIKL_KB_2023_v1.0", "JSON", "*.json"))
                   + glob.glob(os.path.join(root, "NIKL_KB_2024_v1.0", "*.json")))
    rows = []
    for f in files:
        ed = re.search(r"NIKL_KB_(20\d\d)", f).group(1)
        genre = os.path.basename(f)[:2]
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        for p in d["parallel"]:
            doc = p["id"].split(".")[0]
            h = int(hashlib.sha256(doc.encode()).hexdigest(), 16)
            rows.append({
                "id": p["id"], "ed": ed, "genre": genre, "doc": doc,
                "split": "test" if h % 5 == 0 else "dev",
                "src": p["source"], "tgt": p["target"],
            })
    return rows


def _init(tables_dir: str, table: str) -> None:
    global _LOUIS, _TABLES
    os.environ["LOUIS_TABLEPATH"] = ",".join(
        [tables_dir, f"{LIBLOUIS}/share/liblouis/tables"])
    import louis  # noqa: PLC0415
    _LOUIS = louis
    _TABLES = ["unicode.dis", table]


def _tr(text: str, options: dict | None = None) -> str:
    if options is not None:
        from ubt.ko2024_options import translate
        return translate(_LOUIS, _TABLES, text, options)
    return _LOUIS.translateString(_TABLES, text, mode=0).replace("⠀", " ")


def preflight(tables_dir: str, table: str) -> None:
    """Reject an invalid table once, before workers or corpus loading."""
    _init(os.path.abspath(tables_dir), table)
    _LOUIS.checkTable(_TABLES)


def work(r: dict) -> dict:
    tgt = r["tgt"].replace("⠀", " ")
    try:
        hyp = _tr(r["src"], r.get("translation_options"))
    except Exception as e:  # noqa: BLE001
        hyp = f"<EXC {type(e).__name__}>"
    src_ws = WS.sub(" ", r["src"]).strip()
    tgt_ws = WS.sub(" ", tgt).strip()
    # Option spans bind to exact source offsets, so the source stays as is; exact_ws normalizes outputs only.
    if src_ws == r["src"] or r.get("translation_options"):
        hyp_ws = hyp
    else:
        try:
            hyp_ws = _tr(src_ws)
        except Exception as e:  # noqa: BLE001
            hyp_ws = f"<EXC {type(e).__name__}>"
    hyp_ws = WS.sub(" ", hyp_ws).strip()
    return {**r, "hyp": hyp, "exact_raw": hyp == tgt,
            "hyp_ws": hyp_ws, "exact_ws": hyp_ws == tgt_ws}


def sha_dir(d: str, pat: str = "ko-*") -> dict:
    out = {}
    for f in sorted(glob.glob(os.path.join(d, pat))):
        out[os.path.basename(f)] = hashlib.sha256(open(f, "rb").read()).hexdigest()
    return out


def engine_identity() -> dict:
    """Hash the liblouis library loaded by this process (reads /proc, Linux only)."""
    with open("/proc/self/maps", encoding="utf-8") as fh:
        paths = {os.path.realpath(parts[5]) for line in fh
                 if len(parts := line.rstrip().split(maxsplit=5)) == 6
                 and os.path.basename(parts[5]).startswith("liblouis.so")}
    if len(paths) != 1:
        raise RuntimeError(f"Expected one mapped liblouis library, found {sorted(paths)}")
    library = paths.pop()
    with open(library, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    with open(_LOUIS.__file__, "rb") as fh:
        binding_digest = hashlib.sha256(fh.read()).hexdigest()
    return {"version": _LOUIS.version(), "character_bytes": _LOUIS.wideCharBytes,
            "library_path": library,
            "library_sha256": digest, "python_binding_sha256": binding_digest}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", required=True)
    ap.add_argument("--table", default="ko-2020-g2.ctb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=200)
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"])
    args = ap.parse_args()

    t0 = time.time()
    preflight(args.tables_dir, args.table)
    engine = engine_identity()
    rows = load_pairs()
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=_init,
                  initargs=(args.tables_dir, args.table)) as pool:
        res = pool.map(work, rows, chunksize=64)

    os.makedirs(args.out, exist_ok=True)
    with gzip.open(os.path.join(args.out, "rows.jsonl.gz"), "wt", encoding="utf-8") as fh:
        for r in res:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def rate(sel, key):
        n = len(sel)
        return {"n": n, "exact": sum(r[key] for r in sel),
                "pct": round(100 * sum(r[key] for r in sel) / n, 4) if n else None}

    by = defaultdict(list)
    for r in res:
        by[("split", r["split"])].append(r)
        by[("ed", r["ed"])].append(r)
        by[("genre", r["ed"] + r["genre"])].append(r)
    try:
        fork_sha = subprocess.check_output(
            ["git", "-C", args.tables_dir, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        fork_sha = None
    summary = {
        "table": args.table, "tables_dir": args.tables_dir,
        "tables_git_sha": fork_sha, "table_sha256": sha_dir(args.tables_dir),
        "louis_version": _louis_version(args.tables_dir, args.table),
        "engine": engine,
        "split": args.split, "n_pairs": len(res),
        "n_unique_src": len({r["src"] for r in res}),
        "exceptions": sum(r["hyp"].startswith("<EXC") for r in res),
        "all_raw": rate(res, "exact_raw"), "all_ws": rate(res, "exact_ws"),
        "by": {f"{k[0]}={k[1]}": {"raw": rate(v, "exact_raw"), "ws": rate(v, "exact_ws")}
               for k, v in sorted(by.items())},
        "wall_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: summary[k] for k in
                      ("n_pairs", "n_unique_src", "exceptions", "all_raw", "all_ws", "wall_sec")},
                     ensure_ascii=False))


def _louis_version(tables_dir: str, table: str) -> str:
    _init(tables_dir, table)
    return _LOUIS.version()


if __name__ == "__main__":
    main()
