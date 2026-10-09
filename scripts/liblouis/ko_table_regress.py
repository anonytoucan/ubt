"""Regression harness for the Korean 2024 table repairs.

One run measures three checks:
  NIKL dev (all splits with --unseal): exact_raw and rows fixed/broken against the reference run (--ref)
  rule examples: testable rows of resources/korean_braille_2024/examples.jsonl, if present
  liblouis yaml tests (tests/braille-specs/ko-2020.yaml), with expected values the rules overturn in YAML_OVERRIDES

Usage:
  python scripts/liblouis/ko_table_regress.py --tables-dir work/ko2024_tables --name r001 [--ref baseline]
Output: work/ko_table_regress/<name>/{rows.jsonl.gz, report.json, fixed.txt, broken.txt, j1_fail.txt}
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import multiprocessing as mp
import os
import sys
import time

import yaml

sys.path[:0] = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "*", "")))
import ko_table_nikl_check as V  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from ubt.paths import LIBLOUIS_SRC  # noqa: E402

YAML = f"{LIBLOUIS_SRC}/tests/braille-specs/ko-2020.yaml"
EXAMPLES = "resources/korean_braille_2024/examples.jsonl"
YAML_OVERRIDES = "resources/korean_braille_2024/yaml_overrides.json"


def _tr_display(args):
    text, disp = args
    return V._LOUIS.translateString([disp, V._TABLES[1]], text, mode=0)


def load_ref(name: str) -> dict:
    path = f"work/ko_table_regress/{name}/rows.jsonl.gz" if name == "baseline" else f"work/ko_table_regress/{name}/rows.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {r["id"]: r["exact_raw"] for r in map(json.loads, fh)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", required=True)
    ap.add_argument("--table", default="ko-2020-g2.ctb")
    ap.add_argument("--name", required=True)
    ap.add_argument("--ref", default="baseline")
    ap.add_argument("--options-jsonl", help="source-bound transcription options; "
                    "dev annotations may not select test rows")
    ap.add_argument("--workers", type=int, default=200)
    ap.add_argument("--unseal", action="store_true", help="include the sealed test split")
    ap.add_argument("--examples", default=EXAMPLES, help="rule-example JSONL")
    ap.add_argument("--j1-only", action="store_true", help="skip NIKL; only rule examples + yaml")
    ap.add_argument("--filter", default=None,
                    help="regex on src; keep matching sentences (union with --sample)")
    ap.add_argument("--sample", type=int, default=0,
                    help="also include a fixed sample of N dev sentences")
    args = ap.parse_args()
    t0 = time.time()
    out = f"work/ko_table_regress/{args.name}"
    os.makedirs(out, exist_ok=True)
    tables_dir = os.path.abspath(args.tables_dir)

    V.preflight(tables_dir, args.table)
    engine = V.engine_identity()
    rows = V.load_pairs()
    if not args.unseal:
        rows = [r for r in rows if r["split"] == "dev"]
    options_report = {"contract": "sentence"}
    if args.options_jsonl:
        from ubt.ko2024_options import attach_options
        rows, options_report = attach_options(rows, args.options_jsonl)
    if args.j1_only:
        rows = []
    if args.filter or args.sample:
        import hashlib  # noqa: PLC0415
        import re  # noqa: PLC0415
        rx = re.compile(args.filter) if args.filter else None
        keep = []
        for r in rows:
            hit = bool(rx and rx.search(r["src"]))
            if not hit and args.sample:
                h = int(hashlib.md5(r["id"].encode()).hexdigest()[:8], 16)
                hit = h % max(1, len(rows) // args.sample) == 0
            if hit:
                keep.append(r)
        rows = keep
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=V._init, initargs=(tables_dir, args.table)) as pool:
        res = pool.map(V.work, rows, chunksize=64)

        # rule examples
        j1 = []
        if os.path.exists(args.examples):
            ex = [json.loads(line) for line in open(args.examples, encoding="utf-8")]
            ex = [e for e in ex if e.get("testable")]
            outs = pool.map(V.work, [{"id": f"ex{i}", "src": e["print"], "tgt": e["braille"].replace("⠀", " ")}
                                     for i, e in enumerate(ex)], chunksize=8)
            for e, o in zip(ex, outs):
                j1.append({**e, "hyp": o["hyp"], "ok": o["exact_raw"]})

        # liblouis yaml tests
        spec = yaml.safe_load(open(YAML, encoding="utf-8"))
        overrides = json.load(open(YAML_OVERRIDES)) if os.path.exists(YAML_OVERRIDES) else {}
        cases = [(t[0], overrides.get(t[0], t[1])) for t in spec["tests"]]
        youts = pool.map(_tr_display, [(c[0], spec["display"]) for c in cases])
    yfail = [(c[0], c[1], o) for c, o in zip(cases, youts) if o != c[1]]

    with gzip.open(f"{out}/rows.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in res:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    ref = load_ref(args.ref) if rows else {}
    fixed = [r for r in res if r["exact_raw"] and not ref.get(r["id"], False)]
    broken = [r for r in res if not r["exact_raw"] and ref.get(r["id"], False)]
    for name, lst in (("fixed", fixed), ("broken", broken)):
        with open(f"{out}/{name}.txt", "w", encoding="utf-8") as fh:
            for r in lst:
                fh.write(f"[{r['id']}]\n  SRC {r['src']}\n  TGT {r['tgt']}\n  HYP {r['hyp']}\n")
    with open(f"{out}/j1_fail.txt", "w", encoding="utf-8") as fh:
        for e in j1:
            if not e["ok"]:
                fh.write(f"[{e.get('doc')} p{e.get('pdf_page')} {e.get('article')}] {e['print']}\n"
                         f"  RULE {e['braille']}\n  HYP  {e['hyp']}\n")
    with open(f"{out}/yaml_fail.txt", "w", encoding="utf-8") as fh:
        for a, b, c in yfail:
            fh.write(f"{a}\n  WANT {b}\n  GOT  {c}\n")
    n = len(res)
    ex = sum(r["exact_raw"] for r in res)
    report = {
        "name": args.name, "ref": args.ref, "tables_dir": tables_dir,
        "input_contract": options_report,
        "engine": engine,
        "table_sha256": V.sha_dir(tables_dir), "split": "all" if args.unseal else "dev",
        "filter": args.filter, "sample": args.sample,
        "examples": args.examples,
        "nikl": {"n": n, "exact": ex, "pct": round(100 * ex / n, 4) if n else None, "wrong": n - ex,
                 "fixed_vs_ref": len(fixed), "broken_vs_ref": len(broken)},
        "j1": {"n": len(j1), "ok": sum(e["ok"] for e in j1)},
        "yaml": {"n": len(cases), "fail": len(yfail)},
        "wall_sec": round(time.time() - t0, 1),
    }
    json.dump(report, open(f"{out}/report.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(report["nikl"] | {"j1": report["j1"], "yaml": report["yaml"],
                                       "wall": report["wall_sec"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
