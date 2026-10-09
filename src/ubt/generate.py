"""Training-set generation driver.

Usage:
    python -m ubt.generate --n 360000 --workers 48 \
        --config configs/generation.yaml [--out-dir out/data_v2] [--seed 20260706]

Math documents are not generated here (math.enabled=false); their share is logged in the report as a gap.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from collections import Counter, defaultdict

import yaml

from ubt.formats import dedup_key, make_record
from ubt.plan import Planner
from ubt.report import write_report
from ubt.sources import CombinedSource
from ubt.tables import (
    apply_status_overrides,
    assign_weights,
    derive_confusable_groups,
    scan_tables,
)
from ubt.worker import init_worker, process_plan

MAX_ROUNDS = 40
STATUS_OVERRIDES_PATH = "configs/currency_overrides.yaml"


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_status_overrides(cfg: dict) -> dict:
    path = cfg.get("status_overrides_path", STATUS_OVERRIDES_PATH)
    if not os.path.isabs(path):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path = os.path.join(repo_root, path)
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def build_inventory(cfg: dict, train: bool):
    """Returns (tables, groups, source) for the requested side."""
    extra_dirs = list(cfg.get("table_dirs", {}).values())
    tables = scan_tables(extra_dirs)
    source = CombinedSource(cfg["data_root"], train=train, head_langs=cfg["head_langs"])
    langs = sorted({t.base_lang for t in tables})
    sourced = source.sourced_langs(langs)
    tables = assign_weights(
        tables, sourced, cfg["head_langs"], cfg["head_share"],
        cfg["holdout_languages"], cfg["inter_mixed"]["anchor_langs"],
    )
    tables = apply_status_overrides(
        tables, load_status_overrides(cfg),
        status_factors=cfg.get("status_factors"),
        alias_excluded=(cfg.get("empirical_alias_dedup") or {}).get("exclude"),
        alpha=cfg.get("alpha", 1.0),
    )
    groups = derive_confusable_groups(tables, cfg["script_groups"])
    return tables, groups, source


class GenerationRun:
    """Shared machinery: worker pool, dedup, accounting, record assembly."""

    def __init__(self, cfg: dict, workers: int, seed: int, train: bool, id_prefix: str):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.tables, self.groups, self.source = build_inventory(cfg, train)
        self.planner = Planner(cfg, self.tables, self.groups, self.source, self.rng)
        self.records: list[dict] = []
        self.rejections: Counter = Counter()
        self.per_table_accepted: Counter = Counter()
        self.per_table_singles: Counter = Counter()  # coverage-floor metric
        self.per_table_rejected: dict[str, Counter] = defaultdict(Counter)
        self.status_by_id = {t.table_id: t.status for t in self.tables}
        self.seen_keys: set = set()
        self.id_prefix = id_prefix
        self.shortfalls: list[dict] = []
        self._id_seq = 0
        extra_dirs = list(cfg.get("table_dirs", {}).values())
        ctx = mp.get_context("fork")
        self.pool = ctx.Pool(workers, initializer=init_worker, initargs=(extra_dirs,))

    def close(self) -> None:
        self.pool.close()
        self.pool.join()

    def _accept(self, result: dict, tag: str) -> bool:
        plan = result["plan"]
        if not result["ok"]:
            reason = result["reason"]
            self.rejections[reason] += 1
            self.per_table_rejected[result["fail_table"]][reason] += 1
            return False
        key = dedup_key(result["segments"])
        if key in self.seen_keys:
            self.rejections["duplicate"] += 1
            return False
        self.seen_keys.add(key)
        self._id_seq += 1
        rec = make_record(
            rec_id=f"{self.id_prefix}-{self._id_seq:08d}",
            doc_type=plan["doc_type"],
            regime=plan["regime"],
            switch_density=plan["switch_density"],
            segments=result["segments"],
            source=plan["source"],
            n_sentences=plan["n_sentences"],
            extra={"tag": tag},
        )
        rec["table_status"] = [self.status_by_id.get(t, "current")
                               for t in rec["tables"]]
        self.records.append(rec)
        for t in rec["tables"]:
            self.per_table_accepted[t] += 1
        if rec["doc_type"] == "single":
            self.per_table_singles[rec["tables"][0]] += 1
        return True

    def fill(self, target: int, plan_fn, tag: str, chunksize: int = 32) -> int:
        """Generate until `target` docs of this kind are accepted (or stall)."""
        accepted = 0
        for _ in range(MAX_ROUNDS):
            if accepted >= target:
                break
            deficit = target - accepted
            plans = []
            for _ in range(int(deficit * 1.25) + 8):
                p = plan_fn()
                if p is None:
                    self.rejections["no_source"] += 1
                else:
                    plans.append(p)
            if not plans:
                break
            progress = 0
            for result in self.pool.imap_unordered(process_plan, plans, chunksize):
                if accepted < target and self._accept(result, tag):
                    accepted += 1
                    progress += 1
            if progress == 0:
                break
        return accepted


def stratified_dev_split(records: list[dict], n_dev: int, rng: random.Random):
    """Carve a stratified mini-grid dev set out of the accepted records."""
    strata: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        strata[(r["doc_type"], r["regime"], r["len_bucket"])].append(i)
    dev_idx: set[int] = set()
    total = len(records)
    for key, idxs in sorted(strata.items()):
        take = max(1, round(n_dev * len(idxs) / total)) if idxs else 0
        take = min(take, len(idxs))
        dev_idx.update(rng.sample(idxs, take))
    while len(dev_idx) > n_dev:
        dev_idx.pop()
    dev = [records[i] for i in sorted(dev_idx)]
    train = [r for i, r in enumerate(records) if i not in dev_idx]
    return train, dev


def write_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="UBT - training set")
    ap.add_argument("--config", default="configs/generation.yaml")
    ap.add_argument("--n", type=int, default=360_000,
                    help="nominal total, including the skipped math share")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--out-dir", default="out/data_v2")
    ap.add_argument("--dev-size", type=int, default=2000)
    ap.add_argument("--tables-dir", default=None,
                    help="override the ko2020 extra tables dir")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config(args.config)
    if args.tables_dir:
        cfg["table_dirs"]["ko2020"] = args.tables_dir
    out_dir = os.path.join(cfg["data_root"], args.out_dir)

    mix = cfg["doc_mixture"]
    targets = {
        "single": round(args.n * mix["single"]),
        "intra_switch": round(args.n * mix["intra_switch"]),
        "inter_k2": round(args.n * mix["inter_k2"]),
        "inter_k3": round(args.n * mix["inter_k3"]),
        "inter_k4": round(args.n * mix["inter_k4"]),
    }
    math_gap = round(args.n * mix["math"]) if not cfg["math"]["enabled"] else 0

    run = GenerationRun(cfg, args.workers, args.seed, train=True, id_prefix="train")
    p = run.planner
    got = {}
    got["single"] = run.fill(targets["single"], p.plan_single, "single")
    print(f"[gen] single: {got['single']}/{targets['single']}", flush=True)
    got["intra_switch"] = run.fill(targets["intra_switch"], p.plan_intra_switch, "intra_switch")
    print(f"[gen] intra_switch: {got['intra_switch']}/{targets['intra_switch']}", flush=True)
    for k in (2, 3, 4):
        name = f"inter_k{k}"
        got[name] = run.fill(targets[name], lambda k=k: p.plan_inter(k), name)
        print(f"[gen] {name}: {got[name]}/{targets[name]}", flush=True)

    # Coverage-floor top-up: the floor counts single-code documents per table, so the top-up generates singles.
    floor = cfg["coverage_floor"]["min_docs_per_table"]
    shortfall_before, topup_total = {}, 0
    for t in sorted(p.weighted, key=lambda t: t.table_id):
        have = run.per_table_singles.get(t.table_id, 0)
        if have >= floor:
            continue
        shortfall_before[t.table_id] = floor - have
        added = run.fill(floor - have, lambda t=t: p.plan_single(table=t), "floor_topup")
        topup_total += added
    floor_shortfall_after = {
        t.table_id: floor - run.per_table_singles.get(t.table_id, 0)
        for t in p.weighted if run.per_table_singles.get(t.table_id, 0) < floor
    }
    print(f"[gen] floor top-up: +{topup_total}; remaining shortfalls: "
          f"{len(floor_shortfall_after)}", flush=True)

    train_recs, dev_recs = stratified_dev_split(run.records, args.dev_size, run.rng)

    # Zeroshot: holdout-language tables, never in train
    run.id_prefix = "zeroshot"
    zs_run_records_start = len(run.records)
    for t in sorted(p.holdout_tables, key=lambda t: t.table_id):
        run.fill(200, lambda t=t: p.plan_single(table=t), "zeroshot")
    zeroshot_recs = run.records[zs_run_records_start:]
    run.close()

    write_jsonl(os.path.join(out_dir, "train.jsonl"), train_recs)
    write_jsonl(os.path.join(out_dir, "dev.jsonl"), dev_recs)
    write_jsonl(os.path.join(out_dir, "zeroshot.jsonl"), zeroshot_recs)

    singles = [r for r in train_recs + dev_recs if r["doc_type"] == "single"]
    head = set(cfg["head_langs"])
    head_share = (sum(1 for r in singles if r["langs"][0] in head) / len(singles)
                  if singles else 0.0)
    seg_status = Counter(s for r in train_recs + dev_recs for s in r["table_status"])
    doc_status = Counter(
        "+".join(sorted(set(r["table_status"]))) for r in train_recs + dev_recs
    )
    floor_ok = sum(
        1 for t in p.weighted if run.per_table_singles.get(t.table_id, 0) >= floor
    )
    counts = {
        "targets": targets, "accepted": got,
        "math_gap_deferred_phase_b": math_gap,
        "floor_metric": "singles_per_table",
        "floor_tables_at_or_above_300": floor_ok,
        "floor_weighted_tables_total": len(p.weighted),
        "floor_topup_added": topup_total,
        "floor_shortfall_after": floor_shortfall_after,
        "status_composition_segments": dict(seg_status.most_common()),
        "status_composition_docs": dict(doc_status.most_common()),
        "status_overrides_provisional": load_status_overrides(cfg).get("provisional", False),
        "train": len(train_recs), "dev": len(dev_recs),
        "zeroshot": len(zeroshot_recs),
        "distinct_tables_accepted": len(run.per_table_accepted),
        "head_share_of_singles": round(head_share, 4),
        "wall_clock_sec": round(time.time() - t0, 1),
        "workers": args.workers,
        "seed": args.seed,
    }
    write_report(out_dir, cfg, counts, run.rejections,
                 run.per_table_accepted, run.per_table_rejected,
                 extra={"shortfall_before_topup": shortfall_before})
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
