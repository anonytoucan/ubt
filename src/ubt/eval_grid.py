"""Evaluation grid from eval sources only (FLORES-200 devtest, wiki_txt_B); shortfalls are logged with a reason.

Usage:
    python -m ubt.eval_grid --workers 48 [--config configs/generation.yaml]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from collections import Counter

import yaml

from ubt.generate import GenerationRun, write_jsonl
from ubt.noise import flip_dots
from ubt.report import write_report

SNIPPET_BUCKETS = ("snippet_5_10", "snippet_11_20", "snippet_21_40",
                   "snippet_41_80", "sentence")
_BUCKET_WORDS = {"snippet_5_10": (1, 2), "snippet_11_20": (2, 4),
                 "snippet_21_40": (3, 8), "snippet_41_80": (6, 16)}
NOISE_LEVELS = (0.005, 0.01, 0.02)


def pick_snippet_groups(groups: dict[str, list[str]], n: int,
                        rng: random.Random) -> dict[str, list[str]]:
    """Every script group, then the largest language groups until there are n."""
    script = {g: m for g, m in groups.items() if g.startswith("script:")}
    lang = sorted(
        ((g, m) for g, m in groups.items() if g.startswith("lang:")),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )
    out = dict(script)
    for g, m in lang:
        if len(out) >= n:
            break
        out[g] = m
    return out


def gen_core_per_code(run: GenerationRun, head_n: int = 45,
                      head_docs: int = 200, tail_docs: int = 100) -> None:
    ranked = sorted(run.planner.weighted, key=lambda t: -t.weight)
    for i, t in enumerate(ranked):
        target = head_docs if i < head_n else tail_docs
        got = run.fill(target, lambda t=t: run.planner.plan_single(table=t, n_sents=1),
                       "core_per_code")
        if got < target:
            run.shortfalls.append({"set": "core_per_code", "cell": t.table_id,
                                   "got": got, "target": target,
                                   "reason": "source_exhaustion_or_charset"})


def gen_snippets(run: GenerationRun, groups: dict[str, list[str]],
                 per_cell: int = 300) -> None:
    p, rng = run.planner, run.rng
    for gid, members in sorted(groups.items()):
        tabs = [p.mixer.by_id[m] for m in members if m in p.mixer.by_id]
        if not tabs:
            continue
        need = {b: per_cell for b in SNIPPET_BUCKETS}
        for _ in range(30):
            if not any(need.values()):
                break
            plans = []
            for bucket, deficit in need.items():
                for _ in range(int(deficit * 1.4) + 4):
                    t = rng.choice(tabs)
                    if bucket == "sentence":
                        plan = p.plan_single(table=t, n_sents=1)
                    else:
                        plan = p.plan_snippet(t, max_words=rng.randint(*_BUCKET_WORDS[bucket]))
                    if plan is None:
                        run.rejections["no_source"] += 1
                    else:
                        plans.append(plan)
            if not plans:
                break
            progress = 0
            for result in run.pool.imap_unordered(run_process, plans, 32):
                if not result["ok"]:
                    run.rejections[result["reason"]] += 1
                    run.per_table_rejected[result["fail_table"]][result["reason"]] += 1
                    continue
                before = len(run.records)
                if run._accept(result, f"snippet:{gid}"):
                    rec = run.records[-1]
                    b = rec["len_bucket"]
                    b = "sentence" if b in ("sentence", "paragraph") else b
                    if need.get(b, 0) > 0:
                        need[b] -= 1
                        rec["tag"] = f"snippet:{gid}:{b}"
                        progress += 1
                    else:  # bucket already full: drop the record
                        run.records.pop()
                        for tb in rec["tables"]:
                            run.per_table_accepted[tb] -= 1
                        run.seen_keys.discard(tuple((s["table"], s["text"]) for s in
                                                    (rec["segments"] or [{"table": rec["tables"][0], "text": rec["text"]}])))
                        assert len(run.records) == before
            if progress == 0:
                break
        for b, deficit in need.items():
            if deficit > 0:
                run.shortfalls.append({"set": "snippet", "cell": f"{gid}:{b}",
                                       "got": per_cell - deficit, "target": per_cell,
                                       "reason": "source_exhaustion_or_bucket_miss"})


def run_process(plan: dict) -> dict:
    from ubt.worker import process_plan  # noqa: PLC0415
    return process_plan(plan)


def gen_mixed_grid(run: GenerationRun, per_cell: int = 400) -> None:
    for k in (2, 3, 4):
        for regime in ("anchor", "uniform", "confusable"):
            got = run.fill(per_cell,
                           lambda k=k, r=regime: run.planner.plan_inter(k, regime=r),
                           f"mixed_grid:k{k}:{regime}")
            if got < per_cell:
                run.shortfalls.append({"set": "mixed_grid", "cell": f"k{k}:{regime}",
                                       "got": got, "target": per_cell,
                                       "reason": "source_exhaustion"})


def gen_k2_paragraph(run: GenerationRun, per_cell: int = 400) -> None:
    # k=2 documents with paragraph-length segments (4-8 sentences)
    for regime in ("anchor", "uniform"):
        got = run.fill(per_cell,
                       lambda r=regime: run.planner.plan_inter(2, regime=r, seg_sents=(4, 8)),
                       f"k2_paragraph:{regime}")
        if got < per_cell:
            run.shortfalls.append({"set": "k2_paragraph", "cell": regime,
                                   "got": got, "target": per_cell,
                                   "reason": "source_exhaustion"})


def gen_intra_switch_eval(run: GenerationRun, per_cell: int = 500) -> None:
    for density in ("light", "heavy"):
        got = run.fill(per_cell,
                       lambda d=density: run.planner.plan_intra_switch(density=d),
                       f"intra_switch_eval:{density}")
        if got < per_cell:
            run.shortfalls.append({"set": "intra_switch_eval", "cell": density,
                                   "got": got, "target": per_cell,
                                   "reason": "source_exhaustion"})


def gen_ko_discrimination(run: GenerationRun, n_pairs: int = 500) -> None:
    """The same Latin-embedded Korean sentences under ko-2020-g2 and ko-2006-g2, paired by pair_id."""
    p, rng = run.planner, run.rng
    ko2020 = p.by_id.get("ko-2020-g2.ctb")
    ko2006 = p.by_id.get("ko-2006-g2.ctb")
    if not (ko2020 and ko2006):
        run.shortfalls.append({"set": "ko_discrimination", "cell": "all",
                               "got": 0, "target": n_pairs * 2,
                               "reason": "ko tables missing"})
        return
    done = 0
    for _ in range(30):
        if done >= n_pairs:
            break
        flat_plans = []
        for pair_no in range(int((n_pairs - done) * 1.3) + 4):
            base = p.plan_intra_switch(table=ko2020)
            if base is None:
                run.rejections["no_source"] += 1
                continue
            shared_text = base["segments"][0]["text"]
            for t in (ko2020, ko2006):
                plan = copy.deepcopy(base)
                # rebuild the segment: retabling alone would keep ko-2020's siblings in the twin's confusable_set
                plan["segments"][0] = p._segment(t, shared_text)
                flat_plans.append(plan)
        if not flat_plans:
            break
        results = run.pool.map(run_process, flat_plans, 16)
        progress = 0
        for i in range(0, len(results) - 1, 2):
            pair = results[i : i + 2]
            if done >= n_pairs:
                break
            if all(r["ok"] for r in pair):
                accepted = [run._accept(r, "ko_discrimination") for r in pair]
                if all(accepted):
                    pid = f"kodisc-{done:05d}"
                    run.records[-1]["pair_id"] = pid
                    run.records[-2]["pair_id"] = pid
                    done += 1
                    progress += 1
                elif any(accepted):  # half-accepted pair: drop the orphan
                    orphan = run.records.pop()
                    for tb in orphan["tables"]:
                        run.per_table_accepted[tb] -= 1
            else:
                for r in pair:
                    if not r["ok"]:
                        run.rejections[r["reason"]] += 1
                        run.per_table_rejected[r["fail_table"]][r["reason"]] += 1
        if progress == 0:
            break
    if done < n_pairs:
        run.shortfalls.append({"set": "ko_discrimination", "cell": "pairs",
                               "got": done, "target": n_pairs,
                               "reason": "rejections"})


def gen_zeroshot_eval(run: GenerationRun, per_table: int = 200,
                      n_mixed: int = 500) -> None:
    p = run.planner
    for t in sorted(p.holdout_tables, key=lambda t: t.table_id):
        got = run.fill(per_table, lambda t=t: p.plan_single(table=t),
                       f"zeroshot_eval:{t.table_id}")
        if got < per_table:
            run.shortfalls.append({"set": "zeroshot_eval", "cell": t.table_id,
                                   "got": got, "target": per_table,
                                   "reason": "source_exhaustion_or_charset"})

    def plan_zeroshot_mixed() -> dict | None:
        t_hold = run.rng.choice(p.holdout_tables)
        others = p.mixer.choose_tables("anchor", 1, run.rng)
        segs = []
        for t in (t_hold, others[0]):
            got = p._sample_text(t.base_lang, run.rng.randint(1, 2))
            if got is None:
                return None
            text, src_name, _ = got
            seg = p._segment(t, text)
            seg["_src"] = src_name
            segs.append(seg)
        return {"doc_type": "inter", "regime": "anchor", "switch_density": "none",
                "n_sentences": 2, "source": "+".join(sorted({s["_src"] for s in segs}))
                if all("_src" in s for s in segs) else src_name,
                "segments": [{k: v for k, v in s.items() if k != "_src"}
                             for s in segs]}

    got = run.fill(n_mixed, plan_zeroshot_mixed, "zeroshot_eval:mixed")
    if got < n_mixed:
        run.shortfalls.append({"set": "zeroshot_eval", "cell": "mixed",
                               "got": got, "target": n_mixed,
                               "reason": "source_exhaustion"})


def gen_noise_variants(run: GenerationRun, n_base: int = 2000) -> list[dict]:
    core = [r for r in run.records if r["tag"] == "core_per_code"]
    base = run.rng.sample(core, min(n_base, len(core)))
    if len(base) < n_base:
        run.shortfalls.append({"set": "noise_variants", "cell": "base",
                               "got": len(base), "target": n_base,
                               "reason": "not enough core docs"})
    out = []
    for p_flip in NOISE_LEVELS:
        for r in base:
            noisy = dict(r)
            noisy["id"] = f"{r['id']}-noise{p_flip}"
            noisy["braille"] = flip_dots(r["braille"], p_flip, run.rng)
            noisy["noise_p"] = p_flip
            noisy["noise_base_id"] = r["id"]
            noisy["tag"] = "noise_variants"
            out.append(noisy)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="UBT - eval grid")
    ap.add_argument("--config", default="configs/generation.yaml")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--out-dir", default="out/eval_grid")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale all cell targets (smoke runs)")
    ap.add_argument("--tables-dir", default=None,
                    help="override the ko2020 extra tables dir")
    args = ap.parse_args()

    t0 = time.time()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    # eval intra-switch cells must hit the requested density exactly
    cfg_eval = copy.deepcopy(cfg)
    cfg_eval["intra_switch"]["natural_first"] = False
    if args.tables_dir:
        cfg_eval["table_dirs"]["ko2020"] = args.tables_dir
    out_dir = os.path.join(cfg["data_root"], args.out_dir)
    s = args.scale

    run = GenerationRun(cfg_eval, args.workers, args.seed, train=False,
                        id_prefix="eval")

    snippet_groups = pick_snippet_groups(run.groups, 20, run.rng)
    gen_core_per_code(run, head_docs=max(1, int(200 * s)), tail_docs=max(1, int(100 * s)))
    print(f"[eval] core_per_code done: {len(run.records)}", flush=True)
    gen_snippets(run, snippet_groups, per_cell=max(1, int(300 * s)))
    print(f"[eval] snippets done: {len(run.records)}", flush=True)
    gen_mixed_grid(run, per_cell=max(1, int(400 * s)))
    gen_k2_paragraph(run, per_cell=max(1, int(400 * s)))
    gen_intra_switch_eval(run, per_cell=max(1, int(500 * s)))
    print(f"[eval] mixed/k2p/intra done: {len(run.records)}", flush=True)
    gen_ko_discrimination(run, n_pairs=max(1, int(500 * s)))
    gen_zeroshot_eval(run, per_table=max(1, int(200 * s)), n_mixed=max(1, int(500 * s)))
    print(f"[eval] kodisc/zeroshot done: {len(run.records)}", flush=True)
    noise_recs = gen_noise_variants(run, n_base=max(1, int(2000 * s)))
    run.close()

    records = run.records + noise_recs
    write_jsonl(os.path.join(out_dir, "eval_grid.jsonl"), records)

    by_set = Counter(r["tag"].split(":")[0] for r in records)
    counts = {
        "total": len(records),
        "by_set": dict(by_set.most_common()),
        "snippet_groups_used": sorted(snippet_groups),
        "shortfalls": run.shortfalls,
        "wall_clock_sec": round(time.time() - t0, 1),
        "workers": args.workers,
        "seed": args.seed,
    }
    write_report(out_dir, cfg, counts, run.rejections,
                 run.per_table_accepted, run.per_table_rejected)
    print(json.dumps({k: v for k, v in counts.items() if k != "shortfalls"},
                     indent=2), flush=True)
    print(f"shortfall cells: {len(run.shortfalls)}", flush=True)


if __name__ == "__main__":
    main()
