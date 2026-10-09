#!/usr/bin/env python3
"""data-u1 builder: train, dev, eval and manifest.

    python scripts/data/build_unified.py --out datasets/data_u1 \
        --scale 1.0 --workers 48 --seed 20260923

--out and --scale are required. Above SMOKE_MAX_SCALE a build needs a verified ko-2024 certificate and, when a
Korean-math target is configured, ubt.kmath; --override-gates skips this but the build can then never be frozen.
out_default accepts only --scale 1.0.

Steps (train is disjoint from eval and dev by construction):
  1. engine: pin liblouis, stage and certify the ko-2024 table, build the strict shadow tables;
  2. sources and the eval hash registry (FLORES devtest, wiki_B, sealed NIKL test, rule examples);
  3. table inventory and engine fingerprint;
  4. reserve S1b pools, carve dev-only source blocks, probe S1 source capacity;
  5. eval grid;
  6. dev strata from the dev pools;
  7. train strata; a row hitting the eval or dev registry, or repeating a (tables, text) key, is rejected;
  8. files, registry.npz (sealed NIKL test only as digests) and manifest.json.
Verify with scripts/data/verify_unified.py. Smoke: --scale 0.02.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

import yaml  # noqa: E402

from ubt.eval_grid import pick_snippet_groups  # noqa: E402
from ubt.math_norm import nu_math  # noqa: E402
from ubt.plan import Planner  # noqa: E402
from ubt.sources import load_eval_exclusions  # noqa: E402
from ubt.u1 import engine as EN  # noqa: E402
from ubt.u1 import evalsets as EV  # noqa: E402
from ubt.u1 import strata as ST  # noqa: E402
from ubt.u1.inventory import build_inventory, probe_capacity  # noqa: E402
from ubt.u1.nemeth import paper_formula_set  # noqa: E402
from ubt.u1.nikl import load_nikl, sample_dev  # noqa: E402
from ubt.u1.registry import (HashRegistry, build_line_registry, eval_source_files,  # noqa: E402
                             kmath_example_latex, rule_example_texts, save_registry)
from ubt.u1.nikl import NearDupScreen  # noqa: E402
from ubt.u1.provenance import provenance  # noqa: E402
from ubt.u1.run import U1Run  # noqa: E402
from ubt.u1.sources import SOURCE_NORMALISATION, U1Source  # noqa: E402

NON_MATH = ("S1", "S1b", "S2", "S3_k2", "S3_k3", "S3_k4", "S3p", "S5", "S6", "S7")
MATH = ("S4a", "S4b")
EVAL_FILES = ("core_per_code", "snippets", "mixed_grid", "k2_paragraph",
              "intra_switch_eval", "kodisc", "zeroshot_eval", "noise_variants", "math_eval")
SMOKE_MAX_SCALE = 0.02
POINTER_REL = "pointers/nikl_test_pointer.jsonl"   # not under eval/: different schema
TEXT_NORMAL_FORM = ("NFC, except bn/as/mni text: U+09DC/09DD/09DF kept precomposed "
                    "(NFC composition exclusions; bn/as/mni.tbl define only the precomposed "
                    "letters). A consumer that NFC-normalises such text must re-apply "
                    "ubt.u1.sources.normalise_for_lang before re-encoding.")


def sc(n: float, s: float) -> int:
    """Scaled count, at least 1; a target of 0 stays 0."""
    return 0 if n == 0 else max(1, round(n * s))


def apportion(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Largest-remainder split of `total` over keys by weight."""
    if total <= 0 or not weights:
        return {k: 0 for k in weights}
    z = sum(weights.values())
    raw = {k: total * w / z for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: -(raw[k] - out[k]))[: total - sum(out.values())]:
        out[k] += 1
    return out


def sha256_path(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: str, rows: list[dict]) -> dict:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"n_docs": len(rows), "bytes": os.path.getsize(path), "sha256": sha256_path(path)}




class Clock:
    def __init__(self):
        self.t0 = time.time()
        self.last = self.t0
        self.phases: dict[str, float] = {}

    def lap(self, name: str, msg: str = "") -> None:
        now = time.time()
        self.phases[name] = round(now - self.last, 1)
        self.last = now
        print(f"[u1 {now - self.t0:8.1f}s] {name}: {self.phases[name]}s {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="data-u1 builder")
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "unified_u1.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--eval-scale", type=float, default=None)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty --out")
    ap.add_argument("--override-gates", action="store_true",
                    help="build above smoke scale despite a failed gate (never freezable)")
    ap.add_argument("--alias-cache", default=None,
                    help="reuse inventory.alias_dedup from a previous build's inventory.json")
    args = ap.parse_args()

    clock = Clock()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.workers > int(cfg.get("max_workers", 48)) or args.workers < 1:
        sys.exit(f"--workers must be 1..{cfg.get('max_workers', 48)}")
    s = args.scale
    es = args.eval_scale if args.eval_scale is not None else s
    if not (0 < s <= 1.0 and 0 < es <= 1.0):
        sys.exit("--scale / --eval-scale must be in (0, 1]")
    out = os.path.abspath(args.out)
    reserved = os.path.abspath(cfg["out_default"])
    if out == reserved and (s != 1.0 or es != 1.0):
        sys.exit(f"{reserved} is reserved for the full build (--scale 1.0); "
                 f"write smoke/partial builds elsewhere")
    above_smoke = s > SMOKE_MAX_SCALE or es > SMOKE_MAX_SCALE
    seed = args.seed

    # ---------------------------------------------------------------- 1. engine
    spec = EN.spec_from_config(cfg, out)
    # gates are checked before anything is written; kmath only when a Korean-math target is configured
    kmath_status = EN.kmath_status()
    kmath_required = bool(cfg["train_targets"].get("S4b", 0) or cfg["eval"].get("math_kmath", 0))
    if above_smoke:
        ko_decl = spec.ko2024_certified
        cert_path = os.path.join(spec.ko2024_src_dir, EN.CERT_NAME)
        problems = []
        if not ko_decl:
            problems.append("ko-2024 table not certified (KO2024_CERTIFIED unset)")
        elif not os.path.isfile(cert_path):
            problems.append(f"no certificate {cert_path}")
        if kmath_required and not kmath_status["available"]:
            problems.append(f"ubt.kmath unavailable ({kmath_status.get('error')})")
        if problems and not args.override_gates:
            sys.exit("refusing to build above smoke scale "
                     f"(--scale {s}, --eval-scale {es} > {SMOKE_MAX_SCALE}): "
                     + "; ".join(problems) + "  [gates; --override-gates to force]")
    if os.path.isdir(out) and os.listdir(out) and not args.force:
        sys.exit(f"{out} is not empty (use --force)")
    os.makedirs(out, exist_ok=True)
    for sub in ("eval", "engine", "pointers"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    for f in ("train.jsonl", "dev.jsonl", "manifest.json", "inventory.json", "verify_report.json",
              "registry.npz", os.path.join("eval", "nikl_test_pointer.jsonl")):
        if os.path.exists(os.path.join(out, f)):
            os.unlink(os.path.join(out, f))
    staging = EN.stage_ko2024(spec)
    cert = EN.check_certificate(spec, staging)       # raises if declared but invalid
    spec.ko2024_certified = cert["certified"]
    shadow = EN.build_strict_shadow(spec)
    EN.apply_env(spec)
    EN.preload_kmath()                                # before the fork: workers inherit it
    kmath_fp = EN.kmath_fingerprint()                 # frozen for the whole build
    clock.lap("engine_stage", f"ko2024 <- {spec.ko2024_src_dir}/{spec.ko2024_src_file} "
                              f"(certified={spec.ko2024_certified}) strict shadow: "
                              f"{len(shadow['files_changed'])} files with `undefined`; "
                              f"kmath={kmath_status}")

    # ---------------------------------------------------------------- 2. sources + registry
    data_root = cfg["data_root"]
    data_dir = os.path.join(data_root, "data")
    train_src = U1Source(data_root, True, cfg["head_langs"], cfg.get("extra_wiki_first"))
    eval_src = U1Source(data_root, False, cfg["head_langs"], cfg.get("extra_wiki_first"))
    nikl_rows = load_nikl(REPO)
    reg = HashRegistry()
    excl = load_eval_exclusions(data_dir)
    for path, name in eval_source_files(data_dir):
        reg.add_lines_file(path, name, excl)
    nikl_test_texts = {r["src"] for r in nikl_rows if r["split"] == "test"}
    for t in nikl_test_texts:
        reg.add_text(t, "nikl_test_sealed", sealed=True)
    for t in rule_example_texts(REPO):
        reg.add_text(t, "rule_examples")
    clock.lap("registry", f"{len(reg)} hashes {reg.sources}")

    # ---------------------------------------------------------------- 3. pool + inventory
    from ubt.u1.inventory import scan_u1_tables  # noqa: PLC0415
    langs = sorted({t.base_lang for t in scan_u1_tables(spec)[0]})
    train_src.sourced_langs(langs)
    eval_src.sourced_langs(langs)       # load everything before the fork
    gc.collect()
    gc.freeze()
    pool = mp.get_context("fork").Pool(args.workers, initializer=EN.init_worker,
                                       initargs=(spec.to_dict(),))
    alias_result = None
    if args.alias_cache:
        with open(args.alias_cache) as fh:
            alias_result = json.load(fh)["alias_dedup"]
    inv = build_inventory(cfg, spec, train_src, REPO, pool=pool, alias_result=alias_result)
    used = [t.table_id for t in inv.tables if t.weight > 0 or t.holdout]
    shadow.update(EN.check_strict_shadow(spec, used))
    engine_id, engine_map = EN.fingerprint(spec, used, extra={"kmath": kmath_fp})
    clock.lap("inventory", f"weighted={len(inv.weighted)} holdout={len(inv.holdout)} "
                           f"alias_exclude={inv.alias['exclude']} engine={engine_id} "
                           f"undefined-rule tables used={len(shadow['used_tables_with_undefined_rule'])}")

    # floors, S1b tables (config ids -> inventory ids; unknown id = hard error)
    fl = cfg["floor"]
    failing = {}
    for code in fl["failing_codes"]:
        tid = inv.resolve_code(code, skip_unusable=True)
        if tid is not None:
            failing[tid] = code
    floors = {t.table_id: sc(fl["failing_min"] if t.table_id in failing else fl["other_min"], s)
              for t in inv.weighted}
    s1b_tables = {lang: [x for x in (inv.resolve_code(t, skip_unusable=True) for t in lst)
                         if x is not None]
                  for lang, lst in cfg["s1b"]["tables"].items()}

    # ---------------------------------------------------------------- targets
    tt = cfg["train_targets"]
    targets = {k: sc(v, s) for k, v in tt.items()}
    dev_q = apportion(sc(cfg["dev"]["n"], s), {k: tt[k] for k in NON_MATH})
    dev_q.update(apportion(sc(cfg["dev"]["math"], s), {k: tt[k] for k in MATH}))

    # ---------------------------------------------------------------- 4. S1b pools, dev carve
    s1b_all = [t for lang in sorted(s1b_tables) for t in s1b_tables[lang]]
    per_table_train = apportion(targets["S1b"], {t: 1.0 for t in s1b_all})
    per_table_dev = apportion(dev_q["S1b"], {t: 1.0 for t in s1b_all})
    pools_train, pools_dev = {}, {}
    for i, lang in enumerate(sorted(s1b_tables)):
        n_tr = sum(per_table_train[t] for t in s1b_tables[lang])
        n_dv = sum(per_table_dev[t] for t in s1b_tables[lang])
        over = cfg["s1b"]["reserve_overdraw"].get(lang, 1.5)
        want = int((n_tr + n_dv) * over) + 64
        lines = train_src.reserve_wiki_blocks(
            lang, want, random.Random(seed + 100 + i), block=cfg["s1b"].get("block", 64),
            max_frac=cfg["s1b"].get("reserve_max_frac", 0.45))
        # reserved lines are in a seeded random order: the dev share is a prefix
        k_dev = min(len(lines) // 4, int(n_dv * over) + 32) if n_dv else 0
        pools_dev[lang], pools_train[lang] = lines[:k_dev], lines[k_dev:]
    dcfg = cfg["dev"].get("source", {})
    dev_src, dev_carve = train_src.carve_dev(
        langs, seed + 13, frac=dcfg.get("frac", 0.03), min_lines=dcfg.get("min_lines", 40),
        max_lines=dcfg.get("max_lines", 4000), max_frac=dcfg.get("max_frac", 0.10),
        block=dcfg.get("block", 8))
    clock.lap("dev_carve", f"{len(dev_carve)} pools, "
                           f"{sum(v['dev_lines'] for v in dev_carve.values())} dev lines; "
                           f"S1b dev/train lines {({k: (len(pools_dev[k]), len(pools_train[k])) for k in pools_train})}")

    # S1 source capacity of small pools: which full-scale floors are reachable
    capacity = probe_capacity(pool, [t for t in inv.tables if t.base_lang not in cfg["head_langs"]],
                              train_src, max_lines=250000)
    tt_s1 = cfg["train_targets"]["S1"]
    full_floor = {tid: fl["failing_min"] if tid in failing else fl["other_min"] for tid in floors}
    wsum = sum(t.weight for t in inv.weighted)
    s1_weighted_full = max(0, tt_s1 - sum(full_floor.values()))
    cap_report = {}
    for tid, c in capacity["per_table"].items():
        exp_w = s1_weighted_full * inv.by_id[tid].weight / wsum
        cap_report[tid] = {**c, "floor_full_scale": full_floor.get(tid),
                           "expected_s1_full_scale": round(full_floor.get(tid, 0) + exp_w),
                           "floor_reachable_full_scale": c["capacity"] >= full_floor.get(tid, 0)}
    capacity["per_table"] = cap_report
    capacity["projected_floor_shortfall_full_scale"] = {
        tid: {"floor": v["floor_full_scale"], "capacity": v["capacity"]}
        for tid, v in sorted(cap_report.items()) if not v["floor_reachable_full_scale"]}
    capacity["projected_weighted_saturation_full_scale"] = sorted(
        tid for tid, v in cap_report.items()
        if v["floor_reachable_full_scale"] and v["capacity"] < v["expected_s1_full_scale"])
    clock.lap("capacity", f"{len(cap_report)} small-pool tables; projected full-scale floor "
                          f"shortfall: {sorted(capacity['projected_floor_shortfall_full_scale'])}")

    cfg_train = copy.deepcopy(cfg)
    cfg_eval = copy.deepcopy(cfg)
    cfg_eval["intra_switch"]["natural_first"] = False
    planner_eval = Planner(cfg_eval, inv.tables, inv.groups, eval_src, random.Random(seed + 1))
    planner_train = Planner(cfg_train, inv.tables, inv.groups, train_src, random.Random(seed + 2))
    planner_dev = Planner(copy.deepcopy(cfg), inv.tables, inv.groups, dev_src,
                          random.Random(seed + 14))
    run = U1Run(pool, engine_id, inv.status_by_id(), reg, random.Random(seed + 3))
    en_sibs = planner_train._segment(planner_train.by_id["en-ueb-g2.ctb"], "")["siblings"]
    ko_sibs = planner_train._segment(planner_train.by_id[EN.KO2024_LABEL], "")["siblings"]
    clock.lap("plan_setup", "")

    # ---------------------------------------------------------------- 5. EVAL first
    ecfg = cfg["eval"]
    ev: dict[str, list[dict]] = {}
    ev["core_per_code"] = EV.gen_core(run, planner_eval, ecfg, es)
    clock.lap("eval.core_per_code", str(len(ev["core_per_code"])))
    groups = pick_snippet_groups(inv.groups, ecfg["snippet_groups"], random.Random(seed + 4))
    ev["snippets"] = EV.gen_snippets(run, planner_eval, groups, sc(ecfg["snippet_per_cell"], es))
    clock.lap("eval.snippets", str(len(ev["snippets"])))
    ev["mixed_grid"] = EV.gen_mixed(run, planner_eval, sc(ecfg["mixed_per_cell"], es))
    ev["k2_paragraph"] = EV.gen_k2_paragraph(run, planner_eval, sc(ecfg["k2p_per_cell"], es))
    ev["intra_switch_eval"] = EV.gen_intra(run, planner_eval, sc(ecfg["intra_per_cell"], es))
    ev["kodisc"] = EV.gen_kodisc(run, planner_eval, sc(ecfg["kodisc_pairs"], es),
                                 tuple(inv.resolve_code(t) for t in ecfg["kodisc_tables"]))
    ev["zeroshot_eval"] = EV.gen_zeroshot(run, planner_eval, sc(ecfg["zeroshot_per_table"], es),
                                          sc(ecfg["zeroshot_mixed"], es))
    ev["noise_variants"] = EV.gen_noise(run, ev["core_per_code"], sc(ecfg["noise_base"], es),
                                        random.Random(seed + 5))
    clock.lap("eval.grid_rest", str({k: len(v) for k, v in ev.items()}))
    paper_canon = {nu_math(f) for f in paper_formula_set(0)}
    # the Korean math regulation examples certify ubt.kmath: never generated
    kmath_ex_canon = {nu_math(x) for x in kmath_example_latex(REPO)}
    nem_eval = ST.gen_nemeth(run, spec, sc(ecfg["math_nemeth"], es), form="Emath", split="eval",
                             rng=random.Random(seed + 6), exclude_canon=set(paper_canon),
                             include_paper=False, tag_prefix="math_eval",
                             carrier_siblings=en_sibs)
    km_eval = ST.gen_kmath(run, sc(ecfg["math_kmath"], es), form="Emath", split="eval",
                           rng=random.Random(seed + 7),
                           exclude_canon=set(paper_canon) | kmath_ex_canon, kmath_fp=kmath_fp,
                           tag_prefix="math_eval")
    ev["math_eval"] = nem_eval + km_eval
    eval_canon = {r["latex_canonical"] for r in ev["math_eval"]}
    pointer = EV.nikl_pointer_rows(nikl_rows)
    clock.lap("eval.math", f"nemeth={len(nem_eval)} kmath={len(km_eval)} "
                           f"nikl_pointer={len(pointer)}")

    # ---------------------------------------------------------------- 6./7. DEV then TRAIN
    screen = NearDupScreen(nikl_test_texts)
    rows: dict[str, dict[str, list[dict]]] = {"dev": {}, "train": {}}
    nikl_stats: dict[str, dict] = {}
    floor_report: dict = {}
    s1_parts: dict[str, tuple[list, list]] = {}
    used_canon = set(eval_canon)
    s7_dev_ids: set[str] = set()
    for split, planner, tgt, pools_s1b, per_s1b, rng_off in (
            ("dev", planner_dev, dev_q, pools_dev, per_table_dev, 200),
            ("train", planner_train, targets, pools_train, per_table_train, 0)):
        tr = rows[split]
        f_rows, w_rows, frep = ST.gen_s1(run, planner, tgt["S1"],
                                          floors if split == "train" else {}, split=split)
        s1_parts[split] = (f_rows, w_rows)
        tr["S1"] = f_rows + w_rows
        if split == "train":
            floor_report = frep
        clock.lap(f"{split}.S1", f"floor={len(f_rows)} weighted={len(w_rows)}")
        tr["S1b"] = ST.gen_s1b(run, planner, pools_s1b, s1b_tables, per_s1b, split=split)
        tr["S2"] = ST.gen_s2(run, planner, tgt["S2"], split=split)
        for k in (2, 3, 4):
            key = f"S3_k{k}"
            tr[key] = ST.gen_s3(run, planner, k, tgt[key], split=split)
        tr["S3p"] = ST.gen_s3(run, planner, 2, tgt["S3p"], paragraph=True, split=split)
        clock.lap(f"{split}.S1b_S2_S3", str({k: len(tr[k]) for k in
                                             ("S1b", "S2", "S3_k2", "S3_k3", "S3_k4", "S3p")}))
        tr["S4a"] = ST.gen_nemeth(run, spec, tgt["S4a"], form="S4a", split=split,
                                  rng=random.Random(seed + 8 + rng_off),
                                  exclude_canon=set(used_canon),
                                  include_paper=split == "train", carrier_siblings=en_sibs)
        used_canon |= {r["latex_canonical"] for r in tr["S4a"]}
        tr["S4b"] = ST.gen_kmath(run, tgt["S4b"], form="S4b", split=split,
                                 rng=random.Random(seed + 9 + rng_off),
                                 exclude_canon=set(used_canon) | kmath_ex_canon,
                                 kmath_fp=kmath_fp)
        used_canon |= {r["latex_canonical"] for r in tr["S4b"]}
        tr["S5"] = ST.gen_s5(run, planner, tgt["S5"], split=split)
        tr["S6"] = ST.gen_s6(run, planner, tgt["S6"], split=split)
        strata7, st7 = sample_dev(nikl_rows, tgt["S7"], random.Random(seed + 10 + rng_off),
                                  nikl_test_texts, screen=screen, exclude_ids=s7_dev_ids)
        tr["S7"], s7_report = ST.gen_s7(run, strata7, tgt["S7"], split=split, siblings=ko_sibs)
        st7.update(s7_report)
        nikl_stats[split] = st7
        if split == "dev":
            s7_dev_ids = {r["nikl_id"] for r in tr["S7"]}
        clock.lap(f"{split}.S4_S7", str({k: len(tr[k]) for k in ("S4a", "S4b", "S5", "S6", "S7")}))
    # line hashes of every source (both sides) for the persisted registry
    line_reg = build_line_registry(data_dir, nikl_rows, excl, pool=pool)
    prov = provenance(REPO, data_dir, pool=pool)
    pool.close()
    pool.join()

    # ---------------------------------------------------------------- 8. files
    STRATA = ("S1", "S1b", "S2", "S3_k2", "S3_k3", "S3_k4", "S3p", "S4a", "S4b", "S5", "S6", "S7")
    train_rows = [r for key in STRATA for r in rows["train"][key]]
    dev_rows = [r for key in STRATA for r in rows["dev"][key]]
    random.Random(seed + 12).shuffle(train_rows)
    per_stratum = {key: {"generated": len(rows["train"][key]) + len(rows["dev"][key]),
                         "train": len(rows["train"][key]), "dev": len(rows["dev"][key])}
                   for key in STRATA}

    files = {}
    files["train.jsonl"] = write_jsonl(os.path.join(out, "train.jsonl"), train_rows)
    files["dev.jsonl"] = write_jsonl(os.path.join(out, "dev.jsonl"), dev_rows)
    for name in EVAL_FILES:
        files[f"eval/{name}.jsonl"] = write_jsonl(os.path.join(out, "eval", f"{name}.jsonl"), ev[name])
    files[POINTER_REL] = write_jsonl(os.path.join(out, POINTER_REL), pointer)
    inv_json = inv.to_json(out_dir=out)
    with open(os.path.join(out, "inventory.json"), "w") as fh:
        json.dump(inv_json, fh, ensure_ascii=False, indent=1)
    files["inventory.json"] = {"sha256": sha256_path(os.path.join(out, "inventory.json"))}
    clock.lap("write", f"train={len(train_rows)} dev={len(dev_rows)}")

    # persisted registry: the eval units every train/dev row was screened against + non-sealed line hashes
    reg_info = save_registry(os.path.join(out, "registry.npz"), reg, line_reg)
    files["registry.npz"] = {"sha256": sha256_path(os.path.join(out, "registry.npz"))}
    del line_reg
    clock.lap("registry_persist", f"{reg_info['n_eval_units']} eval units, "
                                  f"line overlap train/eval = {reg_info['line_overlap_train_eval']}")

    # ---------------------------------------------------------------- manifest
    kmath_changed = EN.kmath_fingerprint() != kmath_fp
    if kmath_changed:
        print("[u1] WARNING: ubt.kmath changed during the build; S4b/math_eval-ko used the "
              "build-start state (see manifest kmath_changed_during_build)", flush=True)
    status = {}
    for key, v in per_stratum.items():
        tgt = targets[key]
        dv = dev_q.get(key, 0)
        st = "ok" if v["train"] >= tgt and v["dev"] >= dv else "shortfall"
        if key == "S4b" and v["generated"] == 0 and not kmath_fp.get("available"):
            st = "pending"
        if tgt == 0 and dv == 0 and v["generated"] == 0:
            st = "excluded"
        status[key] = {**v, "target_train": tgt, "target_dev": dv, "status": st}
    s1_train = s1_parts["train"][0] + s1_parts["train"][1]
    s1_train_by_table = Counter(r["tables"][0] for r in s1_train)
    floor_out = {tid: {"floor": floors[tid], "failing": tid in failing,
                       "train_rows": s1_train_by_table.get(tid, 0),
                       "met": s1_train_by_table.get(tid, 0) >= floors[tid]}
                 for tid in sorted(floors)}
    k_dist = Counter(r["k"] for r in train_rows)
    eval_targets = {
        "core_per_code": {"head_n": ecfg["core_head_n"], "head_docs": sc(ecfg["core_head_docs"], es),
                          "tail_docs": sc(ecfg["core_tail_docs"], es)},
        "snippets": {"groups": sorted(groups), "per_cell": sc(ecfg["snippet_per_cell"], es),
                     "buckets": 5},
        "mixed_grid": {"per_cell": sc(ecfg["mixed_per_cell"], es), "cells": "k2..4 x 3 regimes"},
        "k2_paragraph": {"per_cell": sc(ecfg["k2p_per_cell"], es), "cells": "anchor, uniform"},
        "intra_switch_eval": {"per_cell": sc(ecfg["intra_per_cell"], es)},
        "kodisc": {"pairs": sc(ecfg["kodisc_pairs"], es), "tables": ecfg["kodisc_tables"]},
        "zeroshot_eval": {"per_table": sc(ecfg["zeroshot_per_table"], es),
                          "mixed": sc(ecfg["zeroshot_mixed"], es)},
        "noise_variants": {"base": sc(ecfg["noise_base"], es), "levels": [0.005, 0.01, 0.02]},
        "math_eval": {"nemeth": sc(ecfg["math_nemeth"], es), "kmath": sc(ecfg["math_kmath"], es),
                      "embed_frac": 0.5},
    }
    gates = {
        "ko2024_certified": spec.ko2024_certified,
        "kmath_required": kmath_required,
        "kmath_available": bool(kmath_fp.get("available")),
        "kmath_frozen": not kmath_changed,
        "pending_strata": sorted(k for k, v in status.items() if v["status"] == "pending"),
        "full_scale": s == 1.0 and es == 1.0,
        "gates_overridden": bool(args.override_gates and above_smoke),
        "note": "Korean rows only from a certified ko-2024 table (C1+C2+C3, verified certificate); "
                "Korean math (S4b, math_eval-ko) is excluded — kmath is a gate only if a Korean-math "
                "target is configured; only a full-scale build without overrides can be frozen",
    }
    manifest = {
        "dataset": "data-u1",
        "smoke": not gates["full_scale"],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv, "seed": seed, "scale": s, "eval_scale": es, "workers": args.workers,
        "config": os.path.relpath(args.config, REPO), "config_sha256": sha256_path(args.config),
        "ubt_git_sha": (prov.get("git_head") or "unknown") + ("+dirty" if prov.get("code_dirty") else ""),
        "provenance": prov,
        "engine_id": engine_id, "engines": {engine_id: engine_map},
        "engine_spec": spec.to_dict(), "ko2024": staging, "ko2024_certificate": cert,
        "strict_shadow": shadow, "kmath_status": kmath_status,
        "downstream": {"UBT_TABLE_FALLBACK_DIRS": "<out>/engine/ko2024",
                       "note": "tools outside the builder (eval harness, bCER) resolve the "
                               "ko-2024-g2.ctb label via ubt.translate's "
                               "UBT_TABLE_FALLBACK_DIRS; UBT_LOUIS_TABLES/UBT_LOUIS_LIB pin "
                               "the liblouis build"},
        "text_normal_form": TEXT_NORMAL_FORM,
        "files": files,
        "train_targets": targets, "dev_quota": dev_q, "strata": status,
        "dev_source_carve": {"params": dcfg, "per_lang": dev_carve,
                             "s1b_dev_lines": {k: len(v) for k, v in pools_dev.items()}},
        "k_distribution_train": {str(k): v for k, v in sorted(k_dist.items())},
        "eval_targets": eval_targets,
        "eval_counts": {k: len(v) for k, v in ev.items()},
        "nikl": {"n_pairs": len(nikl_rows), "n_test_sealed": len(pointer),
                 "n_test_texts": len(nikl_test_texts), "dev": nikl_stats["dev"],
                 "train": nikl_stats["train"],
                 "s7_f_consistent": sum(1 for r in rows["train"]["S7"] + rows["dev"]["S7"]
                                        if r["f_consistent"]),
                 "s7_rows": len(rows["train"]["S7"]) + len(rows["dev"]["S7"]),
                 "test_seal": {
                     "rows": "test rows never written; pointers/nikl_test_pointer.jsonl ids only",
                     "near_dup_screen": {"key": "registry.loose", "shingle": 20, "frac": 0.8},
                     "registry": "test units screened in memory, persisted only as a digest",
                     "table_edits_informed_by_test": cert.get("test_informed_edits"),
                     "heldout_for_models_trained_on_u1":
                         None if cert.get("test_informed_edits") is None
                         else not cert["test_informed_edits"],
                     "note": "None = no ko-2024 certificate yet (unknown); True in "
                             "table_edits_informed_by_test means NIKL test must NOT be "
                             "reported as held-out for models trained on u1"}},
        "floor": {"failing_codes_source": fl["failing_codes_source"],
                  "failing_codes_config": fl["failing_codes"],
                  "failing_resolved": {v: k for k, v in failing.items()},
                  "unusable_codes_skipped": inv.unusable_codes_skipped,
                  "per_table": floor_out,
                  "unmet": sorted(t for t, v in floor_out.items() if not v["met"])},
        "floor_phase_report": floor_report,
        "s1_source_capacity": capacity,
        "s1b": {"tables": s1b_tables, "per_table_target": per_table_train,
                "per_table_target_dev": per_table_dev,
                "pool_lines": {k: len(v) for k, v in pools_train.items()},
                "per_table_generated": dict(Counter(r["tables"][0] for r in rows["train"]["S1b"]))},
        "registry": {"n_hashes": len(reg), "sources": reg.sources, "persisted": reg_info,
                     "dev_registry_units": len(run.dev_registry)},
        "source_normalisation": SOURCE_NORMALISATION,
        "freeze_gates": gates,
        "rejections": {k: dict(v.most_common()) for k, v in run.rejections.items()},
        "rejection_examples": {f: dict(v) for f, v in run.examples.items()
                               if f not in ("S7",) and v},
        "kmath_example_canon_excluded": len(kmath_ex_canon),
        "kmath_changed_during_build": kmath_changed,
        "rejected_tables_top": {t: dict(c) for t, c in sorted(
            run.rejected_tables.items(), key=lambda kv: -sum(kv[1].values()))[:40]},
        "shortfalls": run.shortfalls,
        "n_train": len(train_rows), "n_dev": len(dev_rows),
        "n_eval": sum(len(v) for v in ev.values()),
        "wall_clock_sec": round(time.time() - clock.t0, 1),
        "phase_sec": clock.phases,
    }
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"out": out, "engine": engine_id, "n_train": len(train_rows),
                      "n_dev": len(dev_rows), "n_eval": manifest["n_eval"],
                      "strata": {k: (v["train"], v["target_train"], v["dev"], v["target_dev"],
                                     v["status"]) for k, v in status.items()},
                      "n_shortfall_cells": len(run.shortfalls),
                      "freeze_gates": gates,
                      "wall_clock_sec": manifest["wall_clock_sec"]}, ensure_ascii=False, indent=1),
          flush=True)


if __name__ == "__main__":
    main()
