#!/usr/bin/env python3
"""data-u1 post-build verifier -> <data>/verify_report.json.

    python scripts/data/verify_unified.py --data datasets/data_u1 --workers 48

Checks (overall pass iff all pass; pending strata are reported, not failed):
  engine        the staged engine re-fingerprints to the manifest's engine id
  files         every file's sha256 / n_docs match the manifest
  schema        record fields, id u1-<form>-<seq>, split = file, engine id, cells/len_bucket recomputed, segments join
                back, no span lists its own table as confusable, table_status = inventory status
  tables        every label in the inventory (or a math pseudo-table); no ko-2020-*; no held-out table in train/dev
  f_consistency every synthetic row re-transcribed: braille == f(text); noise rows skipped, NIKL via f_consistent
  undefined     no span changes under noUndefined; braille is U+2800-U+28FF ('\\n' only between segments)
  dedup         (tables, text) unique across all files; noise variants exempt
  disjoint      train/dev ∩ eval = 0 under the registry rule (eval rows, FLORES devtest, wiki_B, sealed NIKL test,
                Korean rule examples); source-level FLORES dev/devtest and wiki_A/wiki_B disjoint minus eval_exclusions;
                no NIKL test id/text in train/dev; pointer rows = the test split
  registry      registry.npz = the recomputed eval unit set; train- and eval-side source line hashes do not meet
  noise         every noise row points at an existing core row of the same length
  counts        per stratum vs targets; eval cells and S1 floors recounted against the cell design, and a short cell
                without a matching manifest shortfall fails
  s1b_one_table every S1b source line has one table and no other train/dev stratum uses it, even as a sub-sentence
  kodisc_pairs  every pair_id = the same text under exactly the two kodisc tables
  cjk_joiner    no ASCII space between joined CJK sentences inside cmn/yue/ja spans
  dev_disjoint  train ∩ dev = 0 under the registry rule; no shared NIKL id or math canonical form
  nikl_near_dup no S7 row is a near-duplicate of a sealed NIKL test sentence (loose-exact or 20-char shingles >= 0.8)
  provenance    config and every input file re-hash to the manifest; code changes since the build are only reported
The strict (undefined-char) pass uses a shadow of the tables that the verifier builds itself, not the builder's copy.
freezable = verify pass, ko-2024 certified, kmath frozen if required, no pending stratum, full scale, no gate override.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

import tempfile  # noqa: E402

from ubt.eval_grid import SNIPPET_BUCKETS  # noqa: E402
from ubt.sources import load_eval_exclusions, sentence_hash  # noqa: E402
from ubt.u1 import engine as EN  # noqa: E402
from ubt.u1 import kmath_adapter as KM  # noqa: E402
from ubt.u1 import nemeth as NM  # noqa: E402
from ubt.u1.nikl import NearDupScreen, load_nikl  # noqa: E402
from ubt.u1.provenance import code_files, digest, hash_inputs  # noqa: E402
from ubt.u1.provenance import sha256_file as prov_sha  # noqa: E402
from ubt.u1.records import EVAL_FORMS, TRAIN_FORMS, dedup_key, schema_errors  # noqa: E402
from ubt.u1.registry import (HashRegistry, digest as reg_digest, eval_source_files,  # noqa: E402
                             h64, load_registry, row_units, rule_example_texts, sub_sentences,
                             train_source_files, keys as unit_keys)
from ubt.verify import is_braille_only  # noqa: E402

POINTER_REL = "pointers/nikl_test_pointer.jsonl"

CJK_SPACE = re.compile(r"[。！？」』）] (?=[㐀-鿿぀-ヿ])")
MAX_EXAMPLES = 20


class Check:
    def __init__(self, name: str):
        self.name = name
        self.fail = 0
        self.n = 0
        self.examples: list = []
        self.info: dict = {}

    def bad(self, what) -> None:
        self.fail += 1
        if len(self.examples) < MAX_EXAMPLES:
            self.examples.append(what)

    def to_json(self) -> dict:
        return {"pass": self.fail == 0, "n_checked": self.n, "n_fail": self.fail,
                "examples": self.examples, **self.info}


def load_rows(data: str) -> list[tuple[str, dict]]:
    rows = []
    for rel in ["train.jsonl", "dev.jsonl"] + sorted(
            f"eval/{f}" for f in os.listdir(os.path.join(data, "eval"))
            if f.endswith(".jsonl")):
        with open(os.path.join(data, rel), encoding="utf-8") as fh:
            for line in fh:
                rows.append((rel, json.loads(line)))
    return rows


def spans_for_f(rec: dict) -> list[dict]:
    """The spans f is applied to: [{table, text, braille, latex_canonical?}]."""
    if rec.get("segments"):
        return rec["segments"]
    table = rec["tables"][-1] if rec["doc_type"] == "math" else rec["tables"][0]
    return [{"table": table, "text": rec["text"], "braille": rec["braille"],
             "latex_canonical": rec.get("latex_canonical")}]


def sha256_path(path: str) -> str:
    import hashlib  # noqa: PLC0415
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    if not 1 <= args.workers <= 200:
        sys.exit("--workers must be 1..200")
    t0 = time.time()
    data = os.path.abspath(args.data)
    with open(os.path.join(data, "manifest.json")) as fh:
        man = json.load(fh)
    with open(os.path.join(data, "inventory.json")) as fh:
        inv = json.load(fh)
    spec = EN.EngineSpec.from_dict(man["engine_spec"])
    # the staged engine travels with the data dir; the strict shadow goes to a temp dir
    spec.stage_dir = os.path.join(data, "engine", "ko2024")
    spec.cwd_dir = os.path.join(data, "engine", "cwd")
    shadow_tmp = tempfile.mkdtemp(prefix="u1verify-strict-")
    spec.strict_dir = shadow_tmp
    EN.build_strict_shadow(spec)
    EN.apply_env(spec)
    checks = {n: Check(n) for n in ("engine", "files", "provenance", "schema", "tables",
                                    "f_consistency", "undefined", "dedup", "disjoint",
                                    "dev_disjoint", "nikl_near_dup", "registry", "noise",
                                    "counts", "s1b_one_table", "kodisc_pairs", "cjk_joiner")}

    # ------------------------------------------------------------ engine + files
    tinfo = {t["table_id"]: t for t in inv["tables"]}
    used = [t for t, v in tinfo.items() if v["weight"] > 0 or v["holdout"]]
    eng = checks["engine"]
    eng.n = 1
    try:
        eng.info["strict_shadow"] = EN.check_strict_shadow(spec, used)
    except RuntimeError as e:
        eng.bad(f"strict shadow: {e}")
    eid, _emap = EN.fingerprint(spec, used, extra={"kmath": EN.kmath_fingerprint()})
    eng.info.update({"manifest_engine": man["engine_id"], "recomputed": eid})
    if eid != man["engine_id"]:
        eng.bad("engine fingerprint changed since the build")
    staging_man = man.get("ko2024") or {}
    cert_copy = os.path.join(data, "engine", EN.CERT_NAME)
    decl = bool((man.get("ko2024_certificate") or {}).get("declared"))
    spec_decl = EN.EngineSpec.from_dict({**man["engine_spec"], "ko2024_certified": decl})
    try:
        cert = EN.check_certificate(spec_decl, staging_man, cert_path=cert_copy)
        eng.info["ko2024_certified_recomputed"] = cert["certified"]
        if cert["certified"] != bool(man["engine_spec"].get("ko2024_certified")):
            eng.bad("ko-2024 certification state differs from the manifest")
        for name, v in (staging_man.get("staged_files") or {}).items():
            pth = os.path.join(spec.stage_dir, name)
            if not os.path.isfile(pth) or EN.sha256_file(pth) != v["sha256"]:
                eng.bad(f"staged ko2024 file changed: {name}")
    except RuntimeError as e:
        eng.bad(f"ko-2024 certificate: {e}")
    for rel, meta in man["files"].items():
        checks["files"].n += 1
        p = os.path.join(data, rel)
        if not os.path.isfile(p) or sha256_path(p) != meta["sha256"]:
            checks["files"].bad(rel)

    # ------------------------------------------------------------ provenance
    pv = checks["provenance"]
    cfg_path = os.path.join(REPO, man["config"])
    pv.n += 1
    if sha256_path(cfg_path) != man["config_sha256"]:
        pv.bad({"why": "config changed since the build", "config": man["config"]})
    import yaml  # noqa: PLC0415
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    data_dir = os.path.join(cfg["data_root"], "data")
    mprov = man.get("provenance") or {}
    if not mprov.get("inputs"):
        pv.bad({"why": "manifest has no provenance.inputs"})
    else:
        now = hash_inputs(REPO, data_dir)
        pv.n += len(now)
        for k in sorted(set(now) | set(mprov["inputs"])):
            if now.get(k) != mprov["inputs"].get(k):
                pv.bad({"input": k, "why": "missing" if k not in now else
                        "new" if k not in mprov["inputs"] else "sha256 changed"})
        cf = {p: prov_sha(os.path.join(REPO, p)) for p in code_files(REPO)}
        pv.info["code_changed_since_build"] = digest(cf) != mprov.get("code_sha256")
        pv.info["code_files_changed"] = sorted(
            k for k in set(cf) | set(mprov.get("code_files", {}))
            if cf.get(k) != mprov.get("code_files", {}).get(k))[:40]
        pv.info["git_head_at_build"] = mprov.get("git_head")
        pv.info["code_dirty_at_build"] = mprov.get("code_dirty")

    rows = load_rows(data)
    print(f"[verify] {len(rows)} rows loaded ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------ schema / tables / dedup
    ids = Counter(r["id"] for _f, r in rows)
    dup_ids = [i for i, c in ids.items() if c > 1]
    checks["schema"].n = len(rows)
    for i in dup_ids[:MAX_EXAMPLES]:
        checks["schema"].bad(f"duplicate id {i}")
    checks["schema"].fail += max(0, len(dup_ids) - MAX_EXAMPLES)
    seen_keys: dict[bytes, str] = {}
    for rel, r in rows:
        errs = schema_errors(r)
        want_split = "eval" if rel.startswith("eval/") else rel.split(".")[0]
        if r.get("split") != want_split:
            errs.append(f"split!={want_split}")
        if r.get("engine") != man["engine_id"]:
            errs.append("engine_id")
        forms = EVAL_FORMS if want_split == "eval" else TRAIN_FORMS
        if r.get("form") not in forms:
            errs.append("form_for_split")
        for t, st in zip(r.get("tables", []), r.get("table_status", [])):
            if t in tinfo and tinfo[t]["status"] != st:
                errs.append(f"table_status:{t}")
        if errs:
            checks["schema"].bad({"id": r.get("id"), "errors": errs})
        # tables
        checks["tables"].n += 1
        for t in r.get("tables", []):
            if t.startswith("ko-2020"):
                checks["tables"].bad({"id": r["id"], "table": t, "why": "ko-2020 excluded"})
            elif t in EN.PSEUDO_TABLES:
                continue
            elif t not in tinfo:
                checks["tables"].bad({"id": r["id"], "table": t, "why": "not in inventory"})
            elif want_split != "eval" and tinfo[t]["holdout"]:
                checks["tables"].bad({"id": r["id"], "table": t, "why": "held-out table in train/dev"})
            elif want_split != "eval" and tinfo[t]["weight"] <= 0:
                checks["tables"].bad({"id": r["id"], "table": t, "why": "zero-weight table in train"})
        # dedup
        if r.get("form") != "Enoise":
            checks["dedup"].n += 1
            k = dedup_key(r["tables"], r["text"])
            if k in seen_keys:
                checks["dedup"].bad({"id": r["id"], "dup_of": seen_keys[k]})
            else:
                seen_keys[k] = r["id"]
    print(f"[verify] schema/tables/dedup done ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------ f-consistency (all rows)
    lou_jobs: dict[tuple[str, str], int] = {}
    nem_canon: dict[str, int] = {}
    km_items: dict[tuple[str, str], int] = {}
    plan: list[tuple[dict, list]] = []
    nikl_rows_out = []
    noise_rows = []
    for rel, r in rows:
        if r.get("form") == "Enoise":
            noise_rows.append(r)
            continue
        if r.get("source") == "nikl":
            nikl_rows_out.append(r)
            lou_jobs.setdefault((r["tables"][0], r["text"]), len(lou_jobs))
            continue
        parts = []
        if EN.KMATH_LABEL in r["tables"]:
            key = (r["kmath_input"], r["kmath_text"])
            want_text = (r["kmath_text"].replace("$", "$$") if r["kmath_input"] == "sentence"
                         else f"$${r['kmath_text']}$$")
            if r["text"] != want_text:
                parts.append(("bad", "text != kmath_text mapping"))
            km_items.setdefault(key, len(km_items))
            parts.append(("kmath", key))
        else:
            for sp in spans_for_f(r):
                if sp["table"] == EN.NEMETH_LABEL:
                    c = NM.canon_of_text(sp["text"])
                    if c is None or (sp.get("latex_canonical") not in (None, c)):
                        parts.append(("bad", "latex_canonical mismatch"))
                        continue
                    nem_canon.setdefault(c, len(nem_canon))
                    parts.append(("nemeth", c, bool(r.get("segments"))))
                else:
                    lou_jobs.setdefault((sp["table"], sp["text"]), len(lou_jobs))
                    parts.append(("louis", (sp["table"], sp["text"])))
        plan.append((r, parts))
    print(f"[verify] jobs: liblouis={len(lou_jobs)} nemeth={len(nem_canon)} kmath={len(km_items)}",
          flush=True)

    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=EN.init_worker, initargs=(spec.to_dict(),)) as pool:
        lou_keys = list(lou_jobs)
        # one table per spawned process: liblouis output can depend on tables compiled earlier in the process
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import u1_retranscribe as ISO  # noqa: PLC0415
        by_t = defaultdict(list)
        for tk, tx in lou_keys:
            by_t[tk].append(tx)
        itasks = []
        for tk, txs in by_t.items():
            for i in range(0, len(txs), 20000):
                itasks.append((len(itasks), tk, txs[i:i + 20000]))
        lou = {}
        sctx = mp.get_context("spawn")
        with sctx.Pool(args.workers, initializer=ISO._init, initargs=(spec.to_dict(),),
                       maxtasksperchild=1) as ipool:
            for idx, res in ipool.imap_unordered(ISO._task, itasks, chunksize=1):
                _i, tk, txs = itasks[idx]
                for tx, (b, ok) in zip(txs, res):
                    lou[(tk, tx)] = (b, ok, b is not None and is_braille_only(b, allow_newline=False))
        canon_keys = list(nem_canon)
        chunks = [canon_keys[i:i + 200] for i in range(0, len(canon_keys), 200)]
        nem_out: list = []
        for res in pool.imap(NM.nemeth_job, [(c, spec.nemeth_jar, spec.java) for c in chunks], 1):
            nem_out.extend(res)
        nem = dict(zip(canon_keys, nem_out))
        km_keys = list(km_items)
        km = dict(zip(km_keys, pool.map(KM.kmath_job, km_keys, chunksize=16))) if km_keys else {}
    print(f"[verify] re-transcription done ({time.time() - t0:.0f}s)", flush=True)

    fc, ud = checks["f_consistency"], checks["undefined"]
    per_form = defaultdict(lambda: [0, 0])
    for r, parts in plan:
        fc.n += 1
        ud.n += 1
        pieces, ok = [], True
        for p in parts:
            if p[0] == "bad":
                ok = False
                fc.bad({"id": r["id"], "why": p[1]})
                break
            if p[0] == "louis":
                b, strict_ok, bonly = lou[p[1]]
                if b is None:
                    ok = False
                    fc.bad({"id": r["id"], "why": f"translation failed {p[1][0]}"})
                    break
                if not strict_ok or not bonly:
                    ud.bad({"id": r["id"], "table": p[1][0], "strict_ok": strict_ok,
                            "braille_only": bonly})
                pieces.append(b)
            elif p[0] == "nemeth":
                c = nem.get(p[1])
                if c is None:
                    ok = False
                    fc.bad({"id": r["id"], "why": "latex2nemeth failed on re-run"})
                    break
                pieces.append(NM.NEMETH_OPEN + c + NM.NEMETH_CLOSE if p[2] else c)
            elif p[0] == "kmath":
                b, why = km.get(p[1], (None, "missing"))
                if b is None:
                    ok = False
                    fc.bad({"id": r["id"], "why": f"kmath re-run failed: {why}"})
                    break
                pieces.append(b)
        if not ok:
            per_form[r["form"]][1] += 1
            continue
        join = r.get("seg_join", "\n") if r.get("segments") else ""
        f = join.join(pieces)
        allow_nl = bool(r.get("segments")) and join == "\n"
        if not is_braille_only(r["braille"], allow_newline=allow_nl):
            ud.bad({"id": r["id"], "why": "non-braille char in braille"})
        if f != r["braille"]:
            fc.bad({"id": r["id"], "form": r["form"], "tables": r["tables"],
                    "text": r["text"][:120], "stored": r["braille"][:80], "f": f[:80]})
            per_form[r["form"]][1] += 1
        per_form[r["form"]][0] += 1
    fc.info["per_form_ok_fail"] = {k: {"ok": v[0], "fail": v[1]} for k, v in sorted(per_form.items())}

    # NIKL: reported via f_consistent (not enforced), but the flag must be right
    n_cons = 0
    for r in nikl_rows_out:
        b, _s, _bo = lou[(r["tables"][0], r["text"])]
        cons = b == r["braille"]
        n_cons += cons
        if cons != r.get("f_consistent"):
            fc.bad({"id": r["id"], "why": "stored f_consistent is wrong"})
        if not is_braille_only(r["braille"], allow_newline=False):
            ud.bad({"id": r["id"], "why": "NIKL braille not U+2800-block"})
    fc.info["nikl_rows"] = len(nikl_rows_out)
    fc.info["nikl_f_consistent"] = n_cons
    fc.info["nikl_f_consistent_rate"] = round(n_cons / len(nikl_rows_out), 5) if nikl_rows_out else None
    fc.info["excluded_noise_rows"] = len(noise_rows)

    # ------------------------------------------------------------ noise
    core = {r["id"]: r for _f, r in rows if r.get("form") == "Ecore"}
    nz = checks["noise"]
    for r in noise_rows:
        nz.n += 1
        base = core.get(r.get("noise_base_id"))
        if base is None or len(base["braille"]) != len(r["braille"]) or \
                base["text"] != r["text"] or r.get("noise_p") not in (0.005, 0.01, 0.02):
            nz.bad(r["id"])
            continue
        if r["braille"] == base["braille"]:
            nz.bad({"id": r["id"], "why": "identical to its base row"})
        flips = sum(bin((ord(a) - 0x2800) ^ (ord(b) - 0x2800)).count("1")
                    for a, b in zip(base["braille"], r["braille"]) if a != b)
        if flips != r.get("noise_n_flips"):
            nz.bad({"id": r["id"], "why": "noise_n_flips wrong"})
        if r.get("noise_n_dots") == 6 and any(
                (ord(c) - 0x2800) & 0xC0 for c in r["braille"] if 0x2800 <= ord(c) <= 0x28FF):
            nz.bad({"id": r["id"], "why": "dot 7/8 cell in a 6-dot code's noise row"})

    # ------------------------------------------------------------ disjointness
    dj = checks["disjoint"]
    excl = load_eval_exclusions(data_dir)
    reg = HashRegistry()
    for path, name in eval_source_files(data_dir):
        reg.add_lines_file(path, name, excl)
    nikl = load_nikl(REPO)
    test_ids = {r["id"] for r in nikl if r["split"] == "test"}
    test_texts = {r["src"] for r in nikl if r["split"] == "test"}
    for t in test_texts:
        reg.add_text(t, "nikl_test_sealed", sealed=True)
    for t in rule_example_texts(REPO):
        reg.add_text(t, "rule_examples")
    for rel, r in rows:
        if rel.startswith("eval/"):
            reg.add_row(r)
    for rel, r in rows:
        if rel.startswith("eval/"):
            continue
        dj.n += 1
        hit = row_units(r) & reg.hashes
        if hit:
            dj.bad({"id": r["id"], "form": r["form"], "n_hits": len(hit), "text": r["text"][:100]})
        if r.get("nikl_id") in test_ids or (r.get("source") == "nikl" and r["text"] in test_texts):
            dj.bad({"id": r["id"], "why": "NIKL test row in train/dev"})
    # source level
    def hashes(files, skip=frozenset()):
        out = set()
        for p, _n in files:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    s = line.strip()
                    if s:
                        h = sentence_hash(s)
                        if h not in skip:
                            out.add(h)
        return out
    tr_src = hashes(train_source_files(data_dir))
    ev_src = hashes(eval_source_files(data_dir), skip=excl)
    inter = len(tr_src & ev_src)
    dj.info["source_level_overlap"] = inter
    dj.info["n_train_source_lines"] = len(tr_src)
    dj.info["n_eval_source_lines"] = len(ev_src)
    if inter:
        dj.bad({"why": f"train sources ∩ eval sources = {inter} (minus eval_exclusions)"})
    with open(os.path.join(data, POINTER_REL)) as fh:
        ptr = [json.loads(line) for line in fh]
    if {p["nikl_id"] for p in ptr} != test_ids or any(set(p) & {"src", "tgt", "text", "braille"} for p in ptr):
        dj.bad({"why": "nikl_test_pointer is not exactly the sealed test ids (ids only)"})
    dj.info["registry_hashes"] = len(reg)
    dj.info["nikl_test_ids"] = len(test_ids)
    nikl_dev_texts = {r["src"] for r in nikl if r["split"] == "dev"}
    dj.info["nikl_dev_texts_also_in_test"] = len(nikl_dev_texts & test_texts)
    # math canonical forms: eval vs train/dev
    canon = defaultdict(set)
    for rel, r in rows:
        if r.get("latex_canonical"):
            canon["eval" if rel.startswith("eval/") else r["split"]].add(r["latex_canonical"])
    dj.info["math_canon"] = {k: len(v) for k, v in sorted(canon.items())}
    for side in ("train", "dev"):
        ov = canon[side] & canon["eval"]
        if ov:
            dj.bad({"why": f"math canonical forms shared {side}/eval", "n": len(ov)})

    # ------------------------------------------------------------ dev vs train
    dd = checks["dev_disjoint"]
    dev_units: set[int] = set()
    dev_ids, dev_canon = set(), canon["dev"]
    for rel, r in rows:
        if r.get("split") == "dev":
            dev_units |= row_units(r)
            if r.get("nikl_id"):
                dev_ids.add(r["nikl_id"])
    for rel, r in rows:
        if r.get("split") != "train":
            continue
        dd.n += 1
        hit = row_units(r) & dev_units
        if hit:
            dd.bad({"id": r["id"], "form": r["form"], "n_hits": len(hit), "text": r["text"][:100]})
        if r.get("nikl_id") in dev_ids:
            dd.bad({"id": r["id"], "why": "NIKL id also in dev"})
    ov = canon["train"] & dev_canon
    if ov:
        dd.bad({"why": "math canonical forms shared train/dev", "n": len(ov)})
    dd.info["dev_units"] = len(dev_units)

    # ------------------------------------------------------------ NIKL near-duplicates of test
    nd = checks["nikl_near_dup"]
    screen = NearDupScreen(test_texts)
    why_c = Counter()
    for _f, r in rows:
        if r.get("source") == "nikl":
            nd.n += 1
            w = screen.why(r["text"])
            if w is not None:
                why_c[w] += 1
                nd.bad({"id": r["id"], "why": w})
    nd.info["reasons"] = dict(why_c)
    del screen

    # ------------------------------------------------------------ persisted registry
    # registry.npz holds the public eval units; sealed units and NIKL test lines are checked by digest only
    rg = checks["registry"]
    rg.n = 1
    rp = os.path.join(data, "registry.npz")
    if not os.path.isfile(rp):
        rg.bad("registry.npz missing")
    else:
        arr = load_registry(rp)
        units = set(int(x) for x in arr["eval_units"])
        rg.info["n_eval_units_public"] = len(units)
        pub = reg.public()
        if units != pub:
            rg.bad({"why": "eval_units != recomputed public registry",
                    "only_file": len(units - pub), "only_recomputed": len(pub - units)})
        pers = man["registry"]["persisted"]
        sealed_only = reg.hashes - pub
        if pers.get("sealed_units", {}).get("sha256") != reg_digest(sealed_only):
            rg.bad({"why": "sealed unit digest != recomputed"})
        if units & sealed_only:
            rg.bad({"why": "a sealed-only unit is persisted"})
        lines = {k[len("lines__"):].replace("__", ":", 1): set(int(x) for x in v)
                 for k, v in arr.items() if k.startswith("lines__")}
        if "eval:nikl_test" in lines:
            rg.bad({"why": "NIKL test line hashes persisted"})
        test_lines = {h64(t) for t in test_texts}
        if pers.get("sealed_line_sets", {}).get("eval:nikl_test", {}).get("sha256") != \
                reg_digest(test_lines):
            rg.bad({"why": "NIKL test line digest != recomputed"})
        lines["eval:nikl_test"] = test_lines
        rg.info["lines"] = {k: len(v) for k, v in sorted(lines.items())}
        for a in sorted(lines):
            for b in sorted(lines):
                if a.startswith("train:") and b.startswith("eval:"):
                    n = len(lines[a] & lines[b])
                    if n:
                        rg.info.setdefault("line_overlap", {})[f"{a}|{b}"] = n
                        # NIKL dev/test share boilerplate sentences; reported only, such dev rows are never used
                        if not (a == "train:nikl_dev" and b == "eval:nikl_test"):
                            rg.bad({"why": "train/eval source line overlap", "pair": f"{a}|{b}", "n": n})
        s7_units = set()
        for _f, r in rows:
            if r.get("source") == "nikl":
                s7_units |= row_units(r)
        rg.info["s7_rows_hitting_nikl_test_lines"] = len(s7_units & lines.get("eval:nikl_test", set()))
        if rg.info["s7_rows_hitting_nikl_test_lines"]:
            rg.bad("an S7 row contains a sealed NIKL test sentence")

    # ------------------------------------------------------------ counts
    cn = checks["counts"]
    got_split = Counter((r["split"], r["form"]) for _f, r in rows)
    form_of = {"S3_k2": ("S3", 2), "S3_k3": ("S3", 3), "S3_k4": ("S3", 4)}
    got_k = Counter((r["split"], r["form"], r["k"]) for _f, r in rows if r["form"] == "S3")
    table = {}
    for key, st in man["strata"].items():
        if key in form_of:
            f, k = form_of[key]
            n_tr, n_dv = got_k[("train", f, k)], got_k[("dev", f, k)]
        else:
            n_tr, n_dv = got_split[("train", key)], got_split[("dev", key)]
        cn.n += 1
        table[key] = {"train": n_tr, "target_train": st["target_train"], "dev": n_dv,
                      "target_dev": st["target_dev"], "status": st["status"]}
        if (n_tr, n_dv) != (st["train"], st["dev"]):
            cn.bad({"stratum": key, "why": "file counts != manifest"})
        if st["status"] == "shortfall":
            cn.bad({"stratum": key, "why": "shortfall", "train": n_tr, "target": st["target_train"]})
    cn.info["strata"] = table
    cn.info["pending"] = sorted(k for k, v in man["strata"].items() if v["status"] == "pending")
    # --- eval cells recounted from the files against the cell design
    sf_got = {(x["stratum"], x["cell"]): x["got"] for x in man["shortfalls"]}
    et = man["eval_targets"]
    tag_n = Counter(r["tag"] for rel, r in rows if rel.startswith("eval/"))
    cells: list[tuple[str, str, int, int]] = []     # (stratum, cell, recount, target)
    weighted = sorted((t for t in inv["tables"] if t["weight"] > 0),
                      key=lambda t: (-t["weight"], t["table_id"]))
    core_n = Counter(r["tables"][0] for rel, r in rows if r.get("form") == "Ecore")
    for i, t in enumerate(weighted):
        tgt = et["core_per_code"]["head_docs"] if i < et["core_per_code"]["head_n"] \
            else et["core_per_code"]["tail_docs"]
        cells.append(("eval:core_per_code", t["table_id"], core_n.get(t["table_id"], 0), tgt))
    for gid in et["snippets"]["groups"]:
        if ("eval:snippets", f"{gid}:all") in sf_got:
            continue
        for b in SNIPPET_BUCKETS:
            cells.append(("eval:snippets", f"{gid}:{b}", tag_n.get(f"snippet:{gid}:{b}", 0),
                          et["snippets"]["per_cell"]))
    for k in (2, 3, 4):
        for reg_ in ("anchor", "uniform", "confusable"):
            cells.append(("eval:mixed_grid", f"k{k}:{reg_}",
                          tag_n.get(f"mixed_grid:k{k}:{reg_}", 0), et["mixed_grid"]["per_cell"]))
    for reg_ in ("anchor", "uniform"):
        cells.append(("eval:k2_paragraph", reg_, tag_n.get(f"k2_paragraph:{reg_}", 0),
                      et["k2_paragraph"]["per_cell"]))
    for d in ("light", "heavy"):
        cells.append(("eval:intra_switch_eval", d, tag_n.get(f"intra_switch_eval:{d}", 0),
                      et["intra_switch_eval"]["per_cell"]))
    n_pairs = len({r.get("pair_id") for _f, r in rows if r.get("form") == "Ekodisc"})
    cells.append(("eval:kodisc", "pairs", n_pairs, et["kodisc"]["pairs"]))
    for t in sorted(x["table_id"] for x in inv["tables"] if x["holdout"]):
        cells.append(("eval:zeroshot_eval", t, tag_n.get(f"zeroshot_eval:{t}", 0),
                      et["zeroshot_eval"]["per_table"]))
    cells.append(("eval:zeroshot_eval", "mixed", tag_n.get("zeroshot_eval:mixed", 0),
                  et["zeroshot_eval"]["mixed"]))
    n_bases = len({r.get("noise_base_id") for r in noise_rows})
    cells.append(("eval:noise_variants", "base", n_bases, et["noise_variants"]["base"]))
    for p_ in et["noise_variants"]["levels"]:
        cells.append(("eval:noise_variants", f"p={p_}",
                      sum(1 for r in noise_rows if r.get("noise_p") == p_), n_bases))
    ef = et["math_eval"].get("embed_frac", 0.5)
    for eng_, n_ in (("nemeth", et["math_eval"]["nemeth"]), ("kmath", et["math_eval"]["kmath"])):
        if ("eval:math_eval", eng_) in sf_got:          # pending engine: whole cell
            cells.append(("eval:math_eval", eng_,
                          sum(tag_n.get(f"math_eval:{eng_}:{f}", 0) for f in ("embedded", "standalone")),
                          n_))
            continue
        n_emb = round(n_ * ef)
        for f, tgt in (("embedded", n_emb), ("standalone", n_ - n_emb)):
            cells.append(("eval:math_eval", f"{eng_}:{f}", tag_n.get(f"math_eval:{eng_}:{f}", 0), tgt))
    bad_cells = []
    for st_, cell, got, tgt in cells:
        cn.n += 1
        ok = got == tgt or (got < tgt and sf_got.get((st_, cell)) == got)
        if not ok:
            bad_cells.append(f"{st_}:{cell} {got}/{tgt}")
            cn.bad({"cell": f"{st_}:{cell}", "recount": got, "target": tgt,
                    "manifest_shortfall_got": sf_got.get((st_, cell)),
                    "why": "over target" if got > tgt else "short without a matching shortfall"})
    cn.info["eval_cells_recounted"] = len(cells)
    cn.info["eval_cells_short"] = sorted(f"{a}:{b} {g}/{t}" for a, b, g, t in cells if g < t)
    # eval tags agree with the row's k / regime / switch density
    for rel, r in rows:
        tg = r.get("tag", "")
        if tg.startswith("mixed_grid:k"):
            _m, kk, rg_ = tg.split(":")
            if r["k"] != int(kk[1:]) or r["regime"] != rg_:
                cn.bad({"id": r["id"], "why": "mixed_grid tag != k/regime"})
        elif tg.startswith("k2_paragraph:"):
            if r["k"] != 2 or r["regime"] != tg.split(":")[1]:
                cn.bad({"id": r["id"], "why": "k2_paragraph tag != k/regime"})
        elif tg.startswith("intra_switch_eval:"):
            if r["switch_density"] != tg.split(":")[1]:
                cn.bad({"id": r["id"], "why": "intra tag != density"})
    # --- S1 floors recounted from train.jsonl
    s1_by_table = Counter(r["tables"][0] for _f, r in rows
                          if r.get("form") == "S1" and r.get("split") == "train")
    unmet = []
    for tid, v in man["floor"]["per_table"].items():
        cn.n += 1
        if s1_by_table.get(tid, 0) != v["train_rows"]:
            cn.bad({"table": tid, "why": "S1 floor recount != manifest",
                    "recount": s1_by_table.get(tid, 0), "manifest": v["train_rows"]})
        if s1_by_table.get(tid, 0) < v["floor"]:
            unmet.append(tid)
    if sorted(unmet) != sorted(man["floor"]["unmet"]):
        cn.bad({"why": "floor unmet list != recount", "recount": sorted(unmet)})
    cn.info["eval_counts"] = man["eval_counts"]
    cn.info["shortfall_cells"] = man["shortfalls"]
    cn.info["shortfall_cells_non_pending"] = [x for x in man["shortfalls"]
                                             if not str(x["reason"]).startswith("pending")]
    cn.info["floor_unmet"] = {t: man["floor"]["per_table"][t] for t in man["floor"]["unmet"]}
    # A short stratum fails; cell-level shortfalls (S1 floors, S1b tables, eval cells) are listed here and in
    # targets_fully_met, and fail only when they make their stratum short.
    cn.info["cell_shortfalls"] = {
        "floors": sorted(man["floor"]["unmet"]),
        "eval": [f'{x["stratum"]}:{x["cell"]} {x["got"]}/{x["target"]}'
                 for x in cn.info["shortfall_cells_non_pending"] if x["stratum"].startswith("eval:")],
        "train": [f'{x["stratum"]}:{x["cell"]} {x["got"]}/{x["target"]}'
                  for x in cn.info["shortfall_cells_non_pending"]
                  if not x["stratum"].startswith("eval:") and x["stratum"] != "S1:floor"],
    }

    # ------------------------------------------------------------ S1b: one table per line
    ob = checks["s1b_one_table"]
    s1b_tab: dict[str, set[str]] = defaultdict(set)
    other_units: set[int] = set()
    for _f, r in rows:
        if r.get("form") == "S1b":
            s1b_tab[r["text"]].add(r["tables"][0])
        elif r.get("form") in TRAIN_FORMS:
            other_units |= row_units(r)
    for t, tabs in s1b_tab.items():
        ob.n += 1
        # sub-sentences too: S1/S2/S6 join sentences with spaces, which a '\n' split misses
        mine = set(unit_keys(t))
        for ss in sub_sentences(t):
            mine.update(unit_keys(ss))
        if len(tabs) > 1:
            ob.bad({"text": t[:60], "tables": sorted(tabs)})
        elif mine & other_units:
            ob.bad({"text": t[:60], "why": "S1b line (or a sentence of it) reused by another stratum"})
    ob.info["per_table"] = dict(Counter(next(iter(v)) for v in s1b_tab.values()))

    # ------------------------------------------------------------ kodisc twins
    kd = checks["kodisc_pairs"]
    pairs: dict[str, list[dict]] = defaultdict(list)
    for _f, r in rows:
        if r.get("form") == "Ekodisc":
            pairs[r.get("pair_id")].append(r)
    want_tabs = sorted(man["eval_targets"]["kodisc"]["tables"])
    for pid, rs in pairs.items():
        kd.n += 1
        if len(rs) != 2 or rs[0]["text"] != rs[1]["text"] or \
                sorted(r["tables"][0] for r in rs) != want_tabs:
            kd.bad({"pair_id": pid, "n": len(rs)})
    kd.info["pairs"] = len(pairs)

    # ------------------------------------------------------------ CJK joiner
    # A space at a sentence join is a defect but one native to a source line is not, so a match counts only when
    # the text after the space starts a known source line.
    cj = checks["cjk_joiner"]
    cjk_files = []
    for sub, names in (("wiki_txt_A", ("cmn.txt", "ja.txt")), ("wiki_txt_B", ("cmn.txt", "ja.txt"))):
        cjk_files += [os.path.join(data_dir, sub, n) for n in names]
    for split in ("dev", "devtest"):
        for code in ("zho_Hans", "zho_Hant", "yue_Hant", "jpn_Jpan"):
            cjk_files.append(os.path.join(data_dir, "flores200_dataset", split, f"{code}.{split}"))
    by_prefix: dict[str, set[str]] = defaultdict(set)
    import unicodedata  # noqa: PLC0415
    for p in cjk_files:
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    ln = unicodedata.normalize("NFC", line.strip())
                    if len(ln) >= 8:
                        by_prefix[ln[:8]].add(ln)
    n_native = 0
    for _f, r in rows:
        for sp in (r.get("segments") or [{"lang": r["langs"][0], "text": r["text"]}]):
            if sp.get("lang") not in ("cmn", "yue", "ja"):
                continue
            cj.n += 1
            for m in CJK_SPACE.finditer(sp["text"]):
                rest = sp["text"][m.end():]
                if any(rest.startswith(ln) for ln in by_prefix.get(rest[:8], ())):
                    cj.bad({"id": r["id"], "text": sp["text"][:80]})
                    break
                n_native += 1
    cj.info["native_spaces_seen"] = n_native

    report = {
        "data": data, "engine_id": man["engine_id"], "n_rows": len(rows),
        "checks": {k: v.to_json() for k, v in checks.items()},
        "pending_strata": cn.info["pending"],
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    report["pass"] = all(v.fail == 0 for v in checks.values())
    fg = man.get("freeze_gates", {})
    report["freeze_gates"] = {
        "verify_pass": report["pass"],
        "ko2024_certified": bool(fg.get("ko2024_certified")),
        # Korean math excluded: a gate only when required
        "kmath_ok": (not fg.get("kmath_required", True))
                    or (bool(fg.get("kmath_available")) and bool(fg.get("kmath_frozen"))),
        "no_pending_strata": not cn.info["pending"],
        "full_scale": bool(fg.get("full_scale")),
        "no_gate_override": not fg.get("gates_overridden", False),
    }
    report["freezable"] = all(report["freeze_gates"].values())
    report["pass_except_counts"] = all(v.fail == 0 for k, v in checks.items() if k != "counts")
    report["targets_fully_met"] = report["pass"] and not any(cn.info["cell_shortfalls"].values())
    report["cell_shortfalls"] = cn.info["cell_shortfalls"]
    import shutil  # noqa: PLC0415
    shutil.rmtree(shadow_tmp, ignore_errors=True)
    with open(os.path.join(data, "verify_report.json"), "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"pass": report["pass"], "pass_except_counts": report["pass_except_counts"],
                      "targets_fully_met": report["targets_fully_met"],
                      "cell_shortfalls": {k: len(v) for k, v in report["cell_shortfalls"].items()},
                      "checks": {k: (v.fail == 0, v.n, v.fail) for k, v in checks.items()},
                      "pending": report["pending_strata"],
                      "freezable": report["freezable"], "freeze_gates": report["freeze_gates"],
                      "nikl_f_consistent_rate": fc.info["nikl_f_consistent_rate"],
                      "wall_clock_sec": report["wall_clock_sec"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
