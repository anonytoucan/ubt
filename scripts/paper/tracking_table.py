"""Internal tracking table (not for the submission): every measured model x condition x decoding x dataset x group x
metric in one long CSV, plus pivoted views and a coverage matrix of what is still missing.

  models      sft, topup (weak-code top-up), rlvr; baselines liblouis (code given), gpt-5.5 (few-shot, 1k) and the math
              rule readers; backbones byt5 / gemma and their checkpoints; rlvr_*_adapt (fine-tuned on a real corpus)
  conditions  agnostic (code inferred), filled (gold code prefilled = code given), hinted (not trained)
  decoding    greedy, bon_lp (best-of-20 by log-prob), bon_verified (lambda=2), bon_closest (lambda->inf), oracle
  datasets    core, mixed k2/k3/k4, intra, k2p, noise, zeroshot, math, real corpora, BrailleLLM few-shot, Korean pairs
  groups      overall, family, record group, language, code, read-as (code carried by the selected output)
  metrics     cer (_st: no Traditional->Simplified fold; _raw: unnormalised; _norm_*: normal-form chain), bcer, wer,
              exact, strict/conf code-ID, consistent (selected output re-transcribes), truncated (hit the token cap),
              error-kind shares, cer_nows (whitespace removed), bleu, chrf, math exact / canonical CER, Korean-pair
              scores, sim/judge, <metric>_ci95_* (Table 1 bootstrap), braille edition counts
  python scripts/paper/tracking_table.py --out <paper>/internal/tracking
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import glob
import hashlib
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from ubt.paths import DATASETS, WORK  # noqa: E402

R = WORK
INV = f"{DATASETS}/data_u1_v2/inventory.json"
MODELS = ("sft", "topup", "rlvr")
CONDITIONS = ("agnostic", "filled", "hinted")
DEC = {"G": "greedy", "LP": "bon_lp", "V": "bon_verified", "Vinf": "bon_closest", "O": "oracle"}
POOLS = {("sft", "agnostic"): [f"{R}/bon_sft/run/per_doc.jsonl"],
         ("sft", "filled"): [f"{R}/bon_sft/run_prefill/per_doc.jsonl"],
         ("topup", "agnostic"): [f"{R}/evals/topup/bon/per_doc.jsonl", f"{R}/bon_topup/run/per_doc.jsonl"],
         ("topup", "filled"): [f"{R}/evals/topup/prefill/per_doc.jsonl"],
         ("rlvr", "agnostic"): [f"{R}/evals/rlvr/bon/per_doc.jsonl"],
         ("rlvr", "filled"): [f"{R}/evals/rlvr/prefill/per_doc.jsonl"]}
# code given on every segment of the multi-segment sets (bon_gen.py gen --force-codes)
FORCED = {m: f"{R}/evals/{m}/forced/per_doc.jsonl" for m in MODELS}
# k = 3/4 paragraph documents (datasets k3p/k4p, scripts/eval/build_kpara_eval.py), final model only
KPARA = {("rlvr", "agnostic"): f"{R}/evals/rlvr_kpara/bon/per_doc.jsonl",
         ("rlvr", "filled"): f"{R}/evals/rlvr_kpara/forced/per_doc.jsonl"}
REAL = {"sft": f"{R}/real_corpora/run/per_doc.jsonl", "topup": f"{R}/evals/topup/real/per_doc.jsonl",
        "rlvr": f"{R}/evals/rlvr/real/per_doc.jsonl"}
REAL_REPORT = {k: v.replace("per_doc.jsonl", "report.json") for k, v in REAL.items()}
FEWSHOT = {"sft": f"{R}/bllm/run/score_{{m}}", "topup": f"{R}/evals/topup/bllm/score_{{m}}",
           "rlvr": f"{R}/evals/rlvr/bllm/score_{{m}}"}
MATH = {"sft": f"{R}/math_base/bon_score.json", "topup": f"{R}/evals/topup/math/bon_score.json",
        "rlvr": f"{R}/evals/rlvr/math/bon_score.json"}
MATH_BASE = f"{R}/math_base/score.json"
KODISC = {"sft": f"{R}/kodisc_plain/run/kodisc_report.json", "topup": f"{R}/evals/topup/kodisc/kodisc_report.json",
          "rlvr": f"{R}/evals/rlvr/kodisc/kodisc_report.json"}
ZS_ROWS = f"{R}/bon_data/zeroshot.jsonl"
ZS_EXCLUDED = {"sw-ke-g1.utb", "sw-ke-g1-2.ctb"}      # held out but cell-identical to trained codes: not zero-shot
LIBLOUIS = f"{R}/liblouis_bwd/per_doc.jsonl"
GPT = f"{R}/frontier/per_doc.jsonl"
UNNORM = {"sft": f"{R}/paper/unnormalized_per_code.json", "topup": f"{R}/paper/unnormalized_per_code.topup.json",
          "rlvr": f"{R}/paper/unnormalized_per_code.rlvr.json"}
# not measured here; listed in the coverage section so nothing goes missing silently
PENDING = [("hinted (separately trained <|tables|> variant)", "not trained; 'filled' is the code-given condition"),
           ("filled on noise/math/kodisc/zeroshot", "not measured (zeroshot: the held-out code label is unknown to the model)"),
           ("NIKL (sealed test)", "measured once, with the final model"),
           ("sim./judge", "final model (rlvr) and the LibLouis / GPT-5.5 baselines (scripts/eval/extra_metrics_chain.sh); SFT, top-up "
            "and the backbones code-inferred only (extra_rows_build.py --pools-only)"),
           ("GPT-5.5 on the other real corpora", "Bible only (1,000 verses); BrailleLLM / Vision not run (API cost); NIKL may not be sent to an external service"),
           ("snippets (30,000)", "unused in the paper; never run"),
           ("real corpora with a backbone", "not run (Table 4 has the final model and its adaptation runs only)")]
SOURCES = {}
EXTRA = {"rlvr": f"{R}/evals/rlvr/extra"}        # sim/judge: the final model and the baselines (scripts/eval/extra_rows_build.py)
# backbones and intermediate checkpoints, code inferred (scripts/eval/eval_byt5_chain.sh, eval_model_chain.sh)
BACKBONES = {"byt5": f"{R}/evals/byt5/bon/per_doc.jsonl", "gemma": f"{R}/evals/gemma/bon/per_doc.jsonl",
             "byt5_ck5000": f"{R}/evals/byt5_ck5000/bon/per_doc.jsonl",      # ByT5-base, step 5,000 of 14,227
             "gemma_ck6000": f"{R}/evals/gemma_ck6000/bon/per_doc.jsonl"}    # Gemma-2-9B, step 6,000 of 7,849
EXTRA.update({m: f"{R}/evals/{m}/extra" for m in BACKBONES})
EXTRA["topup"] = f"{R}/evals/topup/extra"        # Table 1 'w/o reinforcement learning' row (pools-only rows)
EXTRA["sft"] = f"{R}/evals/sft/extra"            # Table 1 'base training only' row (pools-only rows)
# subset pools: core rows go to dataset <name>, apart from the full split; rlvr is scored on the same documents
TRIALS = {"core_trial400": (f"{R}/evals/byt5_ck4000_trial/data/main.jsonl",     # 400 core documents
                            [("byt5_ck4000", f"{R}/evals/byt5_ck4000_trial/per_doc.jsonl"),
                             ("gemma_ck6000", f"{R}/evals/gemma_ck6000_trial/per_doc.jsonl"),    # step 6000 of 7,849
                             ("rlvr", f"{R}/evals/rlvr/bon/per_doc.jsonl")])}
# final model fine-tuned on a real corpus's training split (scripts/real_corpora/bllm_adapt_chain.sh, vision_adapt_chain.sh)
ADAPT = {"rlvr_bllm_adapt": f"{R}/evals/rlvr_bllm_adapt/real", "rlvr_vision_adapt": f"{R}/evals/rlvr_vision_adapt/real"}
CI = f"{R}/paper/table1_ci.json"                # Table 1 intervals (scripts/paper/table1_ci.py)
CI_ROWS = {"gpt55_fewshot": ("gpt-5.5", "none", "fewshot", "core_1k"), "liblouis_given": ("liblouis", "filled", "rule", "core"),
           "ours_bon": ("rlvr", "agnostic", "bon_verified", "core"), "ours_bon_oracle_code": ("rlvr", "filled", "bon_verified", "core"),
           "ours_greedy": ("rlvr", "agnostic", "greedy", "core"), "ours_bon_wo_rl": ("topup", "agnostic", "bon_verified", "core"),
           "byt5_bon": ("byt5", "agnostic", "bon_verified", "core"), "gemma_bon": ("gemma", "agnostic", "bon_verified", "core"),
           "byt5_ck5000_bon": ("byt5_ck5000", "agnostic", "bon_verified", "core"),
           "gemma_ck6000_bon": ("gemma_ck6000", "agnostic", "bon_verified", "core"),
           "ours_bon_base_only": ("sft", "agnostic", "bon_verified", "core")}
CI_MET = {"bcer": "bcer", "cer": "cer", "strict": "strict_pct", "conf": "conf_pct", "sim": "sim", "judge": "judge"}
ERRORS = f"{R}/paper/error_analysis.json"   # character errors by family and error kind (scripts/paper/error_analysis.py)
EDITION = {"rlvr": f"{R}/braille_edition/rlvr_v2/summary.json"}   # appendix braille edition (scripts/braille_edition/braille_edition.py)
# real-corpora baselines: one-candidate pools scored by real_corpora_score.py, code given only.
# NIKL (sealed, licence-bound) stays out of this table.
REAL_BASELINES = {"liblouis": f"{R}/evals/liblouis/real",
                  "gpt-5.5": f"{R}/evals/gpt55/real"}      # few-shot, Bible only (1,000 of 3,000 verses)


def _extra_names(model: str) -> dict:
    """rows_<name>.jsonl of scripts/eval/extra_rows_build.py -> (model, condition, decoding, dataset)."""
    return {f"{model}_V_agn": (model, "agnostic", "bon_verified", "core"),
            f"{model}_G_agn": (model, "agnostic", "greedy", "core"),
            f"{model}_V_given": (model, "filled", "bon_verified", "core"),
            f"{model}_real_given": (model, "filled", "bon_verified", "real"),
            f"{model}_real_inferred": (model, "agnostic", "bon_verified", "real"),
            "liblouis": ("liblouis", "filled", "rule", "core"),
            "gpt55": ("gpt-5.5", "none", "fewshot", "core_1k")}


def _given_extra(d, who, dec, emit):
    """sim / judge of a code-given real-corpora pool, joined with rows_sim_given.jsonl; empty outputs are judged 1."""
    rp = os.path.join(d, "rows_sim_given.jsonl")
    if not os.path.isfile(rp):
        return
    meta = {json.loads(line)["id"]: json.loads(line) for line in open(rp, encoding="utf-8")}
    for met in ("sim", "judge"):
        p = _use(os.path.join(d, f"{met}_given.json.rows.jsonl"))
        if not p:
            continue
        acc = collections.defaultdict(list)
        for line in open(p, encoding="utf-8"):
            x = json.loads(line)
            if x.get(met) is None:
                continue
            r = meta[x["id"]]
            v = 100.0 * x[met] if met == "sim" else (1.0 if not (r.get("hyp_text") or "").strip() else x[met])
            acc[r["group"]].append(v)
        for grp, vs in acc.items():
            emit(who, "filled", dec, f"real_{grp}", "subset", "all", {met: sum(vs) / len(vs)}, len(vs), p)


def _use(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    st = os.stat(path)
    SOURCES[path] = {"bytes": st.st_size, "mtime_utc": datetime.datetime.utcfromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                     "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    return path


def _first(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def family(inv, t):
    b = inv[t]["base_lang"]
    if b == "ja":
        return "Japanese"
    if t.startswith("zh"):
        return "Chinese"
    if b == "ko":
        return "Korean"
    if t in ("kok.tbl", "sah.utb"):
        return "Konkani+Yakut"
    if inv[t]["contraction"] == "full":
        return "contracted"
    return "other"


def _dataset(d):
    s = d["set"]
    if s == "core":
        return "core"
    if s == "mixed":
        return f"mixed_k{d['k']}"
    if s == "noise":
        return {"p0.0": "noise_p0", "p0.005": "noise_p0.5", "p0.01": "noise_p1", "p0.02": "noise_p2"}.get(d["group"], "noise")
    if s == "zeroshot":
        return f"zeroshot_{d['group']}"
    return s


class Agg:
    def __init__(self):
        self.a = collections.defaultdict(collections.Counter)

    def add(self, key, x, cells):
        a = self.a[key]
        for f in ("e", "n", "e_st", "n_st", "bc", "bc_st", "we", "wn", "e_nows", "n_nows", "bc_nows", "cells_nows"):
            if f in x:
                a[f] += x[f] or 0
        a["cells"] += cells
        a["docs"] += 1
        for f in ("strict", "conf", "exact", "feasible"):
            if f in x:
                a[f] += bool(x[f])
                a["has_" + f] += 1
        if "finish" in x:
            a["trunc"] += x["finish"] == "length"
            a["has_finish"] += 1
        if "kind" in x:
            a["k_" + x["kind"]] += 1
            a["ke_" + x["kind"]] += x.get("e", 0)
            a["kn_" + x["kind"]] += x.get("n", 0)
            a["kb_" + x["kind"]] += x.get("bc", 0)
            a["kc_" + x["kind"]] += cells

    def metrics(self, key):
        a = self.a[key]
        pct = lambda u, v: 100.0 * u / v if v else None  # noqa: E731
        m = {"cer": pct(a["e"], a["n"]), "cer_st": pct(a["e_st"], a["n_st"]), "bcer": pct(a["bc"], a["cells"]),
             "bcer_st": pct(a["bc_st"], a["cells"]) if a["bc_st"] or a["bc"] == 0 else None, "wer": pct(a["we"], a["wn"]),
             "exact_pct": pct(a["exact"], a["has_exact"]), "strict_pct": pct(a["strict"], a["has_strict"]),
             "conf_pct": pct(a["conf"], a["has_conf"]), "consistent_pct": pct(a["feasible"], a["has_feasible"]),
             "cer_nows": pct(a["e_nows"], a["n_nows"]), "truncated_pct": pct(a["trunc"], a["has_finish"])}
        if a["cells_nows"]:                        # real corpora: bCER with whitespace removed (real_corpora_score.py)
            m["bcer_nows"] = pct(a["bc_nows"], a["cells_nows"])
        for k in ("inconsistent", "consistent_misreading", "wrong_code", "correct"):
            if a["k_" + k] or any(a[f"k_{x}"] for x in ("inconsistent", "correct")):
                m[f"share_{k}"] = pct(a["k_" + k], a["docs"])
                m[f"cer_{k}"] = pct(a["ke_" + k], a["kn_" + k])
                m[f"bcer_{k}"] = pct(a["kb_" + k], a["kc_" + k])
        return {k: v for k, v in m.items() if v is not None}, a["docs"]


def read_as(model, cond, per_doc_path, emit):
    """Which code the selected output carries, per gold code (core and held-out codes), greedy and verified."""
    d0 = os.path.dirname(per_doc_path)
    want = {}
    for line in open(per_doc_path, encoding="utf-8"):
        d = json.loads(line)
        ds = _dataset(d)
        if ds in ("core", "zeroshot_single"):
            want[d["id"]] = (ds, d["table"], d["slice"], {p: d["pol"][p]["ci"] for p in ("G", "V") if p in d["pol"]})
    cnt = collections.defaultdict(collections.Counter)
    for stem in sorted({v[2] for v in want.values()}):
        prefix = {"main": "gen_main", "zeroshot": "gen_zeroshot", "core": "gen_core"}.get(stem, f"gen_{stem}")
        for p in glob.glob(os.path.join(d0, f"{prefix}.*.jsonl")):
            for line in open(p, encoding="utf-8"):
                x = json.loads(line)
                w = want.get(x["id"])
                if not w:
                    continue
                for pol, ci in w[3].items():
                    t = x["cands"][ci]["text"]
                    lab = t[1:t.index("\u27e9")] if t.startswith("\u27e8") and "\u27e9" in t else "(unparsed)"
                    cnt[(w[0], w[1], pol)][lab] += 1
    for (ds, code, pol), c in cnt.items():
        n = sum(c.values())
        for lab, k in c.most_common(5):
            emit(model, cond, DEC[pol], ds, "read_as", f"{code} -> {lab}", {"share_pct": 100.0 * k / n}, n, per_doc_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    inv = {t["table_id"]: t for t in json.load(open(INV))["tables"]}
    zs_tables = {}
    if os.path.isfile(ZS_ROWS):
        for line in open(ZS_ROWS, encoding="utf-8"):
            r = json.loads(line)
            zs_tables[r["id"]] = set(r["tables"])
    rows = []                                    # dicts for the long CSV
    have = set()                                 # (model, condition, dataset)

    def emit(model, cond, dec, dataset, gtype, group, mets, docs, source):
        for k, v in mets.items():
            rows.append({"model": model, "condition": cond, "decoding": dec, "dataset": dataset, "group_type": gtype,
                         "group": group, "metric": k, "value": round(v, 4) if isinstance(v, float) else v,
                         "docs": docs, "source": source})
        have.add((model, cond, dataset))

    # 1. best-of-20 pools (bon_eval format)
    pools = [(model, cond, _first(paths), None, None) for (model, cond), paths in POOLS.items()]
    pools += [(model, "filled", path, None, None) for model, path in FORCED.items() if os.path.isfile(path)]
    pools += [(model, "agnostic", path, None, None) for model, path in BACKBONES.items() if os.path.isfile(path)]
    pools += [(model, cond, path, None, None) for (model, cond), path in KPARA.items() if os.path.isfile(path)]
    for sub, (ids_path, members) in TRIALS.items():
        if os.path.isfile(ids_path):
            ids = {json.loads(line)["id"] for line in open(_use(ids_path), encoding="utf-8")}
            pools += [(model, "agnostic", path, sub, ids) for model, path in members if os.path.isfile(path)]
    for model, cond, path, sub, ids in pools:
        p = _use(path or "")
        if not p:
            continue
        agg = Agg()
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            ds = _dataset(d)
            if sub is not None:                              # subset pool: its core documents only, as dataset <sub>
                if ds != "core" or d["id"] not in ids:
                    continue
                keys = [(sub, "overall", "all"), (sub, "family", family(inv, d["table"]))]
                for pol, x in d["pol"].items():
                    for k in keys:
                        agg.add((pol,) + k, x, d["n_cells"])
                continue
            keys = [(ds, "overall", "all")]
            if ds == "core":
                keys += [(ds, "family", family(inv, d["table"])), (ds, "record_group", d["group"]), (ds, "code", d["table"]),
                         (ds, "language", inv[d["table"]]["language"])]
            if ds == "zeroshot_single":
                keys.append((ds, "code", d["table"]))
            if ds.startswith("zeroshot_") and d["id"] in zs_tables:        # the paper's six codes / 360 mixed documents
                excl = bool(zs_tables[d["id"]] & ZS_EXCLUDED)
                keys.append((ds, "subset", ("with_excluded_code" if excl else ("six_codes" if ds == "zeroshot_single" else "mixed_360"))))
            for pol, x in d["pol"].items():
                for k in keys:
                    agg.add((pol,) + k, x, d["n_cells"])
            for k in keys:
                agg.add(("DOC",) + k, {"feasible": d.get("any_feasible")}, d["n_cells"])
        if sub is None:
            read_as(model, cond, p, emit)
        for key in agg.a:
            pol, ds, gt, g = key
            if pol == "DOC":
                mets, n = agg.metrics(key)
                emit(model, cond, "pool", ds, gt, g, {"any_feasible_pct": mets.get("consistent_pct")}, n, p)
                continue
            mets, n = agg.metrics(key)
            emit(model, cond, DEC.get(pol, pol), ds, gt, g, mets, n, p)
    # unnormalized per code (starred codes) and the normal-form chain over all core documents, per model
    for model, path in UNNORM.items():
        up = _use(path)
        if not up:
            continue
        for code, v in json.load(open(up)).items():
            if code == "_chain":
                for k, val in v.items():                     # e.g. inferred_V_nfc_zwsp
                    cond, pol, step = k.split("_", 2)
                    emit(model, {"inferred": "agnostic", "given": "filled"}[cond], DEC[pol], "core", "overall", "all",
                         {f"cer_norm_{step}": val}, None, up)
                continue
            for k, val in v.items():
                if k == "liblouis":
                    if model == "sft":
                        emit("liblouis", "filled", "rule", "core", "code", code, {"cer_raw": val}, None, up)
                    continue
                cond, pol = k.split("_")
                emit(model, {"inferred": "agnostic", "given": "filled"}[cond], DEC[pol], "core", "code", code,
                     {"cer_raw": val}, None, up)
    # 2. real corpora (and the BrailleLLM few-shot pools)
    def real_rows(model, path, dataset_prefix, cond_map):
        agg = Agg()
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            cond = cond_map(d["cond"])
            if cond is None:
                continue
            for pol, x in d["pol"].items():
                for sub in (["all"] + (["f_consistent"] if d.get("f_consistent") else [])):
                    agg.add((cond, pol, d["group"], sub), x, d["n_cells"])
                    if cond == "agnostic" and "label" in x:
                        agg.a[(cond, pol, d["group"], sub)]["label_ok"] += x["label"] == d["table"]
        for (cond, pol, grp, sub), c in agg.a.items():
            mets, n = agg.metrics((cond, pol, grp, sub))
            if cond == "agnostic":
                mets["label_strict_pct"] = 100.0 * c["label_ok"] / max(1, c["docs"])
            emit(model, cond, DEC.get(pol, pol), f"{dataset_prefix}{grp}", "subset", sub, mets, n, path)
    for model, path in REAL.items():
        if _use(path):
            real_rows(model, path, "real_", lambda c: {"inferred": "agnostic", "given": "filled"}[c])
            rp = _use(REAL_REPORT[model])
            if rp:
                bl = json.load(open(rp)).get("braillellm", {})
                for cond, by in bl.items():
                    for pol, v in by.items():
                        mm = {"bleu": v.get("bleu"), "chrf": v.get("chrf++"), "bleu_nows": v.get("bleu_nows"),
                              "chrf_nows": v.get("chrf++_nows")}
                        emit(model, {"inferred": "agnostic", "given": "filled"}[cond], DEC.get(pol, pol),
                             "real_braillellm", "subset", "all", {k: x for k, x in mm.items() if x is not None}, None, rp)
        for m in ("rand", "ret"):
            fp = FEWSHOT[model].format(m=m) + "/per_doc.jsonl"
            if _use(fp):
                real_rows(model, fp, f"fewshot_{m}_", lambda c: "filled" if c == "given" else None)
    for model, d in ADAPT.items():
        if not _use(os.path.join(d, "per_doc.jsonl")):
            continue
        real_rows(model, os.path.join(d, "per_doc.jsonl"), "real_", lambda c: {"inferred": "agnostic", "given": "filled"}[c])
        rp = _use(os.path.join(d, "report.json"))
        for cond, by in (json.load(open(rp)).get("braillellm", {}) if rp else {}).items():
            for pol, v in by.items():
                mm = {"bleu": v.get("bleu"), "chrf": v.get("chrf++"), "bleu_nows": v.get("bleu_nows"), "chrf_nows": v.get("chrf++_nows")}
                emit(model, {"inferred": "agnostic", "given": "filled"}[cond], DEC.get(pol, pol), "real_braillellm", "subset", "all",
                     {k: x for k, x in mm.items() if x is not None}, None, rp)
        _given_extra(d, model, "bon_verified", emit)                    # sim / judge of the code-given verified output
    for who, d in REAL_BASELINES.items():
        pp = _use(os.path.join(d, "per_doc.jsonl"))
        if not pp:
            continue
        agg = Agg()
        for line in open(pp, encoding="utf-8"):
            x = json.loads(line)
            if x["cond"] != "given":
                continue
            for sub in ["all"] + (["f_consistent"] if x.get("f_consistent") else []):
                agg.add((x["group"], sub), x["pol"]["G"], x["n_cells"])        # the pool's only candidate
        for (grp, sub) in agg.a:
            mets, n = agg.metrics((grp, sub))
            emit(who, "filled", "rule", f"real_{grp}", "subset", sub, mets, n, pp)
        rp = _use(os.path.join(d, "report.json"))
        v = (json.load(open(rp)).get("braillellm", {}).get("given", {}).get("G") if rp else None) or {}
        mm = {"bleu": v.get("bleu"), "chrf": v.get("chrf++"), "bleu_nows": v.get("bleu_nows"), "chrf_nows": v.get("chrf++_nows")}
        if any(x is not None for x in mm.values()):
            emit(who, "filled", "rule", "real_braillellm", "subset", "all", {k: x for k, x in mm.items() if x is not None}, None, rp)
        _given_extra(d, who, "rule", emit)
    # 3. math
    for model, path in MATH.items():
        if not _use(path):
            continue
        for pol, by in json.load(open(path)).items():
            for part, v in by.items():
                if isinstance(v, dict) and v.get("docs"):
                    emit(model, "agnostic", DEC.get(pol, pol), f"math_{part}", "overall", "all",
                         {"exact_pct": 100.0 * v["exact"] / v["docs"], "cer_canonical": 100.0 * v["e"] / max(1, v["n"]),
                          "consistent_pct": 100.0 * v.get("consistent", 0) / v["docs"], "fail": v.get("fail", 0)}, v["docs"], path)
    mb = _use(MATH_BASE)
    if mb:
        for sysname, by in json.load(open(mb))["systems"].items():
            name = {"model": "sft"}.get(sysname, f"math:{sysname}")
            for part, v in by.items():
                emit(name, "agnostic" if name == "sft" else "filled", "greedy" if name == "sft" else "rule", f"math_{part}",
                     "overall", "all", {"exact_pct": 100.0 * v["exact"] / v["docs"], "cer_canonical": 100.0 * v["e"] / max(1, v["n"]),
                                        "fail": v.get("fail", 0)}, v["docs"], mb)
    # 4. Korean plain pairs
    for model, path in KODISC.items():
        if not _use(path):
            continue
        for split, v in json.load(open(path)).items():
            base = {"identical_pct": v["identical_%"], "ceiling_pct": v["ceiling_%"]}
            emit(model, "agnostic", "pool", "kodisc_plain", "split", split, base, v["pairs"], path)
            for pol in ("G", "V"):
                emit(model, "agnostic", DEC[pol], "kodisc_plain", "split", split, {"code_acc_pct": v[f"code_acc_{pol}_%"]},
                     v["pairs"], path)
    # 5. baselines on core
    lp = _use(LIBLOUIS)
    if lp:
        agg = Agg()
        for line in open(lp, encoding="utf-8"):
            d = json.loads(line)
            if d["set"] != "core":
                continue
            x = {"e": d["e"], "n": d["n"], "e_st": d["e_st"], "n_st": d["n_st"], "bc": d["bc"], "bc_st": d["bc_st"],
                 "exact": d["exact"], "feasible": d["consistent"]}
            for k in (("overall", "all"), ("family", family(inv, d["table"])), ("record_group", d["group"]), ("code", d["table"]),
                      ("language", inv[d["table"]]["language"])):
                agg.add(k, x, d["cells"])
        for k in agg.a:
            mets, n = agg.metrics(k)
            emit("liblouis", "filled", "rule", "core", k[0], k[1], mets, n, lp)
    gp = _use(GPT)
    if gp:
        agg = Agg()
        for line in open(gp, encoding="utf-8"):
            d = json.loads(line)
            x = {"e": d["e"], "n": d["n"], "e_st": d["e_st"], "n_st": d["n_st"], "bc": d["bc"], "exact": d["exact"],
                 "feasible": d["kind"] != "inconsistent", "kind": d["kind"]}
            for k in (("overall", "all"), ("family", family(inv, d["table"])), ("code", d["table"]),
                      ("language", inv[d["table"]]["language"])):
                agg.add(k, x, d["cells"])
            agg.a[("overall", "all")]["empty"] += bool(d.get("empty"))
        for k in agg.a:
            mets, n = agg.metrics(k)
            if k == ("overall", "all"):
                mets["empty_outputs_pct"] = 100.0 * agg.a[k]["empty"] / max(1, n)
            emit("gpt-5.5", "none", "fewshot", "core_1k", k[0], k[1], mets, n, gp)
    # 6. sim (bge-m3, x 100) and judge (Qwen2.5-72B-AWQ, 1-5): per-doc values joined with rows_<name>.jsonl
    for model, d in EXTRA.items():
        for name, (who, cond, dec, dataset) in _extra_names(model).items():
            rows_path = os.path.join(d, f"rows_{name}.jsonl")
            if not os.path.isfile(rows_path):
                continue
            meta = {}
            for line in open(rows_path, encoding="utf-8"):
                r = json.loads(line)
                meta[r["id"]] = r
            for met in ("sim", "judge"):
                p = _use(os.path.join(d, f"{name}.{met}.json.rows.jsonl"))
                if not p:
                    continue
                acc = collections.defaultdict(list)
                for line in open(p, encoding="utf-8"):
                    x = json.loads(line)
                    if x.get(met) is None:
                        continue
                    r = meta[x["id"]]
                    v = 100.0 * x[met] if met == "sim" else x[met]
                    if met == "judge" and not (r.get("hyp_text") or "").strip():
                        v = 1.0                              # an output with no text is scored 1 (ubt.metrics.extra)
                    if dataset == "real":                    # one pool, three corpora: group = corpus
                        acc[(f"real_{r['group']}", "subset", "all")].append(v)
                        continue
                    for k in (("overall", "all"), ("family", r["group"]), ("kind", r["set"]), ("code", r["code"])):
                        acc[(dataset,) + k].append(v)
                for (ds, gt, g), vs in acc.items():
                    emit(who, cond, dec, ds, gt, g, {met: sum(vs) / len(vs)}, len(vs), p)
                if met == "judge" and dataset != "real" and acc.get((dataset, "overall", "all")):
                    vs = acc[(dataset, "overall", "all")]          # expected score rounded to the nearest integer (x.5 up)
                    cnt = collections.Counter(min(5, max(1, int(v + 0.5))) for v in vs)
                    for sc in range(1, 6):
                        emit(who, cond, dec, dataset, "judge_score", str(sc), {"share_pct": 100.0 * cnt[sc] / len(vs)}, len(vs), p)

    # 7. Table 1 confidence intervals (bootstrap over documents)
    cp = _use(CI)
    if cp:
        for row, by in json.load(open(cp))["rows"].items():
            who, cond, dec, ds = CI_ROWS[row]
            for met, v in by.items():
                emit(who, cond, dec, ds, "overall", "all", {f"{CI_MET[met]}_ci95_lo": v["lo"], f"{CI_MET[met]}_ci95_hi": v["hi"],
                                                          f"{CI_MET[met]}_ci95_half": v["half"]}, v["docs"], cp)
    # 7b. error analysis of the final model: error shares per family and kind, and docs exact under a wrong code
    xp = _use(ERRORS)
    if xp:
        for fam, v in json.load(open(xp))["families"].items():
            mets = {"cer": v["cer"], "code_only_pct": v["code_only_pct"]}
            mets.update({"err_share_" + k.replace(":", "_"): x for k, x in v["segs_pct"].items()})
            emit("rlvr", "agnostic", "bon_verified", "core", "error_family", fam, mets, v["docs"], xp)
    # 8. braille edition of the paper (appendix): units certified / exact, error cells, sentence check
    for model, path in EDITION.items():
        ep = _use(path)
        if not ep:
            continue
        e = json.load(open(ep))
        u, sc = e["units"], e["sentence_check"]
        emit(model, "filled", "bon_verified", "braille_edition", "units", "all",
             {k: u[k] for k in ("certified", "exact", "certified_and_exact", "certified_not_exact", "cells", "err_cells", "err_cells_pct",
                                "unc_cells", "err_cells_outside_uncertified")}, u["units"], ep)
        emit(model, "filled", "bon_verified", "braille_edition", "sentences", "all",
             {"verified": sc["verified"], "exact": sc["exact"], "cer_verified_selection": sc["cer_verified_selection"],
              "cer_greedy": sc["cer_greedy"]}, sc["sentences"], ep)
    # ---- write
    cols = ["model", "condition", "decoding", "dataset", "group_type", "group", "metric", "value", "docs", "source"]
    import gzip  # noqa: PLC0415
    if os.path.exists(os.path.join(a.out, "tracking_long.csv")):
        os.remove(os.path.join(a.out, "tracking_long.csv"))
    with gzip.open(os.path.join(a.out, "tracking_long.csv.gz"), "wt", newline="", encoding="utf-8", compresslevel=9) as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: tuple(str(r[c]) for c in cols[:7])):
            w.writerow(r)
    json.dump({"generated_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"), "sources": SOURCES},
              open(os.path.join(a.out, "sources.json"), "w"), indent=1)
    write_summary(a.out, rows, have)
    print(f"{len(rows)} rows, {len(SOURCES)} sources -> {a.out}")


def write_summary(out, rows, have):
    idx = {}
    for r in rows:
        idx[(r["model"], r["condition"], r["decoding"], r["dataset"], r["group_type"], r["group"], r["metric"])] = r["value"]
    g = lambda *k: idx.get(k)                                     # noqa: E731
    f = lambda v, p=2: "–" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))  # noqa: E731
    L = ["# Internal tracking table (not for the submission)", "",
         f"Generated {datetime.datetime.utcnow().isoformat(timespec='seconds')}Z by `scripts/paper/tracking_table.py` "
         "(code repo). All CER/bCER in % in the scoring normal form unless the metric says `_st` / `_raw`; "
         "`bon_verified` = the paper's verified best-of-20 (λ=2). Full data: `tracking_long.csv.gz`; sources and hashes: "
         "`sources.json`.", ""]
    # coverage
    datasets = ["core", "mixed_k2", "mixed_k3", "mixed_k4", "intra", "k2p", "k3p", "k4p", "noise_p0", "noise_p0.5", "noise_p1", "noise_p2",
                "zeroshot_single", "zeroshot_inter", "math_standalone", "math_embedded", "real_bible", "real_braillellm",
                "real_vision", "fewshot_rand_braillellm", "fewshot_ret_braillellm", "kodisc_plain"]
    L += ["## 1. Coverage (✓ measured, · pending)", "", "| dataset | " + " | ".join(f"{m} {c}" for m in MODELS for c in CONDITIONS) + " |",
          "|---|" + "---|" * (len(MODELS) * len(CONDITIONS))]
    for ds in datasets:
        L.append(f"| {ds} | " + " | ".join("✓" if (m, c, ds) in have else "·" for m in MODELS for c in CONDITIONS) + " |")
    L += ["", "Not in the table yet (and why):", ""] + [f"- **{k}**: {v}" for k, v in PENDING] + [""]
    # core headline
    L += ["## 2. Core (20,071 documents), overall", "",
          "| model | condition | decoding | bCER | CER | CER S/T-sens. | exact % | code-ID strict | conf | consistent % | truncated % |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in MODELS:
        for c in ("agnostic", "filled"):
            for dec in ("greedy", "bon_lp", "bon_verified", "bon_closest", "oracle"):
                if g(m, c, dec, "core", "overall", "all", "cer") is None:
                    continue
                L.append(f"| {m} | {c} | {dec} | " + " | ".join(f(g(m, c, dec, "core", "overall", "all", k)) for k in
                         ("bcer", "cer", "cer_st", "exact_pct", "strict_pct", "conf_pct", "consistent_pct", "truncated_pct")) + " |")
    for name, cond, dec, ds in (("liblouis", "filled", "rule", "core"), ("gpt-5.5", "none", "fewshot", "core_1k")):
        if g(name, cond, dec, ds, "overall", "all", "cer") is not None:
            L.append(f"| {name} | {cond} | {dec} ({ds}) | " + " | ".join(f(g(name, cond, dec, ds, "overall", "all", k)) for k in
                     ("bcer", "cer", "cer_st", "exact_pct", "strict_pct", "conf_pct", "consistent_pct", "truncated_pct")) + " |")
    chain = [(m, c) for m in MODELS for c in ("agnostic", "filled") if g(m, c, "bon_verified", "core", "overall", "all", "cer_norm_none") is not None]
    if chain:
        L += ["", "Normal-form chain (paper Table 8, last row), bon_verified CER: none → (i) NFC → (ii) + U+200B removed → (iii) + S/T fold: " + "; ".join(
              f"{m} {c} " + " → ".join(f(g(m, c, "bon_verified", "core", "overall", "all", f"cer_norm_{k}")) for k in ("none", "nfc", "nfc_zwsp", "all"))
              for m, c in chain)]
    # training stages
    def v1(m, c, dec, ds, gt, grp, met, p=2):
        return f(g(m, c, dec, ds, gt, grp, met), p)
    stage = [("core bCER, greedy / verified", lambda m: f"{v1(m,'agnostic','greedy','core','overall','all','bcer')} / {v1(m,'agnostic','bon_verified','core','overall','all','bcer')}"),
             ("core CER, greedy / verified", lambda m: f"{v1(m,'agnostic','greedy','core','overall','all','cer')} / {v1(m,'agnostic','bon_verified','core','overall','all','cer')}"),
             ("core CER, code given, greedy / verified", lambda m: f"{v1(m,'filled','greedy','core','overall','all','cer')} / {v1(m,'filled','bon_verified','core','overall','all','cer')}"),
             ("core oracle-in-20 CER", lambda m: v1(m, "agnostic", "oracle", "core", "overall", "all", "cer")),
             ("code-ID strict / conf (verified)", lambda m: f"{v1(m,'agnostic','bon_verified','core','overall','all','strict_pct',1)} / {v1(m,'agnostic','bon_verified','core','overall','all','conf_pct',1)}"),
             ("consistent % greedy / verified", lambda m: f"{v1(m,'agnostic','greedy','core','overall','all','consistent_pct',1)} / {v1(m,'agnostic','bon_verified','core','overall','all','consistent_pct',1)}")]
    stage += [(f"{fam} CER (verified)", (lambda fam: lambda m: v1(m, "agnostic", "bon_verified", "core", "family", fam, "cer"))(fam))
              for fam in ("other", "Japanese", "Chinese", "contracted", "Konkani+Yakut", "Korean")]
    stage += [(f"{ds} CER (verified)", (lambda ds: lambda m: v1(m, "agnostic", "bon_verified", ds, "overall", "all", "cer"))(ds))
              for ds in ("mixed_k2", "mixed_k3", "mixed_k4", "intra", "k2p", "k3p", "k4p", "noise_p1")]
    stage += [("zero-shot, six codes / 360 mixed (verified; paper subsets)", lambda m: f"{v1(m,'agnostic','bon_verified','zeroshot_single','subset','six_codes','cer')} / {v1(m,'agnostic','bon_verified','zeroshot_inter','subset','mixed_360','cer')}")]
    stage += [("math exact % standalone / embedded (verified)", lambda m: f"{v1(m,'agnostic','bon_verified','math_standalone','overall','all','exact_pct',1)} / {v1(m,'agnostic','bon_verified','math_embedded','overall','all','exact_pct',1)}"),
              ("Bible CER given / inferred (verified)", lambda m: f"{v1(m,'filled','bon_verified','real_bible','subset','all','cer')} / {v1(m,'agnostic','bon_verified','real_bible','subset','all','cer')}"),
              ("BrailleLLM CER no-space, given (verified)", lambda m: v1(m, "filled", "bon_verified", "real_braillellm", "subset", "all", "cer_nows")),
              ("BrailleLLM few-shot ret, CER no-space (verified)", lambda m: v1(m, "filled", "bon_verified", "fewshot_ret_braillellm", "subset", "all", "cer_nows")),
              ("Vision-Braille CER given (verified)", lambda m: v1(m, "filled", "bon_verified", "real_vision", "subset", "all", "cer")),
              ("Korean pairs code acc greedy / verified", lambda m: f"{v1(m,'agnostic','greedy','kodisc_plain','split','all','code_acc_pct',1)} / {v1(m,'agnostic','bon_verified','kodisc_plain','split','all','code_acc_pct',1)}")]
    def above(m):
        vals = [(r["group"], r["value"]) for r in rows if r["model"] == m and r["condition"] == "agnostic" and r["decoding"] == "bon_verified"
                and r["dataset"] == "core" and r["group_type"] == "code" and r["metric"] == "cer"]
        if not vals:
            return "–"
        return str(sum((v > 5.0) if c.startswith("zh") else (v >= 1.0) for c, v in vals))
    stage.append(("codes above their bar (verified)", above))
    L += ["", "## 2b. Training stages (code inferred unless noted) — the paper's stage table reads these", "",
          "| metric | " + " | ".join(MODELS) + " |", "|---|" + "---:|" * len(MODELS)]
    for name, fn in stage:
        L.append(f"| {name} | " + " | ".join(fn(m) for m in MODELS) + " |")
    # Table 1 intervals, backbones, subset trials, adaptation runs
    L += ["", "## 2c. Table 1 intervals, backbones, trials and adaptation runs", "",
          "Table 1, 95% bootstrap interval half-widths (scripts/paper/table1_ci.py; value ± half):", ""]
    for row, (who, cond, dec, ds) in CI_ROWS.items():
        if g(who, cond, dec, ds, "overall", "all", "cer_ci95_half") is None:
            continue
        L.append(f"- {row}: " + ", ".join(f"{k} {f(g(who, cond, dec, ds, 'overall', 'all', m_))} ± {f(g(who, cond, dec, ds, 'overall', 'all', m_ + '_ci95_half'), 3)}"
                                          for k, m_ in CI_MET.items() if g(who, cond, dec, ds, "overall", "all", m_ + "_ci95_half") is not None))
    bb = [m for m in BACKBONES if g(m, "agnostic", "bon_verified", "core", "overall", "all", "cer") is not None]
    if bb:
        L += ["", "Backbones (core, code inferred): bCER / CER / strict / conf / consistent % / sim / judge", "",
              "| model | decoding | docs | bCER | CER | strict | conf | consistent % | sim | judge |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for m in ["rlvr"] + bb:
            for dec in ("greedy", "bon_verified", "oracle"):
                n_ = next((r["docs"] for r in rows if r["model"] == m and r["condition"] == "agnostic" and r["decoding"] == dec
                           and r["dataset"] == "core" and r["group_type"] == "overall" and r["metric"] == "cer"), None)
                L.append(f"| {m} | {dec} | {n_} | " + " | ".join(f(g(m, "agnostic", dec, "core", "overall", "all", k), p) for k, p in
                         (("bcer", 2), ("cer", 2), ("strict_pct", 1), ("conf_pct", 1), ("consistent_pct", 1), ("sim", 1), ("judge", 2))) + " |")
    for sub, (_ids, members) in TRIALS.items():
        have_m = [m for m, _p in members if g(m, "agnostic", "bon_verified", sub, "overall", "all", "cer") is not None]
        if have_m:
            L += ["", f"{sub}: CER greedy / bon_lp / bon_verified / bon_closest / oracle, consistent % (verified), per family CER (verified)", ""]
            for m in have_m:
                fam = ", ".join(f"{fm} {f(g(m, 'agnostic', 'bon_verified', sub, 'family', fm, 'cer'))}" for fm in
                                ("other", "Japanese", "Chinese", "contracted", "Korean") if g(m, "agnostic", "bon_verified", sub, "family", fm, "cer") is not None)
                L.append(f"- {m}: " + " / ".join(f(g(m, "agnostic", d_, sub, "overall", "all", "cer")) for d_ in
                         ("greedy", "bon_lp", "bon_verified", "bon_closest", "oracle")) +
                         f", consistent {f(g(m, 'agnostic', 'bon_verified', sub, 'overall', 'all', 'consistent_pct'), 1)}%; {fam}")
    ad = [(m, ds) for m in ADAPT for ds in ("real_braillellm", "real_vision") if g(m, "filled", "bon_verified", ds, "subset", "all", "cer") is not None]
    if ad:
        L += ["", "Adaptation on a real corpus's training split (code given), against the final model on the same documents:", "",
              "| corpus | model | bCER | bCER no-space | CER | CER no-space | greedy CER no-space | BLEU V (G) | chrF++ V (G) | sim | consistent % |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for m, ds in ad:
            for mm in ("rlvr", m):
                L.append(f"| {ds[5:]} | {mm} | " + " | ".join(f(g(mm, "filled", "bon_verified", ds, "subset", "all", k)) for k in ("bcer", "bcer_nows", "cer", "cer_nows"))
                         + f" | {f(g(mm, 'filled', 'greedy', ds, 'subset', 'all', 'cer_nows'))}"
                         + f" | {f(g(mm, 'filled', 'bon_verified', ds, 'subset', 'all', 'bleu'))} ({f(g(mm, 'filled', 'greedy', ds, 'subset', 'all', 'bleu'))})"
                         + f" | {f(g(mm, 'filled', 'bon_verified', ds, 'subset', 'all', 'chrf'))} ({f(g(mm, 'filled', 'greedy', ds, 'subset', 'all', 'chrf'))})"
                         + f" | {f(g(mm, 'filled', 'bon_verified', ds, 'subset', 'all', 'sim'), 1)} | {f(g(mm, 'filled', 'bon_verified', ds, 'subset', 'all', 'consistent_pct'), 1)} |")
    for m in EDITION:
        u = lambda k: g(m, "filled", "bon_verified", "braille_edition", "units", "all", k)  # noqa: E731
        if u("certified") is not None:
            sc = lambda k: g(m, "filled", "bon_verified", "braille_edition", "sentences", "all", k)  # noqa: E731
            L += ["", f"Braille edition ({m}): units certified {u('certified')}, exact {u('exact')}, certified and exact "
                  f"{u('certified_and_exact')}; error cells {u('err_cells')} ({f(u('err_cells_pct'))}%), of them outside uncertified "
                  f"units {u('err_cells_outside_uncertified')}; uncertified cells {u('unc_cells')}; sentences verified {sc('verified')}, "
                  f"CER verified selection {f(sc('cer_verified_selection'))} / greedy {f(sc('cer_greedy'))}"]
    # families
    fams = ["other", "Japanese", "Chinese", "contracted", "Konkani+Yakut", "Korean"]
    L += ["", "## 3. Core by Table 1B family — CER (bon_verified / greedy / oracle)", "",
          "| family | " + " | ".join(f"{m} {c}" for m in MODELS for c in ("agnostic", "filled")) + " | LibLouis | GPT-5.5 |",
          "|---|" + "---:|" * (len(MODELS) * 2 + 2)]
    for fam in fams:
        cells = []
        for m in MODELS:
            for c in ("agnostic", "filled"):
                v, gr, o = (g(m, c, d, "core", "family", fam, "cer") for d in ("bon_verified", "greedy", "oracle"))
                cells.append("–" if v is None else f"{f(v)} / {f(gr)} / {f(o)}")
        cells.append(f(g("liblouis", "filled", "rule", "core", "family", fam, "cer")))
        cells.append(f(g("gpt-5.5", "none", "fewshot", "core_1k", "family", fam, "cer")))
        L.append(f"| {fam} | " + " | ".join(cells) + " |")
    # per language
    langs = sorted({r["group"] for r in rows if r["dataset"] == "core" and r["group_type"] == "language"})
    L += ["", "## 3b. Core by language (inventory ISO code) — CER bon_verified (greedy); docs", "",
          "| language | docs | " + " | ".join(f"{m} {c}" for m in MODELS for c in ("agnostic", "filled")) + " | LibLouis | GPT-5.5 |",
          "|---|---:|" + "---:|" * (len(MODELS) * 2 + 2)]
    for lg in langs:
        docs = next((r["docs"] for r in rows if r["dataset"] == "core" and r["group_type"] == "language" and r["group"] == lg
                     and r["model"] == "sft" and r["decoding"] == "bon_verified"), None)
        cells = []
        for m in MODELS:
            for c in ("agnostic", "filled"):
                v = g(m, c, "bon_verified", "core", "language", lg, "cer")
                cells.append("–" if v is None else f"{f(v)} ({f(g(m, c, 'greedy', 'core', 'language', lg, 'cer'))})")
        cells.append(f(g("liblouis", "filled", "rule", "core", "language", lg, "cer")))
        cells.append(f(g("gpt-5.5", "none", "fewshot", "core_1k", "language", lg, "cer")))
        L.append(f"| {lg} | {docs if docs is not None else '–'} | " + " | ".join(cells) + " |")
    # datasets
    L += ["", "## 4. Other datasets — CER (bon_verified / greedy)", "", "| dataset | " + " | ".join(f"{m} {c}" for m in MODELS for c in ("agnostic", "filled")) + " |",
          "|---|" + "---:|" * (len(MODELS) * 2)]
    for ds, gt, grp in [(x, "overall", "all") for x in datasets[1:12]] + [("zeroshot_single", "subset", "six_codes"),
                                                                           ("zeroshot_inter", "subset", "mixed_360")]:
        cells = []
        for m in MODELS:
            for c in ("agnostic", "filled"):
                v, gr = g(m, c, "bon_verified", ds, gt, grp, "cer"), g(m, c, "greedy", ds, gt, grp, "cer")
                cells.append("–" if v is None else f"{f(v)} / {f(gr)}")
        L.append(f"| {ds}{'' if grp == 'all' else ' (' + grp + ', paper)'} | " + " | ".join(cells) + " |")
    # math
    L += ["", "## 5. Mathematics (canonical form): exact % / canonical CER", "", "| system | decoding | standalone | embedded |", "|---|---|---:|---:|"]
    for m in list(MODELS) + ["math:liblouis", "math:umml", "math:a8m", "math:rule"]:
        for dec in ("greedy", "bon_verified", "oracle", "rule"):
            s_, e_ = (g(m, "agnostic" if m in MODELS else "filled", dec, f"math_{p}", "overall", "all", "exact_pct") for p in ("standalone", "embedded"))
            if s_ is None and e_ is None:
                continue
            cs, ce = (g(m, "agnostic" if m in MODELS else "filled", dec, f"math_{p}", "overall", "all", "cer_canonical") for p in ("standalone", "embedded"))
            L.append(f"| {m} | {dec} | {f(s_,1)} / {f(cs)} | {f(e_,1)} / {f(ce)} |")
    # real corpora
    L += ["", "## 6. Real corpora — bon_verified CER / consistent % (greedy CER)", "", "| corpus | subset | " + " | ".join(f"{m} {c}" for m in MODELS for c in ("agnostic", "filled")) + " |",
          "|---|---|" + "---:|" * (len(MODELS) * 2)]
    for ds in ("real_bible", "real_braillellm", "real_vision", "fewshot_rand_braillellm", "fewshot_ret_braillellm"):
        for sub in ("all", "f_consistent"):
            cells = []
            for m in MODELS:
                for c in ("agnostic", "filled"):
                    v = g(m, c, "bon_verified", ds, "subset", sub, "cer")
                    n_ = next((r["docs"] for r in rows if r["model"] == m and r["condition"] == c and r["decoding"] == "bon_verified"
                               and r["dataset"] == ds and r["group"] == sub and r["metric"] == "cer"), None)
                    cells.append("–" if v is None else f"{f(v)} / {f(g(m, c, 'bon_verified', ds, 'subset', sub, 'consistent_pct'),1)} ({f(g(m, c, 'greedy', ds, 'subset', sub, 'cer'))}) n={n_}")
            if any(x != "–" for x in cells):
                L.append(f"| {ds} | {sub} | " + " | ".join(cells) + " |")
    L += ["", "BrailleLLM whitespace-free CER (their metric), bon_verified: " + ", ".join(
          f"{m} {c} {f(g(m, c, 'bon_verified', ds, 'subset', 'all', 'cer_nows'))} ({ds})" for m in MODELS for c in ("agnostic", "filled")
          for ds in ("real_braillellm", "fewshot_rand_braillellm", "fewshot_ret_braillellm") if g(m, c, "bon_verified", ds, "subset", "all", "cer_nows") is not None) + "; BrailleLLM specialist 5.42."]
    for who in REAL_BASELINES:
        cells = [f"{ds[5:]} {sub}: bCER {f(g(who, 'filled', 'rule', ds, 'subset', sub, 'bcer'))}, CER {f(g(who, 'filled', 'rule', ds, 'subset', sub, 'cer'))}, "
                 f"no-space bCER {f(g(who, 'filled', 'rule', ds, 'subset', sub, 'bcer_nows'))} / CER {f(g(who, 'filled', 'rule', ds, 'subset', sub, 'cer_nows'))}, "
                 f"consistent {f(g(who, 'filled', 'rule', ds, 'subset', sub, 'consistent_pct'), 1)}%"
                 for ds in ("real_bible", "real_braillellm", "real_vision") for sub in ("all", "f_consistent")
                 if g(who, "filled", "rule", ds, "subset", sub, "cer") is not None]
        if cells:
            L += ["", f"Baseline {who} (code given) on the real corpora: " + "; ".join(cells) +
                  f"; BrailleLLM BLEU {f(g(who, 'filled', 'rule', 'real_braillellm', 'subset', 'all', 'bleu'))}, chrF++ {f(g(who, 'filled', 'rule', 'real_braillellm', 'subset', 'all', 'chrf'))}."]
    # kodisc
    L += ["", "## 7. Korean plain-sentence pairs (2024 vs 2006)", "", "| model | split | identical % | ceiling % | code acc greedy | code acc verified |", "|---|---|---:|---:|---:|---:|"]
    for m in MODELS:
        for split in ("digit-free", "digit-bearing", "all", "latin-free", "latin-free digit-free", "latin-free digit-bearing"):
            v = g(m, "agnostic", "pool", "kodisc_plain", "split", split, "identical_pct")
            if v is None:
                continue
            L.append(f"| {m} | {split} | {f(v,1)} | {f(g(m, 'agnostic', 'pool', 'kodisc_plain', 'split', split, 'ceiling_pct'),1)} | "
                     f"{f(g(m, 'agnostic', 'greedy', 'kodisc_plain', 'split', split, 'code_acc_pct'),1)} | {f(g(m, 'agnostic', 'bon_verified', 'kodisc_plain', 'split', split, 'code_acc_pct'),1)} |")
    # sim / judge
    systems = [(m, c, dec, "core") for m in MODELS for c, dec in (("agnostic", "bon_verified"), ("filled", "bon_verified"), ("agnostic", "greedy"))] \
        + [("liblouis", "filled", "rule", "core"), ("gpt-5.5", "none", "fewshot", "core_1k")]
    systems = [s for s in systems if g(*s, "overall", "all", "sim") is not None or g(*s, "overall", "all", "judge") is not None]
    if systems:
        groups = [("overall", "all"), ("kind", "inconsistent"), ("kind", "consistent_misreading"), ("kind", "wrong_code"), ("kind", "correct")] \
            + [("family", x) for x in ("other", "Japanese", "Chinese", "contracted", "Konkani+Yakut", "Korean")]
        L += ["", "## 7b. sim (bge-m3, 0-100) / judge (Qwen2.5-72B-AWQ, 1-5)", "",
              "| system | " + " | ".join(grp for _, grp in groups) + " |", "|---|" + "---:|" * len(groups)]
        for s in systems:
            L.append(f"| {s[0]} {s[1]} {s[2]} ({s[3]}) | " + " | ".join(
                f"{f(g(*s, gt, grp, 'sim'), 1)} / {f(g(*s, gt, grp, 'judge'), 2)}" for gt, grp in groups) + " |")
        jd = [s for s in systems + [(m, "agnostic", "bon_verified", "core") for m in BACKBONES]
              if g(*s, "judge_score", "5", "share_pct") is not None]
        if jd:
            L += ["", "## 7c. judge score distribution (% of documents, expected score rounded to the nearest integer; appendix "
                  "fig:judge-dist)", "", "| system | 1 | 2 | 3 | 4 | 5 | ≥3 | ≥4 |", "|---|" + "---:|" * 7]
            for s in jd:
                sh = [g(*s, "judge_score", str(k), "share_pct") or 0.0 for k in range(1, 6)]
                L.append(f"| {s[0]} {s[1]} {s[2]} ({s[3]}) | " + " | ".join(f(x, 2) for x in sh)
                         + f" | {f(sum(sh[2:]), 1)} | {f(sum(sh[3:]), 1)} |")
        segs = ["I_characters", "I_other", "C_characters", "C_case", "C_punct", "C_digits", "W_characters", "W_other"]
        ef = lambda fam, met: g("rlvr", "agnostic", "bon_verified", "core", "error_family", fam, met)  # noqa: E731
        fams = [x for x in ("all", "Chinese", "Japanese", "Indic", "Korean", "contracted", "other") if ef(x, "cer") is not None]
        if fams:
            L += ["", "## 7d. error analysis, rlvr agnostic bon_verified (core): % of each family's character errors by kind "
                  "(I inconsistent, C consistent misreading, W wrong code) and what is wrong; code only = % of documents with "
                  "the text exact under a wrong code (appendix fig:error-families; Indic = the 15 Bharati codes, Yakut in other)", "",
                  "| family | CER | code only | " + " | ".join(segs) + " |", "|---|" + "---:|" * (len(segs) + 2)]
            for fam in fams:
                L.append(f"| {fam} | {f(ef(fam, 'cer'), 2)} | {f(ef(fam, 'code_only_pct'), 1)} | "
                         + " | ".join(f(ef(fam, 'err_share_' + k) or 0.0, 1) for k in segs) + " |")
        real = [(m, c) for m in MODELS for c in ("filled", "agnostic") if g(m, c, "bon_verified", "real_bible", "subset", "all", "sim") is not None]
        if real:
            L += ["", "Real corpora, sim of the verified output: " + "; ".join(
                f"{m} {c} " + ", ".join(f"{ds[5:]} {f(g(m, c, 'bon_verified', ds, 'subset', 'all', 'sim'), 1)}"
                                        for ds in ("real_bible", "real_braillellm", "real_vision")) for m, c in real)]
    # per code
    codes = sorted({r["group"] for r in rows if r["dataset"] == "core" and r["group_type"] == "code"})
    L += ["", "## 8. Per code (core) — CER: bon_verified (oracle) / greedy", "",
          "| code | " + " | ".join(f"{m} {c}" for m in MODELS for c in ("agnostic", "filled")) + " | LibLouis | GPT-5.5 | raw agn V (" + " / ".join(MODELS) + ") |",
          "|---|" + "---:|" * (len(MODELS) * 2 + 3)]
    for code in codes:
        cells = []
        for m in MODELS:
            for c in ("agnostic", "filled"):
                v = g(m, c, "bon_verified", "core", "code", code, "cer")
                cells.append("–" if v is None else f"{f(v)} ({f(g(m, c, 'oracle', 'core', 'code', code, 'cer'))}) / {f(g(m, c, 'greedy', 'core', 'code', code, 'cer'))}")
        cells.append(f(g("liblouis", "filled", "rule", "core", "code", code, "cer")))
        cells.append(f(g("gpt-5.5", "none", "fewshot", "core_1k", "code", code, "cer")))
        raws = [g(m, "agnostic", "bon_verified", "core", "code", code, "cer_raw") for m in MODELS]
        cells.append("–" if all(x is None for x in raws) else " / ".join(f(x) for x in raws))
        L.append(f"| {code} | " + " | ".join(cells) + " |")
    # per-code deltas: top-up - SFT, RLVR - top-up
    for new, old in (("topup", "sft"), ("rlvr", "topup")):
        d = [(code, g(new, "agnostic", "bon_verified", "core", "code", code, "cer"), g(old, "agnostic", "bon_verified", "core", "code", code, "cer"))
             for code in codes]
        d = [(c, a_ - b_, a_, b_) for c, a_, b_ in d if a_ is not None and b_ is not None]
        if not d:
            continue
        d.sort(key=lambda t: t[1])
        L += ["", f"## 8b. Per code, {new} − {old} (verified CER, code inferred): 12 largest gains, and every code worse by ≥ 0.2 points", "",
              f"| code | {old} | {new} | Δ |", "|---|---:|---:|---:|"]
        for c, dd, a_, b_ in d[:12] + [t for t in d if t[1] >= 0.2][::-1]:
            L.append(f"| {c} | {f(b_)} | {f(a_)} | {dd:+.2f} |")
    zs = sorted({r["group"] for r in rows if r["dataset"] == "zeroshot_single" and r["group_type"] == "code"})
    L += ["", "## 9. Held-out codes (zeroshot single) — CER: bon_verified | greedy", "", "| code | " + " | ".join(MODELS) + " |", "|---|" + "---:|" * len(MODELS)]
    for code in zs:
        L.append(f"| {code} | " + " | ".join("–" if g(m, "agnostic", "bon_verified", "zeroshot_single", "code", code, "cer") is None else
                 f"{f(g(m, 'agnostic', 'bon_verified', 'zeroshot_single', 'code', code, 'cer'))} \\| {f(g(m, 'agnostic', 'greedy', 'zeroshot_single', 'code', code, 'cer'))}" for m in MODELS) + " |")
    # read-as: held-out codes and the core codes least often read as themselves
    ra = [r for r in rows if r["group_type"] == "read_as" and r["decoding"] == "bon_verified" and r["condition"] == "agnostic"]
    if ra:
        L += ["", "## 10. Read as (code carried by the verified output), held-out codes and the 15 core codes read under their own code least often", "",
              "| model | set | gold code | read as (share %) |", "|---|---|---|---|"]
        by = collections.defaultdict(list)
        for r in ra:
            gold, lab = r["group"].split(" -> ")
            by[(r["model"], r["dataset"], gold)].append((lab, r["value"]))
        own = {k: dict(v).get(k[2], 0.0) for k, v in by.items()}
        for m in MODELS:
            ks = [k for k in by if k[0] == m and k[1] == "zeroshot_single"]
            ks += sorted([k for k in by if k[0] == m and k[1] == "core"], key=lambda k: own[k])[:15]
            for k in ks:
                L.append(f"| {m} | {k[1]} | {k[2]} | " + ", ".join(f"{lab} ({v:.0f})" for lab, v in sorted(by[k], key=lambda t: -t[1])) + " |")
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
