#!/usr/bin/env python3
"""Score the real-braille corpora with verified best-of-20 (the paper's real-corpora table). CPU only.

Corpora: the embossed Bible (3,000 verses, en_US.tbl), the BrailleLLM test set (1,000, zhcn-cbs.ctb) and
Vision-Braille (1,000, zhcn-g1.ctb). The sealed NIKL test (group "nikl") is scored the same way from its own
directory, and its outputs stay local.

  python scripts/real_corpora/real_corpora_score.py [--dir work/real_corpora/run]
      -> <dir>/report.json, <dir>/report.md, <dir>/per_doc.jsonl

Inputs:
  <dir>/real.jsonl            scripts/real_corpora/build_real_corpora.py rows; f_consistent = forward(gold code, text) == cells
  <dir>/gen_<cond>.<i>.jsonl  scripts/eval/bon_gen.py pools, cands[0] greedy + 19 samples; cond = inferred | given (gold
                              code prefilled)

Scoring reuses scripts/eval/bon_eval.py: candidate consistency under the data's engine, the policies (G greedy, LP max
log p, V verified lambda=2, Vinf, O oracle-in-20) and the per-document metrics in the scoring normal form; bCER aligns
the reference to the corpus cells, with a proportional fallback counted per corpus. Added here:
  - every metric on the f_consistent subset, and consistent vs rest for every policy;
  - predicted code labels, strict accuracy and alias-aware accuracy (strict, or the predicted code writes the
    reference with the gold code's cells);
  - CER and bCER with all whitespace removed (BrailleLLM's space-normalised CER);
  - BrailleLLM corpus BLEU (tokenize='zh') and chrF++, as-is and whitespace-free;
  - diagnostics: edit categories in the consistent set, S/T-fold changes, other-language labels, pool checks.
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU only: the imports pull in torch/bitsandbytes, keep them off the GPUs
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path[:0] = sorted(glob.glob(os.path.join(REPO, "scripts", "*", "")))
from ubt.paths import DATASETS, WORK  # noqa: E402

RUN = f"{WORK}/real_corpora/run"
DATA_U1 = f"{DATASETS}/data_u1_v2"
GROUPS = ("bible", "braillellm", "vision", "nikl")
CONDS = ("inferred", "given")
WS = re.compile(r"\s+")
UNPARSABLE = "<unparsable>"
SUM_KEYS = ("e", "n", "e_st", "n_st", "we", "wn", "bc", "bc_st", "strict", "conf", "exact", "feasible", "code_alias",
            "e_nows", "n_nows", "folded", "bc_nows", "cells_nows")


def _jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _gen_model(log_path: str) -> str | None:
    """Model path from a gen shard's vLLM log, or None."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = re.search(r"'model': '([^']+)'", line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return None


def load_pools(run_dir: str, cond: str) -> tuple[dict, list[str]]:
    files = sorted(glob.glob(os.path.join(run_dir, f"gen_{cond}.*.jsonl")))
    gens = {}
    for p in files:
        for x in _jsonl(p):
            if x["id"] in gens:
                raise SystemExit(f"{p}: duplicate pool id {x['id']}")
            gens[x["id"]] = x
    return gens, files


def row_checks(rows: list[dict]) -> dict:
    """Per-corpus counts of rows that are not single-segment (completion = ⟨code⟩text, prompt cells = braille)."""
    import bon_gen as E  # noqa: PLC0415

    from ubt.wandb_utils import parse_target  # noqa: PLC0415
    out = collections.defaultdict(collections.Counter)
    for r in rows:
        c = out[r["group"]]
        c["rows"] += 1
        c["f_consistent"] += bool(r["f_consistent"])
        c["completion_not_single_gold_segment"] += parse_target(r["completion"]) != [(r["tables"][0], r["text"])]
        c["prompt_cells_ne_braille"] += E._cells(r["prompt"]) != r["braille"]
        c["tables_ne_1"] += len(r["tables"]) != 1
    return {g: dict(c) for g, c in out.items()}


def pool_checks(rows: list[dict], gens: dict, cond: str, inventory: set[str]) -> dict:
    """Per-corpus candidate checks: parse, empty text, labels outside the inventory, cap hits, prefill."""
    from ubt.wandb_utils import parse_target  # noqa: PLC0415
    out = collections.defaultdict(collections.Counter)
    outside = collections.defaultdict(collections.Counter)
    for r in rows:
        c = out[r["group"]]
        cands = gens[r["id"]]["cands"]
        gold = r["tables"][0]
        c["docs"] += 1
        c["cands"] += len(cands)
        c["docs_with_n_cands_ne_20"] += len(cands) != 20
        c["cand0_not_flagged_greedy"] += not cands[0].get("greedy")
        for cand in cands:
            segs = parse_target(cand["text"])
            c["unparsable"] += not segs
            c["multi_segment"] += len(segs) > 1
            c["empty_text"] += bool(segs) and not "".join(t for _, t in segs).strip()
            bad = [lab for lab, _ in segs if lab not in inventory]
            c["label_outside_inventory"] += bool(bad)
            outside[r["group"]].update(bad)
            c["cap_hit"] += cand.get("finish") == "length"
            if cond == "given":
                c["prefill_mismatch"] += not cand["text"].startswith(f"⟨{gold}⟩")
                c["label_ne_gold"] += any(lab != gold for lab, _ in segs)
    return {g: {**dict(c), "labels_outside_inventory": dict(outside[g])} for g, c in out.items()}


def _pct(a: float, b: float) -> float:
    return 100 * a / max(1, b)


QUOTES = ({"\u2019", "\u2018", "'"}, {"\u201c", "\u201d", '"'})


def edit_categories(ref: str, hyp: str) -> collections.Counter:
    """Count character edits by kind: quote typography, letter case only, whitespace, other."""
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415
    out = collections.Counter()
    for op in Levenshtein.editops(ref, hyp):
        a = ref[op.src_pos] if op.tag != "insert" else ""
        b = hyp[op.dest_pos] if op.tag != "delete" else ""
        if op.tag == "replace" and any(a in q and b in q for q in QUOTES):
            out["quote_typography"] += 1
        elif op.tag == "replace" and a.lower() == b.lower():
            out["case"] += 1
        elif (a or b).isspace():
            out["whitespace"] += 1
        else:
            out["other"] += 1
    return out


def agg(ds: list[dict], pol: str, inventory: set[str]) -> dict:
    """bon_eval.cmd_score's agg (same keys, same formulas) plus the real-corpora additions."""
    m = collections.Counter()
    kinds = collections.defaultdict(collections.Counter)
    labels = collections.Counter()
    other_lang = collections.Counter()
    for d in ds:
        p = d["pol"][pol]
        if p["label"] == UNPARSABLE or any(x[:2] != d["table"][:2] for x in p["label"].split("+")):
            other_lang["docs"] += 1
            other_lang["consistent"] += p["feasible"]
        for k in SUM_KEYS:
            m[k] += p[k]
        m["cells"] += d["n_cells"]
        m["any_feasible"] += d["any_feasible"]
        m["fallback"] += d["bcer_fallback"]
        m["cap"] += p["finish"] == "length"
        m["unparsable"] += p["label"] == UNPARSABLE
        m["gt50"] += p["e"] > 0.5 * max(1, p["n"])
        kk = kinds[p["kind"]]
        kk["docs"] += 1
        kk["e"] += p["e"]
        kk["n"] += p["n"]
        kk["bc"] += p["bc"]
        kk["cells"] += d["n_cells"]
        labels[p["label"]] += 1
    n = max(1, len(ds))
    out = {"docs": len(ds), "cer": _pct(m["e"], m["n"]), "cer_st": _pct(m["e_st"], m["n_st"]),
           "bcer": _pct(m["bc"], m["cells"]), "bcer_st": _pct(m["bc_st"], m["cells"]),
           "wer": _pct(m["we"], m["wn"]), "strict": 100 * m["strict"] / n, "conf": 100 * m["conf"] / n,
           "exact": 100 * m["exact"] / n, "consistent": 100 * m["feasible"] / n,
           "any_feasible": 100 * m["any_feasible"] / n,
           "kinds": {k: {"share": 100 * v["docs"] / n, "cer": _pct(v["e"], v["n"]), "bcer": _pct(v["bc"], v["cells"])}
                     for k, v in kinds.items()}}
    out.update({
        "cer_nows": _pct(m["e_nows"], m["n_nows"]), "bcer_nows": _pct(m["bc_nows"], m["cells_nows"]),
        "code_alias": 100 * m["code_alias"] / n,
        "outputs_st_folded": m["folded"],
        "ref_chars": m["n"], "cells": m["cells"], "bcer_fallback_docs": m["fallback"], "cap_hits": m["cap"],
        "unparsable": m["unparsable"], "docs_cer_gt50": m["gt50"],
        "labels": {"top5": [[lab, round(100 * c / n, 2), c] for lab, c in labels.most_common(5)],
                   "n_distinct": len(labels),
                   "outside_inventory": sum(c for lab, c in labels.items()
                                            if lab != UNPARSABLE and any(x not in inventory for x in lab.split("+"))),
                   "other_language": other_lang["docs"], "other_language_consistent": other_lang["consistent"]},
    })
    split = {}
    for name, sub in (("consistent", [d for d in ds if d["pol"][pol]["feasible"]]),
                      ("rest", [d for d in ds if not d["pol"][pol]["feasible"]])):
        s = collections.Counter()
        cats = collections.Counter()
        for d in sub:
            p = d["pol"][pol]
            for k in ("e", "n", "e_st", "n_st", "bc", "exact", "e_nows", "n_nows"):
                s[k] += p[k]
            s["cells"] += d["n_cells"]
            cats.update(edit_categories(d["ref_norm"], p["hyp_norm"]))
        split[name] = {"docs": len(sub), "share": 100 * len(sub) / n, "cer": _pct(s["e"], s["n"]),
                       "cer_st": _pct(s["e_st"], s["n_st"]), "bcer": _pct(s["bc"], s["cells"]),
                       "cer_nows": _pct(s["e_nows"], s["n_nows"]), "exact": _pct(s["exact"], len(sub)),
                       "edits": {k: cats[k] for k in ("quote_typography", "case", "whitespace", "other")}}
    out["by_consistency"] = split
    return out


def bleu_block(ds: list[dict], pol: str) -> dict:
    """Corpus BLEU (zh tokenizer) / chrF++ of one policy's outputs, scoring normal form, as-is and whitespace-free."""
    from ubt.metrics.extra import bleu_chrf  # noqa: PLC0415
    refs = [d["ref_norm"] for d in ds]
    hyps = [d["pol"][pol]["hyp_norm"] for d in ds]
    as_is = bleu_chrf(refs, hyps, tokenize="zh")
    nows = bleu_chrf([WS.sub("", x) for x in refs], [WS.sub("", h) for h in hyps], tokenize="zh")
    return {"bleu": as_is["bleu"], "chrf++": as_is["chrf++"], "bleu_nows": nows["bleu"], "chrf++_nows": nows["chrf++"],
            "hyps_with_whitespace": sum(bool(WS.search(h)) for h in hyps)}


def _signature(metric) -> str:
    """sacrebleu signature (it needs one evaluation to know the number of references)."""
    metric.corpus_score(["a"], [["a"]])
    return str(metric.get_signature())


def _f(x: float) -> str:
    return f"{x:.2f}"


def write_md(path: str, report: dict, samples: dict) -> None:
    R, meta = report["results"], report["meta"]
    pols = meta["policies"]
    L = [f"# Real-braille corpora — best-of-20 scoring ({meta['model'] or 'model: see gen logs'})", "",
         (f"`scripts/real_corpora/real_corpora_score.py`, {meta['created_utc'][:19]}Z, repo {str(meta['repo_git_head'])[:9]}; pools "
          f"`{meta['run_dir']}`. Scoring normal form (NFC, U+200B removed, zh S/T folded); CER/bCER in % (micro). "
          "V = verified lambda=2; consistent = the selected output re-transcribes to the input cells under the code it "
          "carries (data engine, RoutedLouis on data_u1_v2). inferred = code inferred; "
          "given = gold <code> prefilled."), "",
         "## Main (all documents)", "",
         "| corpus | cond | n | " + " | ".join(f"{p} CER / bCER" for p in pols)
         + " | CER S/T-sens. G→V | exact G→V | consistent G→V | any feasible |",
         "|---|---|---:|" + "---:|" * (len(pols) + 4)]
    for g in GROUPS:
        for c in CONDS:
            r = R[g][c]["all"]
            L.append(f"| {g} | {c} | {r['G']['docs']} | "
                     + " | ".join(f"{_f(r[p]['cer'])} / {_f(r[p]['bcer'])}" for p in pols)
                     + f" | {_f(r['G']['cer_st'])} → {_f(r['V']['cer_st'])}"
                     + f" | {r['G']['exact']:.1f} → {r['V']['exact']:.1f}"
                     + f" | {r['G']['consistent']:.1f} → {r['V']['consistent']:.1f} | {r['V']['any_feasible']:.1f} |")
    L += ["", "## V: consistent outputs vs the rest", "",
          ("| corpus | cond | consistent (%) | CER consistent | CER rest | bCER consistent | bCER rest "
           "| exact within consistent (%) | edits in the consistent set: quote typography / case / whitespace / other |"),
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in GROUPS:
        for c in CONDS:
            s = R[g][c]["all"]["V"]["by_consistency"]
            e = s["consistent"]["edits"]
            L.append(f"| {g} | {c} | {s['consistent']['share']:.1f} ({s['consistent']['docs']}) | "
                     f"{_f(s['consistent']['cer'])} | {_f(s['rest']['cer'])} | {_f(s['consistent']['bcer'])} | "
                     f"{_f(s['rest']['bcer'])} | {s['consistent']['exact']:.1f} | "
                     f"{e['quote_typography']} / {e['case']} / {e['whitespace']} / {e['other']} |")
    L += ["", "## Code inferred: predicted labels of the selected outputs", "",
          ("strict = label == gold; alias-aware = strict, or the predicted code writes the reference text with the same "
           "cells as the gold code on that document."), "",
          ("| corpus (gold) | policy | strict (%) | alias-aware (%) | top-5 labels (share %) "
           "| other-language label (docs; consistent) |"),
          "|---|---|---:|---:|---|---:|"]
    for g in GROUPS:
        for p in ("G", "V"):
            r = R[g]["inferred"]["all"][p]
            top = ", ".join(f"{lab} {sh:.1f}" for lab, sh, _ in r["labels"]["top5"])
            L.append(f"| {g} ({meta['gold_code'][g]}) | {p} | {r['strict']:.1f} | {r['code_alias']:.1f} | {top} | "
                     f"{r['labels']['other_language']}; {r['labels']['other_language_consistent']} |")
    L += ["", "## f_consistent subset (forward(gold code, reference) == corpus cells)", "",
          "| corpus | cond | n | " + " | ".join(f"{p} CER / bCER" for p in pols) + " | consistent G→V |",
          "|---|---|---:|" + "---:|" * (len(pols) + 1)]
    for g in GROUPS:
        for c in CONDS:
            r = R[g][c]["f_consistent"]
            L.append(f"| {g} | {c} | {r['G']['docs']} | "
                     + " | ".join(f"{_f(r[p]['cer'])} / {_f(r[p]['bcer'])}" for p in pols)
                     + f" | {r['G']['consistent']:.1f} → {r['V']['consistent']:.1f} |")
    if report["braillellm"]:
        B = report["braillellm"]
        L += ["", "## BrailleLLM test set: their metrics", "",
              ("CER space-norm. = all whitespace removed from both sides (BrailleLLM's CER). BLEU = sacrebleu corpus BLEU, "
               "tokenize='zh'; chrF++ = chrF word_order=2; on the scoring-normal-form texts as they are, and with all "
               "whitespace removed from both sides (ws-free). The model writes the corpus's word-division blank cells as "
               "spaces; the references have none. O = oracle under the headline CER (spaces counted)."), "",
              ("| cond | policy | CER | CER space-norm. | BLEU | chrF++ | BLEU ws-free | chrF++ ws-free "
               "| outputs with whitespace |"),
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for c in CONDS:
            for p in pols:
                r, b = R["braillellm"][c]["all"][p], B[c][p]
                L.append(f"| {c} | {p} | {_f(r['cer'])} | {_f(r['cer_nows'])} | {_f(b['bleu'])} | {_f(b['chrf++'])} | "
                         f"{_f(b['bleu_nows'])} | {_f(b['chrf++_nows'])} | {b['hyps_with_whitespace']} |")
    L += ["", "## Checks", ""]
    A = report["bcer_alignment"]
    L.append("bCER alignment (liblouis inPos of the reference against the corpus cells; proportional fallback "
             "otherwise): "
             + "; ".join(f"{g} fallback on {A[g]['fallback_docs']}/{A[g]['docs']} docs "
                         f"({A[g]['fallback_docs_f_consistent']}/{A[g]['f_consistent_docs']} of the f_consistent ones)"
                         for g in GROUPS) + ".")
    L.append("")
    for c in CONDS:
        for g in GROUPS:
            k = report["checks"]["pools"][c][g]
            out = k["labels_outside_inventory"]
            L.append(f"- pool {c}/{g}: " + ", ".join(f"{a}={b}" for a, b in k.items() if a != "labels_outside_inventory")
                     + (f", outside-inventory labels {out}" if out else ""))
    for g, k in report["checks"]["rows"].items():
        L.append(f"- rows {g}: " + ", ".join(f"{a}={b}" for a, b in k.items()))
    for note in report.get("notes", []):
        L.append(f"- {note}")
    L += ["", "## Samples (reference vs selected outputs; ✓ = consistent)", ""]
    for g in GROUPS:
        for s in samples[g]:
            L.append(f"**{s['id']}** (f_consistent={s['f_consistent']}, {s['why']})  ")
            L.append(f"REF: {s['ref']}  ")
            for c in CONDS:
                for p in ("G", "V", "O"):
                    x = s[c][p]
                    L.append(f"{c[:3]} {p} [{'✓' if x['feasible'] else '✗'} CER {100 * x['cer']:.1f}]: {x['text']}  ")
            L.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main() -> None:
    global GROUPS
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--dir", default=RUN, help="pools gen_<cond>.<i>.jsonl; the reports go here too")
    ap.add_argument("--rows", default=None, help="default <dir>/real.jsonl")
    ap.add_argument("--engine-data", default=DATA_U1)
    ap.add_argument("--samples", type=int, default=3, help="documents per corpus shown in report.md")
    ap.add_argument("--out-dir", default=None, help="report directory (default --dir)")
    ap.add_argument("--limit-per-group", type=int, default=0, help="debug: first N rows per corpus")
    a = ap.parse_args()
    a.dir = os.path.abspath(a.dir)
    a.out_dir = os.path.abspath(a.out_dir or a.dir)
    os.makedirs(a.out_dir, exist_ok=True)
    a.rows = os.path.abspath(a.rows or os.path.join(a.dir, "real.jsonl"))
    a.engine_data = os.path.abspath(a.engine_data)
    t_start = time.time()

    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415

    import bon_eval as B  # noqa: PLC0415
    import bon_gen as E  # noqa: PLC0415
    import main_eval as TC  # noqa: PLC0415
    from ubt.eval_harness import score_hyp  # noqa: PLC0415
    from ubt.metrics.textnorm import _error_cells, _reference_cells, norm_join  # noqa: PLC0415
    from ubt.u1.routed_louis import RoutedLouis  # noqa: PLC0415
    from ubt.wandb_utils import parse_target  # noqa: PLC0415

    rows = _jsonl(a.rows)
    assert len({r["id"] for r in rows}) == len(rows), "duplicate row ids"
    if a.limit_per_group:
        seen = collections.Counter()
        rows = [r for r in rows if (seen.update([r["group"]]) or seen[r["group"]]) <= a.limit_per_group]
    unknown = {r["group"] for r in rows} - set(GROUPS)
    assert not unknown, f"unexpected groups {unknown}"
    GROUPS = tuple(g for g in GROUPS if any(r["group"] == g for r in rows))   # subset runs (few-shot pools)
    inventory = {t["table_id"] for t in _json(os.path.join(a.engine_data, "inventory.json"))["tables"]}
    gold_code = {g: sorted({r["tables"][0] for r in rows if r["group"] == g}) for g in GROUPS}
    assert all(len(v) == 1 for v in gold_code.values()), gold_code
    gold_code = {g: v[0] for g, v in gold_code.items()}

    pools, files = {}, {}
    for cond in CONDS:
        pools[cond], files[cond] = load_pools(a.dir, cond)
        miss = [r["id"] for r in rows if r["id"] not in pools[cond]]
        extra = set() if a.limit_per_group else set(pools[cond]) - {r["id"] for r in rows}
        if miss or extra:
            raise SystemExit(f"{cond}: {len(miss)} rows without a pool (e.g. {miss[:3]}), "
                             f"{len(extra)} pools without a row")
    checks = {"rows": row_checks(rows), "pools": {c: pool_checks(rows, pools[c], c, inventory) for c in CONDS}}

    # 1. consistency of every candidate; runs before _pin_engine, which chdirs and repins the process
    feas = {}
    for cond in CONDS:
        t0 = time.time()
        feas[cond] = E._feasibility(rows, pools[cond], a.engine_data)
        print(f"[feas] {cond}: {len(feas[cond])} candidates, {sum(v[0] for v in feas[cond].values())} consistent, "
              f"{time.time() - t0:.0f}s", flush=True)

    # 2. per-candidate CER (for the oracle), then each policy's pick
    for r in rows:
        ref = norm_join(parse_target(r["completion"]))
        for cond in CONDS:
            for c in pools[cond][r["id"]]["cands"]:
                segs = parse_target(c["text"])
                c["_cer"] = (Levenshtein.distance(ref, norm_join(segs)) / max(1, len(ref))) if segs else 1.0
    picks = {}
    for cond in CONDS:
        for r in rows:
            cands = pools[cond][r["id"]]["cands"]
            f = [feas[cond][(r["id"], i)] for i in range(len(cands))]
            for pol in B.POLICIES:
                picks[(cond, r["id"], pol)] = B._select(cands, f, pol)

    # 3. alias-aware code check: forward the reference under the gold code and every selected label
    need = set()
    for r in rows:
        gold = parse_target(r["completion"])
        need.add(gold[0])
        for cond in CONDS:
            for pol in B.POLICIES:
                hs = parse_target(pools[cond][r["id"]]["cands"][picks[(cond, r["id"], pol)]]["text"])
                if len(hs) == 1:
                    need.add((hs[0][0], gold[0][1]))
    need = sorted(need)
    t0 = time.time()
    Lr = RoutedLouis(a.engine_data, timeout_s=300.0)
    fwd = dict(zip(need, Lr.translate_many(need)))
    Lr.close()
    print(f"[alias] {len(need)} (label, reference) forwards, {time.time() - t0:.0f}s", flush=True)
    for r in rows:                              # the builder's f_consistent, re-derived with the same engine
        gold = parse_target(r["completion"])[0]
        checks["rows"][r["group"]]["f_consistent_recheck_mismatch"] = (
            checks["rows"][r["group"]].get("f_consistent_recheck_mismatch", 0)
            + ((fwd.get(gold) == r["braille"]) != bool(r["f_consistent"])))

    # 4. per-document scoring (bon_eval.cmd_score fields); bCER needs the pinned engine
    TC._pin_engine(a.engine_data)
    align = {}
    for r in rows:
        segs_ref = TC._bcer_segments(r)
        cells, starts, n_cells, fb = _reference_cells(segs_ref)
        segs_nows = [(t, WS.sub("", x), b.replace("\u2800", "")) for t, x, b in segs_ref]
        align[r["id"]] = (segs_ref, cells, starts, n_cells, fb, (segs_nows,) + tuple(_reference_cells(segs_nows)))
    docs = {c: [] for c in CONDS}
    for cond in CONDS:
        for r in rows:
            cands = pools[cond][r["id"]]["cands"]
            f = [feas[cond][(r["id"], i)] for i in range(len(cands))]
            segs_ref, cells, starts, n_cells, fb, (segs_nw, cells_nw, starts_nw, n_cells_nw, _fb_nw) = align[r["id"]]
            gold = parse_target(r["completion"])
            ref_n = norm_join(gold)
            ref_nows = WS.sub("", ref_n)
            gold_fwd = fwd.get(gold[0])
            d = {"id": r["id"], "cond": cond, "group": r["group"], "table": r["tables"][0],
                 "f_consistent": bool(r["f_consistent"]), "n_cells": n_cells, "bcer_fallback": fb,
                 "any_feasible": any(x[0] for x in f), "n_feasible": sum(bool(x[0]) for x in f), "ref_norm": ref_n,
                 "pol": {}}
            for pol in B.POLICIES:
                ci = picks[(cond, r["id"], pol)]
                c = cands[ci]
                row = score_hyp({"id": r["id"], "confusable_set": r["confusable_set"]}, {"completion": r["completion"]},
                                c["text"], None)
                hs = parse_target(c["text"])
                hyp_n = norm_join(hs) if hs else ""
                folded = bool(hs) and norm_join(hs, st_fold=False) != hyp_n
                we, wn = B._wer(ref_n, hyp_n)
                feasible = bool(f[ci][0])
                kind = ("inconsistent" if not feasible else "wrong_code" if not row["strict"]
                        else "correct" if row["hyp_eq_ref"] else "consistent_misreading")
                alias = bool(row["strict"]) or (len(hs) == 1 and gold_fwd is not None
                                                and fwd.get((hs[0][0], gold[0][1])) == gold_fwd)
                d["pol"][pol] = {"ci": ci, "e": row["edit"], "n": row["ref_len"], "e_st": row["edit_st"],
                                 "n_st": row["ref_len_st"], "strict": row["strict"], "conf": row["conf_aware"],
                                 "exact": row["hyp_eq_ref"], "feasible": feasible, "d_cell": f[ci][1], "kind": kind,
                                 "we": we, "wn": wn,
                                 "bc": _error_cells(segs_ref, starts, cells, n_cells, hs, True),
                                 "bc_st": _error_cells(segs_ref, starts, cells, n_cells, hs, False),
                                 "e_nows": Levenshtein.distance(ref_nows, WS.sub("", hyp_n)), "n_nows": len(ref_nows),
                                 "bc_nows": _error_cells(segs_nw, starts_nw, cells_nw, n_cells_nw,
                                                         [(t, WS.sub("", x)) for t, x in hs], True),
                                 "cells_nows": n_cells_nw,
                                 "label": "+".join(t for t, _ in hs) if hs else UNPARSABLE, "code_alias": alias,
                                 "finish": c.get("finish"), "lp": c["lp"], "cer_doc": c["_cer"], "text": c["text"],
                                 "hyp_norm": hyp_n, "folded": folded}
            docs[cond].append(d)
    with open(os.path.join(a.out_dir, "per_doc.jsonl"), "w", encoding="utf-8") as fh:
        for cond in CONDS:
            for d in docs[cond]:
                fh.write(json.dumps({k: v for k, v in d.items() if k != "ref_norm"}, ensure_ascii=False) + "\n")

    # 5. aggregates
    results = {}
    for g in GROUPS:
        results[g] = {}
        for cond in CONDS:
            ds = [d for d in docs[cond] if d["group"] == g]
            fc = [d for d in ds if d["f_consistent"]]
            results[g][cond] = {"all": {p: agg(ds, p, inventory) for p in B.POLICIES},
                                "f_consistent": {p: agg(fc, p, inventory) for p in B.POLICIES}}
    brl = {} if "braillellm" not in GROUPS else {cond: {p: bleu_block([d for d in docs[cond] if d["group"] == "braillellm"], p) for p in B.POLICIES}
           for cond in CONDS}
    alignment = {}
    for g in GROUPS:
        rs = [r for r in rows if r["group"] == g]
        alignment[g] = {"docs": len(rs), "fallback_docs": sum(align[r["id"]][4] for r in rs),
                        "f_consistent_docs": sum(bool(r["f_consistent"]) for r in rs),
                        "fallback_docs_f_consistent": sum(align[r["id"]][4] for r in rs if r["f_consistent"])}

    import sacrebleu  # noqa: PLC0415
    from sacrebleu.metrics import BLEU, CHRF  # noqa: PLC0415
    notes = []
    for g in GROUPS:
        for cond in CONDS:
            ds = [d for d in docs[cond] if d["group"] == g]
            ws_g = sum(bool(WS.search(d["pol"]["G"]["hyp_norm"])) for d in ds)
            ws_r = sum(bool(WS.search(d["ref_norm"])) for d in ds)
            if ws_g and not ws_r:
                notes.append(f"{g}/{cond}: {ws_g}/{len(ds)} greedy outputs contain whitespace, {ws_r} references do; the "
                             "headline CER counts every such space as an insertion (see CER space-norm.)")
            fv = results[g][cond]["all"]["V"]["outputs_st_folded"]
            if fv:
                notes.append(f"{g}/{cond}: {fv}/{len(ds)} V outputs contain characters the S/T fold changes (Traditional "
                             "script against a Simplified reference): headline CER folds them, cer_st does not")
    for g in GROUPS:
        if alignment[g]["fallback_docs_f_consistent"]:
            notes.append(f"{g}: {alignment[g]['fallback_docs_f_consistent']} f_consistent documents still use the bCER "
                         "fallback (the in-process alignment engine disagrees with the isolated RoutedLouis worker)")
    meta = {
        "script": os.path.relpath(os.path.abspath(__file__), REPO), "script_sha256": _sha256(os.path.abspath(__file__)),
        "repo_git_head": _git_head(), "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "argv": sys.argv[1:], "run_dir": a.dir, "limit_per_group": a.limit_per_group,
        "rows": {"path": a.rows, "sha256": _sha256(a.rows), "n": len(rows),
                 "per_group": dict(collections.Counter(r["group"] for r in rows))},
        "pools": {c: {os.path.basename(p): _sha256(p) for p in files[c]} for c in CONDS},
        "model": _gen_model(os.path.join(a.dir, "gen_inferred.0.log")),
        "engine_data": a.engine_data,
        "engine_id": _json(os.path.join(a.engine_data, "manifest.json")).get("engine_id"),
        "gold_code": gold_code, "policies": list(B.POLICIES), "lambda": B.LAMBDA,
        "bleu": {"sacrebleu": sacrebleu.__version__, "bleu_signature": _signature(BLEU(tokenize="zh")),
                 "chrf_signature": _signature(CHRF(word_order=2))},
        "definitions": {
            "cer/cer_st/bcer/bcer_st/exact": "scoring normal form (ubt.metrics.textnorm), micro over documents; "
                                              "*_st = S/T-sensitive",
            "consistent": "% of documents whose selected output re-transcribes to the input cells under its own codes",
            "any_feasible": "% of documents with at least one consistent candidate among the 20",
            "strict": "% predicted labels == gold labels", "code_alias": "% strict, or the (single) predicted code "
                                                                         "writes the reference text with the gold "
                                                                         "code's cells (RoutedLouis)",
            "cer_nows": "CER with all whitespace removed from both sides (space-normalised)",
            "by_consistency": "the documents whose selected output is consistent vs the rest",
            "kinds": "bon_eval error kinds: inconsistent / wrong_code / correct / consistent_misreading",
            "f_consistent": "subset with forward(gold code, reference text) == corpus cells (build_real_corpora.py)",
            "bleu/chrf++": "ubt.metrics.extra.bleu_chrf(tokenize='zh') on norm_join texts; *_nows: whitespace removed"},
        "seconds": None}
    report = {"meta": meta, "checks": checks, "bcer_alignment": alignment, "results": results, "braillellm": brl,
              "notes": notes}

    # 6. samples: the first document and the first where V changes G's pick or is inconsistent
    samples = {}
    by_id = {r["id"]: r for r in rows}
    for g in GROUPS:
        dg = {cond: {d["id"]: d for d in docs[cond] if d["group"] == g} for cond in CONDS}
        ids = [r["id"] for r in rows if r["group"] == g]
        chosen = [(ids[0], "first document")]
        for why, test in (("V differs from G (inferred)", lambda i, dg=dg: dg["inferred"][i]["pol"]["V"]["ci"] != 0),
                          ("V output inconsistent (given)", lambda i, dg=dg: not dg["given"][i]["pol"]["V"]["feasible"]),
                          ("V differs from G (given)", lambda i, dg=dg: dg["given"][i]["pol"]["V"]["ci"] != 0)):
            hit = next((i for i in ids if test(i) and i not in {x for x, _ in chosen}), None)
            if hit:
                chosen.append((hit, why))
        samples[g] = [{"id": i, "why": why, "f_consistent": by_id[i]["f_consistent"], "ref": by_id[i]["text"],
                       **{cond: {p: {k: dg[cond][i]["pol"][p][k] for k in ("text", "feasible")}
                                 | {"cer": dg[cond][i]["pol"][p]["cer_doc"]} for p in ("G", "V", "O")}
                          for cond in CONDS}}
                      for i, why in chosen[: max(1, a.samples)]]
    report["samples"] = samples
    meta["seconds"] = round(time.time() - t_start, 1)
    with open(os.path.join(a.out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    md = os.path.join(a.out_dir, "report.md")
    write_md(md, report, samples)
    with open(md, encoding="utf-8") as fh:
        print(fh.read())


if __name__ == "__main__":
    main()
