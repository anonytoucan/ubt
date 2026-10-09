"""Verified best-of-n and backbone-prior re-selection on the CJK codes.

  build : eval = main_eval core ja/zh documents; dev = --dev-per-table documents per CJK table from the top-up pool
          (never trained on; their ids are written out so the top-up run can exclude them)
  gen   : vLLM greedy output + (n-1) samples (T=0.8, top-p 0.95) per document, each with its cumulative log p(Y|B)
  prior : log p_base(T) of each distinct candidate text under the base Qwen/Qwen2.5-7B, T = segment texts joined
          with '\\n', conditioned on <|endoftext|>
  score : re-transcribe under the predicted codes (feasible = reproduces the input cells, d_cell = cell edit distance)
          and select:
            G      greedy
            V      argmax log p(Y|B) over feasible candidates; none feasible -> argmax log p - 2 d_cell
            Vinf   as V, but none feasible -> min d_cell (lambda -> inf)
            VP(a)  feasible candidates ranked by log p(Y|B) + a * log p_base(T) (a = inf: prior only); else as V
            O      oracle: lowest CER (upper bound)
          a* is chosen on dev (lowest CJK CER) and applied once to eval. CER/bCER use the scoring normal form
          (headline folds S/T; *_st is S/T-sensitive).

  python scripts/eval/bon_gen.py build --out work/bon_cjk/data
  CUDA_VISIBLE_DEVICES=i python scripts/eval/bon_gen.py gen --model <merged> --slice <data>/eval.jsonl --out <dir>/gen_eval.i.jsonl --shard i/4
  CUDA_VISIBLE_DEVICES=i python scripts/eval/bon_gen.py prior --gen '<dir>/gen_*.jsonl' --out <dir>/prior.i.jsonl --shard i/4
  python scripts/eval/bon_gen.py score --data <data> --dir <dir>
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from ubt.paths import DATASETS, WORK  # noqa: E402

ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0, math.inf]
LAMBDA = 2.0
CJK_PREFIX = ("zh", "ja-")


def _load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def _cells(prompt: str) -> str:
    return prompt.split("<|braille|>\n", 1)[1].split("\n<|text|>", 1)[0]


def _text_of(hyp: str):
    from ubt.wandb_utils import parse_target  # noqa: PLC0415
    segs = parse_target(hyp)
    return "\n".join(t for _, t in segs) if segs else None


def cmd_build(a) -> None:
    from transformers import AutoTokenizer  # noqa: PLC0415

    from ubt.task_format import render  # noqa: PLC0415
    os.makedirs(a.out, exist_ok=True)
    keep = ("id", "set", "tables", "k", "regime", "switch_density", "confusable_set", "group", "prompt", "completion",
            "max_tokens")
    ev = [{k: r[k] for k in keep} for r in _load(a.main_eval) if r["set"] == "core" and r["group"] in ("ja", "zh")]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    by = collections.defaultdict(list)
    for line in open(a.topup_cjk, encoding="utf-8"):
        d = json.loads(line)
        by[d["tables"][0]].append(d)
    rng = random.Random(a.seed)
    dev = []
    for t in sorted(by):
        for d in rng.sample(by[t], min(a.dev_per_table, len(by[t]))):
            ex = render(d, hinted=False)
            n_c = len(tok(ex["completion"], add_special_tokens=False)["input_ids"])
            dev.append({"id": d["id"], "set": "core", "tables": d["tables"], "k": d["k"], "regime": d["regime"],
                        "switch_density": d["switch_density"], "confusable_set": d.get("confusable_set") or [],
                        "group": "ja" if t.startswith("ja-") else "zh", "prompt": ex["prompt"],
                        "completion": ex["completion"], "max_tokens": 2 * n_c + 32})
    meta = {}
    for name, rows in (("eval", ev), ("dev", dev)):
        p = os.path.join(a.out, f"{name}.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta[name] = {"n": len(rows), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
                      "per_table": dict(collections.Counter(r["tables"][0] for r in rows))}
    with open(os.path.join(a.out, "dev_ids_exclude_from_topup.txt"), "w") as fh:
        fh.write("\n".join(r["id"] for r in dev) + "\n")
    meta["from"] = {"eval": a.main_eval, "dev": a.topup_cjk, "seed": a.seed, "dev_per_table": a.dev_per_table}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


def cmd_gen(a) -> None:
    import time  # noqa: PLC0415

    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    si, sn = (int(x) for x in a.shard.split("/"))
    rows = _load(a.slice)[si::sn][: a.limit or None]
    tok = AutoTokenizer.from_pretrained(a.model)
    # --prefill-code: start the completion with the first gold <code>, tokenised on its own as in training. A row's
    # own "prefill" (few-shot segments, then the target <code>) wins; candidates keep it, the scorer reads the last one.
    pre = [r.get("prefill") or (f"\u27e8{r['tables'][0]}\u27e9" if a.prefill_code else "") for r in rows]
    ids = [tok(r["prompt"], add_special_tokens=True)["input_ids"] + (tok(p_, add_special_tokens=False)["input_ids"] if p_ else [])
           for r, p_ in zip(rows, pre)]
    t0 = time.time()
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              seed=a.seed, enable_prefix_caching=True)
    eos = [tok.eos_token_id]
    if a.force_codes:
        _gen_forced(a, llm, tok, rows, eos)
        print(f"[gen] {a.slice} shard {a.shard}: {len(rows)} docs x {a.n}, codes forced, {time.time() - t0:.0f}s", flush=True)
        return
    prompts = [{"prompt_token_ids": x} for x in ids]
    greedy = llm.generate(prompts, [SamplingParams(temperature=0.0, max_tokens=r["max_tokens"], stop_token_ids=eos,
                                                   logprobs=0) for r in rows])
    samples = llm.generate(prompts, [SamplingParams(n=a.n - 1, temperature=a.temp, top_p=a.top_p, seed=a.seed + i,
                                                    max_tokens=r["max_tokens"], stop_token_ids=eos, logprobs=0)
                                     for i, r in enumerate(rows)])

    def lp(c):
        if c.cumulative_logprob is not None:
            return float(c.cumulative_logprob)
        return float(sum(d[t].logprob for t, d in zip(c.token_ids, c.logprobs)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r, p_, g, s in zip(rows, pre, greedy, samples):
            cands = [{"text": p_ + g.outputs[0].text, "lp": lp(g.outputs[0]), "finish": g.outputs[0].finish_reason,
                      "greedy": True}]
            cands += [{"text": p_ + c.text, "lp": lp(c), "finish": c.finish_reason, "greedy": False} for c in s.outputs]
            fh.write(json.dumps({"id": r["id"], "cands": cands}, ensure_ascii=False) + "\n")
    print(f"[gen] {a.slice} shard {a.shard}: {len(rows)} docs x {a.n}, {time.time() - t0:.0f}s", flush=True)


def _gen_forced(a, llm, tok, rows, eos) -> None:
    """Code given on every segment: round j appends the gold '\\n<code_j>' and decodes segment j (path 0 greedy, the
    rest sampled); log p counts generated tokens only, and a path that hits the token cap stops there."""
    from vllm import SamplingParams  # noqa: PLC0415
    P = [{"r": ri, "j": j, "text": "", "lp": 0.0, "finish": None, "live": True} for ri in range(len(rows)) for j in range(a.n)]
    K = max(len(r["tables"]) for r in rows)
    pids = [tok(r["prompt"], add_special_tokens=True)["input_ids"] for r in rows]
    for seg in range(K):
        act = [p for p in P if p["live"] and seg < len(rows[p["r"]]["tables"])]
        if not act:
            break
        for p in act:
            code = rows[p["r"]]["tables"][seg]
            p["text"] += ("\n" if seg else "") + f"\u27e8{code}\u27e9"
        prompts = [{"prompt_token_ids": pids[p["r"]] + tok(p["text"], add_special_tokens=False)["input_ids"]} for p in act]
        params = [SamplingParams(temperature=0.0 if p["j"] == 0 else a.temp, top_p=1.0 if p["j"] == 0 else a.top_p,
                                 seed=a.seed + 7919 * p["r"] + 101 * p["j"] + seg, max_tokens=rows[p["r"]]["max_tokens"],
                                 stop=["\n"], stop_token_ids=eos, logprobs=0) for p in act]
        outs = llm.generate(prompts, params)
        for p, o in zip(act, outs):
            c = o.outputs[0]
            p["text"] += c.text
            p["lp"] += float(c.cumulative_logprob) if c.cumulative_logprob is not None else float(
                sum(d[t].logprob for t, d in zip(c.token_ids, c.logprobs)))
            p["finish"] = c.finish_reason
            if c.finish_reason == "length":
                p["live"] = False
        print(f"[gen] forced segment {seg + 1}/{K}: {len(act)} paths", flush=True)
    by = {}
    for p in P:
        by.setdefault(p["r"], []).append(p)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for ri, r in enumerate(rows):
            ps = sorted(by[ri], key=lambda p: p["j"])
            fh.write(json.dumps({"id": r["id"], "cands": [{"text": p["text"], "lp": p["lp"], "finish": p["finish"],
                                                            "greedy": p["j"] == 0} for p in ps]}, ensure_ascii=False) + "\n")


def cmd_prior(a) -> None:
    import time  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    si, sn = (int(x) for x in a.shard.split("/"))
    texts = set()
    for p in sorted(glob.glob(a.gen)):
        for line in open(p, encoding="utf-8"):
            for c in json.loads(line)["cands"]:
                t = _text_of(c["text"])
                if t:
                    texts.add(t)
    texts = sorted(texts)[si::sn][: a.limit or None]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.lm)
    model = AutoModelForCausalLM.from_pretrained(a.lm, dtype=torch.bfloat16).to(a.device).eval()
    start = tok.convert_tokens_to_ids("<|endoftext|>")
    enc = [[start] + tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    out = {}
    i = 0
    with torch.no_grad():
        while i < len(order):
            j, width = i, 0
            while j < len(order) and max(width, len(enc[order[j]])) * (j - i + 1) <= a.batch_tokens:
                width = max(width, len(enc[order[j]]))
                j += 1
            j = max(j, i + 1)
            idx = order[i:j]
            width = max(len(enc[k]) for k in idx)
            x = torch.full((len(idx), width), start, dtype=torch.long)
            m = torch.zeros((len(idx), width), dtype=torch.long)
            for r, k in enumerate(idx):
                x[r, :len(enc[k])] = torch.tensor(enc[k])
                m[r, :len(enc[k])] = 1
            x, m = x.to(a.device), m.to(a.device)
            logits = model(input_ids=x, attention_mask=m).logits[:, :-1].float()
            ll = torch.log_softmax(logits, dim=-1).gather(-1, x[:, 1:, None])[..., 0] * m[:, 1:]
            for r, k in enumerate(idx):
                out[texts[k]] = (float(ll[r].sum()), len(enc[k]) - 1)
            i = j
    with open(a.out, "w", encoding="utf-8") as fh:
        for t, (v, n) in out.items():
            fh.write(json.dumps({"text": t, "lp": v, "n_tok": n}, ensure_ascii=False) + "\n")
    print(f"[prior] shard {a.shard}: {len(out)} texts, {time.time() - t0:.0f}s", flush=True)


def _feasibility(rows, gens, data_dir):
    """(doc id, candidate index) -> (feasible, d_cell), re-encoded with the data's engine."""
    import unicodedata  # noqa: PLC0415

    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415

    from ubt.u1.reencode import reencode_segment  # noqa: PLC0415
    from ubt.u1.routed_louis import RoutedLouis  # noqa: PLC0415
    from ubt.u1.sources import _bengali_fix  # noqa: PLC0415
    from ubt.wandb_utils import parse_target  # noqa: PLC0415
    L = RoutedLouis(data_dir, timeout_s=300.0)     # long zh paragraphs are slow; a timeout would count as infeasible
    plain, special = {}, {}
    plan = {}
    for r in rows:
        for ci, c in enumerate(gens[r["id"]]["cands"]):
            segs = parse_target(c["text"])
            if not segs or len(c["text"]) > 4 * len(r["completion"]) + 64:       # unparsable or runaway
                plan[(r["id"], ci)] = None
                continue
            keys = []
            for tab, text in segs:
                if "$$" in text or tab in ("nemeth", "ko-math-2024"):
                    special.setdefault((tab, text), None)
                    keys.append(("s", tab, text))
                else:
                    k = (tab, _bengali_fix(unicodedata.normalize("NFC", text)))
                    plain.setdefault(k, None)
                    keys.append(("p",) + k)
            plan[(r["id"], ci)] = keys
    items = list(plain)
    for k, b in zip(items, L.translate_many(items)):
        plain[k] = b
    for tab, text in list(special):
        special[(tab, text)] = reencode_segment(L, tab, text)
    L.close()
    out = {}
    for r in rows:
        B = _cells(r["prompt"])
        for ci, _ in enumerate(gens[r["id"]]["cands"]):
            keys = plan[(r["id"], ci)]
            if keys is None:
                out[(r["id"], ci)] = (False, None)
                continue
            parts = [plain[k[1:]] if k[0] == "p" else special[k[1:]] for k in keys]
            if any(p is None for p in parts):
                out[(r["id"], ci)] = (False, None)
                continue
            enc = "\n".join(parts)
            out[(r["id"], ci)] = (enc == B, Levenshtein.distance(enc, B))
    return out


def _homophone_split(rows, gens, picks, name, pols, data_dir):
    """(policy, group) -> (homophone CER, other CER) in % of reference characters; a homophone error substitutes a
    character the code writes with the same cells."""
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415

    from ubt.metrics.textnorm import norm_join  # noqa: PLC0415
    from ubt.u1.routed_louis import RoutedLouis  # noqa: PLC0415
    from ubt.wandb_utils import parse_target  # noqa: PLC0415
    ops, need = [], set()
    for pol in pols:
        for r in rows:
            t = r["tables"][0]
            ref = norm_join(parse_target(r["completion"]))
            segs = parse_target(gens[r["id"]]["cands"][picks[(name, pol, r["id"])]]["text"])
            hyp = norm_join(segs) if segs else ""
            for op in Levenshtein.editops(ref, hyp):
                pair = (ref[op.src_pos], hyp[op.dest_pos]) if op.tag == "replace" else None
                ops.append((pol, r, t, pair, len(ref)))
                if pair:
                    need.update({(t, pair[0]), (t, pair[1])})
    L = RoutedLouis(data_dir, timeout_s=300.0)
    need = sorted(need)
    enc = dict(zip(need, L.translate_many(need)))
    L.close()
    agg = collections.defaultdict(collections.Counter)
    for pol, r, t, pair, _n in ops:
        h = bool(pair) and enc[(t, pair[0])] is not None and enc[(t, pair[0])] == enc[(t, pair[1])]
        for g in (r["group"], t, "cjk"):
            agg[(pol, g)]["homophone" if h else "other"] += 1
    refs = collections.Counter()
    for r in rows:
        n = len(norm_join(parse_target(r["completion"])))
        for g in (r["group"], r["tables"][0], "cjk"):
            refs[g] += n
    return {f"{pol}|{g}": (100 * c["homophone"] / max(1, refs[g]), 100 * c["other"] / max(1, refs[g]))
            for (pol, g), c in agg.items()}


def _select(cands, feas, prior, policy, alpha=None):
    idx = list(range(len(cands)))
    if policy == "G":
        return 0
    if policy == "O":
        return min(idx, key=lambda i: (cands[i]["_cer"], i))
    F = [i for i in idx if feas[i][0]]
    if F:
        if policy in ("V", "Vinf"):
            return max(F, key=lambda i: cands[i]["lp"])
        def key(i):
            p = prior.get(cands[i]["_text"])
            p = p if p is not None else -1e9
            return (p, cands[i]["lp"]) if math.isinf(alpha) else (cands[i]["lp"] + alpha * p,)
        return max(F, key=key)
    D = [i for i in idx if feas[i][1] is not None]
    if not D:
        return 0
    if policy == "Vinf":
        return min(D, key=lambda i: (feas[i][1], -cands[i]["lp"]))
    return max(D, key=lambda i: cands[i]["lp"] - LAMBDA * feas[i][1])


def cmd_score(a) -> None:
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415

    from ubt.eval_harness import score_hyp  # noqa: PLC0415
    from ubt.metrics.textnorm import norm_join  # noqa: PLC0415
    from ubt.wandb_utils import parse_target  # noqa: PLC0415
    a.data, a.dir = os.path.abspath(a.data), os.path.abspath(a.dir)
    prior = {}
    for p in sorted(glob.glob(os.path.join(a.dir, "prior.*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            x = json.loads(line)
            prior[x["text"]] = x["lp"]
    slices = {}
    for name in ("dev", "eval"):
        rows = _load(os.path.join(a.data, f"{name}.jsonl"))
        gens = {}
        for p in sorted(glob.glob(os.path.join(a.dir, f"gen_{name}.*.jsonl"))):
            for line in open(p, encoding="utf-8"):
                x = json.loads(line)
                gens[x["id"]] = x
        rows = [r for r in rows if r["id"] in gens]
        feas = _feasibility(rows, gens, a.engine_data)
        for r in rows:
            gold = parse_target(r["completion"])
            ref = norm_join(gold)
            for c in gens[r["id"]]["cands"]:
                segs = parse_target(c["text"])
                c["_text"] = "\n".join(t for _, t in segs) if segs else None
                c["_cer"] = (Levenshtein.distance(ref, norm_join(segs)) / max(1, len(ref))) if segs else 1.0
        slices[name] = (rows, gens, feas)
    policies = [("G", None), ("V", None), ("Vinf", None)] + [(f"VP{al:g}", al) for al in ALPHAS] + [("O", None)]
    picks = {}
    for name, (rows, gens, feas) in slices.items():
        for pol, al in policies:
            for r in rows:
                cands = gens[r["id"]]["cands"]
                f = [feas[(r["id"], ci)] for ci in range(len(cands))]
                picks[(name, pol, r["id"])] = _select(cands, f, prior, "VP" if pol.startswith("VP") else pol, al)

    def cer_of(name, pol, rows):
        e = n = 0
        for r in rows:
            c = slices[name][1][r["id"]]["cands"][picks[(name, pol, r["id"])]]
            row = score_hyp({"id": r["id"], "confusable_set": r["confusable_set"]}, {"completion": r["completion"]},
                            c["text"], None)
            e += row["edit"]
            n += row["ref_len"]
        return 100 * e / max(1, n)
    dev_rows = slices["dev"][0]
    dev_cer = {pol: cer_of("dev", pol, dev_rows) for pol, al in policies if pol.startswith("VP")}
    best = min(dev_cer, key=lambda p: (round(dev_cer[p], 6), ALPHAS[[f"VP{x:g}" for x in ALPHAS].index(p)]))
    # metrics per slice x policy x group; bCER needs the pinned engine
    split = {n: _homophone_split(slices[n][0], slices[n][1], picks, n, ["G", "V", best], a.engine_data) for n in slices}
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import main_eval as TC  # noqa: PLC0415
    TC._pin_engine(a.engine_data)
    from ubt.metrics.textnorm import _error_cells, _reference_cells  # noqa: PLC0415
    report = {"alpha_dev_cer": dev_cer, "alpha_star": best, "lambda": LAMBDA, "groups": {},
              "homophone_split": {n: {k: {"homophone_cer": v[0], "other_cer": v[1]} for k, v in d.items()}
                                  for n, d in split.items()}}
    lines = []
    for name, (rows, gens, feas) in slices.items():
        per_doc = {}
        for r in rows:
            segs_ref = [(t, x, b) for (t, x), b in zip(parse_target(r["completion"]), _cells(r["prompt"]).split("\n"))]
            cells, starts, n_cells, _fb = _reference_cells(segs_ref)
            cands = gens[r["id"]]["cands"]
            any_f = any(feas[(r["id"], k)][0] for k in range(len(cands)))
            for pol, _ in policies:
                ci = picks[(name, pol, r["id"])]
                if (r["id"], ci) not in per_doc:
                    c = cands[ci]
                    row = score_hyp({"id": r["id"], "confusable_set": r["confusable_set"]},
                                    {"completion": r["completion"]}, c["text"], None)
                    hs = parse_target(c["text"])
                    per_doc[(r["id"], ci)] = {
                        "e": row["edit"], "n": row["ref_len"], "e_st": row["edit_st"], "n_st": row["ref_len_st"],
                        "bc": _error_cells(segs_ref, starts, cells, n_cells, hs, True),
                        "bc_st": _error_cells(segs_ref, starts, cells, n_cells, hs, False), "cells": n_cells,
                        "exact": row["hyp_eq_ref"], "feasible": feas[(r["id"], ci)][0], "any_feasible": any_f}
        groups = collections.defaultdict(list)
        for r in rows:
            for g in (r["group"], r["tables"][0], "cjk"):
                groups[g].append(r)
        for pol, _ in policies:
            for g, rs in groups.items():
                m = collections.Counter()
                for r in rs:
                    for k, v in per_doc[(r["id"], picks[(name, pol, r["id"])])].items():
                        m[k] += v
                d = max(1, len(rs))
                report["groups"][f"{name}|{pol}|{g}"] = {
                    "docs": len(rs), "cer": 100 * m["e"] / max(1, m["n"]), "cer_st": 100 * m["e_st"] / max(1, m["n_st"]),
                    "bcer": 100 * m["bc"] / max(1, m["cells"]), "bcer_st": 100 * m["bc_st"] / max(1, m["cells"]),
                    "exact": m["exact"] / d, "selected_feasible": m["feasible"] / d, "any_feasible": m["any_feasible"] / d}
        show = ["G", "V", "Vinf", best, "VPinf", "O"]
        lines.append(f"\n### {name} (a* = {best} from dev)\n")
        lines.append("| group | " + " | ".join(f"{p} CER / bCER" for p in show) + " | any feasible | CER S/T-sens. G→V→a* |")
        lines.append("|---|" + "---:|" * (len(show) + 2))
        for g in ["cjk", "ja", "zh"] + sorted(k for k in groups if k not in ("cjk", "ja", "zh")):
            cs = [report["groups"][f"{name}|{p}|{g}"] for p in show]
            st = [report["groups"][f"{name}|{p}|{g}"]["cer_st"] for p in ("G", "V", best)]
            lines.append(f"| {g} | " + " | ".join(f"{c['cer']:.2f} / {c['bcer']:.2f}" for c in cs)
                         + f" | {100 * cs[0]['any_feasible']:.0f}% | " + " → ".join(f"{v:.2f}" for v in st) + " |")
        lines.append("\nhomophone (cell-identical) + other CER, G / V / a*: " + "; ".join(
            f"{g} " + " / ".join("{:.2f}+{:.2f}".format(*split[name].get(f"{p}|{g}", (0.0, 0.0))) for p in ("G", "V", best))
            for g in ("cjk", "ja", "zh")))
    report["table_md"] = "\n".join(lines)
    json.dump(report, open(os.path.join(a.dir, "report.json"), "w"), indent=1)
    print("dev CER by alpha:", {k: round(v, 3) for k, v in dev_cer.items()}, "-> a* =", best)
    print(report["table_md"])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--main-eval", default=f"{WORK}/data/main_eval/eval.jsonl")
    b.add_argument("--topup-cjk", default=f"{DATASETS}/topup_cjk/train.jsonl")
    b.add_argument("--tokenizer", default=f"{WORK}/models/qwen25_cell_tokenizer")
    b.add_argument("--dev-per-table", type=int, default=100)
    b.add_argument("--seed", type=int, default=148)
    g = sub.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--slice", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--shard", default="0/1")
    g.add_argument("--n", type=int, default=20)
    g.add_argument("--temp", type=float, default=0.8)   # tau and top-p as in the method section
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--seed", type=int, default=148)
    g.add_argument("--max-model-len", type=int, default=12288)
    g.add_argument("--gpu-mem", type=float, default=0.9)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--prefill-code", action="store_true", help="prefill the first gold <code>")
    g.add_argument("--force-codes", action="store_true",
                   help="give the gold <code> of every segment; decode segment by segment")
    p = sub.add_parser("prior")
    p.add_argument("--gen", required=True, help="glob of gen shards (all slices)")
    p.add_argument("--out", required=True)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--lm", default="Qwen/Qwen2.5-7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-tokens", type=int, default=4096)
    p.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--data", required=True, help="build output (dev.jsonl, eval.jsonl)")
    s.add_argument("--dir", required=True, help="gen_{dev,eval}.*.jsonl and prior.*.jsonl")
    s.add_argument("--engine-data", default=f"{DATASETS}/data_u1_v2")
    a = ap.parse_args()
    {"build": cmd_build, "gen": cmd_gen, "prior": cmd_prior, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
