"""Tokenizer comparison: model A (base Qwen2.5 tokenizer) vs model B (the cell-token tokenizer, --braille-vocab).

  build   full core_per_code, mixed_grid and intra_switch_eval of data_u1_v2/eval, agnostic prompts, greedy cap of
          2 x gold completion tokens + 32 -> <dir>/eval.jsonl, meta.json (low-resource: < 3,000 train docs)
  merge   base + LoRA checkpoint -> merged bf16 dir, since vLLM cannot serve trainable-token adapters
  gen     vLLM greedy on one shard, prompts encoded as in training
  score   per-model metrics, B-A deltas with paired bootstrap CIs, and gates: per-group CER at most +1pp, mixed_grid
          strict and segment match at least -1pp, no braille cell in B's outputs
  decide  B's last eval loss vs A's at --step-a (same wall-clock time), whether to extend B (within 2% and
          converging), step-matched losses and seconds per step

  python scripts/eval/greedy_eval.py build --out work/data/ablation_eval
  python scripts/eval/greedy_eval.py merge --ckpt <run>/checkpoint-400 --tokenizer Qwen/Qwen2.5-7B --out <dir>
  CUDA_VISIBLE_DEVICES=i python scripts/eval/greedy_eval.py gen --model <merged> --eval <dir> --out g.i.jsonl --shard i/4
  python scripts/eval/greedy_eval.py score --eval <dir> --gen A=<a.jsonl> --gen B=<b.jsonl> --out report.json
  python scripts/eval/greedy_eval.py decide --run-a <A run dir> --run-b <B run dir> --out decide.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from ubt.paths import DATASETS, WORK  # noqa: E402

EVAL_DIR = f"{DATASETS}/data_u1_v2/eval"
TRAIN = f"{WORK}/data/fixed_split/train.jsonl"
SETS = {"core": "core_per_code.jsonl", "mixed": "mixed_grid.jsonl", "intra": "intra_switch_eval.jsonl"}
LOWRES_MAX = 3000
G1_PP, G2_PP = 1.0, -1.0
G1_GROUPS = ("nonCJK", "ko", "th", "ja", "zh", "lowres", "mixed_grid", "intra")


def code_group(table: str) -> str:
    from ubt.eval_harness import lang_of  # noqa: PLC0415
    lang = lang_of(table)
    if lang in ("ko", "th", "ja"):
        return lang
    return "zh" if lang.startswith("zh") else "nonCJK"


def cmd_build(a) -> None:
    from transformers import AutoTokenizer  # noqa: PLC0415

    from ubt.task_format import render  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    cnt: Counter = Counter()
    with open(a.train, encoding="utf-8") as fh:
        for line in fh:
            for t in set(json.loads(line)["tables"]):
                cnt[t] += 1
    rows = []
    for name, fn in SETS.items():
        for line in open(os.path.join(a.eval_dir, fn), encoding="utf-8"):
            rec = json.loads(line)
            ex = render(rec, hinted=False)
            n_c = len(tok(ex["completion"], add_special_tokens=False)["input_ids"])
            n_p = len(tok(ex["prompt"], add_special_tokens=True)["input_ids"])
            t0 = rec["tables"][0]
            rows.append({"id": rec["id"], "set": name, "tables": rec["tables"], "k": rec["k"],
                         "regime": rec["regime"], "switch_density": rec["switch_density"],
                         "confusable_set": rec.get("confusable_set") or [],
                         "group": code_group(t0) if name == "core" else name,
                         "lowres": name == "core" and cnt[t0] < LOWRES_MAX,
                         "prompt": ex["prompt"], "completion": ex["completion"],
                         "max_tokens": 2 * n_c + 32, "n_prompt_tok_orig": n_p, "n_completion_tok": n_c})
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "eval.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    core_tables = sorted({r["tables"][0] for r in rows if r["set"] == "core"})
    meta = {"n": len(rows), "per_set": Counter(r["set"] for r in rows),
            "per_group_core": Counter(r["group"] for r in rows if r["set"] == "core"),
            "lowres_docs_core": sum(r["lowres"] for r in rows),
            "lowres_tables": {t: cnt[t] for t in core_tables if cnt[t] < LOWRES_MAX},
            "train_docs_per_table": dict(sorted(cnt.items())), "train": a.train, "eval_dir": a.eval_dir,
            "max_prompt_tok_orig": max(r["n_prompt_tok_orig"] for r in rows),
            "max_prompt_plus_cap": max(r["n_prompt_tok_orig"] + r["max_tokens"] for r in rows),
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in meta.items() if k != "train_docs_per_table"}, ensure_ascii=False, indent=1))


def cmd_merge(a) -> None:
    import torch  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, a.ckpt).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    model.save_pretrained(a.out, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(a.out)
    json.dump({"base": a.base, "ckpt": os.path.abspath(a.ckpt), "tokenizer": a.tokenizer,
               "tokenizer_len": len(tok)}, open(os.path.join(a.out, "merged_from.json"), "w"), indent=1)
    print("merged ->", a.out, "tokenizer_len", len(tok), flush=True)


def cmd_gen(a) -> None:
    import time  # noqa: PLC0415

    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    si, sn = (int(x) for x in a.shard.split("/"))
    rows = [json.loads(line) for line in open(os.path.join(a.eval, "eval.jsonl"), encoding="utf-8")][si::sn]
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = [tok(r["prompt"], add_special_tokens=True)["input_ids"] for r in rows]
    t0 = time.time()
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_mem, seed=0, enable_prefix_caching=False)
    sps = [SamplingParams(temperature=0.0, max_tokens=r["max_tokens"], stop_token_ids=[tok.eos_token_id])
           for r in rows]
    outs = llm.generate([{"prompt_token_ids": x} for x in ids], sps)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r, x, o in zip(rows, ids, outs):
            c = o.outputs[0]
            fh.write(json.dumps({"id": r["id"], "hyp": c.text, "finish": c.finish_reason,
                                 "n_tok": len(c.token_ids), "n_prompt_tok": len(x)}, ensure_ascii=False) + "\n")
    print(f"[gen] shard {a.shard}: {len(rows)} docs, {sum(map(len, ids))} prompt tok, "
          f"{time.time() - t0:.0f}s", flush=True)


def _summ(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    ref = sum(r["ref_len"] for r in rows)
    return {"n": n, "cer": sum(r["edit"] for r in rows) / max(1, ref),
            "strict": sum(r["strict"] for r in rows) / n, "conf": sum(r["conf_aware"] for r in rows) / n,
            "seg_match": sum(r["seg_match"] for r in rows) / n, "parse": sum(r["parse_ok"] for r in rows) / n,
            "leak_docs": sum(r["leak"] > 0 for r in rows), "leak_chars": sum(r["leak"] for r in rows),
            "cap_hits": sum(r["finish"] == "length" for r in rows),
            "gen_tok": sum(r["n_tok"] for r in rows), "prompt_tok": sum(r["n_prompt_tok"] for r in rows)}


def _groups(rows: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = {k: [] for k in G1_GROUPS}
    g.update({"core_all": [], "mixed_k2": [], "mixed_k3": [], "mixed_k4": [], "all": list(rows)})
    for r in rows:
        if r["set"] == "core":
            g[r["group"]].append(r)
            g["core_all"].append(r)
            if r["lowres"]:
                g["lowres"].append(r)
        elif r["set"] == "mixed":
            g["mixed_grid"].append(r)
            g[f"mixed_k{r['k']}"].append(r)
        else:
            g["intra"].append(r)
    return g


def cmd_score(a) -> None:
    import numpy as np  # noqa: PLC0415

    from ubt.eval_harness import score_hyp  # noqa: PLC0415
    ev = [json.loads(line) for line in open(os.path.join(a.eval, "eval.jsonl"), encoding="utf-8")]
    arms = {}
    for spec in a.gen:
        name, pat = spec.split("=", 1)
        gen = {}
        for p in sorted(glob.glob(pat)):
            for line in open(p, encoding="utf-8"):
                x = json.loads(line)
                gen[x["id"]] = x
        missing = [r["id"] for r in ev if r["id"] not in gen]
        if missing:
            sys.exit(f"arm {name}: {len(missing)} eval docs without a generation (e.g. {missing[:3]})")
        rows = []
        for r in ev:
            x = gen[r["id"]]
            rec = {"id": r["id"], "k": r["k"], "regime": r["regime"], "switch_density": r["switch_density"],
                   "confusable_set": r["confusable_set"]}
            row = score_hyp(rec, {"completion": r["completion"]}, x["hyp"], None)
            row.update(set=r["set"], group=r["group"], lowres=r["lowres"], finish=x["finish"],
                       n_tok=x["n_tok"], n_prompt_tok=x["n_prompt_tok"],
                       leak=sum(1 for ch in x["hyp"] if 0x2800 <= ord(ch) <= 0x28FF),
                       seg_match=len(row["pred_tables"]) == len(row["tables"]))
            rows.append(row)
        arms[name] = rows
        with open(os.path.join(os.path.dirname(os.path.abspath(a.out)), f"rows_{name}.jsonl"), "w",
                  encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    rep = {"eval_sha256": hashlib.sha256(open(os.path.join(a.eval, "eval.jsonl"), "rb").read()).hexdigest(),
           "arms": {n: {g: _summ(rs) for g, rs in _groups(rows).items()} for n, rows in arms.items()}}
    if {"A", "B"} <= set(arms):
        rng = np.random.default_rng(0)
        ga, gb = _groups(arms["A"]), _groups(arms["B"])
        delta = {}
        for g in ga:
            ra, rb = ga[g], gb[g]
            if not ra:
                continue
            ref = np.array([r["ref_len"] for r in ra], float)
            d = np.array([rb_["edit"] - ra_["edit"] for ra_, rb_ in zip(ra, rb)], float)
            idx = rng.integers(0, len(ra), size=(1000, len(ra)))
            boot = d[idx].sum(1) / np.maximum(1, ref[idx].sum(1)) * 100
            sa, sb = _summ(ra), _summ(rb)
            delta[g] = {"n": len(ra), "cer_pp": (sb["cer"] - sa["cer"]) * 100,
                        "cer_pp_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                        "strict_pp": (sb["strict"] - sa["strict"]) * 100,
                        "conf_pp": (sb["conf"] - sa["conf"]) * 100,
                        "seg_match_pp": (sb["seg_match"] - sa["seg_match"]) * 100,
                        "prompt_tok_ratio": sb["prompt_tok"] / max(1, sa["prompt_tok"])}
        g1 = {g: delta[g]["cer_pp"] <= G1_PP for g in G1_GROUPS if g in delta}
        g2 = {"strict": delta["mixed_grid"]["strict_pp"] >= G2_PP,
              "seg_match": delta["mixed_grid"]["seg_match_pp"] >= G2_PP}
        g3 = rep["arms"]["B"]["all"]["leak_docs"] == 0
        rep["delta_B_minus_A"] = delta
        rep["gates"] = {"G1": g1, "G1_pass": all(g1.values()), "G2": g2, "G2_pass": all(g2.values()),
                        "G3_leak_docs_B": rep["arms"]["B"]["all"]["leak_docs"], "G3_pass": g3,
                        "all_pass": all(g1.values()) and all(g2.values()) and g3}
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep.get("gates", {}), indent=1))
    for g in G1_GROUPS + ("core_all", "all"):
        line = "  ".join(f"{n}: cer {s[g]['cer'] * 100:6.2f} strict {s[g]['strict'] * 100:5.1f}"
                         for n, s in rep["arms"].items() if s[g].get("n"))
        print(f"{g:11s} n={rep['arms'][next(iter(arms))][g]['n']:6d}  {line}")


def _printed_elapsed(run: str) -> dict[tuple[str, int], float]:
    """(kind, step) -> elapsed_sec from the rows printed in <run>/train.log.
    Needed because transformers v5 copies a row into log_history before the callback adds elapsed_sec."""
    import ast  # noqa: PLC0415
    import re  # noqa: PLC0415
    p = os.path.join(run, "train.log")
    out = {}
    if os.path.exists(p):
        for m in re.findall(r"\{'(?:loss|eval_loss)'[^{}]*\}", open(p, encoding="utf-8", errors="replace").read()):
            r = ast.literal_eval(m)
            out[("eval" if "eval_loss" in r else "train", int(r["steps_this_launch"]))] = float(r["elapsed_sec"])
    return out


def _history(run: str) -> list[dict]:
    """log_history of the run's latest checkpoint, with a cumulative `elapsed` across relaunches."""
    cks = sorted(glob.glob(os.path.join(run, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    if not cks:
        sys.exit(f"no checkpoint in {run}")
    hist = json.load(open(os.path.join(cks[-1], "trainer_state.json")))["log_history"]
    printed = _printed_elapsed(run)
    for h in hist:
        kind = "eval" if "eval_loss" in h else "train" if "loss" in h else None
        if "elapsed_sec" not in h and kind and (kind, h.get("step")) in printed:
            h["elapsed_sec"] = printed[(kind, h["step"])]
    off, last, out = 0.0, 0.0, []
    for h in hist:
        if "elapsed_sec" not in h:
            continue
        if h["elapsed_sec"] < last - 1e-6:          # new launch (resume): continue the clock
            off += last
        last = h["elapsed_sec"]
        out.append({**h, "elapsed": off + h["elapsed_sec"]})
    return out


def _slope(evals: list[dict], s_end: int, key: str) -> float | None:
    pts = [(e[key], e["eval_loss"]) for e in evals if s_end - 100 <= e["step"] <= s_end]
    if len(pts) < 2:
        return None
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return sum((x - mx) * (y - my) for x, y in pts) / den if den else None


def cmd_decide(a) -> None:
    ha, hb = _history(a.run_a), _history(a.run_b)
    ev_a = {h["step"]: h for h in ha if "eval_loss" in h}
    ev_b = {h["step"]: h for h in hb if "eval_loss" in h}
    tr_a = {h["step"]: h for h in ha if "loss" in h}
    tr_b = {h["step"]: h for h in hb if "loss" in h}
    s_a = a.step_a
    T = tr_a[s_a]["elapsed"]
    s_b = max(ev_b)
    la, lb = ev_a[s_a]["eval_loss"], ev_b[s_b]["eval_loss"]
    sl_a = _slope(list(ev_a.values()), s_a, "elapsed")
    sl_b = _slope(list(ev_b.values()), s_b, "elapsed")
    rel = abs(lb - la) / la
    converging = sl_a is not None and sl_b is not None and (lb - la) * (sl_b - sl_a) < 0
    out = {"T_sec": T, "A_step": s_a, "B_stop_step": s_b, "B_elapsed_at_stop": tr_b.get(s_b, {}).get("elapsed"),
           "L_A": la, "L_B_T": lb, "D1_B_not_worse": lb <= la, "rel_diff": rel,
           "slope_per_hour_A": sl_a * 3600 if sl_a else None, "slope_per_hour_B": sl_b * 3600 if sl_b else None,
           "converging": converging, "extend": rel < 0.02 and converging,
           "D2": {"L_A_400": ev_a.get(400, {}).get("eval_loss"), "L_B_400": ev_b.get(400, {}).get("eval_loss"),
                  "slope_per_step_A": _slope(list(ev_a.values()), s_a, "step"),
                  "slope_per_step_B_at400": _slope([e for e in ev_b.values() if e["step"] <= 400], 400, "step"),
                  "gap_by_step": {s: ev_b[s]["eval_loss"] - ev_a[s]["eval_loss"] for s in sorted(ev_a) if s in ev_b}},
           "P1": {"A_sec_per_step": tr_a[s_a]["elapsed"] / s_a,
                  "B_sec_per_step": tr_b[400]["elapsed"] / 400 if 400 in tr_b else None}}
    if out["P1"]["B_sec_per_step"]:
        out["P1"]["B_over_A"] = out["P1"]["B_sec_per_step"] / out["P1"]["A_sec_per_step"]
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--eval-dir", default=EVAL_DIR)
    b.add_argument("--train", default=TRAIN)
    b.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B")
    m = sub.add_parser("merge")
    m.add_argument("--ckpt", required=True)
    m.add_argument("--tokenizer", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--base", default="Qwen/Qwen2.5-7B")
    g = sub.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--eval", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--shard", default="0/1")
    g.add_argument("--max-model-len", type=int, default=8192)
    g.add_argument("--gpu-mem", type=float, default=0.90)
    s = sub.add_parser("score")
    s.add_argument("--eval", required=True)
    s.add_argument("--gen", action="append", required=True, help="MODEL=glob of gen jsonl shards")
    s.add_argument("--out", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--run-a", required=True)
    d.add_argument("--run-b", required=True)
    d.add_argument("--step-a", type=int, default=400)
    d.add_argument("--out", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "merge": cmd_merge, "gen": cmd_gen, "score": cmd_score, "decide": cmd_decide}[a.cmd](a)


if __name__ == "__main__":
    main()
