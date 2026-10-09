"""Shared eval harness: row production and metric summary.

Greedy, best-of-n, the LibLouis-backward baseline and the real corpora all
produce the same rows and use the same reporter, so their numbers are comparable.

Row schema (one per doc):
  id, tables (gold), pred_tables, parse_ok, strict, conf_aware,
  cer, edit, ref_len, table_correct, feasible, cell, hyp_eq_ref, hyp_text
  [+ bon_fallback, bon_n_feasible when BoN is on]
"""

from __future__ import annotations

import os
import time
import unicodedata
from collections import defaultdict

from ubt.wandb_utils import parse_target

BATCH = 8

# Language classes. ja (large cipher, memorization-bound) and zh (large fiber,
# prior-bound) fail for different reasons, so they are tracked apart.
FIBER_CLASS = {
    "latin_small": ("en", "fr", "es", "de", "it", "pt", "haw", "ny", "eo",
                    "da", "no", "vi", "cy", "af", "nb", "sv", "fi", "nl"),
    "indic_label": ("hi", "bh", "mr", "ne", "bn", "as", "gu", "pa", "sa"),
    "ko_code": ("ko",),
    "ja_large_cipher": ("ja",),
    "zh_large_fiber": ("zh", "zhcn", "yue"),
}
_LANG2CLASS = {l: c for c, ls in FIBER_CLASS.items() for l in ls}


def lang_of(tbl: str) -> str:
    return (tbl.split("-")[0].split("_")[0]
            .replace(".tbl", "").replace(".ctb", "").replace(".utb", ""))


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# Runaway guard: a hypothesis far longer than the reference cannot re-encode to
# the prompt braille but can stall liblouis for minutes, so it is scored
# infeasible without the call. The loose cap (4x + 64) skips no legitimate hyp.
RUNAWAY_REENCODE_FACTOR = 4


def reencode(louis, segs: list[tuple[str, str]], rec: dict | None = None) -> str | None:
    """f_θ̂(T̂) under the predicted tables (NFC), joined per segment. data-u1
    records (`rec`) use the u1 f, which also handles math and $$…$$ spans."""
    if louis is None or not segs:
        return None
    if rec is not None:
        from ubt.u1.reencode import is_u1_record, reencode_segment  # noqa: PLC0415
        if is_u1_record(rec):
            parts = []
            for tab, text in segs:
                b = reencode_segment(louis, tab, text)
                if b is None:
                    return None
                parts.append(b)
            return "\n".join(parts)
    parts = []
    for tab, text in segs:
        # an unresolvable predicted table raises RuntimeError: infeasible
        try:
            r = louis.translate(tab, nfc(text), check_undefined=False)
        except RuntimeError:
            return None
        if not r.ok:
            return None
        parts.append(r.braille)
    return "\n".join(parts)


def feasible_reencode(louis, segs: list[tuple[str, str]], braille: str,
                      rec: dict | None = None) -> bool:
    return reencode(louis, segs, rec) == braille


def score_hyp(rec: dict, ex: dict, hyp: str, louis) -> dict:
    """One hypothesis -> one metric row (the single source of truth)."""
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415

    from ubt.metrics.textnorm import norm_join  # noqa: PLC0415

    segs = parse_target(hyp)
    gold = parse_target(ex["completion"])
    ref = "\n".join(t for _, t in gold)
    gold_tables = [t for t, _ in gold]
    pred_tables = [t for t, _ in segs]
    hyp_join = "\n".join(t for _, t in segs) if segs else ""
    # headline uses the scoring normal form (NFC, no U+200B, Chinese S/T
    # folded); S/T-sensitive and raw counts are kept alongside
    ref_n, ref_s = norm_join(gold), norm_join(gold, st_fold=False)
    hyp_n = norm_join(segs) if segs else ""
    hyp_s = norm_join(segs, st_fold=False) if segs else ""
    edit = Levenshtein.distance(ref_n, hyp_n) if segs else len(ref_n)
    edit_st = Levenshtein.distance(ref_s, hyp_s) if segs else len(ref_s)
    edit_raw = Levenshtein.distance(ref, hyp_join) if segs else len(ref)
    cer = edit / max(1, len(ref_n))
    strict = pred_tables == gold_tables
    conf_ok = strict or (
        len(pred_tables) == len(gold_tables) and all(
            p == g or p in rec.get("confusable_set", [])
            for p, g in zip(pred_tables, gold_tables)))
    table_correct = feasible = None
    cell = None
    if segs and louis is not None:
        table_correct = strict
        if len(hyp_join) > RUNAWAY_REENCODE_FACTOR * len(ref) + 64:
            feasible = False  # runaway guard: see RUNAWAY_REENCODE_FACTOR
        else:
            feasible = feasible_reencode(louis, segs, rec["braille"], rec)
        if table_correct and feasible:
            cell = "exact" if hyp_n == ref_n else "within_fiber"
        elif table_correct and not feasible:
            cell = "infeasible_under_correct_code"
        elif not table_correct and feasible:
            cell = "cross_code"
        else:
            cell = "both_wrong"
    return {"id": rec["id"], "hyp": hyp,
            "k": rec.get("k"), "regime": rec.get("regime"),
            "switch_density": rec.get("switch_density"),
            "tables": gold_tables, "pred_tables": pred_tables,
            "parse_ok": bool(segs), "strict": strict, "conf_aware": conf_ok,
            "cer": round(cer, 4), "edit": edit, "ref_len": len(ref_n),
            "edit_st": edit_st, "ref_len_st": len(ref_s),
            "edit_raw": edit_raw, "ref_len_raw": len(ref),
            "table_correct": table_correct, "feasible": feasible, "cell": cell,
            "hyp_eq_ref": bool(segs) and hyp_n == ref_n,
            "hyp_text": hyp_join}  # full text, so scoring can be re-checked


def _generate(model, tokenizer, prompts: list[str], max_new: int,
              no_repeat_ngram: int, sample: bool, temp: float,
              n_return: int = 1):
    """Texts per prompt: list[list[str]]."""
    import torch  # noqa: PLC0415

    batch = tokenizer(prompts, return_tensors="pt", padding=True,
                      add_special_tokens=True).to(model.device)
    kwargs = dict(max_new_tokens=max_new, use_cache=True,
                  pad_token_id=tokenizer.pad_token_id)
    if no_repeat_ngram:
        kwargs["no_repeat_ngram_size"] = no_repeat_ngram
    if sample:
        kwargs.update(do_sample=True, temperature=temp, top_p=0.95,
                      num_return_sequences=n_return)
    else:
        kwargs["do_sample"] = False
        # UBT_BEAM>1: beam search for the primary hypothesis, an axis separate
        # from best-of-n (no verifier involved).
        _b = int(os.environ.get("UBT_BEAM", "1"))
        if _b > 1:
            kwargs.update(num_beams=_b, early_stopping=True)
    # UBT_BON_CHUNK: sample in chunks of this many candidates to bound the
    # prefill memory spike; samples are independent, so results are the same.
    _c = int(os.environ.get("UBT_BON_CHUNK", "0"))
    if sample and _c and _c < n_return:
        outs = []
        for _off in range(0, n_return, _c):
            k2 = dict(kwargs)
            k2["num_return_sequences"] = min(_c, n_return - _off)
            with torch.no_grad():
                outs.append(model.generate(**batch, **k2))
        _maxlen = max(o.shape[1] for o in outs)
        _pad = tokenizer.pad_token_id or 0
        outs = [torch.nn.functional.pad(o, (0, _maxlen - o.shape[1]),
                                        value=_pad) for o in outs]
        # regroup so texts[i*n_return:(i+1)*n_return] belong to prompt i
        import itertools
        n_p = batch["input_ids"].shape[0]
        per_prompt = [[] for _ in range(n_p)]
        for o, _off in zip(outs, range(0, n_return, _c)):
            cn = min(_c, n_return - _off)
            for pi in range(n_p):
                per_prompt[pi].append(o[pi * cn:(pi + 1) * cn])
        gen = torch.cat(list(itertools.chain.from_iterable(
            ([torch.cat(chunks, dim=0)] for chunks in per_prompt))), dim=0)
    else:
        with torch.no_grad():
            gen = model.generate(**batch, **kwargs)
    # decoder-only outputs start with the prompt; encoder-decoder outputs do not
    plen = 0 if getattr(model.config, "is_encoder_decoder", False) \
        else batch["input_ids"].shape[1]
    texts = [tokenizer.decode(g[plen:], skip_special_tokens=True) for g in gen]
    return [texts[i * n_return:(i + 1) * n_return]
            for i in range(len(prompts))]


def _seq_logprobs(model, tokenizer, prompts: list[str],
                  conts: list[str]) -> list[float]:
    """Sum log p(continuation | prompt) by teacher forcing; output_scores would
    run out of memory at a 152k vocab."""
    import torch  # noqa: PLC0415

    out = []
    for i in range(0, len(prompts), BATCH):
        ps, cs = prompts[i:i + BATCH], conts[i:i + BATCH]
        seqs, plens = [], []
        for p, c in zip(ps, cs):
            pi = tokenizer(p, add_special_tokens=True)["input_ids"]
            ci = tokenizer(c, add_special_tokens=False)["input_ids"]
            seqs.append(pi + ci)
            plens.append(len(pi))
        width = max(len(s) for s in seqs)
        pad = tokenizer.pad_token_id
        ids = torch.tensor([[pad] * (width - len(s)) + s for s in seqs],
                           device=model.device)
        attn = torch.tensor([[0] * (width - len(s)) + [1] * len(s)
                             for s in seqs], device=model.device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=attn).logits
            # log_softmax over continuation positions only, one sequence at a
            # time: the full [B, width, V] float32 tensor runs out of memory.
            for j, s in enumerate(seqs):
                off = width - len(s)
                idx = [off + t - 1 for t in range(plens[j], len(s))]
                tgt = [s[t] for t in range(plens[j], len(s))]
                if not idx:
                    out.append(0.0)
                    continue
                sel = torch.log_softmax(logits[j, idx, :].float(), dim=-1)
                tt = torch.tensor(tgt, device=sel.device)
                out.append(sel[range(len(tgt)), tt].sum().item())
            del logits
    return out


def _select(scored: list[dict], policy: str) -> dict:
    """The candidate {hyp, logprob, d_cell} that `policy` selects.
    d_cell None means unparsable or re-encode failure."""
    feas = [c for c in scored if c["d_cell"] == 0]
    if policy == "oracle":
        return min(scored, key=lambda c: c["cer_true"])
    if feas:
        return max(feas, key=lambda c: c["logprob"])
    usable = [c for c in scored if c["d_cell"] is not None] or scored
    if policy == "hard_greedy":       # wipeout -> greedy (candidate 0)
        return scored[0]
    if policy == "lambda2_fbinf":     # product policy: λ=2, λ→∞ on wipeout
        return min(usable, key=lambda c: ((c["d_cell"]
                   if c["d_cell"] is not None else 10**6), -c["logprob"]))
    if policy.startswith("lambda"):   # Eq. (1) MAP: argmax [logprob - λ·d_cell]
        if policy == "lambda_inf":    # λ→∞ limit: min d_cell, tie-break logprob
            return min(usable, key=lambda c: ((c["d_cell"]
                       if c["d_cell"] is not None else 10**6), -c["logprob"]))
        lam = float(policy[len("lambda"):])
        return max(usable, key=lambda c: c["logprob"] - lam * (c["d_cell"]
                   if c["d_cell"] is not None else 10**6))
    raise ValueError(policy)


# preview set; the canonical run passes the full λ sweep
BON_POLICIES = ("hard_greedy", "lambda5", "lambda_inf", "oracle")
BON_POLICIES_CANONICAL = ("hard_greedy", "lambda1", "lambda2", "lambda5",
                          "lambda10", "lambda_inf", "oracle")


def predict_rows(model, tokenizer, records: list[dict], hinted: bool, louis,
                 max_new: int = 768, no_repeat_ngram: int = 0,
                 bon: int = 0, bon_temp: float = 0.8,
                 policies: tuple = BON_POLICIES,
                 log=print,
                 checkpoint=None) -> tuple[list[dict], dict[str, list[dict]]]:
    """Batched decode -> (greedy_rows, {policy: rows}), {} if bon == 0.
    All BoN policies select from one pool: greedy + (n-1) samples (T=bon_temp, top-p .95)."""
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415

    from ubt.metrics.textnorm import norm_join  # noqa: PLC0415
    from ubt.task_format import render  # noqa: PLC0415

    # HF n-gram blocking spans the prompt for decoder-only models and would stop
    # hinted prompts from copying long table ids (e.g. da-dk-*_1993).
    if hinted and no_repeat_ngram:
        raise ValueError(
            "no_repeat_ngram is prompt-inclusive and forbidden for hinted "
            "prompts (blocks verbatim table-id re-emission)")

    # UBT_FORCE_CODE=1 (hinted, single-segment docs): prefill the given table
    # label so the model writes only the text; otherwise alias tables with the
    # same cells (ko-g1, ko-2006-g1) get relabelled to the alias seen most in
    # training.
    force_code = hinted and os.environ.get("UBT_FORCE_CODE") == "1"

    greedy_rows = []
    bon_rows: dict[str, list[dict]] = {p: [] for p in policies} if bon else {}
    t0 = time.time()
    for i in range(0, len(records), BATCH):
        chunk = records[i:i + BATCH]
        exs = [render(r, hinted=hinted) for r in chunk]
        prompts = [e["prompt"] for e in exs]
        # the prefilled "⟨table⟩" is put back on the outputs before scoring
        pre = [""] * len(chunk)
        if force_code:
            pre = ["⟨" + r["tables"][0] + "⟩" for r in chunk]
            prompts = [p + q for p, q in zip(prompts, pre)]
        greedy = _generate(model, tokenizer, prompts, max_new,
                           no_repeat_ngram, sample=False, temp=0.0)
        if force_code:
            greedy = [[q + t for t in g] for q, g in zip(pre, greedy)]
        g_rows = [score_hyp(rec, ex, g[0], louis)
                  for rec, ex, g in zip(chunk, exs, greedy)]
        greedy_rows.extend(g_rows)
        if bon > 0:
            n_samp = max(1, bon - 1)
            sub = max(1, BATCH // max(1, n_samp // 2))
            for j in range(0, len(chunk), sub):
                sc, se = chunk[j:j + sub], exs[j:j + sub]
                spre = pre[j:j + sub]
                sampled = _generate(model, tokenizer,
                                    [e["prompt"] + q
                                     for e, q in zip(se, spre)], max_new,
                                    no_repeat_ngram, sample=True,
                                    temp=bon_temp, n_return=n_samp)
                if force_code:
                    sampled = [[q + t for t in cands]
                               for q, cands in zip(spre, sampled)]
                # one flat logprob pass over every (prompt, candidate)
                flat_p, flat_c = [], []
                pools = []
                for ex, cands, g in zip(se, sampled,
                                        greedy[j:j + sub]):
                    pool = [g[0]] + cands
                    pools.append(pool)
                    flat_p.extend([ex["prompt"]] * len(pool))
                    flat_c.extend(pool)
                lps = _seq_logprobs(model, tokenizer, flat_p, flat_c)
                k = 0
                for rec, ex, pool in zip(sc, se, pools):
                    scored = []
                    ref = norm_join(parse_target(ex["completion"]))
                    for hyp in pool:
                        segs = parse_target(hyp)
                        enc = reencode(louis, segs, rec) if segs else None
                        d_cell = (Levenshtein.distance(enc, rec["braille"])
                                  if enc is not None else None)
                        hyp_join = norm_join(segs) if segs else ""
                        cer_true = (Levenshtein.distance(ref, hyp_join)
                                    / max(1, len(ref)) if segs else 1.0)
                        scored.append({"hyp": hyp, "logprob": lps[k],
                                       "d_cell": d_cell,
                                       "cer_true": cer_true})
                        k += 1
                    n_feas = sum(1 for c in scored if c["d_cell"] == 0)
                    for pol in policies:
                        row = score_hyp(rec, ex, _select(scored, pol)["hyp"],
                                        louis)
                        row["bon_fallback"] = n_feas == 0
                        row["bon_n_feasible"] = n_feas
                        bon_rows[pol].append(row)
        # partial save every ~200 docs: each save rewrites all rows, so saving
        # per batch would cost O(n^2)
        if checkpoint is not None and (i % 200) < BATCH:
            try:
                checkpoint(greedy_rows, bon_rows)
            except Exception as _e:  # never let a save error kill decode
                log(f"  [checkpoint failed: {_e}]")
        if i % 200 < BATCH:
            log(f"  {i + len(chunk)}/{len(records)} ({time.time() - t0:.0f}s)")
    return greedy_rows, bon_rows


def _group(rows: list[dict], key: str) -> dict:
    out = defaultdict(list)
    for r in rows:
        v = r.get(key)
        if v is not None:
            out[v].append(r)
    return out


def summarize(rows: list[dict], noisy_reference: bool = False) -> dict:
    """The single reporter: micro CER overall and per class/lang, the CER
    distribution, and the 2x2 cells with invariant asserts."""
    n = len(rows)
    parsed = sum(r["parse_ok"] for r in rows)

    def micro(rs):
        te, tr = sum(r["edit"] for r in rs), sum(r["ref_len"] for r in rs)
        return round(te / tr, 4) if tr else None

    by_lang, by_class, by_table = (defaultdict(list), defaultdict(list),
                                   defaultdict(list))
    for r in rows:
        lang = lang_of(r["tables"][0])
        by_lang[lang].append(r)
        if lang in _LANG2CLASS:
            by_class[_LANG2CLASS[lang]].append(r)
        by_table[r["tables"][0]].append(r)

    dec = [r for r in rows if r["cell"] is not None]
    cells = {c: sum(1 for r in dec if r["cell"] == c) for c in
             ("exact", "within_fiber", "infeasible_under_correct_code",
              "cross_code", "both_wrong")}
    nd = len(dec)
    n_feas = sum(1 for r in dec if r["feasible"])
    n_tab = sum(1 for r in dec if r["table_correct"])
    assert sum(cells.values()) == nd, "2x2 cells must sum to decomposable N"
    assert (cells["exact"] + cells["within_fiber"] + cells["cross_code"]
            == n_feas), "top row must equal feasible count"
    assert (cells["exact"] + cells["within_fiber"]
            + cells["infeasible_under_correct_code"] == n_tab), \
        "left column must equal strict count"
    true_bug = sum(1 for r in dec if r["table_correct"]
                   and not r["feasible"] and r["hyp_eq_ref"])
    # Right text and table yet infeasible means a table defect (the reference
    # does not round-trip under f). Noise slices are infeasible by design and
    # skip the check; UBT_ALLOW_TRUE_BUG=1 turns the error into a warning.
    if not noisy_reference and true_bug > 0:
        msg = f"TRUE BUG: {true_bug} right-text+right-table infeasible"
        if os.environ.get("UBT_ALLOW_TRUE_BUG") == "1":
            print(f"WARNING (table defect, non-fatal): {msg}", flush=True)
        else:
            raise AssertionError(msg)

    srt = sorted(r["cer"] for r in rows)
    iw = [r for r in rows
          if lang_of(r["tables"][0]) in FIBER_CLASS["indic_label"]
          and not r["strict"]]
    out = {
        "n": n, "n_decomposable": nd,
        "parse_rate": round(parsed / n, 4),
        "code_id_strict": round(sum(r["strict"] for r in rows) / n, 4),
        "code_id_conf_aware": round(sum(r["conf_aware"] for r in rows) / n, 4),
        "cer_micro": micro(rows),
        "cer_micro_per_class": {c: micro(rs) for c, rs in by_class.items()},
        "cer_micro_per_lang": {l: micro(rs) for l, rs in by_lang.items()
                               if len(rs) >= 3},
        "distribution": {
            "cer_macro": round(sum(r["cer"] for r in rows) / n, 4),
            "cer_median": round(srt[n // 2], 4),
            "cer_trim5": round(sum(srt[int(n * .05):int(n * .95)])
                               / max(1, int(n * .95) - int(n * .05)), 4),
            "cer_p90": round(srt[int(n * .9)], 4),
            "text_em": round(sum(1 for r in rows if r["cer"] == 0) / n, 4),
            "joint_em": round(cells["exact"] / max(1, n), 4),
            "runaway_rate": round(sum(1 for r in rows if r["cer"] > 2) / n, 4),
            "note": "runaway magnitude is cap-censored at max_new",
        },
        "cells_2x2": cells,
        "headline_3way": {
            "exact": cells["exact"], "within_fiber": cells["within_fiber"],
            "cross_code": cells["cross_code"],
            "infeasible": (cells["infeasible_under_correct_code"]
                           + cells["both_wrong"])},
        "per_k_micro": {str(k): micro(rs) for k, rs in
                        _group(rows, "k").items()} or None,
        "per_regime_micro": {g: micro(rs) for g, rs in
                             _group(rows, "regime").items()} or None,
        "per_density_micro": {d: micro(rs) for d, rs in
                              _group(rows, "switch_density").items()} or None,
        "indic_wrongtable": {
            "n": len(iw),
            "text_intact_cer_lt_005": sum(1 for r in iw if r["cer"] < 0.05)},
        "zh_per_table": {t: {"micro": micro(rs), "n": len(rs)}
                         for t, rs in by_table.items()
                         if lang_of(t) in FIBER_CLASS["zh_large_fiber"]
                         and len(rs) >= 2},
    }
    if rows and "bon_fallback" in rows[0]:
        # a histogram, not a mean: the count is bimodal
        hist = {"0": 0, "1": 0, "2+": 0}
        for r in rows:
            k = r["bon_n_feasible"]
            hist["0" if k == 0 else "1" if k == 1 else "2+"] += 1
        out["bon"] = {
            "fallback_rate": round(sum(r["bon_fallback"] for r in rows) / n, 4),
            "n_feasible_hist": hist,
        }
    return out
