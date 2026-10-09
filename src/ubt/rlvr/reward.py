"""Verifiable RLVR rewards for braille -> print.

Definitions match the paper evaluator (ubt.eval_harness.score_hyp):
  completion = "\\n".join("⟨table⟩text" per segment)
  CER        = Levenshtein(ref_join, hyp_join) / max(1, len(ref_join)), joins in the scoring normal form
  feasible   = one segment per braille line, and f_table(text) == that line for every segment
  conf_ok    = each predicted label is gold or in the record's confusable_set

Reward (per completion; GRPO normalises within the group, so only differences matter):
  parse failure / empty                       -> R_FAIL
  R = w_cer * max(0, 1 - CER)
    + w_feas * feasible
    + w_exact * (hyp_join == ref_join)
    + w_label * conf_ok
  runaway (hyp much longer than ref) is never re-encoded and gets feasible = 0.

f is the data's per-table-isolated liblouis (ubt.rlvr.louis_pool).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SEG_RE = re.compile(r"⟨([^⟩]+)⟩(.*)", re.S)
RUNAWAY_FACTOR = 3.0

# model label -> engine table, where the data-u1 engine names a table differently
LABEL_TO_ENGINE = {"ko-2020-g2.ctb": "ko-2024-g2.ctb", "hu.tbl": "hu-hu-g1.ctb",
                   "it.tbl": "it-it-comp6.utb"}
ENGINE_TO_LABEL = {v: k for k, v in LABEL_TO_ENGINE.items()}


def parse_completion(text: str) -> list[tuple[str, str]] | None:
    """'⟨t1⟩x1\\n⟨t2⟩x2' -> [(t1,x1),(t2,x2)]; None if malformed."""
    text = text.strip()
    if not text.startswith("⟨"):
        return None
    segs = []
    for chunk in text.split("\n⟨"):
        chunk = chunk if chunk.startswith("⟨") else "⟨" + chunk
        m = SEG_RE.fullmatch(chunk)
        if not m:
            return None
        segs.append((m.group(1).strip(), m.group(2)))
    return segs or None


@dataclass
class RewardConfig:
    w_cer: float = 1.0
    w_feas: float = 0.5
    w_exact: float = 0.25
    w_label: float = 0.25
    r_fail: float = -0.5
    runaway_factor: float = RUNAWAY_FACTOR
    extra: dict = field(default_factory=dict)


def levenshtein(a: str, b: str) -> int:
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415
    return Levenshtein.distance(a, b)


def score_batch(completions: list[str], golds: list[str], brailles: list[str],
                confusables: list[list[str]], pool, cfg: RewardConfig) -> list[dict]:
    """One detail dict per completion, with the reward in 'reward'."""
    from ubt.metrics.textnorm import norm_join  # noqa: PLC0415
    parsed = [parse_completion(c) for c in completions]
    gold_p = [parse_completion(g) or [] for g in golds]
    # one pool call for every re-encoding in the batch
    reqs, where = [], []
    for i, (segs, g, b) in enumerate(zip(parsed, gold_p, brailles)):
        if not segs:
            continue
        ref_join = "\n".join(t for _, t in g)
        hyp_join = "\n".join(t for _, t in segs)
        lines = b.split("\n")
        if len(segs) != len(lines) or len(hyp_join) > cfg.runaway_factor * len(ref_join) + 64:
            continue
        for j, (tab, text) in enumerate(segs):
            reqs.append((LABEL_TO_ENGINE.get(tab, tab), text))
            where.append((i, j))
    outs = pool.translate_many(reqs) if reqs else []
    enc: dict[int, dict[int, str | None]] = {}
    for (i, j), o in zip(where, outs):
        enc.setdefault(i, {})[j] = o
    rows = []
    for i, (segs, g, b, conf) in enumerate(zip(parsed, gold_p, brailles, confusables)):
        if not segs:
            rows.append({"reward": cfg.r_fail, "parse_ok": False, "cer": 1.0, "feasible": False,
                         "exact": False, "conf_ok": False})
            continue
        ref_join = norm_join(g)
        hyp_join = norm_join(segs)
        cer = levenshtein(ref_join, hyp_join) / max(1, len(ref_join))
        lines = b.split("\n")
        feas = i in enc and len(enc[i]) == len(lines) and all(
            enc[i].get(j) is not None and enc[i][j] == lines[j] for j in range(len(lines)))
        exact = hyp_join == ref_join
        gold_tabs = [t for t, _ in g]
        conf_set = set(conf or []) | {ENGINE_TO_LABEL.get(c, c) for c in (conf or [])}
        conf_ok = len(segs) == len(gold_tabs) and all(
            p == q or p in conf_set or LABEL_TO_ENGINE.get(p) == q or ENGINE_TO_LABEL.get(q) == p
            for (p, _), q in zip(segs, gold_tabs))
        r = (cfg.w_cer * max(0.0, 1.0 - cer) + cfg.w_feas * float(feas)
             + cfg.w_exact * float(exact) + cfg.w_label * float(conf_ok))
        rows.append({"reward": r, "parse_ok": True, "cer": cer, "feasible": feas,
                     "exact": exact, "conf_ok": conf_ok})
    return rows
