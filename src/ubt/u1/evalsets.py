"""data-u1 eval grid: the cells of ubt.eval_grid on the pinned 3.38 + ko-2024 engine, from eval sources only.

  core_per_code      top-45 weighted tables x 200, the rest x 100 (1 sentence)
  snippets           20 groups (all script groups + largest language groups) x 5 length buckets x 300
  mixed_grid         k in {2,3,4} x regime {anchor, uniform, confusable} x 400
  k2_paragraph       regime {anchor, uniform} x 400, 4-8 sentences per segment
  intra_switch_eval  density {light, heavy} x 500 (natural_first off)
  kodisc             500 same-text pairs ko-2024-g2 <-> ko-2006-g2 (pair_id)
  zeroshot_eval      8 held-out tables x 200 + 500 held-out/anchor mixed
  noise_variants     2,000 core rows x p in {.005, .01, .02} (f-check exempt; see gen_noise)
  math_eval          Nemeth 1,000 (+ Korean math 1,000 when ubt.kmath exists)
Every cell target is multiplied by --eval-scale (min 1).
"""
from __future__ import annotations

import copy
import random

from ubt.eval_grid import _BUCKET_WORDS, NOISE_LEVELS, SNIPPET_BUCKETS
from ubt.noise import flip_dots
from ubt.u1.engine import KO2024_LABEL
from ubt.u1.work import process


def sc(n: int, s: float) -> int:
    return 0 if n == 0 else max(1, round(n * s))


def snippet_plan(planner, table, max_words: int) -> dict | None:
    """planner.plan_snippet with edge whitespace stripped, so the braille never starts or ends with a blank cell."""
    p = planner.plan_snippet(table, max_words=max_words)
    if p is None:
        return None
    seg = p["segments"][0]
    t = seg["text"].strip()
    if not t:
        return None
    seg["text"] = t
    return p


def gen_core(run, planner, ecfg: dict, s: float) -> list[dict]:
    out = []
    ranked = sorted(planner.weighted, key=lambda t: (-t.weight, t.table_id))
    for i, t in enumerate(ranked):
        target = sc(ecfg["core_head_docs"] if i < ecfg["core_head_n"] else ecfg["core_tail_docs"], s)
        got = run.fill(target, lambda t=t: planner.plan_single(table=t, n_sents=1),
                       form="Ecore", split="eval", tag="core_per_code")
        run.shortfall("eval:core_per_code", t.table_id, len(got), target,
                      "source_exhaustion_or_charset")
        out += got
    return out


def gen_snippets(run, planner, groups: dict[str, list[str]], per_cell: int) -> list[dict]:
    out = []
    rng = run.rng
    for gid, members in sorted(groups.items()):
        tabs = [planner.mixer.by_id[m] for m in members if m in planner.mixer.by_id]
        if not tabs:
            run.shortfall("eval:snippets", f"{gid}:all", 0, per_cell * len(SNIPPET_BUCKETS),
                          "no weighted table in group")
            continue
        need = {b: per_cell for b in SNIPPET_BUCKETS}

        def plan_fn(tabs=tabs, need=need):
            open_b = [b for b, v in need.items() if v > 0]
            if not open_b:
                return None
            b = rng.choices(open_b, weights=[need[x] for x in open_b])[0]
            t = rng.choice(tabs)
            if b == "sentence":
                return planner.plan_single(table=t, n_sents=1)
            return snippet_plan(planner, t, rng.randint(*_BUCKET_WORDS[b]))

        def filt(rec, need=need):
            b = rec["len_bucket"]
            b = "sentence" if b in ("sentence", "paragraph") else b
            if need.get(b, 0) > 0:
                need[b] -= 1
                rec["_bucket"] = b
                return True
            return False

        def tag(rec, gid=gid):
            return f"snippet:{gid}:{rec.pop('_bucket')}"

        got = run.fill(per_cell * len(SNIPPET_BUCKETS), plan_fn, form="Esnip",
                       split="eval", tag=tag, filter_fn=filt, max_rounds=60)
        for b, deficit in need.items():
            run.shortfall("eval:snippets", f"{gid}:{b}", per_cell - deficit, per_cell,
                          "source_exhaustion_or_bucket_miss")
        out += got
    return out


def gen_mixed(run, planner, per_cell: int) -> list[dict]:
    out = []
    for k in (2, 3, 4):
        for regime in ("anchor", "uniform", "confusable"):
            got = run.fill(per_cell, lambda k=k, r=regime: planner.plan_inter(k, regime=r),
                           form="Emix", split="eval", tag=f"mixed_grid:k{k}:{regime}")
            run.shortfall("eval:mixed_grid", f"k{k}:{regime}", len(got), per_cell, "source_exhaustion")
            out += got
    return out


def gen_k2_paragraph(run, planner, per_cell: int) -> list[dict]:
    out = []
    for regime in ("anchor", "uniform"):
        got = run.fill(per_cell, lambda r=regime: planner.plan_inter(2, regime=r, seg_sents=(4, 8)),
                       form="Ek2p", split="eval", tag=f"k2_paragraph:{regime}")
        run.shortfall("eval:k2_paragraph", regime, len(got), per_cell, "source_exhaustion")
        out += got
    return out


def gen_intra(run, planner, per_cell: int) -> list[dict]:
    out = []
    for density in ("light", "heavy"):
        got = run.fill(per_cell, lambda d=density: planner.plan_intra_switch(density=d),
                       form="Eintra", split="eval", tag=f"intra_switch_eval:{density}")
        run.shortfall("eval:intra_switch_eval", density, len(got), per_cell, "source_exhaustion")
        out += got
    return out


def gen_kodisc(run, planner, n_pairs: int, tables: tuple[str, str]) -> list[dict]:
    """The same Latin-spliced Korean text under both tables, each twin with its own sibling list."""
    ta, tb = (planner.by_id.get(t) for t in tables)
    if not (ta and tb):
        run.shortfall("eval:kodisc", "all", 0, n_pairs, f"tables missing {tables}")
        return []
    out: list[dict] = []
    done = 0
    for _ in range(40):
        if done >= n_pairs:
            break
        flat = []
        for _ in range(int((n_pairs - done) * 1.3) + 4):
            base = planner.plan_intra_switch(table=ta)
            if base is None:
                run.reject("Ekodisc", "no_source")
                continue
            text = base["segments"][0]["text"]
            for t in (ta, tb):
                p = copy.deepcopy(base)
                p["segments"][0] = planner._segment(t, text)
                flat.append(p)
        if not flat:
            break
        results = list(run.pool.imap(process, flat, run.chunksize))
        progress = 0
        for i in range(0, len(results) - 1, 2):
            if done >= n_pairs:
                break
            built = [run.build(r, form="Ekodisc", split="eval") for r in results[i:i + 2]]
            if not all(built):
                continue
            pid = f"kodisc-{done:05d}"
            for rec, key in built:
                rec["pair_id"] = pid
                out.append(run.commit(rec, key, form="Ekodisc", split="eval",
                                      tag="ko_discrimination"))
            done += 1
            progress += 1
        if progress == 0:
            break
    run.shortfall("eval:kodisc", "pairs", done, n_pairs, "rejections")
    return out


def gen_zeroshot(run, planner, per_table: int, n_mixed: int) -> list[dict]:
    out = []
    for t in sorted(planner.holdout_tables, key=lambda t: t.table_id):
        got = run.fill(per_table, lambda t=t: planner.plan_single(table=t),
                       form="Ezs", split="eval", tag=f"zeroshot_eval:{t.table_id}")
        run.shortfall("eval:zeroshot_eval", t.table_id, len(got), per_table,
                      "source_exhaustion_or_charset")
        out += got

    def plan_mixed():
        if not planner.holdout_tables:
            return None
        t_hold = run.rng.choice(planner.holdout_tables)
        other = planner.mixer.choose_tables("anchor", 1, run.rng)
        segs, srcs, n = [], [], 0
        for t in (t_hold, other[0]):
            got = planner._sample_text(t.base_lang, run.rng.randint(1, 2))
            if got is None:
                return None
            text, src_name, n_act = got
            segs.append(planner._segment(t, text))
            srcs.append(src_name)
            n += n_act
        return {"doc_type": "inter", "regime": "anchor", "switch_density": "none",
                "n_sentences": n, "source": "+".join(sorted(set(srcs))), "segments": segs}

    got = run.fill(n_mixed, plan_mixed, form="Ezs", split="eval", tag="zeroshot_eval:mixed")
    run.shortfall("eval:zeroshot_eval", "mixed", len(got), n_mixed, "source_exhaustion")
    return out + got


def _eight_dot_tables(rows: list[dict]) -> set[str]:
    """Tables whose rows use a dot-7/8 cell anywhere (8-dot codes)."""
    out = set()
    for r in rows:
        if any((ord(c) - 0x2800) & 0xC0 for c in r["braille"] if 0x2800 <= ord(c) <= 0x28FF):
            out.add(r["tables"][0])
    return out


def _n_flips(a: str, b: str) -> int:
    return sum(bin((ord(x) - 0x2800) ^ (ord(y) - 0x2800)).count("1")
               for x, y in zip(a, b) if x != y)


def gen_noise(run, core: list[dict], n_base: int, rng: random.Random) -> list[dict]:
    """Dot-flip variants of core rows; 6-dot codes (no dot 7/8 in any core row) flip dots 1-6 only. A draw with no
    flip is resampled, so p is the per-dot rate given >= 1 flip; noise_n_flips records the count."""
    base = rng.sample(core, min(n_base, len(core)))
    run.shortfall("eval:noise_variants", "base", len(base), n_base, "not enough core rows")
    eight = _eight_dot_tables(core)
    out = []
    for p in NOISE_LEVELS:
        for r in base:
            nd = 8 if r["tables"][0] in eight else 6
            for _ in range(10000):
                b = flip_dots(r["braille"], p, rng, n_dots=nd)
                if b != r["braille"]:
                    break
            else:
                raise RuntimeError(f"noise: no flip after 10000 draws for {r['id']}")
            noisy = copy.deepcopy(r)
            noisy["id"] = run.next_id("Enoise")
            noisy["form"] = "Enoise"
            noisy["braille"] = b
            noisy["noise_p"] = p
            noisy["noise_n_dots"] = nd
            noisy["noise_n_flips"] = _n_flips(r["braille"], b)
            noisy["noise_base_id"] = r["id"]
            noisy["tag"] = "noise_variants"
            out.append(noisy)
    return out


def nikl_pointer_rows(nikl_rows: list[dict]) -> list[dict]:
    """Sealed NIKL test split: ids only (no text, no braille)."""
    return [{"nikl_id": r["id"], "doc": r["doc"], "ed": r["ed"], "genre": r["genre"],
             "split": "test", "table": KO2024_LABEL}
            for r in nikl_rows if r["split"] == "test"]
