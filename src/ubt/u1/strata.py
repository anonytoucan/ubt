"""data-u1 training strata S1 to S7.

Each generator returns the accepted rows of its stratum for one split (train or dev; the builder runs dev first, from
the carved dev source lines) and records shortfalls on the run. A pending engine (S4b without ubt.kmath) gives no rows
and an explicit `pending` shortfall, never a stand-in.

Math formulas are split by canonical form (math_side): eval draws only canonicals with sha1 % 5 == 0 and train/dev the
rest, so eval, drawn first, cannot drain a small formula family. Standalone and embedded rows fill exact quotas.
"""
from __future__ import annotations

import hashlib
import random

from ubt.eval_grid import _BUCKET_WORDS, SNIPPET_BUCKETS
from ubt.math_norm import nu_math
from ubt.u1 import kmath_adapter as KM
from ubt.u1 import nemeth as NM
from ubt.u1.engine import KMATH_LABEL, KO2024_LABEL, NEMETH_LABEL
from ubt.u1.evalsets import snippet_plan
from ubt.u1.nikl import n_sentences as nikl_n_sentences
from ubt.u1.work import process

MATH_EVAL_MOD = 5


def math_side(canon: str) -> str:
    """'eval' for ~1/5 of canonical formulas, 'train' (train + dev) for the rest."""
    h = int(hashlib.sha1(canon.encode("utf-8")).hexdigest()[:8], 16)
    return "eval" if h % MATH_EVAL_MOD == 0 else "train"


# ----------------------------------------------------------------------- S1
def _sf(stratum: str, split: str) -> str:
    """Shortfall stratum name (dev cells are reported as dev:<stratum>)."""
    return stratum if split == "train" else f"{split}:{stratum}"


def gen_s1(run, planner, target: int, floors: dict[str, int],
           split: str = "train") -> tuple[list[dict], list[dict], dict]:
    """Fill each table's floor, then the rest by inventory weight (dev has no floors).
    Returns (floor_rows, weighted_rows, per-table floor report)."""
    floor_rows: list[dict] = []
    report = {}
    for tid in sorted(floors):
        t = planner.by_id[tid]
        need = floors[tid]
        got = run.fill(need, lambda t=t: planner.plan_single(table=t),
                       form="S1", split=split, tag="S1:floor")
        report[tid] = {"floor": need, "floor_phase": len(got)}
        run.shortfall(_sf("S1:floor", split), tid, len(got), need, "source_exhaustion_or_charset")
        floor_rows += got
    rest = max(0, target - len(floor_rows))
    weighted = run.fill(rest, planner.plan_single, form="S1", split=split,
                        tag="S1:weighted") if rest else []
    run.shortfall(_sf("S1", split), "weighted", len(weighted), rest, "stall")
    return floor_rows, weighted, report


# ----------------------------------------------------------------------- S1b
def gen_s1b(run, planner, pools: dict[str, list[str]], tables: dict[str, list[str]],
            per_table: dict[str, int], split: str = "train") -> list[dict]:
    """Per-table deep pairs: each reserved line is transcribed under every open table of its language and given to
    the acceptor with the largest relative remaining need, so it is used with one table only."""
    out: list[dict] = []
    for lang in sorted(pools):
        lines = pools[lang]
        tabs = [t for t in tables[lang] if t in per_table]
        need = {t: per_table[t] for t in tabs}
        sibs = {t: planner._segment(planner.by_id[t], "")["siblings"] for t in tabs}
        lang_of = {t: planner.by_id[t].base_lang for t in tabs}
        offered = {t: 0 for t in tabs}
        accepted = {t: 0 for t in tabs}
        ptr = 0
        while ptr < len(lines) and any(v > 0 for v in need.values()):
            open_t = [t for t in tabs if need[t] > 0]
            b = max(64, int(sum(need.values()) * 1.25))
            batch = lines[ptr:ptr + b]
            ptr += len(batch)
            plans = []
            for text in batch:
                if run.screened_out(text, "S1b", split):
                    continue
                plans.append({"kind": "s1b", "text": text, "tables": open_t,
                              "lang_of": lang_of, "siblings": sibs,
                              "doc_type": "single", "regime": "single",
                              "switch_density": "none", "n_sentences": 1,
                              "source": f"wiki_A:{lang}:s1b"})
            for res in run.pool.imap(process, plans, run.chunksize):
                for t in res["plan"]["tables"]:
                    offered[t] += 1
                    accepted[t] += t in res["options"]
                cands = [t for t in res["options"] if need[t] > 0]
                if not cands:
                    for t, why in res["reasons"].items():
                        run.reject("S1b", why, t)
                    if res["options"]:
                        run.reject("S1b", "cell_full")
                    continue
                # neediest relative to quota, boosted for tables that accept few lines
                t = max(cands, key=lambda t: (need[t] / per_table[t]
                                              / ((accepted[t] + 1) / (offered[t] + 2)), t))
                rec = run.accept({"ok": True, "plan": res["plan"], "segments": [res["options"][t]]},
                                 form="S1b", split=split, tag=f"S1b:{t}")
                if rec is not None:
                    need[t] -= 1
                    out.append(rec)
        for t in tabs:
            run.shortfall(_sf("S1b", split), t, per_table[t] - need[t], per_table[t],
                          f"reserved {lang} lines exhausted ({ptr}/{len(lines)} used)")
    return out


# ----------------------------------------------------------------------- S2 / S3
def gen_s2(run, planner, target: int, split: str = "train") -> list[dict]:
    got = run.fill(target, planner.plan_intra_switch, form="S2", split=split,
                   tag=lambda r: f"S2:intra_switch:{r['switch_density']}")
    run.shortfall(_sf("S2", split), "all", len(got), target, "stall")
    return got


def gen_s3(run, planner, k: int, target: int, paragraph: bool = False,
           split: str = "train") -> list[dict]:
    form = "S3p" if paragraph else "S3"
    seg = (4, 8) if paragraph else (1, 2)
    got = run.fill(target, lambda: planner.plan_inter(k, seg_sents=seg), form=form,
                   split=split, tag=lambda r: f"{form}:k{k}:{r['regime']}")
    run.shortfall(_sf(form, split), f"k{k}", len(got), target, "stall")
    return got


# ----------------------------------------------------------------------- S5
_HOSTS = ["github.com", "arxiv.org", "aclanthology.org", "openreview.net", "huggingface.co",
          "nfb.org", "www.example.org", "doi.org", "proceedings.mlr.press", "dl.acm.org",
          "liblouis.io", "zenodo.org", "www.w3.org", "brailleauthority.org", "gitlab.com",
          "bitbucket.org", "pypi.org", "docs.python.org", "www.nature.com", "ieeexplore.ieee.org",
          "scholar.google.com", "en.wikipedia.org", "data.gov", "www.who.int", "kaggle.com"]
_USERS = ["alice", "smith", "j.doe", "research", "contact", "first.last", "info", "support",
          "lab-admin", "k.lee", "m_garcia", "noreply", "editor", "chen.wei", "team"]
_TLDS = ["com", "org", "net", "edu", "io", "ac.uk", "de", "fr", "jp", "kr", "ca", "au"]
_PATH = ["abs", "pdf", "datasets", "code", "papers", "blob", "main", "src", "docs", "v2",
         "release", "wiki", "issues", "tree", "files", "index.html", "README.md", "data.csv"]
_CARRIERS = [
    "The code is available at {u} for reproduction.", "See {u} for the full dataset.",
    "Our implementation is released at {u}.", "Data and models: {u}.",
    "Correspondence to {u}.", "Available under {u} (accessed {y}).", "{u}",
    "Cited from {u} in the appendix.", "Contact: {u}", "Preprint: {u}",
    "Please report issues at {u}.", "Source: {u}", "Visit {u} for details.",
    "The paper ({u}) describes the method.", "Download the files from {u} before {y}.",
    "Email {u} with questions.", "Further reading: {u} and references therein.",
    "Mirror: {u}", "Supplementary material is at {u}.", "Retrieved {y} from {u}.",
]


def an_identifier(rng: random.Random) -> str:
    u = rng.random()
    if u < 0.45:
        sch = rng.choice(["https://", "http://", "", "https://www."])
        host = rng.choice(_HOSTS) if rng.random() < 0.7 else \
            f"{rng.choice(['lab', 'my', 'open', 'the', 'data', 'code'])}{rng.choice(['braille', 'project', 'tools', 'hub', 'site'])}.{rng.choice(_TLDS)}"
        if sch.endswith("www.") and host.startswith("www."):
            host = host[4:]
        path = "/".join(rng.choice(_PATH + [str(rng.randint(1, 9999))])
                        for _ in range(rng.randint(0, 3)))
        q = rng.choice(["", "", "?q=1", f"?id={rng.randint(1, 999)}", "#sec", f"#fig{rng.randint(1, 9)}",
                        f"?page={rng.randint(1, 50)}&lang=en"])
        return f"{sch}{host}/{path}{q}".rstrip("/")
    if u < 0.6:
        return f"arXiv:{rng.randint(1501, 2512)}.{rng.randint(1, 29999):05d}" if rng.random() < .5 \
            else f"arxiv.org/abs/{rng.randint(1501, 2512)}.{rng.randint(1000, 9999)}v{rng.randint(1, 4)}"
    if u < 0.75:
        return f"doi:10.{rng.randint(1000, 99999)}/{rng.choice(['', 'j.', 's'])}{rng.randint(100000, 9999999)}"
    if u < 0.92:
        return f"{rng.choice(_USERS)}@{rng.choice(_HOSTS + [f'uni-{rng.randint(1, 99)}.{rng.choice(_TLDS)}'])}"
    return f"ISBN {rng.randint(978, 979)}-{rng.randint(0, 9)}-{rng.randint(100, 999)}-{rng.randint(10000, 99999)}-{rng.randint(0, 9)}"


def gen_s5(run, planner, target: int, split: str = "train") -> list[dict]:
    t = planner.by_id["en-ueb-g2.ctb"]
    rng = run.rng

    def plan_fn():
        text = rng.choice(_CARRIERS).format(u=an_identifier(rng), y=rng.randint(2015, 2026))
        return {"doc_type": "single", "regime": "single", "switch_density": "none",
                "n_sentences": 1, "source": "synthetic_identifier",
                "segments": [planner._segment(t, text)]}

    got = run.fill(target, plan_fn, form="S5", split=split, tag="S5:identifier")
    run.shortfall(_sf("S5", split), "all", len(got), target, "stall")
    return got


# ----------------------------------------------------------------------- S6
def gen_s6(run, planner, target: int, split: str = "train") -> list[dict]:
    """Training analogue of the eval snippets: n-grams in 5 equal-quota length buckets, tables by inventory weight;
    the text is transcribed whole."""
    rng = run.rng
    per = {b: target // len(SNIPPET_BUCKETS) for b in SNIPPET_BUCKETS}
    for b in list(SNIPPET_BUCKETS)[: target - sum(per.values())]:
        per[b] += 1
    need = dict(per)

    def plan_fn():
        open_b = [b for b, v in need.items() if v > 0]
        if not open_b:
            return None
        b = rng.choices(open_b, weights=[need[x] for x in open_b])[0]
        t = planner.pick_weighted_table()
        if b == "sentence":
            return planner.plan_single(table=t, n_sents=1)
        return snippet_plan(planner, t, rng.randint(*_BUCKET_WORDS[b]))

    def filt(rec):
        b = rec["len_bucket"]
        b = "sentence" if b in ("sentence", "paragraph") else b
        if need.get(b, 0) > 0:
            need[b] -= 1
            rec["_bucket"] = b
            return True
        return False

    got = run.fill(target, plan_fn, form="S6", split=split, filter_fn=filt,
                   tag=lambda r: f"S6:snippet:{r.pop('_bucket')}", max_rounds=80)
    for b in SNIPPET_BUCKETS:
        run.shortfall(_sf("S6", split), b, per[b] - need[b], per[b], "bucket_miss")
    return got


# ----------------------------------------------------------------------- S7
def gen_s7(run, strata: list[dict], target: int, split: str = "train",
           siblings: list[str] | None = None) -> tuple[list[dict], dict]:
    """NIKL rows up to each (edition, genre) quota of ubt.u1.nikl.sample_dev; confusable_set lists the ko-2024
    siblings whose forward encoding equals the human braille."""
    plans = [{"kind": "nikl", "table": KO2024_LABEL, "text": r["src"], "row": r,
              "human": r["braille"], "siblings": list(siblings or []),
              "stratum": st["stratum"]}
             for st in strata for r in st["rows"]]
    quota = {st["stratum"]: st["quota"] for st in strata}
    got = {s: 0 for s in quota}
    out = []
    for res in run.pool.imap(process, plans, run.chunksize):
        s = res["plan"]["stratum"]
        if got[s] >= quota[s]:
            continue
        r = res["plan"]["row"]
        f = res["f"]
        result = {"ok": True, "plan": {
            "doc_type": "single", "regime": "single", "switch_density": "none",
            "n_sentences": nikl_n_sentences(r["src"]), "source": "nikl"},
            "segments": [{"table": KO2024_LABEL, "lang": "ko", "text": r["src"],
                          "braille": r["braille"],
                          "confusable_set": res.get("confusable_set", [])}]}
        rec = run.accept(result, form="S7", split=split, tag=f"S7:nikl:{s}",
                         extra={"f_consistent": f == r["braille"], "nikl_id": r["id"],
                                "nikl_doc": r["doc"]})
        if rec is not None:
            got[s] += 1
            out.append(rec)
    for s in sorted(quota):
        run.shortfall(_sf("S7", split), s, got[s], quota[s], "stratum candidates exhausted")
    run.shortfall(_sf("S7", split), "all", len(out), target, "nikl dev sample exhausted")
    return out, {"per_stratum": {s: {"quota": quota[s], "got": got[s]} for s in sorted(quota)}}


# ----------------------------------------------------------------------- math
def _pick_form(rng, embed_frac, need_emb, need_std, plan_emb, plan_std, over=1.3):
    """Form for the next candidate; a form whose quota plus overdraw is planned this round is closed."""
    emb_open = plan_emb < need_emb * over + 2 and need_emb > 0
    std_open = plan_std < need_std * over + 2 and need_std > 0
    if emb_open and std_open:
        return "emb" if rng.random() < embed_frac else "std"
    if emb_open:
        return "emb"
    if std_open:
        return "std"
    return None


def gen_nemeth(run, spec, n: int, *, form: str, split: str, rng: random.Random,
               exclude_canon: set[str], include_paper: bool, embed_frac: float = 0.5,
               tag_prefix: str = "S4a", chunk: int = 200, formula_fn=None,
               carrier_siblings: list[str] | None = None) -> list[dict]:
    """Nemeth rows: round(n*embed_frac) embedded in UEB prose (k=2, inline), the rest standalone $$latex$$ (k=1).
    Formulas come from the split's math_side, except that the paper's own formulas always go to train."""
    formula_fn = formula_fn or NM.sample_formula
    side = "eval" if split == "eval" else "train"
    n_emb = round(n * embed_frac)
    n_std = n - n_emb
    got = {"emb": 0, "std": 0}
    out: list[dict] = []
    seen_canon: set[str] = set()
    paper = NM.paper_formula_set(0) if include_paper else []
    for rnd in range(8):
        if len(out) >= n:
            break
        want = int((n - len(out)) * 1.3) + 16
        cands: list[tuple[str, str, str]] = []
        while paper and len(cands) < want:
            f = paper.pop(0)
            c = nu_math(f)
            if c not in exclude_canon and c not in seen_canon:
                seen_canon.add(c)
                cands.append((f, c, "paper"))
        guard = 0
        while len(cands) < want and guard < want * 80:
            guard += 1
            f, kind = formula_fn(rng)
            c = nu_math(f)
            if c in exclude_canon or c in seen_canon or math_side(c) != side:
                continue
            seen_canon.add(c)
            cands.append((f, c, kind))
        if not cands:
            break
        chunks = [[c for _f, c, _k in cands[i:i + chunk]] for i in range(0, len(cands), chunk)]
        cells: list[str | None] = []
        for res in run.pool.imap(NM.nemeth_job, [(ch, spec.nemeth_jar, spec.java) for ch in chunks], 1):
            cells.extend(res)
        # decide the form of each candidate (quota-aware), then translate all carriers at once
        items, carrier_plans = [], []
        plan_n = {"emb": 0, "std": 0}
        for (f, c, kind), cell in zip(cands, cells):
            if cell is None:
                run.reject(form, "nemeth_fail_or_charset")
                continue
            typ = _pick_form(rng, embed_frac, n_emb - got["emb"], n_std - got["std"],
                             plan_n["emb"], plan_n["std"])
            if typ is None:
                break
            plan_n[typ] += 1
            if typ == "emb":
                pre, post = NM.carrier(rng)
                segs, _todo = NM.embedded_segments(f, c, cell, pre, post)
                for sg in segs:
                    if sg["braille"] is None:
                        sg["siblings"] = list(carrier_siblings or [])
                plan = {"kind": "carrier", "segments": segs, "doc_type": "math",
                        "regime": "math", "switch_density": "none", "n_sentences": 1,
                        "source": f"synthetic_math:{kind}",
                        "extra": {"latex_canonical": c, "embedded": True}}
                items.append(("emb", len(carrier_plans)))
                carrier_plans.append(plan)
            else:
                items.append(("std", (f, c, kind, cell)))
        carrier_res = list(run.pool.imap(process, carrier_plans, run.chunksize)) if carrier_plans else []
        for typ, payload in items:
            if len(out) >= n:
                break
            if got[typ] >= (n_emb if typ == "emb" else n_std):
                run.reject(form, "cell_full")
                continue
            if typ == "std":
                f, c, kind, cell = payload
                result = {"ok": True, "plan": {"doc_type": "math", "regime": "math",
                                               "switch_density": "none", "n_sentences": 1,
                                               "source": f"synthetic_math:{kind}",
                                               "extra": {"latex_canonical": c, "embedded": False}},
                          "segments": [{"table": NEMETH_LABEL, "lang": "math",
                                        "text": f"$${f}$$", "braille": cell,
                                        "confusable_set": []}]}
                rec = run.accept(result, form=form, split=split, tag=f"{tag_prefix}:nemeth:standalone")
            else:
                rec = run.accept(carrier_res[payload], form=form, split=split,
                                 tag=f"{tag_prefix}:nemeth:embedded", seg_join="",
                                 tables=["en-ueb-g2.ctb", NEMETH_LABEL], langs=["en", "math"])
            if rec is not None:
                got[typ] += 1
                out.append(rec)
    sf = form if split == "train" else (f"{split}:{form}" if split == "dev" else "eval:math_eval")
    run.shortfall(sf, "nemeth:embedded", got["emb"], n_emb, "latex2nemeth failures / formula space")
    run.shortfall(sf, "nemeth:standalone", got["std"], n_std, "latex2nemeth failures / formula space")
    return out


def gen_kmath(run, n: int, *, form: str, split: str, rng: random.Random,
              exclude_canon: set[str], kmath_fp: dict, embed_frac: float = 0.5,
              tag_prefix: str = "S4b") -> list[dict]:
    """Korean math rows; availability is frozen in kmath_fp at build start, so nothing enters unfingerprinted.
    Stored text uses $$…$$ for math as S4a does; kmath_text is the engine input, with $…$ in embedded sentences."""
    sf = form if split == "train" else (f"{split}:{form}" if split == "dev" else "eval:math_eval")
    if n <= 0:          # Korean math excluded from data-u1: target 0
        return []
    km, why = KM.load_kmath() if kmath_fp.get("available") else \
        (None, "unavailable at build start")
    if km is None:
        run.shortfall(sf, "kmath", 0, n, f"pending: Korean math engine unavailable ({why})")
        return []
    side = "eval" if split == "eval" else "train"
    n_emb = round(n * embed_frac)
    n_std = n - n_emb
    got = {"emb": 0, "std": 0}
    out: list[dict] = []
    seen: set[str] = set()
    for _ in range(8):
        if len(out) >= n:
            break
        want = int((n - len(out)) * 1.4) + 16
        items = []
        plan_n = {"emb": 0, "std": 0}
        guard = 0
        while len(items) < want and guard < want * 80:
            guard += 1
            f, fam = KM.ko_school_formula(rng)
            c = nu_math(f)
            if c in exclude_canon or c in seen or math_side(c) != side:
                continue
            typ = _pick_form(rng, embed_frac, n_emb - got["emb"], n_std - got["std"],
                             plan_n["emb"], plan_n["std"], over=1.4)
            if typ is None:
                break
            seen.add(c)
            plan_n[typ] += 1
            if typ == "emb":
                items.append(("sentence", KM.ko_sentence(rng, f), c, fam))
            else:
                items.append(("latex", f, c, fam))
        if not items:
            break
        res = list(run.pool.imap(KM.kmath_job, [(k, t) for k, t, _c, _f in items], 16))
        for (kind, ktext, c, fam), (b, why2) in zip(items, res):
            if len(out) >= n:
                break
            if b is None:
                run.reject(form, why2)
                continue
            emb = kind == "sentence"
            typ = "emb" if emb else "std"
            if got[typ] >= (n_emb if emb else n_std):
                run.reject(form, "cell_full")
                continue
            tables = [KO2024_LABEL, KMATH_LABEL] if emb else [KMATH_LABEL]
            langs = ["ko", "math"] if emb else ["math"]
            txt = ktext.replace("$", "$$") if emb else f"$${ktext}$$"
            result = {"ok": True, "plan": {"doc_type": "math", "regime": "math",
                                           "switch_density": "none", "n_sentences": 1,
                                           "source": f"synthetic_kmath:{fam}",
                                           "extra": {"latex_canonical": c, "embedded": emb,
                                                     "kmath_input": kind, "kmath_text": ktext}},
                      "segments": [{"table": KMATH_LABEL, "lang": "math", "text": txt,
                                    "braille": b, "confusable_set": []}]}
            rec = run.accept(result, form=form, split=split,
                             tag=f"{tag_prefix}:kmath:{'embedded' if emb else 'standalone'}",
                             tables=tables, langs=langs)
            if rec is not None:
                got[typ] += 1
                out.append(rec)
    run.shortfall(sf, "kmath:embedded", got["emb"], n_emb, "kmath rejections / formula space")
    run.shortfall(sf, "kmath:standalone", got["std"], n_std, "kmath rejections / formula space")
    return out
