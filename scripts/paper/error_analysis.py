"""Error analysis of the final model (UBT, code inferred, verified best-of-20) on the 20,071 core test documents.

Each character edit between reference and selected output in the scoring normal form (the CER numerator, checked
against the scorer) gets a category (case, punct, digits, characters) and the document's error kind from bon_eval.py
(inconsistent, consistent misreading, wrong code). A document exact under a wrong code counts as 'code only'. Families
follow scripts/paper/tracking_table.py, except that the 15 Indic codes (Bharati braille) form one family and Yakut is 'other'.

  python scripts/paper/error_analysis.py [--paper <paper repo>]
      -> work/paper/error_analysis.json (+ error_analysis.docs.jsonl) and <paper>/figures/errors/error_families.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import unicodedata

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path[:0] = sorted(glob.glob(os.path.join(REPO, "scripts", "*", "")))
from ubt.paths import DATASETS, PAPER, WORK  # noqa: E402
R = WORK
BON = f"{R}/evals/rlvr/bon"
INDIC_LANGS = {"as", "awa", "bn", "gu", "hi", "kn", "kok", "ml", "mni", "mr", "ne", "pa", "sa", "ta", "te"}
KINDS = {"inconsistent": "I", "consistent_misreading": "C", "wrong_code": "W"}
# figure rows: (family, label, [(document id, kind:category, context)]); examples chosen by hand
FIGURE = [
    ("Chinese", "Chinese", [("u1-Ecore-0015216", "C:characters", 7), ("u1-Ecore-0009737", "I:characters", 8)]),
    ("Japanese", "Japanese", [("u1-Ecore-0015052", "I:characters", 7), ("u1-Ecore-0015017", "C:punct", 6)]),
    ("Indic", "Indic", [("u1-Ecore-0005418", "W:characters", 3), ("u1-Ecore-0010010", "W:characters", 3),
                        ("u1-Ecore-0005450", "I:characters", 3)]),
    ("Korean", "Korean", [("u1-Ecore-0019087", "I:characters", 5), ("u1-Ecore-0015977", "W:other", 5),
                          ("u1-Ecore-0018981", "code only", 30)]),
    ("contracted", "Contracted", [("u1-Ecore-0000206", "I:characters", 14), ("u1-Ecore-0014604", "C:characters", 14)]),
    ("other", "Other", [("u1-Ecore-0014303", "I:characters", 12), ("u1-Ecore-0001891", "C:characters", 14),
                        ("u1-Ecore-0017260", "C:case", 16), ("u1-Ecore-0014274", "W:characters", 14)]),
]


def cclass(c):
    o = ord(c)
    if c.isspace():
        return "space"
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x2FFFF:
        return "han"
    if c.isdigit():
        return "digit"
    if unicodedata.category(c)[0] in "PS":
        return "punct"
    return "char"


def category(tag, a, b):
    """case | punct (punctuation, symbols, spaces) | digits | characters"""
    if tag == "replace" and a.lower() == b.lower():
        return "case"
    cs = {cclass(x) for x in (a, b) if x}
    if cs & {"space", "punct"}:
        return "punct"
    if "digit" in cs:
        return "digits"
    return "characters"


def seg(kind, cat):
    k = KINDS.get(kind)
    if k is None:
        return None
    if k == "C":
        return "C:" + cat
    return k + (":characters" if cat == "characters" else ":other")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=PAPER)
    a = ap.parse_args()
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415

    import paper_qualitative as Q  # noqa: PLC0415
    from tracking_table import family  # noqa: PLC0415
    from ubt.metrics.textnorm import norm_join  # noqa: PLC0415
    from ubt.wandb_utils import parse_target  # noqa: PLC0415

    inv = {t["table_id"]: t for t in json.load(open(f"{DATASETS}/data_u1_v2/inventory.json"))["tables"]}
    ev = {r["id"]: r for r in map(json.loads, open(Q.TEX_EVAL, encoding="utf-8")) if r["set"] == "core"}
    pd = {d["id"]: d for d in map(json.loads, open(f"{BON}/per_doc.jsonl", encoding="utf-8"))
          if d["slice"] == "main" and d.get("set") == "core"}
    gen = {}
    for p in glob.glob(f"{BON}/gen_main.*.jsonl"):
        for line in open(p, encoding="utf-8"):
            x = json.loads(line)
            if x["id"] in pd:
                gen[x["id"]] = x
    assert len(pd) == 20071 and set(pd) <= set(ev) and set(pd) <= set(gen)

    fam_of = {}
    for t in {d["table"] for d in pd.values()}:
        f = family(inv, t)
        fam_of[t] = "Indic" if inv[t]["base_lang"] in INDIC_LANGS else ("other" if f == "Konkani+Yakut" else f)
    rows, strings = [], {}
    tot = collections.Counter()
    fams = collections.defaultdict(lambda: {"E": 0, "N": 0, "docs": 0, "code_only": 0, "segs": collections.Counter(),
                                            "codes": set()})
    mismatch = 0
    for i, d in pd.items():
        v = d["pol"]["V"]
        segs = parse_target(gen[i]["cands"][v["ci"]]["text"]) or []
        ref, hyp = norm_join(parse_target(ev[i]["completion"])), (norm_join(segs) if segs else "")
        ops = Levenshtein.editops(ref, hyp)
        mismatch += len(ops) != v["e"]
        cats = collections.Counter()
        for op in ops:
            x = ref[op.src_pos] if op.tag != "insert" else None
            y = hyp[op.dest_pos] if op.tag != "delete" else None
            s = seg(v["kind"], category(op.tag, x, y))
            if s:
                cats[s] += 1
        code = d["table"]
        f = fam_of[code]
        co = int(v["kind"] == "wrong_code" and v["e"] == 0)
        for g in (f, "all"):
            F = fams[g]
            F["E"] += v["e"]; F["N"] += v["n"]; F["docs"] += 1; F["code_only"] += co; F["codes"].add(code)
            F["segs"].update(cats)
        tot.update(cats)
        rows.append({"id": i, "code": code, "family": f, "kind": v["kind"], "e": v["e"], "n": v["n"],
                     "pred": [c for c, _ in segs], "segs": cats})
        strings[i] = (ref, hyp)
    assert mismatch == 0, f"edit count differs from the scorer on {mismatch} documents"
    E = sum(tot.values())
    out = {"what": __doc__.split("\n\n")[0], "edits": E, "chars": sum(r["n"] for r in rows),
           "share_pct": {k: 100.0 * n / E for k, n in tot.most_common()},
           "code_only_docs": sum(r["kind"] == "wrong_code" and r["e"] == 0 for r in rows), "docs": len(rows),
           "families": {f: {"codes": len(F["codes"]), "docs": F["docs"], "cer": 100.0 * F["E"] / F["N"],
                            "code_only_pct": 100.0 * F["code_only"] / F["docs"],
                            "segs_pct": {k: 100.0 * n / max(1, F["E"]) for k, n in F["segs"].items()},
                            "codes_list": sorted(F["codes"])} for f, F in fams.items()}}
    os.makedirs(f"{R}/paper", exist_ok=True)
    json.dump(out, open(f"{R}/paper/error_analysis.json", "w"), indent=1)
    with open(f"{R}/paper/error_analysis.docs.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    fig = []
    for f, label, ex in FIGURE:
        F = out["families"][f]
        items = []
        for i, s, ctx in ex:
            r = next(x for x in rows if x["id"] == i)
            want = "wrong_code" if s == "code only" else {"I": "inconsistent", "C": "consistent_misreading",
                                                            "W": "wrong_code"}[s[0]]
            assert r["kind"] == want and r["family"] == f, (i, r["kind"], r["family"], s, f)
            ref, hyp = strings[i]
            items.append({"id": i, "seg": s, "ctx": ctx, "code": r["code"], "pred": r["pred"], "kind": r["kind"],
                          "ref": ref, "hyp": hyp,       # character and word (Indic) alignments for the highlight
                          "ops": [list(o) for o in Levenshtein.opcodes(ref, hyp).as_list()],
                          "wops": [list(o) for o in Levenshtein.opcodes(ref.split(" "), hyp.split(" ")).as_list()]})
        fig.append({"family": f, "label": label, "codes": F["codes"], "cer": F["cer"], "code_only_pct": F["code_only_pct"],
                    "segs_pct": F["segs_pct"], "examples": items})
    d = os.path.join(a.paper, "figures", "errors")
    os.makedirs(d, exist_ok=True)
    json.dump({"generated_by": "scripts/paper/error_analysis.py (code repo)", "rows": fig},
              open(os.path.join(d, "error_families.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{E} character edits over {out['chars']} characters (CER {100.0 * E / out['chars']:.3f}%); code only "
          f"{out['code_only_docs']} documents")
    for f in ("all", "Chinese", "Japanese", "Indic", "Korean", "contracted", "other"):
        F = out["families"][f]
        print(f"{f:10s} {F['codes']:3d} codes CER {F['cer']:5.2f} code only {F['code_only_pct']:5.1f}% | "
              + " ".join(f"{k} {v:.0f}" for k, v in sorted(F["segs_pct"].items(), key=lambda kv: -kv[1])[:5]))


if __name__ == "__main__":
    main()
