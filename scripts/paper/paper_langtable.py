"""Regenerate the paper's per-code table (tables/langtable.tex, `tab:all-codes`) from the evaluation results.

Columns, in the scoring normal form: bCER (verified output if the best-of-20 pool exists, else greedy), greedy CER,
verified CER (best-of-20, lambda=2) with oracle-in-20 CER, sim. (bge-m3 similarity of the BoN output), and LibLouis
backward CER (code given). Missing values are \\gval{}. Two mathematics rows use the canonical math form of
scripts/eval/math_baselines.py. Marks, on verified CER else greedy: \\dagger above the bar (Chinese <= 5%, others < 1%),
\\circ language outside Qwen2.5's 29, * value depends on a normalisation. Shading: (1,5] mild, > 5 severe.

  python scripts/paper/paper_langtable.py --out <paper>/tables/langtable.tex [--model sft|topup|rlvr]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path[:0] = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "*", "")))
from ubt.paths import DATASETS, WORK  # noqa: E402

TEX_EVAL = f"{WORK}/data/main_eval/eval.jsonl"
GREEDY = f"{WORK}/main_eval/final/gen.*.jsonl"
LIBLOUIS = f"{WORK}/liblouis_bwd/report.json"
MATH = f"{WORK}/math_base/score.json"          # LibLouis math rows (all models); SFT greedy math row
R = WORK
# per model: bon (first existing file), prefill (code given), math_bon, unnorm (w/o-norm. CER), sim (0-100)
SOURCES = {
    "sft": {"bon": [f"{R}/bon_sft/run/report.json", f"{R}/bon_sft/run_main/report.json"],
            "prefill": f"{R}/bon_sft/run_prefill/report.json", "math_bon": f"{R}/math_base/bon_score.json",
            "unnorm": f"{R}/paper/unnormalized_per_code.json", "sim": f"{R}/bon_sft/run/sim_per_code.json"},
    **{m: {"bon": [f"{R}/evals/{m}/bon/report.json"] + ([f"{R}/bon_topup/run/report.json"] if m == "topup" else []),
           "prefill": f"{R}/evals/{m}/prefill/report.json", "math_bon": f"{R}/evals/{m}/math/bon_score.json",
           "unnorm": f"{R}/paper/unnormalized_per_code.{m}.json", "sim": f"{R}/evals/{m}/bon/sim_per_code.json"}
       for m in ("topup", "rlvr")},
}
CACHE = f"{WORK}/paper/greedy_per_code.json"
# the 29 languages Qwen2.5 lists as officially supported
QWEN29 = {"cmn", "yue", "en", "de", "fr", "es", "pt", "it", "nl", "ru", "cs", "pl", "ar", "fa", "he", "tr", "ja", "ko",
          "vi", "th", "id", "ms", "lo", "my", "ceb", "km", "fil", "hi", "bn", "ur"}
NORMALISED = {"zh_CHN.tbl", "zh-tw.ctb", "zh-hk.ctb", "zhcn-g1.ctb", "zhcn-g2.ctb", "zhcn-cbs.ctb", "km-g1.utb",
              "bn.tbl", "as.tbl", "mni.tbl"}
NEW_NAMES = {"en-nz-g1": "English", "en-nz-g2": "English", "et-6dot": "Estonian", "ht-g1": "Haitian Creole",
             "hu-hu-g1": "Hungarian", "it-it-comp6": "Italian", "ja-rokutenkanji": "Japanese", "ko-2024-g2": "Korean",
             "mk-g1": "Macedonian", "sv-6g0d": "Swedish", "sv-6g0p": "Swedish", "sv-6g1d": "Swedish",
             "sv-6g2d": "Swedish"}


def _stem(c: str) -> str:
    return re.sub(r"\.(ctb|utb|tbl)$", "", c)


def _old_names(path: str) -> dict:
    names = {}
    for line in open(path, encoding="utf-8"):
        line = re.sub(r"^\\rowcolor\{\w+\}", "", line)            # shaded rows carry their language too
        if line.startswith("\\"):
            continue
        m = re.match(r"([^&]+?)\s*&\s*([^&$]+?)(?:\$\^\{[^}]*\}\$)*\s*&", line)
        if m and "Language" not in m.group(1):
            names[m.group(2).strip().replace("\\_", "_")] = m.group(1).strip()
    return names


def greedy_per_code() -> dict:
    """Per-code greedy CER / bCER / WER of the final checkpoint (main_eval final), cached in CACHE."""
    if os.path.isfile(CACHE):
        return json.load(open(CACHE))
    import main_eval as TC  # noqa: PLC0415
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415

    from ubt.eval_harness import score_hyp  # noqa: PLC0415
    from ubt.metrics.textnorm import doc_error_cells, norm_join  # noqa: PLC0415
    from ubt.wandb_utils import parse_target  # noqa: PLC0415
    gen = {}
    for p in glob.glob(GREEDY):
        for line in open(p, encoding="utf-8"):
            x = json.loads(line)
            gen[x["id"]] = x["hyp"]
    TC._pin_engine(f"{DATASETS}/data_u1_v2")
    agg = collections.defaultdict(collections.Counter)
    for line in open(TEX_EVAL, encoding="utf-8"):
        r = json.loads(line)
        if r["set"] != "core":
            continue
        h = gen[r["id"]]
        row = score_hyp({"id": r["id"], "confusable_set": r["confusable_set"]}, {"completion": r["completion"]}, h, None)
        e, n, _ = doc_error_cells(TC._bcer_segments(r), parse_target(h))
        ref = norm_join(parse_target(r["completion"]))
        hs = parse_target(h)
        rw, hw = ref.split(), (norm_join(hs) if hs else "").split()
        a = agg[r["tables"][0]]
        a["e"] += row["edit"]
        a["n"] += row["ref_len"]
        a["bc"] += e
        a["cells"] += n
        a["we"] += Levenshtein.distance(rw, hw)
        a["wn"] += len(rw)
        a["docs"] += 1
    out = {t: {"docs": a["docs"], "cer": 100 * a["e"] / max(1, a["n"]), "bcer": 100 * a["bc"] / max(1, a["cells"]),
               "wer": 100 * a["we"] / max(1, a["wn"])} for t, a in agg.items()}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(out, open(CACHE, "w"), indent=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--old", default=None, help="existing langtable.tex for language names (default: --out)")
    ap.add_argument("--model", default="sft", choices=sorted(SOURCES))
    a = ap.parse_args()
    S = SOURCES[a.model]
    a.out = os.path.abspath(a.out)
    a.old = os.path.abspath(a.old) if a.old else None
    names = _old_names(a.old or a.out)
    names.update(NEW_NAMES)
    inv = {t["table_id"]: t for t in json.load(open(f"{DATASETS}/data_u1_v2/inventory.json"))["tables"]}
    g = greedy_per_code()
    bon_path = next((p for p in S["bon"] if os.path.isfile(p)), None)
    bon = json.load(open(bon_path)) if bon_path else {}
    lib = json.load(open(LIBLOUIS)) if os.path.isfile(LIBLOUIS) else {}
    pf = json.load(open(S["prefill"])) if os.path.isfile(S["prefill"]) else {}
    sims = json.load(open(S["sim"])) if os.path.isfile(S["sim"]) else {}
    unnorm = json.load(open(S["unnorm"])) if os.path.isfile(S["unnorm"]) else {}

    def v(x, fmt="{:.2f}"):
        return "\\gval{}" if x is None else fmt.format(x)

    def bo(b_, o_):                                   # "BoN (oracle)" in one cell
        return "\\gval{}" if b_ is None else f"{b_:.2f} ({o_:.2f})" if o_ is not None else f"{b_:.2f}"

    def bo_under(b_, o_first):                        # w/o-norm. line: BoN aligned with the first line's BoN
        return "\\gval{}" if b_ is None else f"{b_:.2f}" + (f"\\phantom{{ ({o_first:.2f})}}" if o_first is not None else "")
    rows = []
    for code in sorted(g, key=lambda c: (names.get(_stem(c), _stem(c)), _stem(c))):
        b = bon.get("code:" + code, {})
        V, O = b.get("V"), b.get("O")
        best = V["cer"] if V else None
        crit = best if best is not None else g[code]["cer"]
        bar = 5.0 if code.startswith("zh") else 1.0
        above = crit > bar if code.startswith("zh") else crit >= bar
        lang = names.get(_stem(code), _stem(code))
        base = (inv.get(code) or {}).get("base_lang", "")
        marks = ("$^{\\dagger}$" if above else "") + ("$^{*}$" if code in NORMALISED else "") \
            + ("$^{\\circ}$" if base not in QWEN29 else "")
        shade = "\\rowcolor{barSevere}" if crit > 5 else "\\rowcolor{barMild}" if crit > 1 else ""
        bc = V["bcer"] if V else g[code]["bcer"]
        sim = sims.get(code, {}).get("V")
        ll = lib.get("code:" + code, {}).get("cer")
        pg = pf.get("code:" + code, {})
        gcer = b["G"]["cer"] if b.get("G") else g[code]["cer"]          # greedy of the same candidate pool when present
        rows.append(f"{shade}{lang} & {_stem(code).replace('_', chr(92) + '_')}{marks} & {bc:.2f} & {gcer:.2f} & "
                    f"{bo(best, O['cer'] if O else None)} & {v(sim, '{:.1f}')} & {v(pg.get('G', {}).get('cer'))} & "
                    f"{bo(pg.get('V', {}).get('cer'), pg.get('O', {}).get('cer'))} & {v(ll)} \\\\")
        if code in NORMALISED and code in unnorm:              # normalized codes report both values
            u = unnorm[code]
            rows.append(f"{shade} & \\multicolumn{{1}}{{l}}{{\\quad\\textit{{w/o norm.}}}} & -- & {v(u.get('inferred_G'))} & "
                        f"{bo_under(u.get('inferred_V'), O['cer'] if O else None)} & -- & {v(u.get('given_G'))} & "
                        f"{bo_under(u.get('given_V'), pg.get('O', {}).get('cer'))} & {v(u.get('liblouis'))} \\\\")
            # no oracle on the second line: it is chosen under the headline metric
    # mathematics rows (canonical math form)
    ms = json.load(open(MATH)) if os.path.isfile(MATH) else {}
    mb = json.load(open(S["math_bon"])) if os.path.isfile(S["math_bon"]) else {}
    sysm = ms.get("systems", {})
    rows.append("\\midrule")                                            # mathematics set apart from the per-code rows
    for key, label in (("standalone", "nemeth"), ("embedded", "en-ueb-g2 $+$ nemeth")):
        mdl = sysm.get("model", {}).get(key) if a.model == "sft" else mb.get("G", {}).get(key)
        ll = sysm.get("liblouis", {}).get(key)
        mv = mb.get("V", {}).get(key)
        mo = mb.get("O", {}).get(key)
        cer = (lambda d: None if not d else 100 * d["e"] / max(1, d["n"]))
        rows.append(f"Mathematics$^{{\\ddagger}}$ & {label} & -- & {v(cer(mdl))} & {bo(cer(mv), cer(mo))} & -- & -- & -- & "
                    f"{v(cer(ll))} \\\\")
    head = ("\\toprule & & \\multicolumn{4}{c}{code inferred} & \\multicolumn{3}{|c}{code given} \\\\ "
            "\\cmidrule(lr){3-6}\\cmidrule(lr){7-9} "
            "Language & \\multicolumn{1}{l}{Code} & bCER & greedy & BoN (orc.) & sim. & \\multicolumn{1}{|r}{greedy} & "
            "BoN (orc.) & LibLouis \\\\ \\midrule")
    cap = ("\\caption{Per-code results of the final checkpoint on the core-per-code split, with the code inferred and given, "
           "all values in \\% and in the scoring normal form of \\S\\ref{sec:metrics}. greedy and BoN (verified best-of-20, "
           "$\\lambda{=}2$) are text CER, with the oracle-in-20 CER of the same pool in parentheses; bCER and sim.\\ (embedding similarity to the reference, 0--100, \\S\\ref{sec:metrics}) are "
           "of the BoN output. Code given: our model with the gold code prefilled "
           "in the output (\\S\\ref{sec:model}), and LibLouis, the CER of backward translation with the code. Rows shaded \\colorbox{barMild}{yellow} have BoN CER in "
           "$(1,5]\\%$, \\colorbox{barSevere}{red} above $5\\%$; $^{\\dagger}$ marks a code above its bar (Chinese "
           "$\\le5\\%$, others $<1\\%$); $^{*}$ marks a value that depends on a scoring normalization (Chinese: "
           "Traditional folded to Simplified; Khmer: U+200B removed; Bengali script: NFC), and the line beneath such a code "
           "(\\textit{w/o norm.}) scores the same outputs on the raw strings (Appendix~\\ref{app:normal-form}); $^{\\circ}$ marks a "
           "language outside Qwen2.5's 29 officially supported languages, whose judge score should be read with "
           "caution (Appendix~\\ref{app:judge}); $^{\\ddagger}$ held-out mathematics set, exact-structure CER on the "
           "canonical MathML form (Appendix~\\ref{app:data}).\\label{tab:all-codes}}\\\\")
    body = ["\\scriptsize\\setlength{\\tabcolsep}{2pt}", "\\begin{longtable}{@{}l>{\\ttfamily}lrrrr|rr|r@{}}", cap, head,
            "\\endfirsthead \\multicolumn{9}{@{}l}{\\small\\itshape Table~\\ref{tab:all-codes} (continued)}\\\\[2pt] "
            + head + " \\endhead",
            "\\midrule \\multicolumn{9}{r}{\\small\\itshape continued on next page}\\\\ \\endfoot \\endlastfoot",
            *rows, "\\bottomrule", "\\end{longtable}"]
    open(a.out, "w", encoding="utf-8").write("\n".join(body) + "\n")
    print(f"{len(rows)} rows -> {a.out} (model {a.model}: {bon_path}; best-of-20 pool {'yes' if bon else 'not yet'}, "
          f"LibLouis {'yes' if lib else 'no'}, math {'yes' if ms else 'no'})")


if __name__ == "__main__":
    main()
