"""Nemeth math: rule-based baselines and one canonical metric for all systems.

Data: data_u1_v2 eval/math_eval.jsonl, standalone Nemeth expressions and English UEB sentences with a Nemeth span
between the code indicators ⠸⠩ … ⠸⠱. Rule-based systems back-translate the prose around a span with LibLouis en-ueb-g2.

Systems:
  liblouis : LibLouis 3.38 back-translation with en-us-mathtext.ctb (Nemeth characters only, no math rules)
  umml     : UnicodeMathML (MIT), braille -> UnicodeMath -> MathML, scripts/eval/math_umml.js
  a8m      : Access8Math (GPL-2.0) Nemeth2LaTeXTranslator -> LaTeX
  rule     : a8m, falling back to umml where a8m's grammar does not parse
  model    : a scripts/eval/greedy_eval.py gen file on `build`'s rows ($$LaTeX$$ spans of the output)

Metric: reference and output are converted to presentation MathML and linearised (base_{..}^{..}, frac{..}{..},
sqrt{..}; msub = munder, msup = mover), dropping spaces and invisible operators and folding math italic to plain
letters; other alphabets stay distinct. Reported: exact match and CER on the canonical strings, conversion failures
(counted wrong, CER 1), and prose CER for embedded rows. Tools: datasets/tools/mathbase (pinned).

  python scripts/eval/math_baselines.py baselines --out work/math_base
  python scripts/eval/math_baselines.py build --out work/data/math_eval        # rows for greedy_eval gen
  python scripts/eval/math_baselines.py score --out work/math_base [--model-gen 'g.*.jsonl']
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from ubt.paths import DATASETS, WORK  # noqa: E402
TOOLS = f"{DATASETS}/tools/mathbase"
EVAL = f"{DATASETS}/data_u1_v2/eval/math_eval.jsonl"
OPEN, CLOSE = "⠸⠩", "⠸⠱"
MATH_RE = re.compile(r"\$\$(.*?)\$\$", re.S)


def _rows():
    return [json.loads(line) for line in open(EVAL, encoding="utf-8")]


def _split_cells(braille: str):
    """English line with embedded Nemeth -> [('prose', cells) | ('math', cells)] in order."""
    out, i = [], 0
    while True:
        j = braille.find(OPEN, i)
        if j < 0:
            out.append(("prose", braille[i:]))
            return [p for p in out if p[1]]
        out.append(("prose", braille[i:j]))
        k = braille.find(CLOSE, j + len(OPEN))
        if k < 0:
            out.append(("math", braille[j + len(OPEN):]))
            return [p for p in out if p[1]]
        out.append(("math", braille[j + len(OPEN):k]))
        i = k + len(CLOSE)


# ---------------------------------------------------------------- canonical math form
_ACC = {"^": "^", "ˆ": "^", "\u0302": "^", "¯": "¯", "‾": "¯", "\u0304": "¯", "\u0305": "¯", "~": "~", "˜": "~",
        "\u0303": "~", "→": "→", "\u20d7": "→", "˙": "˙", "\u0307": "˙", "¨": "¨", "\u0308": "¨", "ˇ": "ˇ",
        "\u030c": "ˇ"}
_FOLD = {"\\": "", "∼": "~", "∣": "|", "−": "-", "⋅": "·", "∗": "*", "ϵ": "ε", "ϕ": "φ", "′": "'", "\u2061": "", "\u2062": "", "\u2063": "",
         "\u2064": "", "\u00a0": "", " ": "", "\u2009": "", "\u2005": "", "\u205f": "", "\u200b": ""}
_VARIANT = {"double-struck": "bb", "script": "cal", "fraktur": "frak", "bold": "bf", "bold-italic": "bf",
            "sans-serif": "sf", "monospace": "tt", "bold-script": "cal", "bold-fraktur": "frak"}


def _letter(ch: str, variant: str | None = None) -> str:
    """Math alphanumeric -> plain letter (italic/normal) or 'bb:E' (other alphabets)."""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return ch
    kind = None
    for key, tag in (("DOUBLE-STRUCK", "bb"), ("BLACK-LETTER", "frak"), ("FRAKTUR", "frak"), ("SCRIPT", "cal"),
                     ("BOLD", "bf"), ("SANS-SERIF", "sf"), ("MONOSPACE", "tt")):
        if key in name and ("MATHEMATICAL" in name or key in ("DOUBLE-STRUCK", "BLACK-LETTER", "SCRIPT")):
            kind = tag
            break
    base = unicodedata.normalize("NFKC", ch) if ("MATHEMATICAL" in name or kind) else ch
    kind = kind or (_VARIANT.get(variant or "") if base.isalpha() else None)
    return f"{kind}:{base}" if kind else base


def _text(s: str, variant: str | None = None) -> str:
    s = unicodedata.normalize("NFD", s or "")
    out = []
    for ch in s:
        if ch in _ACC and unicodedata.combining(ch) and out:
            out.append("^{" + _ACC[ch] + "}")
            continue
        ch = unicodedata.normalize("NFC", ch)
        ch = _FOLD.get(ch, ch)
        out.append(_letter(ch, variant) if len(ch) == 1 and (ch.isalpha()) else ch)
    return "".join(out)


def _lin(e) -> str:
    tag = e.tag.rsplit("}", 1)[-1]
    ch = list(e)
    if tag in ("mi", "mn", "mo", "mtext", "ms"):
        t = "".join(e.itertext())
        if tag == "mo" and t.strip() in _ACC:
            return _ACC[t.strip()]
        return _text(t, e.get("mathvariant"))
    if tag in ("mspace", "none", "mprescripts", "annotation", "annotation-xml", "mphantom"):
        return ""
    if tag == "semantics":
        return _lin(ch[0]) if ch else ""
    if tag in ("msub", "munder") and len(ch) >= 2:
        return _lin(ch[0]) + "_{" + _lin(ch[1]) + "}"
    if tag in ("msup", "mover") and len(ch) >= 2:
        return _lin(ch[0]) + "^{" + _lin(ch[1]) + "}"
    if tag in ("msubsup", "munderover") and len(ch) >= 3:
        return _lin(ch[0]) + "_{" + _lin(ch[1]) + "}^{" + _lin(ch[2]) + "}"
    if tag == "mfrac" and len(ch) >= 2:
        return "frac{" + _lin(ch[0]) + "}{" + _lin(ch[1]) + "}"
    if tag == "msqrt":
        return "sqrt{" + "".join(_lin(c) for c in ch) + "}"
    if tag == "mroot" and len(ch) >= 2:
        return "root{" + _lin(ch[0]) + "}{" + _lin(ch[1]) + "}"
    return "".join(_lin(c) for c in ch)


def canon_mathml(mathml: str | None) -> str | None:
    import xml.etree.ElementTree as ET  # noqa: PLC0415
    if not mathml:
        return None
    try:
        root = ET.fromstring(re.sub(r"&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)", "&amp;", mathml))
    except ET.ParseError:
        return None
    return _lin(root)


def canon_latex(latex: str | None) -> str | None:
    if latex is None:
        return None
    sys.path.insert(0, os.path.join(TOOLS, "pylib"))
    from latex2mathml.converter import convert  # noqa: PLC0415
    try:
        return canon_mathml(convert(latex.strip()))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- systems
def _liblouis():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import main_eval as TC  # noqa: PLC0415
    TC._pin_engine(f"{DATASETS}/data_u1_v2")
    from ubt.translate import build_table_path, preload_louis_lib  # noqa: PLC0415
    os.environ["LOUIS_TABLEPATH"] = build_table_path(None)
    preload_louis_lib()
    import louis  # noqa: PLC0415

    def bt(table, cells):
        try:
            return louis.backTranslateString(["unicode.dis", table], cells)
        except Exception:  # noqa: BLE001
            return None
    return bt


def _a8m():
    sys.path.insert(0, os.path.join(TOOLS, "pylib"))
    sys.path.insert(0, os.path.join(TOOLS, "Access8Math/addon/globalPlugins/Access8Math/lib"))
    from nemeth.translator import Nemeth2LaTeXTranslator  # noqa: PLC0415
    tr = Nemeth2LaTeXTranslator()

    def alarm(_s, _f):
        raise TimeoutError

    signal.signal(signal.SIGALRM, alarm)

    def run(cells):
        signal.alarm(20)
        try:
            return tr.translate(cells), None
        except TimeoutError:
            return None, "timeout"
        except Exception as e:  # noqa: BLE001
            return None, type(e).__name__
        finally:
            signal.alarm(0)
    return run


def cmd_baselines(a) -> None:
    os.makedirs(a.out, exist_ok=True)
    rows = _rows()
    units = []                                   # (doc id, part index, kind, cells)
    for r in rows:
        parts = [("math", r["braille"])] if r["tables"] == ["nemeth"] else _split_cells(r["braille"])
        for pi, (kind, cells) in enumerate(parts):
            units.append((r["id"], pi, kind, cells))
    maths = [u for u in units if u[2] == "math"]
    tmp_in, tmp_out = os.path.join(a.out, "umml_in.json"), os.path.join(a.out, "umml_out.json")
    json.dump([{"id": f"{u[0]}#{u[1]}", "braille": u[3]} for u in maths], open(tmp_in, "w"), ensure_ascii=False)
    subprocess.run(["node", os.path.join(os.path.dirname(os.path.abspath(__file__)), "math_umml.js"),
                    os.path.join(TOOLS, "UnicodeMathML"), tmp_in, tmp_out], check=True)
    umml = {x["id"]: x for x in json.load(open(tmp_out))}
    a8m = _a8m()
    bt = _liblouis()
    out = {}
    for doc, pi, kind, cells in units:
        key = f"{doc}#{pi}"
        if kind == "prose":
            p = bt("en-ueb-g2.ctb", cells)
            out[key] = {"kind": "prose", "liblouis": p, "umml": p, "a8m": p}
            continue
        lat, err = a8m(cells)
        u = umml.get(key, {})
        out[key] = {"kind": "math", "liblouis": bt("en-us-mathtext.ctb", cells),
                    "umml_um": u.get("um"), "umml_mathml": u.get("mathml"), "umml_err": u.get("err"),
                    "a8m_latex": lat, "a8m_err": err}
    json.dump(out, open(os.path.join(a.out, "baselines.json"), "w"), ensure_ascii=False)
    vers = {d: subprocess.run(["git", "-C", os.path.join(TOOLS, d), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip() for d in ("UnicodeMathML", "Access8Math")}
    json.dump({"tools": vers, "pylib": "lark 1.3.1, latex2mathml 3.81.1", "liblouis": "3.38.0 (data engine)",
               "units": len(units), "math_units": len(maths)}, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"{len(units)} units ({len(maths)} math); umml failures {sum(1 for u in umml.values() if u['err'])}, "
          f"a8m failures {sum(1 for k, v in out.items() if v['kind'] == 'math' and v['a8m_err'])}")


def cmd_build(a) -> None:
    from transformers import AutoTokenizer  # noqa: PLC0415

    from ubt.task_format import render  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "eval.jsonl"), "w", encoding="utf-8") as fh:
        for r in _rows():
            ex = render(r, hinted=False)
            n_c = len(tok(ex["completion"], add_special_tokens=False)["input_ids"])
            fh.write(json.dumps({"id": r["id"], "set": "math", "tables": r["tables"], "k": r["k"], "regime": r["regime"],
                                 "switch_density": r["switch_density"], "confusable_set": r.get("confusable_set") or [],
                                 "group": "math", "lowres": False, "prompt": ex["prompt"], "completion": ex["completion"],
                                 "max_tokens": 2 * n_c + 32}, ensure_ascii=False) + "\n")
    print("rows ->", os.path.join(a.out, "eval.jsonl"))


def cmd_score(a) -> None:
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415

    from ubt.metrics.textnorm import norm_line  # noqa: PLC0415
    from ubt.wandb_utils import parse_target  # noqa: PLC0415
    rows = _rows()
    base = json.load(open(os.path.join(a.out, "baselines.json"), encoding="utf-8"))
    model = {}
    for p in sorted(glob.glob(a.model_gen or "")):
        for line in open(p, encoding="utf-8"):
            x = json.loads(line)
            segs = parse_target(x["hyp"])
            model[x["id"]] = "\n".join(t for _, t in segs) if segs else None
    systems = ["liblouis", "umml", "a8m", "rule"] + (["model"] if model else [])
    res = {s: {k: {"docs": 0, "exact": 0, "e": 0, "n": 0, "fail": 0, "pe": 0, "pn": 0}
               for k in ("standalone", "embedded")} for s in systems}
    skipped = []
    for r in rows:
        kind = "standalone" if r["tables"] == ["nemeth"] else "embedded"
        gold_math = MATH_RE.findall(r["text"])
        if any(canon_latex(m) is None for m in gold_math):
            skipped.append(r["id"])              # reference does not convert: excluded for every system
            continue
        gold_prose = MATH_RE.sub(" ", r["text"])
        parts = [("math", None)] if kind == "standalone" else _split_cells(r["braille"])
        for s in systems:
            if s == "model":
                t = model.get(r["id"])
                hm = MATH_RE.findall(t) if t else []
                hp = MATH_RE.sub(" ", t) if t else ""
                hyp_canon = [canon_latex(m) for m in hm]
            else:
                hm, hp_parts = [], []
                for pi, (k, _c) in enumerate(parts):
                    b = base[f"{r['id']}#{pi}"]
                    if k == "prose":
                        hp_parts.append(b["liblouis"] or "")
                    elif s == "liblouis":
                        hm.append(canon_latex(b["liblouis"]) if b["liblouis"] is not None else None)
                    elif s == "umml":
                        hm.append(canon_mathml(b["umml_mathml"]))
                    elif s == "a8m":
                        hm.append(canon_latex(b["a8m_latex"]))
                    else:                                   # rule: a8m, umml where a8m does not parse
                        h = canon_latex(b["a8m_latex"]) if b["a8m_latex"] is not None else None
                        hm.append(h if h is not None else canon_mathml(b["umml_mathml"]))
                hyp_canon = hm
                hp = " ".join(hp_parts)
            g = res[s][kind]
            g["docs"] += 1
            gold_canon = [canon_latex(m) for m in gold_math]
            ok = len(hyp_canon) == len(gold_canon) and all(h is not None and h == gc for h, gc in zip(hyp_canon, gold_canon))
            g["exact"] += ok
            for i, gc in enumerate(gold_canon):
                gc = gc or ""
                h = hyp_canon[i] if i < len(hyp_canon) else None
                g["fail"] += h is None
                g["e"] += len(gc) if h is None else Levenshtein.distance(gc, h)
                g["n"] += len(gc)
            if kind == "embedded":
                gp, hpn = norm_line(gold_prose, "en-ueb-g2.ctb"), norm_line(hp, "en-ueb-g2.ctb")
                gp, hpn = re.sub(r"\s+", " ", gp).strip(), re.sub(r"\s+", " ", hpn).strip()
                g["pe"] += Levenshtein.distance(gp, hpn)
                g["pn"] += len(gp)
    lines = ["| system | standalone: exact / canon. CER / fail | embedded: math exact / canon. CER / fail | embedded prose CER |",
             "|---|---:|---:|---:|"]
    for s in systems:
        a_, b_ = res[s]["standalone"], res[s]["embedded"]
        lines.append(f"| {s} | {100*a_['exact']/max(1,a_['docs']):.1f}% / {100*a_['e']/max(1,a_['n']):.1f}% / {a_['fail']} | "
                     f"{100*b_['exact']/max(1,b_['docs']):.1f}% / {100*b_['e']/max(1,b_['n']):.1f}% / {b_['fail']} | "
                     f"{100*b_['pe']/max(1,b_['pn']):.2f}% |")
    json.dump({"systems": res, "skipped_reference_not_convertible": skipped}, open(os.path.join(a.out, "score.json"), "w"),
              indent=1)
    print(f"docs scored: standalone {res[systems[0]]['standalone']['docs']}, embedded {res[systems[0]]['embedded']['docs']} "
          f"({len(skipped)} excluded: reference LaTeX not convertible)")
    print("\n".join(lines))


def cmd_bon(a) -> None:
    """Best-of-20 on math_eval: re-transcribe each candidate as the data were made (ubt.u1.reencode) and score the
    policies G / V (lambda 2) / Vinf / O in the canonical math form."""
    import multiprocessing as mp  # noqa: PLC0415

    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415

    from ubt.math_norm import nu_math  # noqa: PLC0415
    from ubt.u1.nemeth import NEMETH_CLOSE, NEMETH_OPEN, nemeth_job  # noqa: PLC0415
    from ubt.u1.routed_louis import RoutedLouis  # noqa: PLC0415
    from ubt.u1.sources import _bengali_fix  # noqa: PLC0415
    from ubt.wandb_utils import parse_target  # noqa: PLC0415
    split = re.compile(r"(\$\$.*?\$\$)", re.S)
    rows = [json.loads(line) for line in open(a.rows, encoding="utf-8")]
    gens = {}
    for p_ in sorted(glob.glob(a.gen)):
        for line in open(p_, encoding="utf-8"):
            x = json.loads(line)
            gens[x["id"]] = x
    rows = [r for r in rows if r["id"] in gens]
    formulas, prose, plan = set(), set(), {}
    for r in rows:
        for ci, c in enumerate(gens[r["id"]]["cands"]):
            segs = parse_target(c["text"])
            if not segs or len(c["text"]) > 4 * len(r["completion"]) + 64:
                plan[(r["id"], ci)] = None
                continue
            parts = []
            for tab, text in segs:
                seq = []
                if tab == "nemeth":
                    m = re.fullmatch(r"\$\$(.*)\$\$", text, flags=re.S)
                    if m:
                        f = nu_math(m.group(1))
                        formulas.add(f)
                        seq.append(("f", f, False))
                    else:
                        seq = None
                else:
                    for piece in split.split(text):
                        if not piece:
                            continue
                        if piece.startswith("$$") and piece.endswith("$$") and len(piece) >= 4:
                            f = nu_math(piece[2:-2])
                            formulas.add(f)
                            seq.append(("f", f, True))
                        else:
                            k = (tab, _bengali_fix(unicodedata.normalize("NFC", piece)))
                            prose.add(k)
                            seq.append(("p", k, None))
                parts.append(seq)
            plan[(r["id"], ci)] = parts
    man = json.load(open(f"{DATASETS}/data_u1_v2/manifest.json"))
    jar, java = man["engine_spec"]["nemeth_jar"], man["engine_spec"].get("java", "java")
    fl = sorted(formulas)
    chunks = [fl[i:i + 100] for i in range(0, len(fl), 100)]
    with mp.get_context("fork").Pool(min(48, max(1, len(chunks)))) as pool:
        res = pool.map(nemeth_job, [(ch, jar, java) for ch in chunks])
    nem = dict(zip(fl, [x for part in res for x in part]))
    L = RoutedLouis(f"{DATASETS}/data_u1_v2", timeout_s=300.0)
    pl = sorted(prose)
    pout = dict(zip(pl, L.translate_many(pl)))
    L.close()
    feas = {}
    for r in rows:
        B = r["prompt"].split("<|braille|>\n", 1)[1].split("\n<|text|>", 1)[0]
        for ci in range(len(gens[r["id"]]["cands"])):
            parts = plan[(r["id"], ci)]
            enc = None
            if parts is not None and all(sq is not None for sq in parts):
                lines = []
                for sq in parts:
                    out = []
                    for kind, key, wrap in sq:
                        b = nem.get(key) if kind == "f" else pout.get(key)
                        if b is None:
                            out = None
                            break
                        out.append(NEMETH_OPEN + b + NEMETH_CLOSE if (kind == "f" and wrap) else b)
                    if out is None:
                        lines = None
                        break
                    lines.append("".join(out))
                enc = "\n".join(lines) if lines is not None else None
            feas[(r["id"], ci)] = (enc == B, Levenshtein.distance(enc, B) if enc is not None else None)
    res_ = {pol: {k: {"docs": 0, "exact": 0, "e": 0, "n": 0, "fail": 0, "consistent": 0} for k in ("standalone", "embedded")}
            for pol in ("G", "V", "Vinf", "O")}
    for r in rows:
        kind = "standalone" if r["tables"] == ["nemeth"] else "embedded"
        gold_text = "\n".join(t for _, t in parse_target(r["completion"]))
        gc = [canon_latex(m) for m in MATH_RE.findall(gold_text)]
        if any(g is None for g in gc):
            continue
        cands = gens[r["id"]]["cands"]
        hc_all = []
        for c in cands:
            segs = parse_target(c["text"])
            t = "\n".join(x for _, x in segs) if segs else ""
            hc = [canon_latex(m) for m in MATH_RE.findall(t)]
            e = sum((len(g) if (i >= len(hc) or hc[i] is None) else Levenshtein.distance(g, hc[i])) for i, g in enumerate(gc))
            hc_all.append((hc, e))
        f = [feas[(r["id"], i)] for i in range(len(cands))]
        for pol in res_:
            idx = range(len(cands))
            if pol == "G":
                ci = 0
            elif pol == "O":
                ci = min(idx, key=lambda i: (hc_all[i][1], i))
            else:
                F = [i for i in idx if f[i][0]]
                D = [i for i in idx if f[i][1] is not None]
                if F:
                    ci = max(F, key=lambda i: cands[i]["lp"])
                elif not D:
                    ci = 0
                elif pol == "Vinf":
                    ci = min(D, key=lambda i: (f[i][1], -cands[i]["lp"]))
                else:
                    ci = max(D, key=lambda i: cands[i]["lp"] - 2.0 * f[i][1])
            hc, e = hc_all[ci]
            g = res_[pol][kind]
            g["docs"] += 1
            g["e"] += e
            g["n"] += sum(len(x) for x in gc)
            g["exact"] += len(hc) == len(gc) and all(h == x for h, x in zip(hc, gc))
            g["fail"] += sum(1 for i in range(len(gc)) if i >= len(hc) or hc[i] is None)
            g["consistent"] += f[ci][0]
    json.dump(res_, open(a.out, "w"), indent=1)
    for pol, d in res_.items():
        print(pol, {k: f"exact {100*v['exact']/max(1,v['docs']):.1f}% CER {100*v['e']/max(1,v['n']):.2f}% consistent "
                       f"{100*v['consistent']/max(1,v['docs']):.1f}%" for k, v in d.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baselines")
    b.add_argument("--out", required=True)
    m = sub.add_parser("build")
    m.add_argument("--out", required=True)
    m.add_argument("--tokenizer", default=f"{WORK}/models/qwen25_cell_tokenizer")
    bo = sub.add_parser("bon")
    bo.add_argument("--rows", default=f"{WORK}/data/math_eval/eval.jsonl")
    bo.add_argument("--gen", required=True)
    bo.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--model-gen", default=None)
    a = ap.parse_args()
    {"baselines": cmd_baselines, "build": cmd_build, "score": cmd_score, "bon": cmd_bon}[a.cmd](a)


if __name__ == "__main__":
    main()
