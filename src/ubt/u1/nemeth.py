"""S4a / math_eval: LaTeX -> Nemeth via latex2nemeth.

f_nemeth(latex) = N(latex2nemeth(nu_math(latex))), where N strips, turns each whitespace run into one blank cell U+2800
and requires braille cells only (the jar puts ASCII spaces between Nemeth tokens).

Embedded rows (Nemeth in UEB prose) are en-ueb-g2(pre + " ") + ⠸⠩ + f_nemeth + ⠸⠱ + en-ueb-g2(" " + post), with the
carrier pieces translated under the undefined-character check. Formulas mix the 19 paper formulas (once each),
school-level families and the recursive ML-paper grammar; a misaligned jar batch is bisected rather than dropped.
"""
from __future__ import annotations

import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

from ubt.math_norm import nu_math

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))), "scripts", "data")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from math_grammar import paper_formulas, stratified_formula  # noqa: E402

NEMETH_OPEN = "⠸⠩"
NEMETH_CLOSE = "⠸⠱"
CARRIER_TABLE = "en-ueb-g2.ctb"
BLANK = "⠀"
_WS = re.compile(r"\s+")

JAVA_OPTS = ["-XX:+UseSerialGC", "-XX:ActiveProcessorCount=1", "-Xmx768m",
             "-XX:TieredStopAtLevel=1"]


def normalize_nemeth(raw: str) -> str | None:
    s = _WS.sub(BLANK, raw.strip())
    if not s:
        return None
    if any(not (0x2800 <= ord(c) <= 0x28FF) for c in s):
        return None
    return s


def _run_jar(formulas: list[str], workdir: str, jar: str, java: str) -> list[str] | None:
    body = "\n".join(rf"\begin{{equation}}{f}\end{{equation}}" for f in formulas)
    tex = "\\documentclass{article}\\begin{document}\n" + body + "\n\\end{document}\n"
    for f in os.listdir(workdir):
        os.unlink(os.path.join(workdir, f))
    tp, ap = os.path.join(workdir, "m.tex"), os.path.join(workdir, "m.aux")
    with open(tp, "w", encoding="utf-8") as fh:
        fh.write(tex)
    open(ap, "w").close()
    try:
        subprocess.run([java, *JAVA_OPTS, "-jar", jar, tp, ap], cwd=workdir,
                       capture_output=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None
    out = os.path.join(workdir, "m0.nemeth")
    if not os.path.isfile(out):
        return None
    with open(out, "rb") as fh:
        parts = fh.read().decode("utf-16").split("\n\n")
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) == len(formulas) else None


def batch_nemeth(canon: list[str], jar: str, java: str = "java") -> list[str | None]:
    """f_nemeth for canonical LaTeX strings; one JVM per batch, misaligned batches bisected down to single formulas."""
    wd = tempfile.mkdtemp(prefix="u1nem-")
    try:
        out: list[str | None] = [None] * len(canon)
        stack = [(0, len(canon))]
        while stack:
            lo, hi = stack.pop()
            if lo >= hi:
                continue
            parts = _run_jar(canon[lo:hi], wd, jar, java)
            if parts is None:
                if hi - lo == 1:
                    continue           # this formula fails alone -> None
                mid = (lo + hi) // 2
                stack.extend([(lo, mid), (mid, hi)])
                continue
            for i, p in enumerate(parts):
                out[lo + i] = normalize_nemeth(p)
        return out
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def nemeth_job(args: tuple[list[str], str, str]) -> list[str | None]:
    canon, jar, java = args
    return batch_nemeth(canon, jar, java)


# --------------------------------------------------------------------------- formulas
def _r(rng, seq):
    return rng.choice(seq)


def school_formula(rng: random.Random, level: int) -> str:
    """School-level formula: 0 arithmetic, 1 fraction, 2 algebra, 3 calculus, 4 sets/probability."""
    d = lambda: rng.randint(1, 99)  # noqa: E731
    v = lambda: _r(rng, "xyzabnkmpqt")  # noqa: E731
    if level == 0:
        op = _r(rng, ["+", "-", r"\times", r"\div", r"\cdot"])
        return f"{d()} {op} {d()} = {d()}"
    if level == 1:
        return _r(rng, [
            rf"\frac{{{d()}}}{{{d()}}}",
            rf"\frac{{{v()}+{d()}}}{{{v()}-{d()}}}",
            rf"\frac{{1}}{{1+\frac{{1}}{{{v()}}}}}",
            rf"{d()}.{d()} + {d()}.{d()}",
            rf"\frac{{{d()}}}{{{d()}}} + \frac{{{d()}}}{{{d()}}} = \frac{{{d()}}}{{{d()}}}",
        ])
    if level == 2:
        a, b = v(), v()
        return _r(rng, [
            rf"{a}^{{{rng.randint(2, 9)}}} + {d()}{a} + {d()}",
            rf"({a}+{b})^{{2}} = {a}^{{2}}+2{a}{b}+{b}^{{2}}",
            rf"{a} = \frac{{-b \pm \sqrt{{b^{{2}}-4ac}}}}{{2a}}",
            rf"|{a}-{d()}| \leq {d()}",
            rf"{a}_{{{rng.randint(1, 9)}}} + {b}_{{{rng.randint(1, 9)}}}",
            rf"{d()}{a} - {d()} = {d()}{a} + {d()}",
            rf"\sqrt{{{a}^{{2}} + {b}^{{2}}}}",
            rf"{a}^{{{rng.randint(2, 5)}}} {b}^{{{rng.randint(2, 5)}}}",
        ])
    if level == 3:
        a = v()
        return _r(rng, [
            rf"\sum_{{{a}=1}}^{{{d()}}} {a}^{{2}}",
            rf"\int_{{0}}^{{{d()}}} {a}^{{{rng.randint(2, 5)}}} \, d{a}",
            rf"\lim_{{{a} \to \infty}} \frac{{1}}{{{a}}}",
            rf"\frac{{d}}{{d{a}}} {a}^{{{rng.randint(2, 6)}}}",
            rf"f'({a}) = {d()}{a}^{{{rng.randint(1, 4)}}}",
            rf"\sin^{{2}} {a} + \cos^{{2}} {a} = 1",
            rf"\log_{{{rng.randint(2, 10)}}} {d()}",
        ])
    return _r(rng, [
        rf"{_r(rng, 'ABX')} \cup {_r(rng, 'BCY')}",
        rf"{_r(rng, 'ABX')} \cap {_r(rng, 'BCY')} = \emptyset",
        rf"{v()} \in \mathbb{{R}}",
        rf"\forall {v()} \exists {v()}",
        r"P(A \mid B) = \frac{P(A \cap B)}{P(B)}",
        rf"\binom{{{rng.randint(2, 20)}}}{{{rng.randint(1, 9)}}}",
        rf"{v()} \leq {v()} \leq {v()}",
    ])


def sample_formula(rng: random.Random) -> tuple[str, str]:
    """(latex, kind): 60% recursive ML-paper grammar, 40% school families."""
    if rng.random() < 0.6:
        f, depth = stratified_formula(rng)
        return f, f"gen{depth}"
    lvl = rng.randint(0, 4)
    return school_formula(rng, lvl), f"school{lvl}"


def paper_formula_set(seed: int) -> list[str]:
    return paper_formulas(random.Random(seed))


# --------------------------------------------------------------------------- carriers
_PRE = [
    "We define", "Let", "Consider", "Recall that", "Note that", "It follows that",
    "Substituting, we obtain", "By the lemma above,", "In this case", "The objective is",
    "We minimise", "The estimator satisfies", "Hence", "Therefore", "Suppose that",
    "Assume", "Observe that", "Equation {n} gives", "The bound reads", "Our loss is",
    "The update rule is", "Rearranging yields", "From the definition,", "The expected value is",
    "For the second term,", "Applying the chain rule gives", "The solution is",
    "Plugging in the values gives", "The model predicts", "Using the identity",
    "The constraint is", "We can write", "At step {n}, the error is", "The probability is",
    "The student wrote", "The answer is", "Simplify", "Solve", "Evaluate", "Compute",
    "Prove that", "Show that", "Check whether", "In Section {n} we derived",
    "The teacher asked us to factor",
]
_POST = [
    "for all valid inputs.", "under the model.", "in the limit.", "which is bounded.",
    "for every parameter.", "as expected.", "where the variables are real.", "almost surely.",
    "for sufficiently large n.", "with probability at least 0.{n}.", "by construction.",
    "as shown in Section {n}.", "and the claim follows.", "which completes the proof.",
    "in Table {n}.", "on the training set.", "at convergence.", "for each index.",
    "whenever the denominator is nonzero.", "up to a constant.", "in expectation.",
    "if the condition holds.", "before rounding.", "on page {n}.", "in the exercise.",
]
_PREDICATE = [
    "is the objective.", "holds for all x.", "is minimised at zero.", "gives the result.",
    "is bounded above.", "follows from the lemma.", "was used above.", "denotes the loss.",
    "appears in Equation {n}.", "is the final answer.",
]


def _fill(rng: random.Random, s: str) -> str:
    return s.replace("{n}", str(rng.randint(1, 12)))


def carrier(rng: random.Random) -> tuple[str | None, str | None]:
    """(pre, post) around the formula: 'pre $$f$$ post', '$$f$$ post' (pre None) or 'pre $$f$$.'."""
    u = rng.random()
    if u < 0.5:
        return _fill(rng, rng.choice(_PRE)), _fill(rng, rng.choice(_POST))
    if u < 0.75:
        return None, _fill(rng, rng.choice(_PREDICATE))
    return _fill(rng, rng.choice(_PRE)), "."


def embedded_segments(latex: str, canon: str, cells: str, pre: str | None,
                      post: str | None) -> tuple[list[dict], list[dict]]:
    """(segments with braille for the math span, carrier segments still to translate).
    Carrier texts are pre + " " and " " + post; a bare '.' post attaches directly."""
    segs: list[dict] = []
    if pre is not None:
        segs.append({"table": CARRIER_TABLE, "lang": "en", "text": pre + " ",
                     "braille": None, "confusable_set": []})
    segs.append({"table": "nemeth", "lang": "math", "text": f"$${latex}$$",
                 "braille": NEMETH_OPEN + cells + NEMETH_CLOSE,
                 "latex_canonical": canon, "confusable_set": []})
    if post is not None:
        ptxt = post if post == "." else " " + post
        segs.append({"table": CARRIER_TABLE, "lang": "en", "text": ptxt,
                     "braille": None, "confusable_set": []})
    return segs, [s for s in segs if s["braille"] is None]


def canon_of_text(text: str) -> str | None:
    m = re.search(r"\$\$(.*)\$\$", text, flags=re.S)
    return nu_math(m.group(1)) if m else None
