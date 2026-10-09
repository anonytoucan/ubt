"""Recursive math grammar for synthesizing ML-paper formulas of realistic complexity.

Two sources: paper_formulas, templates of this paper's own formulas, injected at high multiplicity because a
self-transcription of the paper must reconstruct them; and gen_expr, a recursive grammar with depth peaked at 2-4.
"""
from __future__ import annotations

import random

VARS = list("xyztijklmn") + [r"\theta", r"\lambda", r"\mu", r"\sigma",
                             r"\alpha", r"\beta", r"\gamma", r"\epsilon"]
CAPS = list("XYZABTN") + [r"\mathcal{F}", r"\mathcal{L}", r"\mathcal{T}",
                          r"\mathcal{B}"]
SETS = [r"\mathbb{R}", r"\mathbb{N}", r"\mathbb{E}", r"\Theta", r"\mathcal{X}"]


def _v(rng): return rng.choice(VARS)
def _n(rng): return str(rng.randint(1, 20))


def gen_expr(rng: random.Random, depth: int) -> str:
    """Random expression nested `depth` levels deep (0 = atom)."""
    if depth <= 0:
        return rng.choice([_v(rng), _n(rng), _v(rng),
                           rng.choice(CAPS)])
    sub = lambda: gen_expr(rng, depth - 1)
    forms = [
        lambda: f"{sub()} + {sub()}",
        lambda: f"{sub()} - {sub()}",
        lambda: f"{sub()} {sub()}",                         # juxtaposition
        lambda: rf"\frac{{{sub()}}}{{{sub()}}}",
        lambda: f"{sub()}^{{{gen_expr(rng, max(0, depth-2))}}}",
        lambda: f"{sub()}_{{{gen_expr(rng, max(0, depth-2))}}}",
        lambda: rf"\sqrt{{{sub()}}}",
        lambda: f"({sub()})",
        lambda: rf"\| {sub()} \|",
        lambda: rf"| {sub()} |",
        lambda: rf"\hat{{{_v(rng)}}}",
        lambda: rf"\log {sub()}",
        lambda: rf"\exp({sub()})",
        lambda: rf"\sum_{{{_v(rng)}=1}}^{{{_v(rng)}}} {sub()}",
        lambda: rf"\int_{{0}}^{{{_v(rng)}}} {sub()} \, d{_v(rng)}",
        lambda: rf"\prod_{{{_v(rng)}}} {sub()}",
        lambda: rf"\mathbb{{E}}[{sub()}]",
        lambda: rf"\mathbb{{E}}_{{{_v(rng)}}}[{sub()}]",
        lambda: rf"P({sub()} \mid {sub()})",
        lambda: rf"\arg\max_{{{_v(rng)}}} {sub()}",
        lambda: rf"\arg\min_{{{_v(rng)}}} {sub()}",
        lambda: rf"\nabla_{{{_v(rng)}}} {sub()}",
    ]
    return rng.choice(forms)()


def gen_relation(rng: random.Random, depth: int) -> str:
    lhs = gen_expr(rng, depth)
    rhs = gen_expr(rng, depth)
    rel = rng.choice([r"=", r"\leq", r"\geq", r"\neq", r"\approx",
                      r"\sim", r"\propto", r"\in", r"\subseteq",
                      r"\rightarrow"])
    return f"{lhs} {rel} {rhs}"


# ---- this paper's own formulas (must cover) ------------------
def paper_formulas(rng: random.Random) -> list[str]:
    v = _v(rng)
    return [
        # forward operator / composition
        r"\mathcal{F}_\theta(B) = T",
        r"\mathcal{F}_\theta : \mathcal{T} \rightarrow \mathcal{B}",
        r"f_{math} = \text{latex2nemeth} \circ \nu_{math}",
        r"f = g \circ \nu",
        # MAP / Eq.(1)
        r"\hat{T} = \arg\max_{T} P(T \mid B)",
        r"\hat{T} = \arg\max_{T} P(T) P(B \mid T)",
        r"P(T \mid B) = \frac{P(B \mid T) P(T)}{P(B)}",
        # fiber / Prop 1
        r"\mathcal{F}_\theta^{-1}(B) = \{ T : \mathcal{F}_\theta(T) = B \}",
        r"| \mathcal{F}_\theta^{-1}(B) | > 1",
        r"T_1, T_2 \in \mathcal{F}_\theta^{-1}(B)",
        # KL / objective
        r"D_{KL}(p \| q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}",
        r"\mathcal{L}(\theta) = \mathbb{E}_{T}[- \log P_\theta(T \mid B)]",
        r"\nabla_\theta \mathcal{L}(\theta)",
        # BoN / lambda-fallback
        r"\hat{T} = \arg\max_{i} \mathbb{1}[\mathcal{F}_\theta(T_i) = B]",
        r"\lambda \in [0, 1]",
        # bCER / metrics
        r"\text{bCER} = \frac{\sum_{c} e_c}{| B |}",
        r"\text{CER} = \frac{\sum \text{edit}}{\sum | T |}",
        r"\min(1\%, \text{floor}_\ell + \epsilon)",
        # oracle-in-n
        r"\text{oracle@}n = \min_{i \leq n} \text{CER}(T_i, T)",
    ]


def stratified_formula(rng: random.Random) -> tuple[str, int]:
    """(latex, depth 0-4), with depth peaked at 2-4."""
    depth = rng.choices([0, 1, 2, 3, 4],
                        weights=[5, 15, 30, 30, 20])[0]
    if rng.random() < 0.6:
        return gen_relation(rng, depth), depth
    return gen_expr(rng, max(1, depth)), depth
