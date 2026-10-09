"""Dot-flip noise for the eval-only noise variants; never applied to training data.

A cell's dots are the bits of (codepoint - U+2800); each flips i.i.d. with probability p.
"""

from __future__ import annotations

import random

from ubt.verify import BRAILLE_HI, BRAILLE_LO


def flip_dots(braille: str, p: float, rng: random.Random, n_dots: int = 8) -> str:
    """n_dots=6 flips only dots 1-6, so a 6-dot code never gains a dot-7/8
    cell; the default 8 flips every dot."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"invalid flip probability {p}")
    if n_dots not in (6, 8):
        raise ValueError(f"n_dots must be 6 or 8, got {n_dots}")
    out = []
    for ch in braille:
        code = ord(ch)
        if BRAILLE_LO <= code <= BRAILLE_HI:
            bits = code - BRAILLE_LO
            for d in range(n_dots):
                if rng.random() < p:
                    bits ^= 1 << d
            ch = chr(BRAILLE_LO + bits)
        out.append(ch)
    return "".join(out)
