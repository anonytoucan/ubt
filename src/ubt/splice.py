"""Intra-sentence code-switch splicer.

Embeds English terms, short phrases or digits/dates into an L1 sentence, which
is then translated with the single L1 table (its indicator rules do the work).
Currency symbols are never generated here (see configs/currency_overrides.yaml).
"""

from __future__ import annotations

import importlib.resources
import random
import re

_LATIN_RUN = re.compile(r"[A-Za-z]{2,}(?:[ '-][A-Za-z]{2,})*")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")


def load_wordlist() -> tuple[list[str], list[str]]:
    """Returns (terms, phrases) from the bundled en_terms.txt."""
    text = (importlib.resources.files("ubt") / "data" / "en_terms.txt").read_text("utf-8")
    entries = [ln.strip() for ln in text.splitlines() if ln.strip()]
    terms = [e for e in entries if " " not in e]
    phrases = [e for e in entries if " " in e]
    return terms, phrases


def natural_latin_runs(text: str) -> int:
    """Count naturally occurring Latin token runs (>=2 letters)."""
    return len(_LATIN_RUN.findall(text))


def is_latin_script(text: str) -> bool:
    """True for mostly Latin-script text, where embedded-Latin detection is moot."""
    alpha = [ch for ch in text if ch.isalpha()]
    if not alpha:
        return True
    latin = sum(1 for ch in alpha if "a" <= ch.lower() <= "z")
    return latin / len(alpha) > 0.5


def make_digits_dates(rng: random.Random) -> str:
    kind = rng.randrange(7)
    if kind == 0:
        return str(rng.randint(1900, 2029))                       # year
    if kind == 1:
        return f"{rng.randint(1, 999) * 1000:,}"                  # 12,000
    if kind == 2:
        return f"{rng.randint(0, 99)}.{rng.randint(0, 99):02d}"   # decimal
    if kind == 3:
        return (f"{rng.randint(1990, 2026)}-{rng.randint(1, 12):02d}"
                f"-{rng.randint(1, 28):02d}")                     # ISO date
    if kind == 4:
        return f"{rng.choice(_MONTHS)} {rng.randint(1, 28)}, {rng.randint(1950, 2026)}"
    if kind == 5:
        return f"{rng.randint(1, 100)}%"
    return f"{rng.randint(0, 23)}:{rng.randint(0, 59):02d}"       # time


def sample_span(rng: random.Random, splice_weights: dict[str, float],
                terms: list[str], phrases: list[str]) -> str:
    kinds, weights = zip(*splice_weights.items())
    kind = rng.choices(kinds, weights=weights)[0]
    if kind == "en_term":
        return rng.choice(terms)
    if kind == "en_phrase":
        return rng.choice(phrases) if phrases else rng.choice(terms)
    return make_digits_dates(rng)


def insert_spans(text: str, spans: list[str], rng: random.Random) -> str:
    """Insert spans at word boundaries (or char boundaries for unspaced
    scripts), never before index 1 or at the very end."""
    out = text
    for span in spans:
        if " " in out:
            words = out.split(" ")
            pos = rng.randint(1, max(1, len(words) - 1))
            out = " ".join(words[:pos] + [span] + words[pos:])
        else:
            pos = rng.randint(1, max(1, len(out) - 1))
            out = out[:pos] + " " + span + " " + out[pos:]
    return out


def splice(
    text: str,
    density: str,  # "light" (1 span) | "heavy" (2+)
    rng: random.Random,
    splice_weights: dict[str, float],
    terms: list[str],
    phrases: list[str],
    natural_first: bool = True,
) -> tuple[str, str]:
    """Returns (spliced_text, actual_density). With natural_first, a non-Latin
    sentence that already has Latin runs counts as intra-switch unspliced."""
    if natural_first and not is_latin_script(text):
        runs = natural_latin_runs(text)
        if runs >= 1:
            return text, ("light" if runs == 1 else "heavy")
    n = 1 if density == "light" else rng.randint(2, 3)
    spans = [sample_span(rng, splice_weights, terms, phrases) for _ in range(n)]
    return insert_spans(text, spans, rng), density
