"""data-u1 source views over the data root (sources already built).

  data/flores200_dataset/{dev,devtest}   dev = train side, devtest = eval side
  data/wiki_txt_A/*.txt                  train side
  data/wiki_txt_B/*.txt                  eval side (minus data/eval_exclusions.txt)

Differences from ubt.sources.CombinedSource (the base class):
  * wiki-first languages: the 15 head languages plus extras with a wiki shard and a thin FLORES pool (th); weighting
    still uses the 15 head languages only.
  * Thai ZWSP (U+200B) -> space, since the th tables emit a space cell for it.
  * Bengali-script nukta letters (bn, as, mni) are recomposed after NFC, which decomposes them, because the tables
    define only the precomposed letters; the text stays canonically equivalent.
  * S1b reservation: wiki_A blocks carved out for the per-table stratum leave the general pool, so each line is used
    with one table only.
  * Dev carve: the dev split comes from its own source lines, carved from every train-side pool before any train row
    exists, so dev and train share no sentence.
"""
from __future__ import annotations

import random
import re

from ubt.sources import CombinedSource, SentenceSource

_MULTISPACE = re.compile(r" {2,}")
ZWSP_TO_SPACE_LANGS = ("th",)
BENGALI_SCRIPT_LANGS = ("bn", "as", "mni")
_BENGALI_RECOMPOSE = (("ড়", "ড়"), ("ঢ়", "ঢ়"),
                      ("য়", "য়"))
SOURCE_NORMALISATION = {
    "th": "ZWSP (U+200B) -> space, space runs collapsed, re-dedup, >= 20 chars",
    "bn,as,mni": "NFC then recompose U+09A1/09A2/09AF + U+09BC -> U+09DC/09DD/09DF",
}


def _zwsp_fix(s: str) -> str:
    return _MULTISPACE.sub(" ", s.replace("​", " ")).strip()


def _bengali_fix(s: str) -> str:
    for a, b in _BENGALI_RECOMPOSE:
        s = s.replace(a, b)
    return s


def normalise_for_lang(lang: str, s: str) -> str:
    """The u1 source view of one raw line (after ubt.sources' NFC)."""
    if lang in ZWSP_TO_SPACE_LANGS:
        return _zwsp_fix(s)
    if lang in BENGALI_SCRIPT_LANGS:
        return _bengali_fix(s)
    return s


class DevSource:
    """The u1 dev split's source view: lang -> the carved dev pool."""

    def __init__(self, pools: dict[str, SentenceSource]):
        self.pools = pools

    def for_lang(self, base_lang: str) -> SentenceSource | None:
        return self.pools.get(base_lang)

    def sourced_langs(self, langs: list[str]) -> set[str]:
        return {l for l in langs if (s := self.for_lang(l)) and len(s) > 0}


def _carve_blocks(lines: list[str], n: int, rng: random.Random, block: int) -> set[int]:
    n_blocks = (len(lines) + block - 1) // block
    order = list(range(n_blocks))
    rng.shuffle(order)
    take, got = set(), 0
    for b in order:
        if got >= n:
            break
        take.add(b)
        got += min(block, len(lines) - b * block)
    return take


class U1Source(CombinedSource):
    def __init__(self, data_root: str, train: bool, head_langs: list[str],
                 extra_wiki_first: list[str] | None = None):
        super().__init__(data_root, train=train, head_langs=head_langs)
        # CombinedSource uses self.head_langs only for source preference
        self.head_langs = set(head_langs) | set(extra_wiki_first or [])
        self.reserved: dict[str, list[str]] = {}
        self._fixed: set[str] = set()

    def for_lang(self, base_lang: str) -> SentenceSource | None:
        src = super().for_lang(base_lang)
        if src is not None and id(src) not in self._fixed and (
                base_lang in ZWSP_TO_SPACE_LANGS or base_lang in BENGALI_SCRIPT_LANGS):
            fixed, seen = [], set()
            for s in src.sentences:
                t = normalise_for_lang(base_lang, s)
                if len(t) >= 20 and t not in seen:
                    seen.add(t)
                    fixed.append(t)
            src.sentences = fixed
            self._fixed.add(id(src))
        return src

    def reserve_wiki_blocks(self, lang: str, n_lines: int, rng: random.Random,
                            block: int = 64, max_frac: float = 0.45) -> list[str]:
        """Carve about n_lines of this language's wiki pool for S1b in blocks of consecutive lines, so the rest keeps
        coherent runs; returns the reserved lines shuffled."""
        wiki = self.wiki.for_lang(lang)
        if wiki is None or len(wiki) == 0:
            return []
        if lang in ZWSP_TO_SPACE_LANGS or lang in BENGALI_SCRIPT_LANGS:
            self.for_lang(lang)  # normalise first (same object when wiki-first)
            if id(wiki) not in self._fixed:
                wiki.sentences = [normalise_for_lang(lang, s) for s in wiki.sentences]
                self._fixed.add(id(wiki))
        lines = wiki.sentences
        n_lines = min(n_lines, int(len(lines) * max_frac))
        n_blocks = (len(lines) + block - 1) // block
        order = list(range(n_blocks))
        rng.shuffle(order)
        take = set()
        got = 0
        for b in order:
            if got >= n_lines:
                break
            take.add(b)
            got += min(block, len(lines) - b * block)
        reserved = [s for i, s in enumerate(lines) if (i // block) in take]
        wiki.sentences = [s for i, s in enumerate(lines) if (i // block) not in take]
        rng.shuffle(reserved)
        self.reserved[lang] = reserved
        return reserved

    def carve_dev(self, langs: list[str], seed: int, frac: float = 0.03,
                  min_lines: int = 40, max_lines: int = 4000, max_frac: float = 0.10,
                  block: int = 8) -> tuple[DevSource, dict]:
        """Move seeded blocks of consecutive lines (clamp(frac*len, min_lines, max_lines), at most max_frac) of every
        train pool into a dev-only pool; a pool shared by two languages is carved once. Returns (DevSource, report)."""
        import hashlib  # noqa: PLC0415
        pools: dict[str, SentenceSource] = {}
        report: dict[str, dict] = {}
        done: dict = {}          # pool identity/content key -> (dev lines, removed set)
        for lang in sorted(set(langs)):
            src = self.for_lang(lang)
            if src is None or len(src) == 0:
                continue
            ck = (src.name, len(src.sentences), src.sentences[0], src.sentences[-1])
            key = done.get(id(src)) or done.get(ck)
            if key is not None:
                dev_lines, removed = key
                src.sentences = [x for x in src.sentences if x not in removed]
                pools[lang] = SentenceSource(lang, src.name, list(dev_lines))
                report[lang] = {"pool": src.name, "dev_lines": len(dev_lines),
                                "train_lines": len(src.sentences), "shared_carve": True}
                done[id(src)] = key
                continue
            lines = src.sentences
            n = max(min_lines, min(max_lines, round(frac * len(lines))))
            n = min(n, int(max_frac * len(lines)))
            if n <= 0:
                report[lang] = {"pool": src.name, "dev_lines": 0, "train_lines": len(lines)}
                continue
            h = hashlib.sha256(f"{seed}|{src.name}|{len(lines)}|{lines[0]}".encode()).hexdigest()
            take = _carve_blocks(lines, n, random.Random(int(h[:16], 16)), block)
            dev_lines = [x for i, x in enumerate(lines) if (i // block) in take]
            removed = set(dev_lines)
            src.sentences = [x for x in lines if x not in removed]
            pools[lang] = SentenceSource(lang, src.name, dev_lines)
            report[lang] = {"pool": src.name, "dev_lines": len(dev_lines),
                            "train_lines": len(src.sentences)}
            done[id(src)] = (dev_lines, removed)
            done[ck] = (dev_lines, removed)
        return DevSource(pools), report
