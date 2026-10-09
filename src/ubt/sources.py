"""Source corpora: FLORES-200 (tail + eval) and Wikipedia shards (head).

    data/flores200_dataset/dev/{code}.dev          train
    data/flores200_dataset/devtest/{code}.devtest  eval only
    data/wiki_txt_A/{lang}.txt                     train (head langs)
    data/wiki_txt_B/{lang}.txt                     eval only
dev/devtest and A/B are never mixed across train/eval.
"""

from __future__ import annotations

import hashlib
import os
import random
import unicodedata

MIN_LINE_CHARS = 20
MAX_LINE_CHARS = 600

# liblouis base_lang -> FLORES-200 code where pycountry cannot resolve it (macrolanguages, non-1:1 codes)
_FLORES_SPECIAL = {
    "ar": "arb_Arab", "fa": "pes_Arab", "ps": "pbt_Arab", "cmn": "zho_Hans",
    "yue": "yue_Hant", "ms": "zsm_Latn", "nb": "nob_Latn", "az": "azj_Latn",
    "uz": "uzn_Latn", "sq": "als_Latn", "sw": "swh_Latn", "kok": "gom_Deva",
    "ku": "kmr_Latn", "mn": "khk_Cyrl", "et": "est_Latn", "gn": "grn_Latn",
    "or": "ory_Orya", "qu": "quy_Latn", "ml": "mal_Mlym", "yi": "ydd_Hebr",
    "fil": "tgl_Latn", "hr": "hrv_Latn", "mt": "mlt_Latn", "lv": "lvs_Latn",
    "ne": "npi_Deva",
    # approximation, flagged in the report: Bihari -> Bhojpuri
    "bh": "bho_Deva",
}

# preferred script where FLORES has a language in several scripts
_SCRIPT_PREF = {
    "sr": "Cyrl", "ur": "Arab", "hi": "Deva", "bn": "Beng", "pa": "Guru",
    "kk": "Cyrl", "ky": "Cyrl", "tg": "Cyrl", "be": "Cyrl", "bg": "Cyrl",
    "ru": "Cyrl", "uk": "Cyrl", "mk": "Cyrl", "ko": "Hang", "el": "Grek",
    "he": "Hebr", "ta": "Taml", "te": "Telu", "kn": "Knda", "gu": "Gujr",
    "mr": "Deva", "ne": "Deva", "si": "Sinh", "th": "Thai", "lo": "Laoo",
    "km": "Khmr", "my": "Mymr", "am": "Ethi", "ka": "Geor", "hy": "Armn",
}


def flores_codes(flores_root: str) -> list[str]:
    dev = os.path.join(flores_root, "dev")
    return sorted(f[: -len(".dev")] for f in os.listdir(dev) if f.endswith(".dev"))


def map_lang_to_flores(base_lang: str, codes: list[str]) -> str | None:
    """Map a normalized liblouis language code to a FLORES-200 code."""
    special = _FLORES_SPECIAL.get(base_lang)
    if special:
        return special if special in codes else None
    try:
        import pycountry  # noqa: PLC0415

        rec = (
            pycountry.languages.get(alpha_2=base_lang)
            if len(base_lang) == 2
            else pycountry.languages.get(alpha_3=base_lang)
        )
        alpha3 = rec.alpha_3 if rec else None
    except Exception:
        alpha3 = None
    if alpha3 is None:
        alpha3 = base_lang if len(base_lang) == 3 else None
    if alpha3 is None:
        return None
    cands = [c for c in codes if c.split("_")[0] == alpha3]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    pref = _SCRIPT_PREF.get(base_lang)
    for c in cands:
        if pref and c.endswith("_" + pref):
            return c
    for c in cands:
        if c.endswith("_Latn"):
            return c
    return cands[0]


def sentence_hash(sentence: str) -> str:
    norm = unicodedata.normalize("NFC", sentence.strip())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _load_lines(path: str, exclude: frozenset[str] = frozenset()) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            s = unicodedata.normalize("NFC", line.strip())
            if not (MIN_LINE_CHARS <= len(s) <= MAX_LINE_CHARS):
                continue
            h = sentence_hash(s)
            if h in seen or h in exclude:
                continue
            seen.add(h)
            out.append(s)
    return out


def load_eval_exclusions(data_dir: str) -> frozenset[str]:
    """Hashes of eval-source lines that collide with a train source (from scripts/data/fix_disjoint.py)."""
    path = os.path.join(data_dir, "eval_exclusions.txt")
    if not os.path.isfile(path):
        return frozenset()
    with open(path) as fh:
        return frozenset(ln.strip() for ln in fh if ln.strip())


class SentenceSource:
    """A pool of verified-clean source sentences for one language."""

    def __init__(self, lang: str, name: str, sentences: list[str]):
        self.lang = lang
        self.name = name  # e.g. "flores_dev", "wiki_A"
        self.sentences = sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def sample(self, rng: random.Random, n: int) -> list[str]:
        """n sentences; consecutive run when possible so paragraphs cohere."""
        if not self.sentences:
            return []
        if n == 1:
            return [rng.choice(self.sentences)]
        if len(self.sentences) <= n:
            return list(self.sentences)
        start = rng.randrange(len(self.sentences) - n + 1)
        return self.sentences[start : start + n]


class FloresSource:
    """FLORES-200 loader. split: 'dev' (train) or 'devtest' (eval)."""

    def __init__(self, flores_root: str, split: str,
                 exclude: frozenset[str] = frozenset()):
        if split not in ("dev", "devtest"):
            raise ValueError(f"invalid FLORES split: {split}")
        self.root = flores_root
        self.split = split
        self.exclude = exclude
        self.codes = flores_codes(flores_root)
        self._cache: dict[str, SentenceSource] = {}

    def for_lang(self, base_lang: str) -> SentenceSource | None:
        if base_lang in self._cache:
            return self._cache[base_lang]
        code = map_lang_to_flores(base_lang, self.codes)
        src = None
        if code:
            path = os.path.join(self.root, self.split, f"{code}.{self.split}")
            if os.path.isfile(path):
                src = SentenceSource(base_lang, f"flores_{self.split}",
                                     _load_lines(path, self.exclude))
        self._cache[base_lang] = src
        return src


class WikiSource:
    """Wikipedia text-dir loader. shard: 'A' (train) or 'B' (eval)."""

    def __init__(self, data_root: str, shard: str):
        if shard not in ("A", "B"):
            raise ValueError(f"invalid wiki shard: {shard}")
        self.dir = os.path.join(data_root, f"wiki_txt_{shard}")
        self.shard = shard
        self._cache: dict[str, SentenceSource] = {}

    def for_lang(self, base_lang: str) -> SentenceSource | None:
        if base_lang in self._cache:
            return self._cache[base_lang]
        path = os.path.join(self.dir, f"{base_lang}.txt")
        src = None
        if os.path.isfile(path):
            src = SentenceSource(base_lang, f"wiki_{self.shard}", _load_lines(path))
        self._cache[base_lang] = src
        return src


class CombinedSource:
    """Train or eval view over wiki (head langs) + FLORES, wiki preferred."""

    def __init__(self, data_root: str, train: bool, head_langs: list[str]):
        split, shard = ("dev", "A") if train else ("devtest", "B")
        data_dir = os.path.join(data_root, "data")
        exclude = frozenset() if train else load_eval_exclusions(data_dir)
        self.flores = FloresSource(os.path.join(data_dir, "flores200_dataset"),
                                   split, exclude)
        self.wiki = WikiSource(data_dir, shard)
        self.head_langs = set(head_langs)

    def for_lang(self, base_lang: str) -> SentenceSource | None:
        if base_lang in self.head_langs:
            src = self.wiki.for_lang(base_lang)
            if src and len(src) > 0:
                return src
        flores = self.flores.for_lang(base_lang)
        if flores and len(flores) > 0:
            return flores
        # language missing from FLORES-200: use a wiki shard if one was built
        return self.wiki.for_lang(base_lang)

    def sourced_langs(self, langs: list[str]) -> set[str]:
        return {l for l in langs if (s := self.for_lang(l)) and len(s) > 0}
