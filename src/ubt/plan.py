"""Doc-plan construction: sampling tables, source text and splices.

A plan is everything a worker needs to translate one document; workers never sample, so a seed reproduces generation.
"""

from __future__ import annotations

import random
import re

from ubt.mixer import MixerV2
from ubt.sources import CombinedSource
from ubt.splice import load_wordlist, splice
from ubt.tables import TableInfo, confusable_siblings

# Unspaced scripts get no space between sentences (it would be a braille cell native text never has); a boundary with
# CJK/fullwidth characters on both sides is also joined without one, whatever the language code.
UNSPACED_LANGS = frozenset({"cmn", "yue", "ja", "zh", "lzh", "wuu"})
_CJK_CHAR = re.compile(r"[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")


def join_sentences(lang: str, sents: list[str]) -> str:
    if lang in UNSPACED_LANGS:
        return "".join(sents)
    out = sents[0] if sents else ""
    for s in sents[1:]:
        if _CJK_CHAR.match(out[-1:]) and _CJK_CHAR.match(s[:1]):
            out += s
        else:
            out += " " + s
    return out


class Planner:
    def __init__(
        self,
        cfg: dict,
        tables: list[TableInfo],
        groups: dict[str, list[str]],
        source: CombinedSource,
        rng: random.Random,
    ):
        self.cfg = cfg
        self.rng = rng
        self.source = source
        self.groups = groups
        self.by_id = {t.table_id: t for t in tables}
        self.weighted = [t for t in tables if t.weight > 0]
        self.weights = [t.weight for t in self.weighted]
        self.holdout_tables = [t for t in tables if t.holdout]
        self.mixer = MixerV2(
            self.weighted, groups,
            cfg["inter_mixed"]["anchor_langs"], cfg["inter_mixed"]["regimes"],
        )
        self.terms, self.phrases = load_wordlist()

    # -- helpers ---------------------------------------------------------

    def _sample_text(self, lang: str, n_sents: int) -> tuple[str, str, int] | None:
        src = self.source.for_lang(lang)
        if src is None or len(src) == 0:
            return None
        sents = src.sample(self.rng, n_sents)
        if not sents:
            return None
        return join_sentences(lang, sents), src.name, len(sents)

    def _n_sentences(self) -> int:
        classes = self.cfg["single_length"]
        cls = self.rng.choices(list(classes), weights=list(classes.values()))[0]
        if cls == "one_sent":
            return 1
        if cls == "two_three_sent":
            return self.rng.randint(2, 3)
        return self.rng.randint(4, 8)  # paragraph

    def _segment(self, table: TableInfo, text: str) -> dict:
        return {
            "table": table.table_id,
            "lang": table.base_lang,
            "text": text,
            "siblings": confusable_siblings(table, self.groups),
        }

    def pick_weighted_table(self) -> TableInfo:
        return self.rng.choices(self.weighted, weights=self.weights)[0]

    # -- plan builders ---------------------------------------------------

    def plan_single(self, table: TableInfo | None = None,
                    n_sents: int | None = None) -> dict | None:
        t = table or self.pick_weighted_table()
        n = n_sents if n_sents is not None else self._n_sentences()
        got = self._sample_text(t.base_lang, n)
        if got is None:
            return None
        text, src_name, n_actual = got
        return {
            "doc_type": "single", "regime": "single", "switch_density": "none",
            "n_sentences": n_actual, "source": src_name,
            "segments": [self._segment(t, text)],
        }

    def plan_intra_switch(self, table: TableInfo | None = None,
                          density: str | None = None) -> dict | None:
        t = table or self.pick_weighted_table()
        got = self._sample_text(t.base_lang, self.rng.randint(1, 2))
        if got is None:
            return None
        text, src_name, n_actual = got
        icfg = self.cfg["intra_switch"]
        if density is None:
            dens = icfg["density"]
            density = self.rng.choices(list(dens), weights=list(dens.values()))[0]
        spliced, actual_density = splice(
            text, density, self.rng, icfg["splice"], self.terms, self.phrases,
            natural_first=icfg.get("natural_first", True),
        )
        return {
            "doc_type": "intra_switch", "regime": "single",
            "switch_density": actual_density, "n_sentences": n_actual,
            "source": src_name,
            "segments": [self._segment(t, spliced)],
        }

    def plan_inter(self, k: int, regime: str | None = None,
                   seg_sents: tuple[int, int] = (1, 2)) -> dict | None:
        regime = regime or self.mixer.choose_regime(self.rng)
        tables = self.mixer.choose_tables(regime, k, self.rng)
        if len(tables) < k:
            return None
        segments, src_names, n_total = [], [], 0
        for t in tables:
            got = self._sample_text(t.base_lang, self.rng.randint(*seg_sents))
            if got is None:
                return None
            text, src_name, n_actual = got
            segments.append(self._segment(t, text))
            src_names.append(src_name)
            n_total += n_actual
        return {
            "doc_type": "inter", "regime": regime, "switch_density": "none",
            "n_sentences": n_total, "source": "+".join(sorted(set(src_names))),
            "segments": segments,
        }

    def plan_snippet(self, table: TableInfo, max_words: int) -> dict | None:
        """A word n-gram of one sentence, translated whole; braille is never sliced."""
        got = self._sample_text(table.base_lang, 1)
        if got is None:
            return None
        sent, src_name, _ = got
        words = sent.split(" ")
        if len(words) > max_words:
            n = self.rng.randint(1, max_words)
            start = self.rng.randrange(len(words) - n + 1)
            text = " ".join(words[start : start + n])
        elif len(words) == 1 and len(sent) > max_words * 4:
            # unspaced script: slice the text by characters
            n = self.rng.randint(2, max_words * 4)
            start = self.rng.randrange(len(sent) - n + 1)
            text = sent[start : start + n].strip()
        else:
            text = sent
        if not text.strip():
            return None
        return {
            "doc_type": "single", "regime": "single", "switch_density": "none",
            "n_sentences": 1, "source": src_name,
            "segments": [self._segment(table, text)],
        }
