"""Mixer: table choice for inter-mixed documents; every segment is translated fresh and joined with "\\n".

Regimes:
  anchor:     segment 1's table by inventory weight; the other segments' languages by anchor_langs, each mapped to
              its current-status table.
  uniform:    k distinct tables, uniform over literary tables with weight > 0.
  confusable: 2 distinct tables from one confusable group; for k > 2 the rest come from the anchor procedure.
"""

from __future__ import annotations

import random

from ubt.tables import TableInfo


def _grade_score(t: TableInfo) -> tuple:
    contraction_rank = {"full": 3, "partial": 2, "no": 1}.get(t.contraction, 0)
    try:
        grade = float(t.grade)
    except ValueError:
        grade = -1.0
    return (grade, contraction_rank)


def current_status_table(lang: str, by_lang: dict[str, list[TableInfo]]) -> TableInfo | None:
    """Map a language to its highest-grade table with status "current", or of any status if none is current."""
    tabs = by_lang.get(lang)
    if not tabs:
        return None
    current = [t for t in tabs if t.status == "current"]
    return max(current or tabs, key=_grade_score)


class MixerV2:
    def __init__(
        self,
        weighted_tables: list[TableInfo],
        groups: dict[str, list[str]],
        anchor_langs: dict[str, float],
        regimes: dict[str, float],
    ):
        if not weighted_tables:
            raise ValueError("mixer needs a non-empty weighted table inventory")
        self.tables = weighted_tables
        self.by_id = {t.table_id: t for t in weighted_tables}
        self.by_lang: dict[str, list[TableInfo]] = {}
        for t in weighted_tables:
            self.by_lang.setdefault(t.base_lang, []).append(t)
        # Only groups fully inside the weighted inventory are sampleable.
        self.groups = {
            g: [m for m in members if m in self.by_id]
            for g, members in groups.items()
        }
        self.groups = {g: m for g, m in self.groups.items() if len(m) >= 2}
        self.anchor_langs = {
            l: w for l, w in anchor_langs.items() if l in self.by_lang
        }
        self.regimes = regimes
        self._weights = [t.weight for t in weighted_tables]

    def choose_regime(self, rng: random.Random) -> str:
        names, weights = zip(*self.regimes.items())
        return rng.choices(names, weights=weights)[0]

    def _anchor_fill(self, rng: random.Random, n: int, used: set[str]) -> list[TableInfo]:
        """n additional tables via anchor-language sampling, distinct from used."""
        out: list[TableInfo] = []
        langs = dict(self.anchor_langs)
        attempts = 0
        while len(out) < n and attempts < 50:
            attempts += 1
            pool_langs, pool_w = zip(*langs.items()) if langs else ((), ())
            if not pool_langs:
                # anchor langs exhausted: fall back to weighted sampling
                cand = rng.choices(self.tables, weights=self._weights)[0]
            else:
                lang = rng.choices(pool_langs, weights=pool_w)[0]
                langs.pop(lang, None)  # without replacement over languages
                cand = current_status_table(lang, self.by_lang)
            if cand and cand.table_id not in used:
                used.add(cand.table_id)
                out.append(cand)
        return out

    def choose_tables(self, regime: str, k: int, rng: random.Random) -> list[TableInfo]:
        """Pick k distinct tables for one mixed doc."""
        if regime == "anchor":
            first = rng.choices(self.tables, weights=self._weights)[0]
            used = {first.table_id}
            return [first] + self._anchor_fill(rng, k - 1, used)
        if regime == "uniform":
            return rng.sample(self.tables, min(k, len(self.tables)))
        if regime == "confusable":
            gid = rng.choice(sorted(self.groups))
            pair_ids = rng.sample(self.groups[gid], 2)
            picked = [self.by_id[tid] for tid in pair_ids]
            used = set(pair_ids)
            picked += self._anchor_fill(rng, k - 2, used)
            return picked
        raise ValueError(f"unknown regime {regime}")
