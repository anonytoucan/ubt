"""S7: NIKL human Korean braille (news, transcribed under the 2024 rules).

Split as scripts/liblouis/ko_table_nikl_check.py (sha256(doc) % 5 == 0 -> test). Test rows are sealed: never written to
train/dev and referenced only by id in eval/nikl_test_pointer.jsonl. Their hashes enter the in-memory eval registry so
no train row can contain one; the persisted registry holds 64-bit hashes only.

S7 rows are sampled by document from the dev split, stratified by (edition, genre) to keep the dev mix, because NIKL
documents are whole corpus files of very different sizes. Each stratum gets a quota proportional to its dev size,
filled with whole documents in seeded order; the document that overflows the quota contributes one contiguous run.

Each row: label ko-2024-g2.ctb, the human braille with ASCII space -> U+2800, f_consistent = (braille == f(text))
under the pinned ko-2024 table (reported, not enforced). Rows with other non-braille characters or with text in the
test split are dropped and counted.

Near-duplicate screen: dev and the sealed test share republished sentences with small edits, so a dev row is also
dropped when, on the loose form (registry.loose), it equals a test sentence, >= NEAR_DUP_FRAC of its shingles occur in
test text, or a test sentence has >= NEAR_DUP_FRAC of its shingles inside the row.
"""
from __future__ import annotations

import os
import random
import re
import sys

NIKL_REL = "data_nikl/raw/nikl"
MARGIN = 0.15        # candidate overdraw per stratum (dedup/registry/charset drops)
SHINGLE = 20         # chars of the loose form
NEAR_DUP_FRAC = 0.8
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=\S)")


def n_sentences(text: str) -> int:
    """Sentence count of one NIKL unit: sentence-final punctuation followed by more text starts a new one."""
    return 1 + len(_SENT_SPLIT.findall(text.strip()))


def _shingles(lz: str) -> set[int]:
    if len(lz) < SHINGLE:
        return {hash(lz)} if lz else set()
    return {hash(lz[i:i + SHINGLE]) for i in range(len(lz) - SHINGLE + 1)}


class NearDupScreen:
    """Loose-exact and shingle-containment screen against the sealed test texts; in memory only."""

    def __init__(self, test_texts):
        from ubt.u1.registry import loose  # noqa: PLC0415
        self._loose = loose
        self.exact: set[str] = set()
        self.all_sh: set[int] = set()
        self.index: dict[int, list[int]] = {}
        self.n_sh: list[int] = []
        for i, t in enumerate(sorted(test_texts)):
            lz = loose(t)
            if not lz:
                self.n_sh.append(0)
                continue
            self.exact.add(lz)
            sh = _shingles(lz)
            self.n_sh.append(len(sh))
            self.all_sh |= sh
            for h in sh:
                self.index.setdefault(h, []).append(i)

    def why(self, text: str) -> str | None:
        """None if clean, else the reason ('loose_exact' | 'row_in_test' | 'test_in_row')."""
        lz = self._loose(text)
        if not lz:
            return None
        if lz in self.exact:
            return "loose_exact"
        sh = _shingles(lz)
        if not sh:
            return None
        hit = [h for h in sh if h in self.all_sh]
        if len(hit) >= NEAR_DUP_FRAC * len(sh):
            return "row_in_test"
        cnt: dict[int, int] = {}
        for h in hit:
            for i in self.index.get(h, ()):
                cnt[i] = cnt.get(i, 0) + 1
        for i, c in cnt.items():
            if self.n_sh[i] and c >= NEAR_DUP_FRAC * self.n_sh[i]:
                return "test_in_row"
        return None


def load_nikl(repo_root: str) -> list[dict]:
    sys.path.insert(0, os.path.join(repo_root, "scripts", "liblouis"))
    from ko_table_nikl_check import load_pairs  # noqa: PLC0415
    return load_pairs(os.path.join(repo_root, NIKL_REL))


def human_braille(tgt: str) -> str | None:
    b = tgt.replace(" ", "⠀")
    if any(not (0x2800 <= ord(c) <= 0x28FF) for c in b):
        return None
    return b


def _apportion(total: int, sizes: dict[str, int]) -> dict[str, int]:
    z = sum(sizes.values())
    if total <= 0 or z == 0:
        return {k: 0 for k in sizes}
    raw = {k: total * v / z for k, v in sizes.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[: total - sum(out.values())]:
        out[k] += 1
    return out


def sample_dev(rows: list[dict], n: int, rng: random.Random,
               test_texts: set[str], screen: NearDupScreen | None = None,
               exclude_ids: set[str] | frozenset = frozenset()) -> tuple[list[dict], dict]:
    """(strata, stats); each stratum {"stratum", "quota", "rows"} holds candidate rows in document order, quota +
    MARGIN where possible. exclude_ids: NIKL ids already used (the dev split's S7 rows when sampling train)."""
    by_doc: dict[str, list[dict]] = {}
    for r in rows:
        if r["split"] == "dev" and r["id"] not in exclude_ids:
            by_doc.setdefault(r["doc"], []).append(r)
    by_stratum: dict[str, list[str]] = {}
    for d, rs in by_doc.items():
        by_stratum.setdefault(f"{rs[0]['ed']}{rs[0]['genre']}", []).append(d)
    sizes = {s: sum(len(by_doc[d]) for d in docs) for s, docs in by_stratum.items()}
    quotas = _apportion(n, sizes)
    stats = {"dev_docs": len(by_doc), "dev_rows": sum(sizes.values()),
             "strata_dev_rows": dict(sorted(sizes.items())), "quota": dict(sorted(quotas.items())),
             "docs_whole": 0, "docs_partial": 0, "dropped_charset": 0, "dropped_in_test": 0,
             "dropped_near_test": {}, "excluded_ids": len(exclude_ids)}
    strata = []
    for s in sorted(by_stratum):
        quota = quotas[s]
        if quota <= 0:
            continue
        want = quota + int(quota * MARGIN) + 8
        docs = sorted(by_stratum[s])
        rng.shuffle(docs)
        cand: list[dict] = []
        for d in docs:
            if len(cand) >= want:
                break
            rs = by_doc[d]
            room = want - len(cand)
            if len(rs) <= room:
                take = rs
                stats["docs_whole"] += 1
            else:
                start = rng.randrange(len(rs) - room + 1)
                take = rs[start:start + room]
                stats["docs_partial"] += 1
            for r in take:
                if r["src"] in test_texts:
                    stats["dropped_in_test"] += 1
                    continue
                why = screen.why(r["src"]) if screen is not None else None
                if why is not None:
                    stats["dropped_near_test"][why] = stats["dropped_near_test"].get(why, 0) + 1
                    continue
                b = human_braille(r["tgt"])
                if b is None:
                    stats["dropped_charset"] += 1
                    continue
                cand.append({**r, "braille": b})
        strata.append({"stratum": s, "quota": quota, "rows": cand})
    return strata, stats
