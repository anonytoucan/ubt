"""Generation machinery for data-u1: pool dispatch, acceptance, global
dedup, the eval-registry gate, ids and accounting.

Plans come from one seeded RNG in the main process and results are read
with ordered imap, so every id and row is a function of (seed, config,
engine); imap_unordered would break byte-identical rebuilds.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

from ubt.u1.records import dedup_key, make_u1_record
from ubt.u1.registry import HashRegistry
from ubt.u1.work import process


class U1Run:
    def __init__(self, pool, engine_id: str, status_by_id: dict[str, str],
                 registry: HashRegistry, rng: random.Random, chunksize: int = 8):
        self.pool = pool
        self.engine_id = engine_id
        self.status_by_id = status_by_id
        self.registry = registry
        # units of accepted dev rows: a train row hitting one is rejected
        # (dev_overlap), a second net behind dev's own source lines
        self.dev_registry = HashRegistry()
        self.rng = rng
        self.chunksize = chunksize
        self.seen: set[bytes] = set()
        self.seq: Counter = Counter()
        self.rejections: dict[str, Counter] = defaultdict(Counter)
        self.rejected_tables: dict[str, Counter] = defaultdict(Counter)
        self.shortfalls: list[dict] = []
        self.examples: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    # ------------------------------------------------------------------ ids
    def next_id(self, form: str) -> str:
        self.seq[form] += 1
        return f"u1-{form}-{self.seq[form]:07d}"

    def reject(self, form: str, reason: str, table: str | None = None,
               example: str | None = None) -> None:
        """Count a rejection. eval_overlap texts are never kept as examples:
        they hold eval units, possibly from the sealed NIKL test split."""
        self.rejections[form][reason] += 1
        if table:
            self.rejected_tables[table][reason] += 1
        if example is not None and reason != "eval_overlap" \
                and len(self.examples[form][reason]) < 5:
            self.examples[form][reason].append(example[:160])

    def shortfall(self, stratum: str, cell: str, got: int, target: int, reason: str) -> None:
        if got < target:
            self.shortfalls.append({"stratum": stratum, "cell": cell, "got": got,
                                    "target": target, "missing": target - got,
                                    "reason": reason})

    # ------------------------------------------------------------------ accept
    def build(self, result: dict, *, form: str, split: str, filter_fn=None,
              dedup: bool = True, extra: dict | None = None, seg_join: str = "\n",
              tables: list[str] | None = None, langs: list[str] | None = None):
        """All acceptance checks, no side effects except rejection counts and
        `filter_fn` bookkeeping. Returns (record, dedup_key) or None."""
        if not result.get("ok"):
            self.reject(form, result.get("reason", "fail"), result.get("fail_table"))
            return None
        plan = result["plan"]
        segs = result["segments"]
        tabs = tables if tables is not None else [s["table"] for s in segs]
        text = seg_join.join(s["text"] for s in segs)
        key = dedup_key(tabs, text)
        if dedup and key in self.seen:
            self.reject(form, "duplicate")
            return None
        rec = make_u1_record(
            rec_id="PENDING", form=form, split=split, engine=self.engine_id,
            doc_type=plan["doc_type"], regime=plan["regime"],
            switch_density=plan["switch_density"], segments=segs,
            source=plan["source"], n_sentences=plan["n_sentences"], tag="",
            status_by_id=self.status_by_id, seg_join=seg_join, tables=tables,
            langs=langs, extra={**(plan.get("extra") or {}), **(extra or {})})
        if split != "eval" and self.registry.hits(rec):
            self.reject(form, "eval_overlap", example=text)
            return None
        if split == "train" and self.dev_registry.hits(rec):
            self.reject(form, "dev_overlap")
            return None
        if filter_fn is not None and not filter_fn(rec):
            self.reject(form, "cell_full")
            return None
        return rec, key

    def commit(self, rec: dict, key: bytes, *, form: str, split: str, tag,
               dedup: bool = True) -> dict:
        rec["tag"] = tag(rec) if callable(tag) else tag
        rec["id"] = self.next_id(form)
        if dedup:
            self.seen.add(key)
        if split == "eval":
            self.registry.add_row(rec)
        elif split == "dev":
            self.dev_registry.add_row(rec, source="dev_rows")
        return rec

    def accept(self, result: dict, *, form: str, split: str, tag, filter_fn=None,
               dedup: bool = True, extra: dict | None = None, seg_join: str = "\n",
               tables: list[str] | None = None, langs: list[str] | None = None) -> dict | None:
        got = self.build(result, form=form, split=split, filter_fn=filter_fn,
                         dedup=dedup, extra=extra, seg_join=seg_join, tables=tables,
                         langs=langs)
        if got is None:
            return None
        return self.commit(got[0], got[1], form=form, split=split, tag=tag, dedup=dedup)

    def _pre_reject(self, plan: dict, form: str, split: str, batch_keys: set) -> bool:
        """Dedup and registry checks before translation, for plans whose
        (tables, text) is known up front."""
        segs = plan.get("segments")
        if plan.get("kind", "louis") != "louis" or not segs:
            return False
        key = dedup_key([sg["table"] for sg in segs], "\n".join(sg["text"] for sg in segs))
        if key in self.seen or key in batch_keys:
            self.reject(form, "duplicate")
            return True
        if split != "eval":
            text = "\n".join(sg["text"] for sg in segs)
            if self.registry.hits(text):
                self.reject(form, "eval_overlap", example=text)
                return True
            if split == "train" and self.dev_registry.hits(text):
                self.reject(form, "dev_overlap")
                return True
        batch_keys.add(key)
        return False

    def screened_out(self, text: str, form: str, split: str) -> bool:
        """Registry screen for plans built outside fill() (S1b lines)."""
        if split != "eval" and self.registry.hits(text):
            self.reject(form, "eval_overlap", example=text)
            return True
        if split == "train" and self.dev_registry.hits(text):
            self.reject(form, "dev_overlap")
            return True
        return False

    # ------------------------------------------------------------------ fill
    def fill(self, target: int, plan_fn, *, form: str, split: str, tag,
             filter_fn=None, max_rounds: int = 60, max_overdraw: float = 40.0,
             extra: dict | None = None, round_hook=None, patience: int = 4) -> list[dict]:
        """Generate until `target` rows are accepted, the plan source runs out,
        or `patience` consecutive rounds accept nothing."""
        out: list[dict] = []
        rate = 0.8
        idle = 0
        for _ in range(max_rounds):
            if len(out) >= target:
                break
            if round_hook is not None:
                round_hook()
            deficit = target - len(out)
            n_plans = max(64 * (1 + idle),
                          int(deficit * min(max_overdraw, 1.15 / max(rate, 1e-3))) + 4)
            plans = []
            batch_keys: set[bytes] = set()
            nones = 0
            for _ in range(n_plans):
                p = plan_fn()
                if p is None:
                    self.reject(form, "no_source")
                    nones += 1
                    if nones >= 2000:      # source exhausted / cell closed
                        break
                    continue
                nones = 0
                if self._pre_reject(p, form, split, batch_keys):
                    continue
                plans.append(p)
            if not plans:
                idle += 1
                rate /= 2
                if idle >= patience:
                    break
                continue
            got = 0
            for res in self.pool.imap(process, plans, self.chunksize):
                if len(out) >= target:
                    continue
                rec = self.accept(res, form=form, split=split, tag=tag,
                                  filter_fn=filter_fn, extra=extra)
                if rec is not None:
                    out.append(rec)
                    got += 1
            if got == 0:
                idle += 1
                rate /= 2
                if idle >= patience:
                    break
                continue
            idle = 0
            rate = 0.5 * rate + 0.5 * (got / len(plans))
        return out


def carve_dev(records: list[dict], n: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Stratified carve of n rows -> (train, dev). Unused by the builder:
    such dev rows can share sentences with train rows."""
    if n <= 0 or not records:
        return records, []
    n = min(n, len(records))
    strata: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        strata[(r["doc_type"], r["regime"], r["len_bucket"], r["k"])].append(i)
    total = len(records)
    pick: list[int] = []
    for key in sorted(strata):
        idxs = strata[key]
        take = min(len(idxs), max(1, round(n * len(idxs) / total)))
        pick.extend(rng.sample(idxs, take))
    rng.shuffle(pick)
    chosen = set(pick[:n])
    if len(chosen) < n:   # rounding shortfall: top up uniformly
        rest = [i for i in range(total) if i not in chosen]
        chosen.update(rng.sample(rest, n - len(chosen)))
    dev = [records[i] for i in sorted(chosen)]
    train = [r for i, r in enumerate(records) if i not in chosen]
    for r in dev:
        r["split"] = "dev"
    return train, dev
