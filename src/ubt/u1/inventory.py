"""data-u1 table inventory on the pinned liblouis 3.38 build.

  scan 3.38 system tables -> drop the fork's ko-2020 family, add the staged ko-2024-g2.ctb
  -> weights (head share, anchor weights; holdout hr/mt/sw = 0)
  -> status (configs/currency_overrides.yaml + u1 rules) -> empirical alias dedup (measure_aliases)
  -> p ∝ (status factor × usage)^α -> confusable groups.

Config table ids go through `resolve_code`: tables removed in 3.38 map to their SUCCESSORS, and an id still missing
from the inventory is an error, since a skipped typo would give its code 0 docs.
"""
from __future__ import annotations

import copy
import fnmatch
import itertools
import os
from dataclasses import dataclass, field

import yaml

from ubt.tables import (
    TableInfo,
    apply_status_overrides,
    assign_weights,
    derive_confusable_groups,
    scan_tables,
)
from ubt.u1.engine import EXCLUDED_TABLE_GLOBS, KO2024_LABEL, EngineSpec

# tables removed or renamed between 3.34 and 3.38 -> (successor, provenance)
SUCCESSORS = {
    "hu.tbl": ("hu-hu-g1.ctb", "liblouis 65242b25 'Drop hu.tbl file — the metadata has been moved to hu-hu-g1.ctb'"),
    "it.tbl": ("it-it-comp6.utb", "liblouis NEWS 3.37: 'Removed it.tbl — use it-it-comp6.utb instead' (e3d213a9)"),
    "ko-2020-g2.ctb": (KO2024_LABEL, "ko-2020-g2 excluded (2020 rule examples 74.8%); ko-2024-g2 is the current Korean table"),
}

_RANK = {"current": 3, "parallel": 2, "superseded": 1}
U1_STATUS_RULES = {   # prepended to the per-locale rules of currency_overrides.yaml
    "ko": [{"pattern": "ko-2024-*", "status": "current"},
           {"pattern": "ko-2020-*", "status": "excluded"}],
    # en-nz-g1/g2 (new in 3.38) would default to `current` and win the alphabetical grade-2 tie against
    # en-ueb-g2; as a regional variant they are `parallel`, which keeps en-ueb-g2.ctb the English anchor.
    "en": [{"pattern": "en-nz-*", "status": "parallel"}],
}


@dataclass
class Inventory:
    tables: list[TableInfo]
    groups: dict[str, list[str]]
    excluded_files: list[str]
    alias: dict
    usability: dict = field(default_factory=dict)
    successors_applied: dict[str, str] = field(default_factory=dict)
    unusable_codes_skipped: dict[str, str] = field(default_factory=dict)

    @property
    def by_id(self) -> dict[str, TableInfo]:
        return {t.table_id: t for t in self.tables}

    @property
    def weighted(self) -> list[TableInfo]:
        return [t for t in self.tables if t.weight > 0]

    @property
    def holdout(self) -> list[TableInfo]:
        return [t for t in self.tables if t.holdout]

    def status_by_id(self) -> dict[str, str]:
        return {t.table_id: t.status for t in self.tables}

    def resolve_code(self, code: str, *, allow_holdout: bool = False,
                     skip_unusable: bool = False) -> str | None:
        """Config table id -> inventory id (successor/alias-mapped); raises unless weighted or allowed holdout.
        With skip_unusable, a table failing the usability probe returns None; an unknown id still raises."""
        tid = code
        if tid not in self.by_id and tid in SUCCESSORS:
            tid = SUCCESSORS[tid][0]
            self.successors_applied[code] = tid
        t = self.by_id.get(tid)
        if t is None:
            raise KeyError(f"table {code!r} (-> {tid!r}) is not in the 3.38 inventory")
        if t.status == "duplicate-excluded":   # empirical alias -> its keeper
            keep = next((p["keep"] for p in self.alias.get("alias_pairs", [])
                         if p["exclude"] == tid and p["keep"] not in self.alias.get("exclude", [])),
                        None)
            if keep is None:
                raise KeyError(f"table {code!r} is alias-excluded with no keeper")
            self.successors_applied[code] = keep
            tid, t = keep, self.by_id[keep]
        if skip_unusable and tid in self.usability.get("excluded", {}):
            self.unusable_codes_skipped[code] = self.usability["excluded"][tid]
            return None
        if t.weight <= 0 and not (allow_holdout and t.holdout):
            raise KeyError(f"table {code!r} (-> {tid!r}) has zero weight "
                           f"(status={t.status}, holdout={t.holdout})")
        return tid

    def to_json(self, out_dir: str | None = None) -> dict:
        """Paths under `out_dir` are written relative to it, so the inventory sha256 does not depend on it."""
        def rel(p: str) -> str:
            if out_dir:
                ap, root = os.path.abspath(p), os.path.abspath(out_dir)
                if ap.startswith(root + os.sep):
                    return "<out>/" + os.path.relpath(ap, root)
            return p
        return {
            "n_scanned_literary": len(self.tables) + len(self.excluded_files),
            "n_weighted": len(self.weighted),
            "n_holdout": len(self.holdout),
            "excluded_files": self.excluded_files,
            "successors": {k: {"successor": v[0], "provenance": v[1]} for k, v in SUCCESSORS.items()},
            "successors_applied": self.successors_applied,
            "alias_dedup": self.alias,
            "usability_probe": self.usability,
            "unusable_codes_skipped": self.unusable_codes_skipped,
            "tables": [{
                "table_id": t.table_id, "path": rel(t.path), "language": t.language,
                "base_lang": t.base_lang, "grade": t.grade,
                "contraction": t.contraction, "direction": t.direction,
                "status": t.status, "weight": t.weight, "holdout": t.holdout,
                "groups": t.groups,
            } for t in sorted(self.tables, key=lambda t: t.table_id)],
            "groups": self.groups,
        }


def load_status_overrides(repo_root: str, rel: str = "configs/currency_overrides.yaml") -> dict:
    path = rel if os.path.isabs(rel) else os.path.join(repo_root, rel)
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    locales = cfg.setdefault("locales", {})
    for lang, rules in U1_STATUS_RULES.items():
        locales[lang] = list(rules) + list(locales.get(lang) or [])
    return cfg


def scan_u1_tables(spec: EngineSpec) -> tuple[list[TableInfo], list[str]]:
    tables = scan_tables([])          # system dir only (UBT_LOUIS_TABLES pinned)
    kept, excluded = [], []
    for t in tables:
        if any(fnmatch.fnmatch(t.table_id, g) for g in EXCLUDED_TABLE_GLOBS):
            excluded.append(t.table_id)
        else:
            kept.append(t)
    if any(t.table_id == KO2024_LABEL for t in kept):
        raise RuntimeError(f"{KO2024_LABEL} unexpectedly present in the system dir")
    kept.append(TableInfo(
        table_id=KO2024_LABEL, path=os.path.join(spec.stage_dir, KO2024_LABEL),
        language="ko", base_lang="ko", type="literary", grade="2",
        contraction="full", direction="forward"))
    return kept, sorted(excluded)


def _weights_and_status(tables, cfg, sourced, overrides, alias_excluded):
    tables = assign_weights(tables, sourced, cfg["head_langs"], cfg["head_share"],
                            cfg["holdout_languages"], cfg["inter_mixed"]["anchor_langs"])
    return apply_status_overrides(tables, overrides,
                                  status_factors=cfg.get("status_factors"),
                                  alias_excluded=alias_excluded,
                                  alpha=cfg.get("alpha", 1.0))


def _alias_job(job):
    import ubt.worker as w  # noqa: PLC0415
    a, b, texts = job
    ok = eq = 0
    for text in texts:
        ra = w._TR.translate(a, text, check_undefined=True)
        rb = w._TR.translate(b, text, check_undefined=True)
        if ra.ok and rb.ok:
            ok += 1
            eq += ra.braille == rb.braille
    return ok, eq


def measure_aliases(pool, tables: list[TableInfo], source, n_sent: int = 2000,
                    threshold: float = 0.999, min_ok: int = 200) -> dict:
    """Empirical aliases: same-base_lang weighted pairs whose translations agree on >= threshold of the source
    sentences both translate. The higher status rank (then alphabetical) is kept; the other is duplicate-excluded."""
    by_lang: dict[str, list[TableInfo]] = {}
    for t in tables:
        if t.weight > 0:
            by_lang.setdefault(t.base_lang, []).append(t)
    jobs, meta = [], []
    for lang, tabs in sorted(by_lang.items()):
        if len(tabs) < 2:
            continue
        src = source.for_lang(lang)
        if src is None or len(src) == 0:
            continue
        texts = src.sentences[:n_sent]
        for a, b in itertools.combinations(sorted(t.table_id for t in tabs), 2):
            jobs.append((a, b, texts))
            meta.append((lang, a, b))
    results = pool.map(_alias_job, jobs, chunksize=1)
    by_id = {t.table_id: t for t in tables}
    pairs, measured = [], []
    for (lang, a, b), (ok, eq) in zip(meta, results):
        rate = eq / ok if ok else 0.0
        measured.append({"lang": lang, "a": a, "b": b, "n_ok": ok,
                         "collision_rate": round(rate, 5)})
        if ok < min_ok or rate < threshold:
            continue
        keep, drop = sorted((by_id[a], by_id[b]),
                            key=lambda t: (-_RANK.get(t.status, 0), t.table_id))
        pairs.append({"lang": lang, "a": a, "b": b, "n": ok,
                      "collision_rate": round(rate, 5),
                      "keep": keep.table_id, "exclude": drop.table_id})
    # an excluded table must not be anyone's keeper (chains a~b~c keep one)
    exclude: set[str] = set()
    for p in sorted(pairs, key=lambda p: (p["keep"], p["exclude"])):
        if p["keep"] in exclude:
            continue
        exclude.add(p["exclude"])
    return {"threshold": threshold, "n_sentences": n_sent, "min_ok": min_ok,
            "n_pairs_measured": len(measured), "alias_pairs": pairs,
            "exclude": sorted(exclude), "measured": measured}


def _probe_job(job):
    import ubt.worker as w  # noqa: PLC0415
    from ubt.verify import translate_verified  # noqa: PLC0415
    table, texts = job
    ok = 0
    reasons: dict[str, int] = {}
    for text in texts:
        v = translate_verified(w._TR, table, text)
        if v.ok:
            ok += 1
        else:
            reasons[v.reason] = reasons.get(v.reason, 0) + 1
    return ok, reasons


def probe_usability(pool, tables: list[TableInfo], source, n: int = 400,
                    min_accept: float = 0.02) -> dict:
    """Share of each table's source sentences that pass the strict generation checks under this engine. Tables
    below `min_accept` can never yield rows and are excluded, with the measurement recorded."""
    jobs, ids = [], []
    for t in tables:
        if t.weight <= 0 and not t.holdout:
            continue
        src = source.for_lang(t.base_lang)
        texts = src.sentences[:n] if src is not None else []
        jobs.append((t.table_id, texts))
        ids.append(t.table_id)
    res = pool.map(_probe_job, jobs, chunksize=1)
    per, excluded = {}, {}
    for tid, (job, (ok, reasons)) in zip(ids, zip(jobs, res)):
        n_t = len(job[1])
        rate = ok / n_t if n_t else 0.0
        per[tid] = {"n": n_t, "ok": ok, "rate": round(rate, 4), "reasons": reasons}
        if rate < min_accept:
            excluded[tid] = f"unusable under engine: {ok}/{n_t} source sentences pass ({reasons})"
    return {"n_probe": n, "min_accept": min_accept, "per_table": per, "excluded": excluded}


def build_inventory(cfg: dict, spec: EngineSpec, source, repo_root: str,
                    pool=None, alias_result: dict | None = None) -> Inventory:
    """Full u1 inventory; needs `pool` (engine workers, aliases measured) or a precomputed `alias_result`."""
    base, excluded = scan_u1_tables(spec)
    langs = sorted({t.base_lang for t in base})
    sourced = source.sourced_langs(langs)
    overrides = load_status_overrides(repo_root, cfg.get("status_overrides_path",
                                                         "configs/currency_overrides.yaml"))
    if alias_result is None:
        if pool is None:
            raise ValueError("need a worker pool to measure aliases")
        pre = _weights_and_status(copy.deepcopy(base), cfg, sourced, overrides, [])
        alias_result = measure_aliases(
            pool, pre, source,
            n_sent=(cfg.get("empirical_alias_dedup") or {}).get("n_sentences", 2000),
            threshold=(cfg.get("empirical_alias_dedup") or {}).get("min_pairwise_collision", 0.999))
    alias_result = dict(alias_result)
    alias_result["prior_3_34_exclude"] = (cfg.get("empirical_alias_dedup") or {}).get("prior_exclude", [])
    tables = _weights_and_status(base, cfg, sourced, overrides, alias_result["exclude"])
    ucfg = cfg.get("usability_probe") or {}
    usability = probe_usability(pool, tables, source, n=ucfg.get("n_sentences", 400),
                                min_accept=ucfg.get("min_accept", 0.02)) if pool is not None \
        else {"excluded": {}, "per_table": {}, "skipped": "no pool"}
    for t in tables:
        if t.table_id in usability["excluded"]:
            if t.table_id == KO2024_LABEL:
                raise RuntimeError(f"{KO2024_LABEL} failed the usability probe")
            t.status = "excluded"
            t.weight = 0.0
            if t.holdout:
                t.holdout = False       # a held-out table that yields nothing
    groups = derive_confusable_groups(tables, cfg["script_groups"])
    inv = Inventory(tables=tables, groups=groups, excluded_files=excluded,
                    alias=alias_result, usability=usability)
    ko = inv.by_id[KO2024_LABEL]
    if ko.status != "current" or ko.weight <= 0:
        raise RuntimeError(f"{KO2024_LABEL} must be current/weighted, got {ko.status}/{ko.weight}")
    held = sorted(t.table_id for t in inv.holdout)
    if any(t.weight > 0 for t in inv.holdout):
        raise RuntimeError("a holdout table has train weight")
    exp = cfg.get("expected_holdout_tables")
    if exp is not None and sorted(exp) != held:
        raise RuntimeError(f"holdout tables {held} != expected {sorted(exp)}")
    return inv


def _clean_lines_job(job):
    import ubt.worker as w  # noqa: PLC0415
    from ubt.verify import translate_verified  # noqa: PLC0415
    table, texts = job
    return [translate_verified(w._TR, table, t).ok for t in texts]


def probe_capacity(pool, tables: list[TableInfo], source, max_lines: int = 50000,
                   chunk: int = 250) -> dict:
    """S1 capacity of each weighted table with a small source pool: clean lines plus clean runs of 2-8 lines.
    Shows before generation which floors a build can meet; pools above max_lines are reported as 'large'."""
    jobs, meta = [], []
    large = {}
    for t in tables:
        if t.weight <= 0:
            continue
        src = source.for_lang(t.base_lang)
        if src is None:
            continue
        if len(src) > max_lines:
            large[t.table_id] = len(src)
            continue
        lines = src.sentences
        for i in range(0, len(lines), chunk):
            jobs.append((t.table_id, lines[i:i + chunk]))
            meta.append(t.table_id)
    flags: dict[str, list[bool]] = {}
    for tid, res in zip(meta, pool.imap(_clean_lines_job, jobs, 1)):
        flags.setdefault(tid, []).extend(res)
    out = {}
    for tid, ok in flags.items():
        n = len(ok)
        runs = {}
        for k in range(1, 9):
            runs[k] = sum(1 for i in range(n - k + 1) if all(ok[i:i + k]))
        out[tid] = {"pool_lines": n, "clean_lines": runs[1],
                    "capacity_1": runs[1], "capacity_2_3": runs[2] + runs[3],
                    "capacity_4_8": sum(runs[k] for k in range(4, 9)),
                    "capacity": sum(runs.values())}
    return {"max_lines": max_lines, "per_table": out, "large_pools": large}
