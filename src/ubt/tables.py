"""liblouis table inventory from table file metadata.

Selects forward-capable literary tables, assigns sampling weights and derives confusable groups: weighted tables of one
base_lang, or of one script group in the config. No pan-Latin group is built.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from ubt.translate import discover_system_tables_dir

EXCLUDED_FILES = {"th-comp8-backward.utb"}  # backward table: generation is forward-only
# Tables without metadata that holdout coverage needs.
METADATA_OVERRIDES = {
    "mt.ctb": {"language": "mt", "type": "literary", "grade": "1"},
}
_META_RE = re.compile(r"^#\+([\w-]+)\s*:\s*(.+?)\s*$")
_TABLE_EXTS = (".ctb", ".utb", ".tbl")
_INCLUDE_RE = re.compile(r"^include\s+(\S+)")


def primary_include_target(path: str) -> str | None:
    """First included entry table (.ctb/.utb/.tbl): what a wrapper table resolves to."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = _INCLUDE_RE.match(line)
                if m and m.group(1).endswith(_TABLE_EXTS):
                    return m.group(1)
    except OSError:
        pass
    return None


VALID_STATUSES = ("current", "parallel", "pending", "superseded", "excluded",
                  "duplicate-excluded")
_ZERO_WEIGHT_STATUSES = ("excluded", "duplicate-excluded")
_YEAR_TOKEN = re.compile(r"(?:^|[-_])(?:19|20)\d\d")


@dataclass
class TableInfo:
    table_id: str          # filename, e.g. "en-ueb-g2.ctb"
    path: str
    language: str          # raw #+language value
    base_lang: str         # normalized primary subtag, e.g. "en", "cmn"
    type: str
    grade: str
    contraction: str
    direction: str
    weight: float = 0.0
    holdout: bool = False
    status: str = "current"  # standard-recency adjudication (currency_overrides)
    groups: list[str] = field(default_factory=list)  # confusable group ids


def _read_metadata(path: str, max_lines: int = 60) -> dict[str, str]:
    meta: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i >= max_lines:
                    break
                m = _META_RE.match(line)
                if m:
                    meta.setdefault(m.group(1).lower(), m.group(2))
    except OSError:
        pass
    return meta


_LANG_NORMALIZE = {"zh": "cmn", "zho": "cmn", "no": "nb", "iw": "he", "in": "id"}


def normalize_lang(raw: str) -> str:
    base = raw.strip().lower().split("-")[0].split("_")[0]
    return _LANG_NORMALIZE.get(base, base)


def scan_tables(extra_table_dirs: list[str] | None = None) -> list[TableInfo]:
    """Scan table dirs for forward-capable literary tables; extra dirs win over the system dir on duplicate ids."""
    dirs = [os.path.expanduser(d) for d in (extra_table_dirs or [])]
    dirs.append(discover_system_tables_dir())
    seen: dict[str, TableInfo] = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname in EXCLUDED_FILES or fname in seen:
                continue
            if not fname.endswith(_TABLE_EXTS):
                continue
            meta = _read_metadata(os.path.join(d, fname))
            meta = {**meta, **METADATA_OVERRIDES.get(fname, {})}
            ttype = meta.get("type", "")
            direction = meta.get("direction", "both")
            language = meta.get("language", "")
            if ttype != "literary" or not language:
                continue
            if "forward" not in direction and direction != "both":
                continue  # generation is forward-only
            seen[fname] = TableInfo(
                table_id=fname,
                path=os.path.join(d, fname),
                language=language,
                base_lang=normalize_lang(language),
                type=ttype,
                grade=meta.get("grade", ""),
                contraction=meta.get("contraction", ""),
                direction=direction,
            )
    return list(seen.values())


def assign_weights(
    tables: list[TableInfo],
    sourced_langs: set[str],
    head_langs: list[str],
    head_share: float,
    holdout_languages: list[str],
    anchor_langs: dict[str, float],
) -> list[TableInfo]:
    """Set sampling weights in place: head languages share `head_share` by anchor_langs weight, sourced tail languages
    the rest uniformly, split evenly over each language's tables; holdout languages get 0."""
    holdouts = set(holdout_languages)
    by_lang: dict[str, list[TableInfo]] = {}
    for t in tables:
        t.holdout = t.base_lang in holdouts
        by_lang.setdefault(t.base_lang, []).append(t)

    heads = [l for l in head_langs if l in by_lang and l in sourced_langs and l not in holdouts]
    tails = sorted(
        l for l in by_lang
        if l not in heads and l in sourced_langs and l not in holdouts
    )
    anchor_floor = min(anchor_langs.values()) if anchor_langs else 1.0
    head_raw = {l: anchor_langs.get(l, anchor_floor) for l in heads}
    head_total = sum(head_raw.values()) or 1.0
    lang_mass: dict[str, float] = {l: head_share * w / head_total for l, w in head_raw.items()}
    tail_share = (1.0 - head_share) if heads else 1.0
    for l in tails:
        lang_mass[l] = tail_share / len(tails)

    for lang, tabs in by_lang.items():
        mass = lang_mass.get(lang, 0.0)
        for t in tabs:
            t.weight = mass / len(tabs)
    return tables


def apply_status_overrides(
    tables: list[TableInfo],
    overrides_cfg: dict,
    status_factors: dict[str, float] | None = None,
    alias_excluded: list[str] | None = None,
    alpha: float = 1.0,
) -> list[TableInfo]:
    """Set standard-recency statuses from configs/currency_overrides.yaml, then reweight; call after assign_weights.

    Status: first matching glob of `global_rules`, then of the table's locale; otherwise a year-stamped table whose
    language has an unmarked sibling is superseded, else current. Weight: (status factor x weight)^alpha, 0 if excluded.
    """
    import fnmatch  # noqa: PLC0415

    global_rules = overrides_cfg.get("global_rules") or []
    # YAML 1.1 parses the bare locale key `no` (Norwegian) as boolean False.
    locales = {
        normalize_lang("no" if k is False else "yes" if k is True else k): v
        for k, v in (overrides_cfg.get("locales") or {}).items()
    }
    for rule in global_rules + [r for rs in locales.values() for r in rs]:
        if rule["status"] not in VALID_STATUSES:
            raise ValueError(f"invalid table status {rule['status']!r}")

    def match(t: TableInfo) -> str | None:
        for rule in global_rules:
            if fnmatch.fnmatch(t.table_id, rule["pattern"]):
                return rule["status"]
        for rule in locales.get(t.base_lang, []):
            if fnmatch.fnmatch(t.table_id, rule["pattern"]):
                return rule["status"]
        return None

    def match_id(table_id: str, base_lang: str) -> str | None:
        for rule in global_rules:
            if fnmatch.fnmatch(table_id, rule["pattern"]):
                return rule["status"]
        for rule in locales.get(base_lang, []):
            if fnmatch.fnmatch(table_id, rule["pattern"]):
                return rule["status"]
        return None

    by_table_id = {t.table_id: t for t in tables}

    def inherited_status(t: TableInfo) -> str | None:
        """Status of the primary include target, up to 3 hops; a cross-language include only borrows definitions
        and is ignored."""
        path = t.path
        for _ in range(3):
            target = primary_include_target(path)
            if target is None:
                return None
            info = by_table_id.get(target)
            if info is not None and info.base_lang != t.base_lang:
                return None
            status = match_id(target, t.base_lang)
            if status is not None:
                return status
            path = os.path.join(os.path.dirname(path), target)
            if not os.path.isfile(path):
                return None
        return None

    by_lang: dict[str, list[TableInfo]] = {}
    for t in tables:
        by_lang.setdefault(t.base_lang, []).append(t)
    pending = []
    for t in tables:
        status = match(t)
        if status is None:
            status = inherited_status(t)  # explicit overrides still win
        if status is None:
            pending.append(t)
        else:
            t.status = status
    for t in pending:  # generic year rule, then default `current`
        has_unmarked_sibling = any(
            o is not t and not _YEAR_TOKEN.search(o.table_id)
            for o in by_lang[t.base_lang]
        )
        t.status = ("superseded"
                    if _YEAR_TOKEN.search(t.table_id) and has_unmarked_sibling
                    else "current")

    for tid in alias_excluded or []:
        for t in tables:
            if t.table_id == tid:
                t.status = "duplicate-excluded"

    factors = {"current": 1.0, "parallel": 1.0, "superseded": 1.0,
               **(status_factors or {})}
    for t in tables:
        factor = 0.0 if t.status in _ZERO_WEIGHT_STATUSES else factors.get(t.status, 1.0)
        t.weight = (t.weight * factor) ** alpha if factor > 0 and t.weight > 0 else 0.0
    return tables


def derive_confusable_groups(
    tables: list[TableInfo], script_groups: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Derive confusable groups from tables with weight > 0 and record each table's `groups` in place."""
    eligible = [t for t in tables if t.weight > 0]
    by_lang: dict[str, list[TableInfo]] = {}
    for t in eligible:
        by_lang.setdefault(t.base_lang, []).append(t)

    groups: dict[str, list[str]] = {}
    for lang, tabs in sorted(by_lang.items()):
        if len(tabs) >= 2:
            groups[f"lang:{lang}"] = sorted(t.table_id for t in tabs)
    for script, langs in script_groups.items():
        members = sorted(
            t.table_id for l in langs for t in by_lang.get(l, [])
        )
        if len(members) >= 2:
            groups[f"script:{script}"] = members

    by_id = {t.table_id: t for t in tables}
    for gid, members in groups.items():
        for tid in members:
            by_id[tid].groups.append(gid)
    return groups


def confusable_siblings(table: TableInfo, groups: dict[str, list[str]]) -> list[str]:
    """All distinct tables sharing at least one confusable group with `table`."""
    sibs: set[str] = set()
    for gid in table.groups:
        sibs.update(groups[gid])
    sibs.discard(table.table_id)
    return sorted(sibs)
