"""One hash registry for train/eval separation.

A text's units are the whole text, each '\\n'-separated line and each sub-sentence of >= 16 characters (>= 8 in
unspaced scripts). A train/dev row is rejected when any unit hits the eval registry; the verifier checks that
train ∩ eval is empty.

The eval registry holds every eval-row unit, every eval-source line (FLORES devtest, wiki_B minus eval_exclusions),
the sealed NIKL test sentences and the Korean regulation examples. Each unit gets two 64-bit sha1-prefix keys, and a
hit on either rejects the row (a collision only over-rejects):
  exact  sha1(NFC, whitespace-collapsed)
  loose  sha1("~" + NFKC-casefold without punctuation, symbol, separator, control and format characters), only when
         >= LOOSE_MIN_CHARS characters remain; it catches lines that differ only in case, punctuation or spacing.

Sealed units (NIKL test) are screened in memory, but registry.npz stores only their count and sha256 digest, so the
saved registry cannot tell whether a sentence is in the sealed test.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata

_WS = re.compile(r"\s+")
_SUB_SPLIT = re.compile(r"(?<=[.!?…؟।])\s+|(?<=[。！？；;])")
MIN_SUB_CHARS = 8          # unspaced scripts (e.g. zh)
MIN_SUB_CHARS_SPACED = 16  # spaced scripts: keeps generic fragments such as 'århundrede.' out
_UNSPACED = re.compile(r"[\u0E00-\u0EFF\u1000-\u109F\u1780-\u17FF\u3040-\u30FF\u3400-\u9FFF\uF900-\uFAFF]")


LOOSE_MIN_CHARS = 8


def _loose_drop_class() -> re.Pattern:
    """Regex char class (as ranges) of every code point in categories P*, S*, Z*, Cc, Cf."""
    import sys  # noqa: PLC0415
    ranges, start, prev = [], None, None
    for cp in range(sys.maxunicode + 1):
        cat = unicodedata.category(chr(cp))
        drop = cat[0] in "PSZ" or cat in ("Cc", "Cf")
        if drop and start is None:
            start = cp
        if not drop and start is not None:
            ranges.append((start, prev))
            start = None
        prev = cp
    if start is not None:
        ranges.append((start, prev))

    def esc(c: int) -> str:
        return re.escape(chr(c))
    body = "".join(esc(a) if a == b else f"{esc(a)}-{esc(b)}" for a, b in ranges)
    return re.compile(f"[{body}]+")


_LOOSE_DROP = _loose_drop_class()


def norm(s: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", s.strip()))


def loose(s: str) -> str:
    """Letters, marks and digits only, NFKC-casefolded (the near-duplicate key)."""
    return _LOOSE_DROP.sub("", unicodedata.normalize("NFKC", s).casefold())


def _sha64(b: bytes) -> int:
    return int.from_bytes(hashlib.sha1(b).digest()[:8], "big")


def h64(s: str) -> int:
    return _sha64(norm(s).encode("utf-8"))


def h64_loose(s: str) -> int | None:
    lz = loose(s)
    return _sha64(("~" + lz).encode("utf-8")) if len(lz) >= LOOSE_MIN_CHARS else None


def keys(s: str) -> list[int]:
    """Exact key + (if long enough) loose key of one unit."""
    lz = h64_loose(s)
    return [h64(s)] if lz is None else [h64(s), lz]


def sub_sentences(s: str, min_chars: int = MIN_SUB_CHARS) -> list[str]:
    out = []
    for p in _SUB_SPLIT.split(s):
        p = p.strip()
        if len(p) >= MIN_SUB_CHARS_SPACED or (len(p) >= min_chars and _UNSPACED.search(p)):
            out.append(p)
    return out


def text_units(text: str) -> set[int]:
    """Exact + loose hashes of the whole text, each line, and each sub-sentence."""
    out: set[int] = set()
    if not text or not text.strip():
        return out
    out.update(keys(text))
    for line in text.split("\n"):
        if line.strip():
            out.update(keys(line))
            for ss in sub_sentences(line):
                out.update(keys(ss))
    return out


def row_units(rec: dict) -> set[int]:
    """Units of a row's text and, for newline-joined rows, of each segment. Inline (carrier + math) rows add only
    the whole text, since their pieces are template fragments, not source sentences."""
    u = text_units(rec.get("text") or "")
    if rec.get("seg_join", "\n") == "\n":
        for s in rec.get("segments") or []:
            if s.get("text"):
                u |= text_units(s["text"])
    return u


class HashRegistry:
    def __init__(self) -> None:
        self.hashes: set[int] = set()        # everything screened against
        self.public_units: set[int] = set()  # units from some non-sealed source
        self.sources: dict[str, int] = {}

    def add_text(self, text: str, source: str, sealed: bool = False) -> None:
        u = text_units(text)
        self.hashes |= u
        if not sealed:
            self.public_units |= u
        self.sources[source] = self.sources.get(source, 0) + 1

    def public(self) -> set[int]:
        """Units that may be persisted (sealed-only units are not)."""
        return self.public_units

    def mark_public(self, units: set[int]) -> None:
        self.public_units |= units

    def add_row(self, rec: dict, source: str = "eval_rows") -> None:
        u = row_units(rec)
        self.hashes |= u
        self.mark_public(u)
        self.sources[source] = self.sources.get(source, 0) + 1

    def add_lines_file(self, path: str, source: str, exclude_sha1: frozenset[str] = frozenset(),
                       transform=None) -> int:
        """Register every line of a raw source file, skipping lines whose sentence_hash is in `exclude_sha1`."""
        from ubt.sources import sentence_hash  # noqa: PLC0415
        n = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                if exclude_sha1 and sentence_hash(s) in exclude_sha1:
                    continue
                if transform:
                    s = transform(s)
                u = text_units(s)
                if "\u200b" in s:   # the th view replaces ZWSP by a space
                    from ubt.u1.sources import _zwsp_fix  # noqa: PLC0415
                    u |= text_units(_zwsp_fix(s))
                self.hashes |= u
                self.mark_public(u)
                n += 1
        self.sources[source] = self.sources.get(source, 0) + n
        return n

    def hits(self, rec_or_text) -> int:
        u = row_units(rec_or_text) if isinstance(rec_or_text, dict) else text_units(rec_or_text)
        return len(u & self.hashes)

    def __contains__(self, text: str) -> bool:
        return bool(text_units(text) & self.hashes)

    def __len__(self) -> int:
        return len(self.hashes)


def eval_source_files(data_dir: str) -> list[tuple[str, str]]:
    """(path, source-name) for every eval-only raw source under data/."""
    out = []
    dt = os.path.join(data_dir, "flores200_dataset", "devtest")
    for f in sorted(os.listdir(dt)):
        if f.endswith(".devtest"):
            out.append((os.path.join(dt, f), "flores_devtest"))
    wb = os.path.join(data_dir, "wiki_txt_B")
    for f in sorted(os.listdir(wb)):
        if f.endswith(".txt"):
            out.append((os.path.join(wb, f), "wiki_B"))
    return out


def train_source_files(data_dir: str) -> list[tuple[str, str]]:
    out = []
    dv = os.path.join(data_dir, "flores200_dataset", "dev")
    for f in sorted(os.listdir(dv)):
        if f.endswith(".dev"):
            out.append((os.path.join(dv, f), "flores_dev"))
    wa = os.path.join(data_dir, "wiki_txt_A")
    for f in sorted(os.listdir(wa)):
        if f.endswith(".txt"):
            out.append((os.path.join(wa, f), "wiki_A"))
    return out


EXAMPLE_FILES = ("resources/korean_braille_2024/examples.jsonl", "resources/korean_braille_2024/examples_math.jsonl")


def rule_example_texts(repo_root: str) -> list[str]:
    """Korean regulation example texts and math LaTeX, the eval-only certification sets of ko-2024 and kmath.
    A missing file is an error, or its eval-only registration and kmath exclusion would vanish silently."""
    texts = []
    for rel in EXAMPLE_FILES:
        p = os.path.join(repo_root, rel)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"rule example file missing: {p}")
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("row") not in (None, "example"):
                    continue
                for k in ("print", "print_text", "text", "print_mixed", "print_plain", "latex"):
                    v = r.get(k)
                    if isinstance(v, str) and len(v.strip()) >= 2:
                        texts.append(v)
    return texts


# --------------------------------------------------------------------------- persisted registry
def line_hashes(path: str, exclude_sha1: frozenset[str] = frozenset()) -> set[int]:
    """h64 of every non-empty line of a raw source file (line level only)."""
    from ubt.sources import sentence_hash  # noqa: PLC0415
    out = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s and not (exclude_sha1 and sentence_hash(s) in exclude_sha1):
                out.add(h64(s))
    return out


def kmath_example_latex(repo_root: str) -> list[str]:
    """LaTeX of the Korean math regulation examples (kmath certification set)."""
    p = os.path.join(repo_root, "resources/korean_braille_2024/examples_math.jsonl")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"kmath example file missing: {p}")
    out = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("row") == "example" and isinstance(r.get("latex"), str) and r["latex"].strip():
                out.append(r["latex"])
    if not out:
        raise ValueError(f"no kmath example latex in {p}")
    return out


def _line_hashes_job(job: tuple[str, frozenset]) -> list[int]:
    path, excl = job
    return list(line_hashes(path, excl))


def build_line_registry(data_dir: str, nikl_rows: list[dict],
                        exclude_sha1: frozenset[str], pool=None) -> dict[str, set[int]]:
    """Line-level hashes of every source: train side flores_dev, wiki_A, nikl_dev; eval side flores_devtest (minus
    eval_exclusions), wiki_B, nikl_test. A pool hashes the files in parallel."""
    side: dict[str, set[int]] = {"train:flores_dev": set(), "train:wiki_A": set(),
                                 "eval:flores_devtest": set(), "eval:wiki_B": set()}
    jobs = [(p, f"train:{n}", frozenset()) for p, n in train_source_files(data_dir)] + \
           [(p, f"eval:{n}", exclude_sha1) for p, n in eval_source_files(data_dir)]
    args = [(p, ex) for p, _n, ex in jobs]
    results = pool.imap(_line_hashes_job, args, 1) if pool is not None \
        else map(_line_hashes_job, args)
    for (_p, name, _ex), hs in zip(jobs, results):
        side[name].update(hs)
    side["train:nikl_dev"] = {h64(r["src"]) for r in nikl_rows if r["split"] == "dev"}
    side["eval:nikl_test"] = {h64(r["src"]) for r in nikl_rows if r["split"] == "test"}
    return side


SEALED_LINE_SETS = ("eval:nikl_test",)


def digest(hs: set[int]) -> str:
    h = hashlib.sha256()
    for x in sorted(hs):
        h.update(int(x).to_bytes(8, "big"))
    return h.hexdigest()


def save_registry(path: str, reg: HashRegistry, lines: dict[str, set[int]]) -> dict:
    """Write registry.npz: the public eval units (exact + loose keys) and line hashes of every non-sealed source.
    Sealed units and NIKL test lines appear only as count + sha256 digest; no text is stored."""
    import numpy as np  # noqa: PLC0415
    public = reg.public()
    sealed_only = reg.hashes - public
    arrays = {"eval_units": np.array(sorted(public), dtype=np.uint64)}
    for name, hs in sorted(lines.items()):
        if name in SEALED_LINE_SETS:
            continue
        arrays["lines__" + name.replace(":", "__")] = np.array(sorted(hs), dtype=np.uint64)
    np.savez(path, **arrays)
    tr = set().union(*(v for k, v in lines.items() if k.startswith("train:")))
    ev = set().union(*(v for k, v in lines.items() if k.startswith("eval:")))
    return {"path": os.path.basename(path), "n_eval_units": len(reg.hashes),
            "n_eval_units_public": len(public),
            "sealed_units": {"n": len(sealed_only), "sha256": digest(sealed_only)},
            "sealed_line_sets": {k: {"n": len(lines[k]), "sha256": digest(lines[k])}
                                 for k in SEALED_LINE_SETS if k in lines},
            "lines": {k: len(v) for k, v in sorted(lines.items())},
            "line_overlap_train_eval": len(tr & ev),
            "line_overlap_by_pair": {
                f"{a}|{b}": len(lines[a] & lines[b])
                for a in sorted(lines) if a.startswith("train:")
                for b in sorted(lines) if b.startswith("eval:") and lines[a] & lines[b]},
            "hash": "units: sha1(NFC, ws-collapsed)[:8] + sha1('~'+loose)[:8]; "
                    "lines: sha1(NFC, ws-collapsed)[:8]; big-endian uint64"}


def load_registry(path: str) -> dict:
    import numpy as np  # noqa: PLC0415
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
