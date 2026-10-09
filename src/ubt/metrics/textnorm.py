"""Scoring normal form of text for CER / bCER / exact match.

Reference and output are compared after three steps, since the braille carries none of these distinctions:

  1. NFC on both sides. data-u1 references keep precomposed nukta letters, but Qwen's tokenizer NFC-normalises the
     training targets, so the model can only emit the decomposed spelling.
  2. U+200B (zero width space) removed: an invisible Khmer word separator that no table writes as a cell.
  3. Headline only (st_fold=True): on 'zh*' lines, Traditional characters are mapped to Simplified one by one
     (OpenCC TSCharacters, first candidate; data/NOTICE), since the Chinese codes write both with the same cells.
     The S/T-sensitive variant (steps 1-2) is reported next to it.

A line is folded by its own label: gold labels for the reference, predicted labels for the output.
"""

from __future__ import annotations

import os
import unicodedata
from functools import lru_cache

ZWSP = "\u200b"
_TS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "opencc_TSCharacters.txt")


def is_chinese_label(label: str) -> bool:
    """True for the Chinese tables; every one in the inventory starts with 'zh'."""
    return label.startswith("zh")


@lru_cache(maxsize=1)
def ts_map() -> dict[str, str]:
    """Traditional -> Simplified, one code point to one (length-preserving)."""
    m = {}
    with open(_TS_PATH, encoding="utf-8") as fh:
        for line in fh:
            src, _, dst = line.rstrip("\n").partition("\t")
            cand = dst.split(" ")[0] if dst else ""
            if len(src) == 1 and len(cand) == 1:
                m[src] = cand
    return m


def fold_st(text: str) -> str:
    m = ts_map()
    return "".join(m.get(c, c) for c in text)


def norm_line(text: str, label: str, st_fold: bool = True) -> str:
    s = unicodedata.normalize("NFC", text).replace(ZWSP, "")
    return fold_st(s) if st_fold and is_chinese_label(label) else s


def norm_join(segs: list[tuple[str, str]], st_fold: bool = True) -> str:
    """[(label, text)] -> the scoring string: normalised lines joined with '\\n'."""
    return "\n".join(norm_line(t, lab, st_fold) for lab, t in segs)


def _index_map(src: str, dst: str) -> list[int]:
    """For every index of dst, an index into src (dst is a normalisation of src; aligned by edit ops)."""
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415
    if src == dst:
        return list(range(len(dst)))
    last = max(0, len(src) - 1)
    out = [0] * len(dst)
    for op in Levenshtein.opcodes(src, dst):
        for k in range(op.dest_start, op.dest_end):
            if op.tag in ("equal", "replace") and op.src_end > op.src_start:
                out[k] = min(op.src_start + (k - op.dest_start), op.src_end - 1)
            else:                                   # inserted by the normalisation: nearest source char
                out[k] = min(op.src_start, last)
    return out


def norm_line_with_map(text: str, label: str, st_fold: bool = True) -> tuple[str, list[int]]:
    """norm_line plus, for each character of the result, the index of the character of `text` it came from."""
    s1 = unicodedata.normalize("NFC", text)
    m1 = _index_map(text, s1)
    keep = [i for i, c in enumerate(s1) if c != ZWSP]
    s2 = "".join(s1[i] for i in keep)
    m2 = [m1[i] for i in keep]
    return (fold_st(s2) if st_fold and is_chinese_label(label) else s2), m2      # the fold is length-preserving


def _reference_cells(segments: list[tuple[str, str, str]]):
    """Reference char -> cells by liblouis inPos (proportional fallback): (cells per char, '\\n' joiners empty;
    segment starts; total cells; used_fallback)."""
    from ubt.bcer import _align_cells, _proportional_cells  # noqa: PLC0415
    cells: list[list[int]] = []
    starts, base, used_fallback = [], 0, False
    for j, (table, text, braille) in enumerate(segments):
        seg_cells = _align_cells(table, text, braille)
        if seg_cells is None:
            seg_cells = _proportional_cells(text, braille)
            used_fallback = True
        if j:
            cells.append([])
        starts.append(len(cells))
        cells.extend([[base + c for c in cc] for cc in seg_cells])
        base += len(braille)
    return cells, starts, base, used_fallback


def _error_cells(segments, starts, cells, n_cells, hyp_segs, st_fold: bool) -> int:
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415
    ref_n, ref_map = [], []
    for j, (table, text, _b) in enumerate(segments):
        if j:
            ref_n.append("\n")
            ref_map.append(starts[j] - 1)
        s, m = norm_line_with_map(text, table, st_fold)
        ref_n.extend(s)
        ref_map.extend(starts[j] + i for i in m)
    ref_s, hyp_s = "".join(ref_n), norm_join(hyp_segs, st_fold)
    err: set[int] = set()

    def touch(k: int) -> None:
        k = min(max(k, 0), len(cells) - 1)
        if cells[k]:
            err.update(cells[k])
            return
        for back in range(k - 1, -1, -1):
            if cells[back]:
                err.add(cells[back][-1])
                return
        if n_cells:
            err.add(0)

    if cells:
        for op in Levenshtein.editops(ref_s, hyp_s):
            touch(ref_map[op.src_pos] if op.src_pos < len(ref_map) else len(cells) - 1)
    return len(err)


def doc_error_cells(segments: list[tuple[str, str, str]], hyp_segs: list[tuple[str, str]],
                    st_fold: bool = True) -> tuple[int, int, bool]:
    """bCER (error cells, cells, used_fallback) under the scoring normal form: as ubt.bcer.doc_error_cells, but edit
    ops are taken between the normalised texts and mapped back to the original reference characters."""
    cells, starts, n, fb = _reference_cells(segments)
    return _error_cells(segments, starts, cells, n, hyp_segs, st_fold), n, fb


def doc_error_cells_both(segments: list[tuple[str, str, str]], hyp_segs: list[tuple[str, str]]
                         ) -> tuple[int, int, int, bool]:
    """(error cells headline, error cells S/T-sensitive, cells, used_fallback) with one alignment."""
    cells, starts, n, fb = _reference_cells(segments)
    return (_error_cells(segments, starts, cells, n, hyp_segs, True),
            _error_cells(segments, starts, cells, n, hyp_segs, False), n, fb)
