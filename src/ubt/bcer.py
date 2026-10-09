"""bCER: braille-referenced character error rate.

bCER(B, T_hat; T, tables) = (# cells covered by erroneous text spans) / |B|

The cell is the one unit shared by every script, unlike the character, whose weight differs per language. Character
edit operations are projected onto cells through liblouis inPos alignment per segment, with a whitespace-anchored
proportional fallback when positions are unreliable. Only reference text and braille, hypothesis text and tables are
needed, so stored runs can be re-scored on CPU.
"""

from __future__ import annotations

import os

from ubt.translate import DISPLAY_TABLE, build_table_path

_louis = None


def _get_louis():
    global _louis
    if _louis is None:
        # build_table_path keeps an inherited LOUIS_TABLEPATH first, then appends the system dir and
        # $UBT_TABLE_FALLBACK_DIRS, where data-u1 labels resolve; assigned so that a stale value is replaced
        os.environ["LOUIS_TABLEPATH"] = build_table_path(None)
        from ubt.translate import preload_louis_lib  # noqa: PLC0415
        preload_louis_lib()
        import louis  # noqa: PLC0415
        _louis = louis
    return _louis


def _align_cells(table: str, text: str, ref_braille: str) -> list[list[int]] | None:
    """For each input char, the cell indices it produced; None means use the proportional fallback."""
    louis = _get_louis()
    try:
        braille, in_pos, _out_pos, _ = louis.translate([DISPLAY_TABLE, table], text)
    except Exception:
        return None
    if braille != ref_braille or len(in_pos) != len(braille):
        return None
    cells: list[list[int]] = [[] for _ in text]
    for j, i in enumerate(in_pos):
        if 0 <= i < len(text):
            cells[i].append(j)
    return cells


def _proportional_cells(text: str, braille: str) -> list[list[int]]:
    """Fallback: proportional map word by word when the word counts match, else over the whole string."""
    twords = text.split(" ")
    bwords = braille.replace(" ", "⠀").split("⠀")
    cells: list[list[int]] = [[] for _ in text]

    def spread(t0: int, tlen: int, b0: int, blen: int) -> None:
        if tlen == 0 or blen == 0:
            return
        for k in range(blen):
            i = t0 + min(tlen - 1, (k * tlen) // blen)
            cells[i].append(b0 + k)

    if len(twords) == len(bwords):
        ti = bi = 0
        for tw, bw in zip(twords, bwords):
            spread(ti, len(tw), bi, len(bw))
            ti += len(tw) + 1
            bi += len(bw) + 1
    else:
        spread(0, len(text), 0, len(braille))
    return cells


def doc_error_cells(segments: list[tuple[str, str, str]], hyp_text: str,
                    ) -> tuple[int, int, bool]:
    """Return (n_error_cells, n_cells, used_fallback); segments are (table, ref_text, ref_braille), joined by '\n'."""
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415

    ref_text = "\n".join(t for _, t, _ in segments)
    # global char index -> cell ids in the concatenated cell space
    cells: list[list[int]] = []
    cell_base = 0
    used_fallback = False
    for table, text, braille in segments:
        seg_cells = _align_cells(table, text, braille)
        if seg_cells is None:
            seg_cells = _proportional_cells(text, braille)
            used_fallback = True
        cells.extend([[cell_base + j for j in c] for c in seg_cells])
        cells.append([])  # the '\n' joiner: no cells of its own
        cell_base += len(braille)
    cells = cells[:-1] if segments else cells  # drop trailing joiner slot
    n_cells = cell_base

    err: set[int] = set()

    def touch(i: int) -> None:
        """Mark the cells of ref char i; a char without cells marks the nearest preceding cell, so every edit counts."""
        k = min(i, len(cells) - 1)
        if cells[k]:
            err.update(cells[k])
            return
        for back in range(k - 1, -1, -1):
            if cells[back]:
                err.add(cells[back][-1])
                return
        if n_cells:
            err.add(0)

    for op in Levenshtein.editops(ref_text, hyp_text):
        touch(op.src_pos)
    return len(err), n_cells, used_fallback


def doc_error_cells_by_segment(segments: list[tuple[str, str, str]],
                               hyp_text: str,
                               ) -> tuple[list[tuple[int, int]], bool]:
    """Like doc_error_cells, with errors attributed to the segment owning each cell, so per-table bCER is exact.
    Returns ([(err_i, cells_i) per segment], used_fallback)."""
    err, n_cells, fb = doc_error_cells(segments, hyp_text)
    # recompute with per-segment bucketing
    from rapidfuzz.distance import Levenshtein  # noqa: PLC0415
    ref_text = "\n".join(t for _, t, _ in segments)
    bounds = []
    base = 0
    for _, _, br in segments:
        bounds.append((base, base + len(br)))
        base += len(br)

    cells: list[list[int]] = []
    cell_base = 0
    for table, text, braille in segments:
        seg_cells = _align_cells(table, text, braille)
        if seg_cells is None:
            seg_cells = _proportional_cells(text, braille)
        cells.extend([[cell_base + j for j in c] for c in seg_cells])
        cells.append([])
        cell_base += len(braille)
    cells = cells[:-1] if segments else cells

    errset: set[int] = set()

    def touch(i: int) -> None:
        k = min(i, len(cells) - 1)
        if cells[k]:
            errset.update(cells[k])
            return
        for back in range(k - 1, -1, -1):
            if cells[back]:
                errset.add(cells[back][-1])
                return
        if cell_base:
            errset.add(0)

    for op in Levenshtein.editops(ref_text, hyp_text):
        touch(op.src_pos)
    out = []
    for lo, hi in bounds:
        out.append((sum(1 for c in errset if lo <= c < hi), hi - lo))
    return out, fb


def record_segments(rec: dict) -> list[tuple[str, str, str]]:
    """Extract (table, text, braille) segments from a dataset or eval record.

    Inline rows (seg_join "") are one braille line, so they form one segment under the carrier table."""
    if rec.get("segments") and rec.get("seg_join", "\n") == "\n":
        return [(s["table"], s["text"], s["braille"]) for s in rec["segments"]]
    if rec.get("segments"):
        return [(rec["tables"][0], rec["text"], rec["braille"])]
    table = rec["tables"][0] if "tables" in rec else rec["table"]  # external format
    # external corpora (NIKL) separate braille words with ASCII space; the forward translator emits U+2800
    return [(table, rec["text"], rec["braille"].replace(" ", "⠀"))]
