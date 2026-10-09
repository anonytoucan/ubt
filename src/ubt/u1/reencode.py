"""Re-encode a predicted (label, text) segment of a data-u1 record the way the builder made the gold braille.

  'nemeth', '$$latex$$'        f_nemeth(nu_math(latex))
  'ko-math-2024', '$$latex$$'  ubt.kmath.latex_to_braille, if installed
  carrier label with '$$…$$'   liblouis for the prose and ⠸⠩ f_nemeth ⠸⠱ for each formula; for ko-2024-g2,
                               kmath.sentence_to_braille
  anything else                liblouis on the u1 source normal form (NFC, Bengali nukta letters recomposed)
Returns the braille, or None if infeasible. Nemeth results are cached per canonical form.
"""
from __future__ import annotations

import os
import re
import unicodedata
from functools import lru_cache

from ubt.math_norm import nu_math
from ubt.u1.engine import DEFAULT_JAR, KMATH_LABEL, KO2024_LABEL, NEMETH_LABEL
from ubt.u1.nemeth import NEMETH_CLOSE, NEMETH_OPEN, batch_nemeth
from ubt.u1.sources import _bengali_fix

_MATH = re.compile(r"(\$\$.*?\$\$)", re.S)


def is_u1_record(rec: dict | None) -> bool:
    return bool(rec) and str(rec.get("engine", "")).startswith("u1e-")


@lru_cache(maxsize=65536)
def _nemeth(canon: str) -> str | None:
    jar = os.environ.get("UBT_LATEX2NEMETH_JAR") or DEFAULT_JAR
    return batch_nemeth([canon], jar)[0]


def _louis(louis, table: str, text: str) -> str | None:
    try:
        r = louis.translate(table, _bengali_fix(unicodedata.normalize("NFC", text)),
                            check_undefined=False)
    except RuntimeError:
        return None
    return r.braille if r.ok else None


def reencode_segment(louis, table: str, text: str) -> str | None:
    if table == NEMETH_LABEL:
        m = re.fullmatch(r"\$\$(.*)\$\$", text, flags=re.S)
        return _nemeth(nu_math(m.group(1))) if m else None
    if table == KMATH_LABEL or (table == KO2024_LABEL and "$$" in text):
        try:
            import ubt.kmath as km  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return None
        try:
            if table == KMATH_LABEL:
                m = re.fullmatch(r"\$\$(.*)\$\$", text, flags=re.S)
                return km.latex_to_braille(m.group(1), strict=True) if m else None
            return km.sentence_to_braille(text.replace("$$", "$"), strict=True)
        except Exception:  # noqa: BLE001
            return None
    if "$$" in text:
        out = []
        for piece in _MATH.split(text):
            if not piece:
                continue
            if piece.startswith("$$") and piece.endswith("$$") and len(piece) >= 4:
                cell = _nemeth(nu_math(piece[2:-2]))
                if cell is None:
                    return None
                out.append(NEMETH_OPEN + cell + NEMETH_CLOSE)
            else:
                b = _louis(louis, table, piece)
                if b is None:
                    return None
                out.append(b)
        return "".join(out)
    return _louis(louis, table, text)
