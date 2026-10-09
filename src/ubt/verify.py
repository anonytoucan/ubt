"""Verification of generated braille (forward translation only).

Checks: every output char is Unicode braille U+2800-U+28FF ("\\n" only between segments of mixed docs), and a second
forward translation reproduces the first (nondeterminism is a bug).
"""

from __future__ import annotations

from dataclasses import dataclass

from ubt.translate import ForwardTranslator

BRAILLE_LO = 0x2800
BRAILLE_HI = 0x28FF


def is_braille_only(braille: str, allow_newline: bool = True) -> bool:
    for ch in braille:
        if ch == "\n" and allow_newline:
            continue
        if not (BRAILLE_LO <= ord(ch) <= BRAILLE_HI):
            return False
    return True


def cell_count(braille: str) -> int:
    """Number of braille cells (newline separators excluded)."""
    return sum(1 for ch in braille if BRAILLE_LO <= ord(ch) <= BRAILLE_HI)


@dataclass(frozen=True)
class VerifiedTranslation:
    ok: bool
    braille: str
    reason: str  # "" | exception | undefined_char | charset | inconsistent | empty_input


def translate_verified(tr: ForwardTranslator, table: str, text: str) -> VerifiedTranslation:
    """Forward-translate `text` whole and run all verifier checks."""
    first = tr.translate(table, text, check_undefined=True)
    if not first.ok:
        return VerifiedTranslation(False, first.braille, first.reason)
    if not is_braille_only(first.braille, allow_newline=False):
        return VerifiedTranslation(False, first.braille, "charset")
    second = tr.translate(table, text, check_undefined=False)
    if not second.ok or second.braille != first.braille:
        return VerifiedTranslation(False, first.braille, "inconsistent")
    return VerifiedTranslation(True, first.braille, "")
