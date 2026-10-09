"""Translator tests, including the regression that LOUIS_TABLEPATH keeps the system tables dir."""

import os

from ubt.translate import build_table_path, discover_system_tables_dir
from ubt.verify import BRAILLE_HI, BRAILLE_LO, cell_count, is_braille_only, translate_verified


def test_system_tables_discoverable():
    d = discover_system_tables_dir()
    assert os.path.isfile(os.path.join(d, "unicode.dis"))


def test_bug1_table_path_always_includes_system_dir(fork_tables):
    """Regression: extra dirs must not shadow the system tables dir."""
    path = build_table_path([fork_tables])
    parts = path.split(",")
    assert parts[0] == os.path.expanduser(fork_tables)
    assert discover_system_tables_dir() in parts


def test_bug1_regression_ko_and_ueb_both_translate(translator):
    """With a fork tables dir, both ko-2020-g2.ctb (fork) and en-ueb-g2.ctb (system) translate."""
    ko = translator.translate("ko-2020-g2.ctb", "그러나 아이들은 책을 읽는다.")
    en = translator.translate("en-ueb-g2.ctb", "However, the children read books.")
    assert ko.ok and en.ok
    assert is_braille_only(ko.braille) and is_braille_only(en.braille)


def test_translation_is_unicode_braille(translator):
    r = translator.translate("fr-bfu-g2.ctb", "Les enfants lisent des livres.")
    assert r.ok
    assert all(BRAILLE_LO <= ord(c) <= BRAILLE_HI for c in r.braille)


def test_undefined_char_detected(translator):
    # en-ueb-g2 does not define Hangul
    r = translator.translate("en-ueb-g2.ctb", "hello 안녕 world")
    assert not r.ok
    assert r.reason == "undefined_char"


def test_verified_translation_deterministic(translator):
    v1 = translate_verified(translator, "en-ueb-g2.ctb", "Determinism check.")
    v2 = translate_verified(translator, "en-ueb-g2.ctb", "Determinism check.")
    assert v1.ok and v2.ok and v1.braille == v2.braille


def test_cell_count_excludes_newline():
    assert cell_count("⠁⠂\n⠃") == 3
