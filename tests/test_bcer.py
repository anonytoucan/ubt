import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ubt.bcer import doc_error_cells, doc_error_cells_by_segment

SEG = [("en-ueb-g2.ctb", "Hello world", "⠠⠓⠑⠇⠇⠕⠀⠸⠺")]

def test_identity_zero():
    err, cells, fb = doc_error_cells(SEG, "Hello world")
    assert err == 0 and cells == 9 and not fb

def test_substitution_positive_and_monotonic():
    e1, _, _ = doc_error_cells(SEG, "Hallo world")
    e2, _, _ = doc_error_cells(SEG, "Hallo warld")
    assert 0 < e1 <= e2

def test_bounded():
    err, cells, _ = doc_error_cells(SEG, "zzz")
    assert err <= cells

def test_per_segment_attribution():
    segs = SEG + [("fr-bfu-g2.ctb", "bonjour", "⠃⠛")]
    per, _ = doc_error_cells_by_segment(segs, "Hello world\nbonjour")
    assert per[0] == (0, 9) and per[1][0] == 0

def test_external_space_normalization():
    from ubt.bcer import record_segments
    segs = record_segments({"table": "en-ueb-g2.ctb", "text": "a b",
                            "braille": "⠁ ⠃"})
    assert " " not in segs[0][2]
