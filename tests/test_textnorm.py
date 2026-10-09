"""Scoring normal form (ubt.metrics.textnorm): NFC, U+200B removed, Chinese S/T fold."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ubt.metrics import textnorm as TN  # noqa: E402

YYA_PRE, YYA_DEC = "য়", "য\u09bc"          # য\u09bc precomposed (u1 references) vs the model's NFC spelling


def _cer(ref_segs, hyp_segs, st_fold=True):
    from rapidfuzz.distance import Levenshtein
    r, h = TN.norm_join(ref_segs, st_fold), TN.norm_join(hyp_segs, st_fold)
    return Levenshtein.distance(r, h) / max(1, len(r))


def test_nfc_equivalent_nukta_is_no_error():
    ref = [("bn.tbl", f"মখো{YYA_PRE}ব\u09c1")]
    hyp = [("bn.tbl", f"মখো{YYA_DEC}ব\u09c1")]
    assert _cer(ref, hyp) == 0.0


def test_zwsp_ignored_everywhere():
    assert _cer([("km-g1.utb", "ន\u17b7ង\u200bPryce")], [("km-g1.utb", "ន\u17b7ងPryce")]) == 0.0
    assert _cer([("en-ueb-g1.ctb", "a\u200bb")], [("en-ueb-g1.ctb", "ab")]) == 0.0


def test_st_fold_only_on_chinese_lines_and_only_in_headline():
    ref, hyp = [("zh_CHN.tbl", "學校")], [("zh_CHN.tbl", "学校")]
    assert _cer(ref, hyp) == 0.0
    assert _cer(ref, hyp, st_fold=False) == 1.0 / 2
    # a Japanese line is never folded (國 kyujitai vs 国 is a real difference there)
    assert _cer([("ja-kantenji-ucs2.utb", "國")], [("ja-kantenji-ucs2.utb", "国")]) == 1.0
    # each side is folded by its own label
    assert TN.norm_join([("zh-tw.ctb", "國"), ("ja-kantenji-ucs2.utb", "國")]) == "国\n國"


def test_fold_is_length_preserving():
    m = TN.ts_map()
    assert len(m) > 3000 and all(len(k) == 1 and len(v) == 1 for k, v in m.items())
    s = "臺灣的國語教學與發展" * 3
    assert len(TN.fold_st(s)) == len(s)


def test_index_map_points_into_original():
    text = f"a{YYA_PRE}\u200bb"
    s, m = TN.norm_line_with_map(text, "bn.tbl")
    assert s == f"a{YYA_DEC}b"
    assert len(m) == len(s) and m[0] == 0 and m[1] == 1 and m[2] == 1 and m[-1] == 3


def test_bcer_normalised_error_cells(monkeypatch):
    # stub ubt.bcer's alignment to one cell per reference character (no liblouis)
    import ubt.bcer as BC
    monkeypatch.setattr(BC, "_align_cells", lambda table, text, braille: [[i] for i in range(len(text))])
    ref = [("zh_CHN.tbl", "學校", "⠁⠃"), ("km-g1.utb", "ក\u200bខ", "⠉⠙⠑")]
    hyp = [("zh_CHN.tbl", "学校"), ("km-g1.utb", "កខ")]
    err, err_st, n, fb = TN.doc_error_cells_both(ref, hyp)
    assert (err, n, fb) == (0, 5, False)
    assert err_st == 1                      # the S/T-sensitive variant counts 學->学 (one cell)
    err2, _, _ = TN.doc_error_cells(ref, [("zh_CHN.tbl", "学x"), ("km-g1.utb", "កខ")])
    assert err2 == 1                        # 校->x marks the cell of 校 only


def test_score_hyp_uses_normal_form():
    from ubt.eval_harness import score_hyp
    rec = {"id": "t", "confusable_set": []}
    row = score_hyp(rec, {"completion": f"⟨zh-tw.ctb⟩國語\n⟨bn.tbl⟩মখো{YYA_PRE}ব\u09c1"},
                    f"⟨zh-tw.ctb⟩国语\n⟨bn.tbl⟩মখো{YYA_DEC}ব\u09c1", None)
    assert row["edit"] == 0 and row["hyp_eq_ref"]
    assert row["edit_st"] == 2              # S/T-sensitive: 國->国, 語->语
    assert row["edit_raw"] == 4             # + the nukta spelling (1 substitution + 1 insertion)


def test_reward_uses_normal_form():
    from ubt.rlvr.reward import RewardConfig, score_batch

    class NoPool:
        def translate_many(self, items):
            return [None] * len(items)
    rows = score_batch(["⟨zh_CHN.tbl⟩学校"], ["⟨zh_CHN.tbl⟩學校"], ["⠁⠃"], [[]], NoPool(), RewardConfig())
    assert rows[0]["cer"] == 0.0 and rows[0]["exact"]
