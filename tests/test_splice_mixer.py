"""Splicer, mixer (MixerV2), record stamps, noise, dedup."""

import random

import pytest

from ubt.formats import dedup_key, len_bucket, make_record
from ubt.mixer import MixerV2
from ubt.noise import flip_dots
from ubt.splice import load_wordlist, natural_latin_runs, splice
from ubt.tables import assign_weights, derive_confusable_groups, scan_tables

SPLICE_W = {"en_term": 0.6, "en_phrase": 0.2, "digits_dates": 0.2}


@pytest.fixture(scope="module")
def wordlist():
    return load_wordlist()


def test_wordlist_bundled(wordlist):
    terms, phrases = wordlist
    assert len(terms) + len(phrases) >= 1500
    assert phrases and all(" " in p for p in phrases)


def test_splice_light_adds_one_span(wordlist):
    terms, phrases = wordlist
    rng = random.Random(1)
    text = "그러나 아이들은 책을 읽는다."
    out, density = splice(text, "light", rng, SPLICE_W, terms, phrases,
                          natural_first=False)
    assert density == "light"
    assert out != text and natural_latin_runs(out) >= 1 or any(c.isdigit() for c in out)


def test_splice_natural_first_no_modification(wordlist):
    terms, phrases = wordlist
    rng = random.Random(2)
    text = "이 문서는 Wikipedia 프로젝트의 일부다."
    out, density = splice(text, "light", rng, SPLICE_W, terms, phrases,
                          natural_first=True)
    assert out == text  # naturally occurring Latin preserved, no splicing
    assert density == "light"


def test_splice_heavy_multiple_spans(wordlist):
    terms, phrases = wordlist
    rng = random.Random(3)
    text = "하늘은 맑고 바람은 시원하다."
    out, density = splice(text, "heavy", rng, SPLICE_W, terms, phrases,
                          natural_first=False)
    assert density == "heavy"
    assert len(out) > len(text)


@pytest.fixture(scope="module")
def mixer(cfg):
    tabs = scan_tables([cfg["table_dirs"]["ko2020"]])
    langs = sorted({t.base_lang for t in tabs})
    tabs = assign_weights(tabs, set(langs), cfg["head_langs"], cfg["head_share"],
                          cfg["holdout_languages"], cfg["inter_mixed"]["anchor_langs"])
    groups = derive_confusable_groups(tabs, cfg["script_groups"])
    weighted = [t for t in tabs if t.weight > 0]
    return MixerV2(weighted, groups, cfg["inter_mixed"]["anchor_langs"],
                   cfg["inter_mixed"]["regimes"])


@pytest.mark.parametrize("regime,k", [("anchor", 2), ("anchor", 4),
                                      ("uniform", 3), ("confusable", 2),
                                      ("confusable", 3)])
def test_mixer_regimes_distinct_tables(mixer, regime, k):
    rng = random.Random(42)
    for _ in range(20):
        picked = mixer.choose_tables(regime, k, rng)
        assert len(picked) == k
        assert len({t.table_id for t in picked}) == k


def test_mixer_confusable_pair_shares_group(mixer):
    rng = random.Random(7)
    for _ in range(20):
        picked = mixer.choose_tables("confusable", 2, rng)
        shared = set(picked[0].groups) & set(picked[1].groups)
        assert shared, f"{picked[0].table_id} / {picked[1].table_id}"


def test_len_bucket_stamps():
    assert len_bucket(7, 1) == "snippet_5_10"
    assert len_bucket(20, 1) == "snippet_11_20"
    assert len_bucket(40, 1) == "snippet_21_40"
    assert len_bucket(80, 1) == "snippet_41_80"
    assert len_bucket(120, 1) == "sentence"
    assert len_bucket(300, 5) == "paragraph"


def test_make_record_stamps_and_join():
    segs = [
        {"table": "a.ctb", "lang": "aa", "text": "t1", "braille": "⠁⠂",
         "confusable_set": ["x.ctb"]},
        {"table": "b.ctb", "lang": "bb", "text": "t2", "braille": "⠃",
         "confusable_set": []},
    ]
    rec = make_record("r1", "inter", "anchor", "none", segs, "flores_dev", 2)
    assert rec["k"] == 2
    assert rec["braille"] == "⠁⠂\n⠃"
    assert rec["cells"] == 3
    assert rec["fiber_ambiguous"] is False
    assert rec["confusable_set"] == ["x.ctb"]
    for f in ("k", "len_bucket", "regime", "switch_density", "confusable_set",
              "fiber_ambiguous"):
        assert f in rec


def test_dedup_key_per_segment():
    segs1 = [{"table": "a", "text": "x"}, {"table": "b", "text": "y"}]
    segs2 = [{"table": "a", "text": "x"}, {"table": "b", "text": "z"}]
    assert dedup_key(segs1) != dedup_key(segs2)
    assert dedup_key(segs1) == dedup_key([dict(s) for s in segs1])


def test_noise_flip_dots_stays_in_braille_block():
    rng = random.Random(9)
    braille = "⠁⠃⠉⠙⠑\n⠋⠛⠓"
    noisy = flip_dots(braille, 0.5, rng)
    assert len(noisy) == len(braille)
    assert noisy[5] == "\n"
    for ch in noisy.replace("\n", ""):
        assert 0x2800 <= ord(ch) <= 0x28FF
    assert flip_dots(braille, 0.0, rng) == braille


def test_splice_latin_script_l1_always_splices(wordlist):
    """A Latin-script L1 such as Uzbek does not count as natural code-switching, so a splice is always made."""
    terms, phrases = wordlist
    rng = random.Random(4)
    text = "Kuchlar muvozanati barcha davlatlarining tizim edi."
    out, density = splice(text, "light", rng, SPLICE_W, terms, phrases,
                          natural_first=True)
    assert out != text
    assert density == "light"
