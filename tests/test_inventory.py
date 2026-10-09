"""Table inventory, weights, and auto-derived confusable groups."""

import pytest

from ubt.tables import (
    EXCLUDED_FILES,
    assign_weights,
    confusable_siblings,
    derive_confusable_groups,
    scan_tables,
)


@pytest.fixture(scope="module")
def inventory(cfg):
    tabs = scan_tables([cfg["table_dirs"]["ko2020"]])
    langs = sorted({t.base_lang for t in tabs})
    # treat every language as sourced, so the test needs no source data
    tabs = assign_weights(
        tabs, set(langs), cfg["head_langs"], cfg["head_share"],
        cfg["holdout_languages"], cfg["inter_mixed"]["anchor_langs"],
    )
    groups = derive_confusable_groups(tabs, cfg["script_groups"])
    return tabs, groups


def test_backward_table_never_scanned(inventory):
    tabs, _ = inventory
    ids = {t.table_id for t in tabs}
    assert not (ids & EXCLUDED_FILES)
    assert all("forward" in t.direction or t.direction == "both" for t in tabs)


def test_ko2020_present_and_weighted(inventory):
    tabs, _ = inventory
    ko = {t.table_id: t for t in tabs if t.base_lang == "ko"}
    assert "ko-2020-g2.ctb" in ko
    assert ko["ko-2020-g2.ctb"].weight > 0


def test_holdout_languages_have_zero_weight(inventory, cfg):
    tabs, _ = inventory
    for t in tabs:
        if t.base_lang in cfg["holdout_languages"]:
            assert t.holdout and t.weight == 0.0


def test_confusable_groups_no_pan_latin(inventory):
    tabs, groups = inventory
    # the Hangul group holds both ko-2020 and ko-2006
    hangul = groups.get("script:Hangul", [])
    assert "ko-2020-g2.ctb" in hangul and "ko-2006-g2.ctb" in hangul
    # No group mixes unrelated Latin-script languages.
    by_id = {t.table_id: t for t in tabs}
    for gid, members in groups.items():
        if gid.startswith("lang:"):
            assert len({by_id[m].base_lang for m in members}) == 1


def test_confusable_siblings_symmetric(inventory):
    tabs, groups = inventory
    by_id = {t.table_id: t for t in tabs}
    t = by_id["ko-2020-g2.ctb"]
    sibs = confusable_siblings(t, groups)
    assert "ko-2006-g2.ctb" in sibs
    assert "ko-2020-g2.ctb" not in sibs
    assert "ko-2020-g2.ctb" in confusable_siblings(by_id["ko-2006-g2.ctb"], groups)


def test_status_overrides_applied(inventory, cfg):
    import yaml

    from ubt.mixer import current_status_table
    from ubt.tables import apply_status_overrides

    tabs, _ = inventory
    with open("configs/currency_overrides.yaml") as fh:
        ov = yaml.safe_load(fh)
    tabs = apply_status_overrides(tabs, ov)
    by_id = {t.table_id: t for t in tabs}
    # explicit locale rules (first matching glob wins)
    assert by_id["ko-2020-g2.ctb"].status == "current"
    assert by_id["ko-2006-g2.ctb"].status == "superseded"
    assert by_id["en-ueb-g2.ctb"].status == "current"
    assert by_id["en-us-g1.ctb"].status == "superseded"
    assert by_id["zhcn-cbs.ctb"].status == "current"
    assert by_id["zhcn-g2.ctb"].status == "superseded"
    assert by_id["zh-hk.ctb"].status == "current"      # yue locale
    assert by_id["vi-saigon-g1.ctb"].status == "parallel"
    # YAML `no` locale key (parses as False) still reaches nb tables
    assert by_id["no-no-g2.ctb"].status == "current"
    # generic year rule: year-stamped Danish with unmarked siblings
    assert by_id["da-dk-g26_1993.ctb"].status == "superseded"
    # fall-through default
    assert by_id["fr-bfu-g2.ctb"].status == "current"
    by_lang = {}
    for t in tabs:
        if t.weight > 0:
            by_lang.setdefault(t.base_lang, []).append(t)
    assert current_status_table("ko", by_lang).table_id == "ko-2020-g2.ctb"
    assert current_status_table("en", by_lang).table_id == "en-ueb-g2.ctb"


def test_status_excluded_and_alias_zero_weight(inventory):
    from ubt.tables import apply_status_overrides

    tabs, _ = inventory
    tabs = apply_status_overrides(
        tabs,
        {"locales": {"en": [{"pattern": "en-ueb-g2*", "status": "excluded"}]}},
        alias_excluded=["sr-g1.ctb"],
    )
    by_id = {t.table_id: t for t in tabs}
    assert by_id["en-ueb-g2.ctb"].weight == 0.0
    assert by_id["sr-g1.ctb"].status == "duplicate-excluded"
    assert by_id["sr-g1.ctb"].weight == 0.0
    assert by_id["sr-Cyrl.ctb"].weight > 0.0


def test_wrapper_inherits_primary_target_status(inventory, cfg):
    """A wrapper table that no explicit rule matches inherits the status of the table it includes."""
    import yaml

    from ubt.tables import apply_status_overrides

    tabs, _ = inventory
    with open("configs/currency_overrides.yaml") as fh:
        ov = yaml.safe_load(fh)
    # Drop the explicit en_US*/en_GB* rules to exercise pure inheritance.
    ov["locales"]["en"] = [r for r in ov["locales"]["en"]
                           if not r["pattern"].startswith("en_")]
    tabs = apply_status_overrides(tabs, ov)
    by_id = {t.table_id: t for t in tabs}
    # en_US.tbl includes en-us-g2.ctb (en-us-g* -> superseded)
    assert by_id["en_US.tbl"].status == "superseded"
    # en_GB.tbl includes en-GB-g2.ctb (en-GB-g2* -> superseded)
    assert by_id["en_GB.tbl"].status == "superseded"
