"""Language/script categories shared by the training probe, the grid aggregator and the paper tables."""

from __future__ import annotations

from functools import lru_cache

_CAT_OF_LANG = {}
for langs, cat in [
    (("zh", "cmn", "ja", "yue"), "CJK"),
    (("ko",), "Hangul"),
    (("ar", "fa", "ur", "ckb", "ks", "sd", "ps"), "Arabic"),
    (("hi", "mr", "ne", "bn", "as", "gu", "pa", "ta", "te", "kn", "ml",
      "or", "si", "awa", "bho", "mag", "mai", "kok", "sa", "dra"), "Indic"),
    (("ru", "uk", "bg", "be", "sr", "mk", "kk", "ky", "mn", "ba", "cv",
      "tt", "ug"), "Cyrillic"),
    (("el",), "Greek"),
    (("he", "iw", "yi"), "Hebrew"),
    (("th", "lo", "km", "my", "bo", "dz"), "SEA-Tib"),
    (("am", "ti"), "Ethiopic"),
    (("hy", "ka"), "Caucasus"),
]:
    for l in langs:
        _CAT_OF_LANG[l] = cat


@lru_cache(maxsize=1)
def _table_maps() -> tuple[dict, dict]:
    from ubt.tables import scan_tables
    lang_of, cat_of = {}, {}
    for t in scan_tables():
        lang = (t.base_lang or t.language or "").lower()
        lang_of[t.table_id] = lang or t.table_id
        cat_of[t.table_id] = _CAT_OF_LANG.get(lang, "Latin")
    return lang_of, cat_of


def _prefix_lang(table_id: str) -> str:
    """Fallback for tables outside the scanned inventory (e.g. the extra ko-2020 tables)."""
    head = table_id.split("-")[0].split("_")[0].split(".")[0].lower()
    return {"ko": "ko", "zhcn": "zh", "zh": "zh", "ja": "ja"}.get(head, head)


def category_of_table(table_id: str) -> str:
    cat = _table_maps()[1].get(table_id)
    if cat is not None:
        return cat
    return _CAT_OF_LANG.get(_prefix_lang(table_id), "Latin")


def lang_of_table(table_id: str) -> str:
    lang = _table_maps()[0].get(table_id)
    if lang and lang != table_id:
        return lang
    return _prefix_lang(table_id)


def doc_category(tables: list[str]) -> str:
    """Document category: the first non-Latin category among its tables, so
    mixed Latin+CJK documents count as CJK."""
    cats = [category_of_table(t) for t in tables]
    for c in cats:
        if c != "Latin":
            return c
    return "Latin"
