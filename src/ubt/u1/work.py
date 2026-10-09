"""Pool-side jobs for data-u1; each worker owns one translator (ubt.worker._TR, set by ubt.u1.engine.init_worker)."""
from __future__ import annotations

from ubt.verify import is_braille_only, translate_verified


def _tr():
    import ubt.worker as w  # noqa: PLC0415
    assert w._TR is not None, "worker not initialised"
    return w._TR


def process(plan: dict) -> dict:
    """Dispatch on plan['kind'] (default 'louis')."""
    kind = plan.get("kind", "louis")
    if kind == "louis":
        from ubt.worker import process_plan  # noqa: PLC0415
        return process_plan(plan)
    if kind == "s1b":
        return _process_s1b(plan)
    if kind == "carrier":
        return _process_carrier(plan)
    if kind == "nikl":
        tr = _tr()
        r = tr.translate(plan["table"], plan["text"], check_undefined=False)
        # confusable = siblings whose forward encoding equals the human braille
        conf = [sib for sib in plan.get("siblings", [])
                if (x := tr.translate(sib, plan["text"], check_undefined=False)).ok
                and x.braille == plan["human"]]
        return {"plan": plan, "ok": True, "f": r.braille if r.ok else None,
                "confusable_set": conf}
    raise ValueError(f"unknown plan kind {kind}")


def _siblings_confusable(table_sibs: list[str], text: str, braille: str) -> list[str]:
    tr = _tr()
    out = []
    for sib in table_sibs:
        r = tr.translate(sib, text, check_undefined=False)
        if r.ok and r.braille == braille:
            out.append(sib)
    return out


def _process_s1b(plan: dict) -> dict:
    """One corpus line x every open table in its pool -> the tables that transcribe it cleanly, with confusable sets.
    The builder gives the line to the neediest one, so it serves one table and none is wasted on a full table."""
    tr = _tr()
    text = plan["text"]
    cache: dict[str, str | None] = {}

    def plain(t: str) -> str | None:
        if t not in cache:
            r = tr.translate(t, text, check_undefined=False)
            cache[t] = r.braille if r.ok else None
        return cache[t]

    options, reasons = {}, {}
    for t in plan["tables"]:
        v = translate_verified(tr, t, text)
        if not v.ok:
            reasons[t] = v.reason
            continue
        cache[t] = v.braille
        conf = [sib for sib in plan["siblings"].get(t, []) if plain(sib) == v.braille]
        options[t] = {"table": t, "lang": plan["lang_of"][t], "text": text,
                      "braille": v.braille, "confusable_set": conf}
    return {"plan": plan, "ok": bool(options), "options": options, "reasons": reasons}


def _process_carrier(plan: dict) -> dict:
    """Translate the carrier spans of an embedded-math row (strict)."""
    tr = _tr()
    out = []
    for seg in plan["segments"]:
        if seg.get("braille") is not None:
            out.append(seg)
            continue
        v = translate_verified(tr, seg["table"], seg["text"])
        if not v.ok:
            return {"plan": plan, "ok": False, "reason": v.reason, "fail_table": seg["table"]}
        conf = _siblings_confusable(seg.get("siblings", []), seg["text"], v.braille)
        out.append({**seg, "braille": v.braille, "confusable_set": conf})
    return {"plan": plan, "ok": True, "segments": out}


# --------------------------------------------------------------------------- verifier
def retranscribe(item: tuple[str, str]) -> tuple[str | None, bool, bool]:
    """(table, text) -> (f(text) | None, undefined_free, braille_only)."""
    table, text = item
    r = _tr().translate(table, text, check_undefined=True)
    if r.reason in ("exception", "empty_input"):
        return None, False, False
    # r.braille is the normal-mode output even when the strict (noUndefined) pass disagreed; then r.ok is False
    return r.braille, r.ok, is_braille_only(r.braille, allow_newline=False)
