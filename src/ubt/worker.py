"""Multiprocessing worker: forward translation plus verification.

liblouis is not thread-safe, so each worker process owns one ForwardTranslator, which sets LOUIS_TABLEPATH before
`import louis`.
"""

from __future__ import annotations

from ubt.translate import ForwardTranslator
from ubt.verify import translate_verified

_TR: ForwardTranslator | None = None


def init_worker(extra_table_dirs: list[str], quiet: bool = True) -> None:
    global _TR
    if quiet:  # liblouis warns to stderr on every call
        import os  # noqa: PLC0415
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
    _TR = ForwardTranslator(extra_table_dirs)


def process_plan(plan: dict) -> dict:
    """Translate and verify every segment of one document plan; stop at the first failure.

    confusable_set: the siblings whose forward translation of the same text gives the same braille."""
    assert _TR is not None, "worker not initialized"
    out_segments = []
    for seg in plan["segments"]:
        v = translate_verified(_TR, seg["table"], seg["text"])
        if not v.ok:
            return {"plan": plan, "ok": False, "reason": v.reason,
                    "fail_table": seg["table"]}
        confusable = []
        for sib in seg.get("siblings", []):
            r = _TR.translate(sib, seg["text"], check_undefined=False)
            if r.ok and r.braille == v.braille:
                confusable.append(sib)
        out_segments.append({
            "table": seg["table"], "lang": seg["lang"], "text": seg["text"],
            "braille": v.braille, "confusable_set": confusable,
        })
    return {"plan": plan, "ok": True, "segments": out_segments}
