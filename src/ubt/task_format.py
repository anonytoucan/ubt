"""Canonical SFT task format, built from the datagen record schema.

PROMPT   = [ "<|tables|>" + ",".join(gold_tables) + "\\n" ]   # hinted format only
           + "<|braille|>\\n" + "\\n".join(B_i) + "\\n<|text|>\\n"
COMPLETION = "\\n".join( "⟨" + table_i + "⟩" + text_i )        # (+ EOS at tokenization)

Angle brackets are U+27E8 / U+27E9; braille segments are joined with "\\n" as stored in `braille`. Loss is masked on
the prompt (labels -100) and overlong examples are dropped, never truncated.
"""

from __future__ import annotations

TABLE_OPEN, TABLE_CLOSE = "⟨", "⟩"  # ⟨ ⟩
TABLES_TAG, BRAILLE_TAG, TEXT_TAG = "<|tables|>", "<|braille|>", "<|text|>"


def record_segments(record: dict) -> list[dict]:
    """Per-segment view of a record; k=1 and inline records (seg_join "", e.g. prose with a Nemeth span) are one line
    of braille and give one segment labelled with the carrier table."""
    if record.get("segments") and record.get("seg_join", "\n") == "\n":
        return record["segments"]
    return [{"table": record["tables"][0], "text": record["text"],
             "braille": record["braille"]}]


def render(record: dict, hinted: bool) -> dict:
    """Render one datagen record into {prompt, completion} (both str)."""
    segs = record_segments(record)
    prompt = ""
    if hinted:
        prompt += TABLES_TAG + ",".join(s["table"] for s in segs) + "\n"
    prompt += BRAILLE_TAG + "\n" + record["braille"] + "\n" + TEXT_TAG + "\n"
    completion = "\n".join(
        TABLE_OPEN + s["table"] + TABLE_CLOSE + s["text"] for s in segs
    )
    return {"prompt": prompt, "completion": completion, "id": record["id"],
            "doc_type": record["doc_type"], "k": record["k"],
            "regime": record["regime"],
            "switch_density": record["switch_density"]}
