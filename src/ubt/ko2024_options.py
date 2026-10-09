"""Explicit, source-bound input options for Korean liblouis experiments.

Options change translation inputs and never select or copy a reference target.
The only option, roman_no_contract, disables contractions in given Roman spans.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

CONTRACT = "ko2024-options-v1"


def validate_options(text: str, options: dict) -> list[tuple[int, int]]:
    if not isinstance(options, dict) or set(options) - {"roman_no_contract"}:
        raise ValueError("options must contain only roman_no_contract")
    spans = options.get("roman_no_contract", [])
    if not isinstance(spans, list):
        raise ValueError("roman_no_contract must be a list of [start, end] spans")
    valid = []
    previous_end = 0
    for span in spans:
        if (not isinstance(span, (list, tuple)) or len(span) != 2
                or any(type(x) is not int for x in span)):
            raise ValueError("span offsets must be two integers")
        start, end = span
        if not (previous_end <= start < end <= len(text)):
            raise ValueError("spans must be sorted, non-overlapping, and within the source")
        if not re.fullmatch(r"[A-Za-z]+", text[start:end]):
            raise ValueError("roman_no_contract spans must contain ASCII Roman letters only")
        if (start and text[start-1].isascii() and text[start-1].isalpha()
                or end < len(text) and text[end].isascii() and text[end].isalpha()):
            raise ValueError("roman_no_contract must cover a whole Roman run")
        valid.append((start, end))
        previous_end = end
    return valid


def translate(louis, tables: list[str], text: str, options: dict | None = None) -> str:
    spans = validate_options(text, options if options is not None else {})
    if not spans:
        return louis.translateString(tables, text, mode=0).replace("⠀", " ")
    forms = [0] * len(text)
    for start, end in spans:
        forms[start:end] = [louis.no_contract] * (end - start)
    return louis.translateString(tables, text, typeform=forms, mode=0).replace("⠀", " ")


def attach_options(rows: list[dict], path: str) -> tuple[list[dict], dict]:
    """Attach a JSONL options sidecar to rows, binding each entry to its exact source by sha256.

    Provenance is source_metadata (available at inference) or dev_annotation (dev rows only, target-informed).
    Rows without an entry use the default translation; unknown or duplicate ids fail.
    """
    raw = Path(path).read_bytes()
    entries = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("row ids must be unique")
    attached = {}
    provenance_counts = {"source_metadata": 0, "dev_annotation": 0}
    for entry in entries:
        allowed = {"id", "source_sha256", "options", "provenance", "evidence"}
        if not isinstance(entry, dict) or set(entry) != allowed:
            raise ValueError(f"each sidecar entry must have exactly {sorted(allowed)}")
        rid = entry["id"]
        if rid not in by_id or rid in attached:
            raise ValueError(f"unknown or duplicate options id: {rid}")
        row = by_id[rid]
        digest = hashlib.sha256(row["src"].encode()).hexdigest()
        if entry["source_sha256"] != digest:
            raise ValueError(f"source hash mismatch: {rid}")
        provenance = entry["provenance"]
        if provenance not in provenance_counts:
            raise ValueError(f"unsupported options provenance: {provenance}")
        if provenance == "dev_annotation" and row["split"] != "dev":
            raise ValueError("dev annotations cannot select options for sealed test rows")
        if not isinstance(entry["evidence"], str) or not entry["evidence"].strip():
            raise ValueError("option evidence must identify the metadata or annotation basis")
        validate_options(row["src"], entry["options"])
        attached[rid] = entry
        provenance_counts[provenance] += 1
    result = []
    for row in rows:
        entry = attached.get(row["id"])
        result.append({**row, "translation_options": entry["options"] if entry else {}})
    return result, {"contract": CONTRACT, "sidecar_sha256": hashlib.sha256(raw).hexdigest(),
                    "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "annotated_rows": len(attached), "default_rows": len(rows)-len(attached),
                    "provenance_counts": provenance_counts,
                    "target_informed": bool(provenance_counts["dev_annotation"])}
