import hashlib
import json

import pytest

from ubt.ko2024_options import attach_options, translate, validate_options


class LouisStub:
    no_contract = 4096

    def translateString(self, tables, text, **kwargs):
        self.call = (tables, text, kwargs)
        return "⠁⠀⠃"


def test_only_requested_roman_run_receives_typeform():
    louis = LouisStub()
    source = "한글 HACCP와 ISO"
    assert translate(louis, ["ko.ctb"], source, {"roman_no_contract": [[3, 8]]}) == "⠁ ⠃"
    assert louis.call[2]["typeform"] == [0]*3 + [4096]*5 + [0]*5
    translate(louis, ["ko.ctb"], source)
    assert "typeform" not in louis.call[2]


@pytest.mark.parametrize("options", [
    {"roman_no_contract": [[1, 3]]}, {"roman_no_contract": [[0, 1]]},
    {"roman_no_contract": [[0, 3], [0, 3]]}, {"roman_no_contract": [[0, 9]]},
    {"roman_no_contract": [[False, 3]]}, {"target": "⠁"},
])
def test_invalid_spans_and_target_injection_rejected(options):
    with pytest.raises(ValueError):
        validate_options("ABC 한글", options)


def test_sidecar_binding_provenance_and_seal(tmp_path):
    row = {"id": "r", "src": "HACCP", "split": "dev", "tgt": "unused"}
    entry = {"id": "r", "source_sha256": hashlib.sha256(b"HACCP").hexdigest(),
             "options": {"roman_no_contract": [[0, 5]]},
             "provenance": "dev_annotation", "evidence": "dev-only diagnostic"}
    path = tmp_path / "options.jsonl"
    path.write_text(json.dumps(entry) + "\n")
    out, report = attach_options([row], str(path))
    assert out[0]["translation_options"] == entry["options"]
    assert report["target_informed"]
    assert "translation_options" not in row
    with pytest.raises(ValueError, match="sealed test"):
        attach_options([{**row, "split": "test"}], str(path))
    with pytest.raises(ValueError, match="source hash"):
        attach_options([{**row, "src": "HACCP!"}], str(path))
    path.write_text((json.dumps(entry) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        attach_options([row], str(path))


def test_regression_worker_preserves_annotated_source_offsets(monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "liblouis"))
    import ko_table_nikl_check as verifier
    louis = LouisStub()
    monkeypatch.setattr(verifier, "_LOUIS", louis)
    monkeypatch.setattr(verifier, "_TABLES", ["ko.ctb"])
    source = "  HACCP  "
    result = verifier.work({"id": "r", "src": source, "tgt": "⠁ ⠃",
                            "translation_options": {"roman_no_contract": [[2, 7]]}})
    assert result["exact_raw"]
    assert louis.call[1] == source
    assert louis.call[2]["typeform"] == [0, 0] + [4096]*5 + [0, 0]
