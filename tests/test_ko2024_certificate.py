"""Certificate/API consistency checks of the pluggable Korean 2024 table (ubt.u1.engine.check_certificate)."""
import json

import pytest

from ubt.u1.engine import EngineSpec, check_certificate


@pytest.fixture
def certificate(tmp_path):
    spec = EngineSpec("sys", None, "3.38.0", str(tmp_path), "ko.ctb",
                      "stage", "cwd", "jar", ko2024_certified=True)
    staging = {"staged_files": {"ko.ctb": {"from": "ko.ctb", "sha256": "abc"}}}
    cert = {"table_file": "ko.ctb", "closure_sha256": {"ko.ctb": "abc"},
            "input_contract": "sentence", "test_informed_edits": False,
            "C1": {"pass": True, "metric": "exact_raw", "n": 290479,
                   "exact": 290479, "excluded": 0,
                   "split_counts": {"dev": 251319, "test": 39160}},
            "C2": {"pass": True}, "C3": {"pass": True}}
    def check(value):
        path = tmp_path / "certificate.json"
        path.write_text(json.dumps(value))
        return check_certificate(spec, staging, str(path))
    return cert, check


def test_consistent_exact_certificate(certificate):
    cert, check = certificate
    assert check(cert)["certified"]


@pytest.mark.parametrize("change", [
    {"exact": 290466}, {"excluded": 13}, {"n": 251319, "exact": 251319},
    {"metric": "exception_adjusted"}, {"split_counts": {"dev": 290479}},
    {"excluded": False},
])
def test_adjusted_or_partial_certificate_rejected(certificate, change):
    cert, check = certificate
    cert["C1"].update(change)
    with pytest.raises(RuntimeError, match="C1"):
        check(cert)


def test_bare_pass_flag_and_context_api_mismatch_rejected(certificate):
    cert, check = certificate
    cert["input_contract"] = "context_options_v1"
    with pytest.raises(RuntimeError, match="context/options"):
        check(cert)
    cert["input_contract"] = "sentence"
    cert["C1"] = {"pass": True}
    with pytest.raises(RuntimeError, match="C1"):
        check(cert)


def test_non_object_certificate_rejected(certificate):
    _, check = certificate
    with pytest.raises(RuntimeError, match="JSON object"):
        check([])
