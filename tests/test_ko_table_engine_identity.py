"""Evaluation provenance must name the mapped library, not an environment hint."""
import hashlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "liblouis"))
import ko_table_nikl_check as verifier


def fake_maps(monkeypatch, text):
    real_open = open

    def mapped_open(path, *args, **kwargs):
        if str(path) == "/proc/self/maps":
            return io.StringIO(text)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", mapped_open)


def test_identity_hashes_mapped_library_and_binding(tmp_path, monkeypatch):
    library = tmp_path / "liblouis.so.20.1.3"
    library.write_bytes(b"loaded library")
    binding = tmp_path / "binding.py"
    binding.write_bytes(b"loaded binding")
    monkeypatch.setattr(verifier, "_LOUIS", SimpleNamespace(
        version=lambda: "3.38.0", wideCharBytes=4, __file__=str(binding)), raising=False)
    monkeypatch.setenv("UBT_LOUIS_LIB", "/unused/liblouis.so")
    mapping = f"1000-2000 r-xp 0000 00:00 1 {library}\n"
    fake_maps(monkeypatch, mapping + mapping)
    result = verifier.engine_identity()
    assert result == {
        "version": "3.38.0", "character_bytes": 4, "library_path": str(library),
        "library_sha256": hashlib.sha256(b"loaded library").hexdigest(),
        "python_binding_sha256": hashlib.sha256(b"loaded binding").hexdigest(),
    }


@pytest.mark.parametrize("mapping", ["", (
    "1000-2000 r-xp 0000 00:00 1 /one/liblouis.so.20\n"
    "2000-3000 r-xp 0000 00:00 2 /two/liblouis.so.20\n"
)])
def test_identity_rejects_missing_or_ambiguous_library(monkeypatch, mapping):
    fake_maps(monkeypatch, mapping)
    with pytest.raises(RuntimeError, match="Expected one mapped"):
        verifier.engine_identity()
