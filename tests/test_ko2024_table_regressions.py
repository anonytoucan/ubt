"""Real-liblouis regressions for the Korean 2024 table repairs.

Needs KO2024_TABLES_DIR and the installed liblouis library. The cases are expectations only; the translator never
reads them.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "patches/ko2024/regression_cases.json").read_text())
_NIKL = ROOT / "patches/ko2024/regression_cases_nikl.local.json"   # NIKL sentences: local only, not in git
if _NIKL.exists():
    CASES += json.loads(_NIKL.read_text())


@pytest.fixture(scope="module")
def translator():
    table_dir = os.environ.get("KO2024_TABLES_DIR")
    if not table_dir:
        pytest.skip("Set KO2024_TABLES_DIR to a repaired Korean liblouis table directory")
    spec = importlib.util.spec_from_file_location(
        "ko_table_regression_translator", ROOT / "scripts/liblouis/ko_table_nikl_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.preflight(table_dir, "ko-2020-g2.ctb")
    return module._tr


@pytest.mark.parametrize("case", CASES, ids=[str(i) for i in range(len(CASES))])
def test_translation_regression(translator, case):
    assert translator(case["source"], case.get("options")) == case["expected"], (
        case["source"], case["reference"]
    )
