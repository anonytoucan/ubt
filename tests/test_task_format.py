"""Canonical SFT task format."""

from ubt.task_format import render


SINGLE = {"id": "t-1", "doc_type": "single", "k": 1, "regime": "single",
          "switch_density": "none", "tables": ["ko-2020-g2.ctb"],
          "text": "안녕", "braille": "⠣⠒⠉⠻", "segments": None}
MIXED = {"id": "t-2", "doc_type": "inter", "k": 2, "regime": "anchor",
         "switch_density": "none", "tables": ["a.ctb", "b.ctb"],
         "text": "x\ny", "braille": "⠁\n⠃",
         "segments": [{"table": "a.ctb", "text": "x", "braille": "⠁"},
                      {"table": "b.ctb", "text": "y", "braille": "⠃"}]}


def test_agnostic_prompt_has_no_tables_tag():
    ex = render(SINGLE, hinted=False)
    assert ex["prompt"] == "<|braille|>\n⠣⠒⠉⠻\n<|text|>\n"
    assert ex["completion"] == "⟨ko-2020-g2.ctb⟩안녕"


def test_hinted_prompt_lists_gold_tables_in_order():
    ex = render(MIXED, hinted=True)
    assert ex["prompt"].startswith("<|tables|>a.ctb,b.ctb\n<|braille|>\n")
    assert ex["prompt"].endswith("⠁\n⠃\n<|text|>\n")


def test_mixed_completion_one_tagged_line_per_segment():
    ex = render(MIXED, hinted=False)
    assert ex["completion"] == "⟨a.ctb⟩x\n⟨b.ctb⟩y"
    assert ex["completion"].count("⟨") == 2  # U+27E8


def test_brackets_are_math_angle_codepoints():
    ex = render(SINGLE, hinted=False)
    assert ord(ex["completion"][0]) == 0x27E8
    assert "⟩" in ex["completion"]
