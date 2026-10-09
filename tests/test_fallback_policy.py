"""lambda2_fbinf's wipeout fallback must run and be able to disagree with lambda2.

With every candidate infeasible, lambda2 (argmax logprob - 2*d_cell) picks A
(higher logprob, larger d_cell); the fbinf fallback (min d_cell) must pick B.
"""
from ubt.eval_harness import _select


def test_branches_disagree_on_wipeout():
    pool = [
        {"hyp": "A", "logprob": -1.0, "d_cell": 3, "cer_true": 0.5},  # λ2: -7
        {"hyp": "B", "logprob": -9.0, "d_cell": 1, "cer_true": 0.1},  # λ2: -11
    ]
    a = _select(pool, "lambda2")
    b = _select(pool, "lambda2_fbinf")
    assert a["hyp"] == "A" and b["hyp"] == "B"
    assert a["hyp"] != b["hyp"]


def test_lambda2_scores_as_documented():
    # scores well apart (-7 vs -11) to avoid a fragile tie
    pool = [
        {"hyp": "A", "logprob": -1.0, "d_cell": 3, "cer_true": 0.5},  # -7
        {"hyp": "B", "logprob": -9.0, "d_cell": 1, "cer_true": 0.1},  # -11
    ]
    assert _select(pool, "lambda2")["hyp"] == "A"
    assert _select(pool, "lambda2_fbinf")["hyp"] == "B"


def test_feasible_path_identical_between_policies():
    # with a feasible candidate present, both policies must pick the same one
    pool = [
        {"hyp": "F", "logprob": -5.0, "d_cell": 0, "cer_true": 0.0},
        {"hyp": "A", "logprob": -1.0, "d_cell": 3, "cer_true": 0.5},
    ]
    assert _select(pool, "lambda2")["hyp"] == "F"
    assert _select(pool, "lambda2_fbinf")["hyp"] == "F"
