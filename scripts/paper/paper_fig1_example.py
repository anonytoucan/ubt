"""Paper Figure 1: a two-line document (English with an inline Nemeth span, then Korean) and every system's reading.

  line 1  en-ueb-g2 prose + Nemeth span between the code indicators
  line 2  ko-2024-g2

Panel (a), code given per segment: liblouis back-translation for the prose and the Korean line, Access8Math's Nemeth
reader for the span. Panel (b): the final SFT model with the code inferred, greedy + 19 samples (tau 0.8, top-p 0.95)
and the verified selection (lambda = 2), run on CPU so it does not need a GPU.

  python scripts/paper/paper_fig1_example.py --out work/paper/fig1_example.json [--model <merged model dir>]
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path[:0] = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "*", "")))
from ubt.paths import DATASETS, WORK  # noqa: E402
DATA = f"{DATASETS}/data_u1_v2"
MODEL = f"{WORK}/models/sft"
PROSE = ("The area is ", ".")
LATEX = r"\int_{0}^{1}x^{2}\,dx=\frac{1}{3}"
KOREAN = "이 넓이는 3분의 1이다."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=96)
    ap.add_argument("--seed", type=int, default=148)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    import liblouis_backward_eval as LB  # noqa: PLC0415
    import math_baselines as M  # noqa: PLC0415
    from ubt.math_norm import nu_math  # noqa: PLC0415
    from ubt.task_format import render  # noqa: PLC0415
    from ubt.u1.nemeth import NEMETH_CLOSE, NEMETH_OPEN  # noqa: PLC0415
    from ubt.u1.reencode import _nemeth  # noqa: PLC0415
    from ubt.u1.routed_louis import RoutedLouis  # noqa: PLC0415

    # --- the document, transcribed by the data engines
    L = RoutedLouis(DATA, timeout_s=300.0)
    pre, post, ko = L.translate_many([("en-ueb-g2.ctb", PROSE[0]), ("en-ueb-g2.ctb", PROSE[1]),
                                      ("ko-2024-g2.ctb", KOREAN)])
    L.close()
    span = _nemeth(nu_math(LATEX))
    line1 = pre + NEMETH_OPEN + span + NEMETH_CLOSE + post
    text1 = PROSE[0] + "$$" + LATEX + "$$" + PROSE[1]
    rec = {"id": "fig1-example", "doc_type": "inter", "k": 2, "regime": "mixed", "switch_density": "none",
           "text": text1 + "\n" + KOREAN, "braille": line1 + "\n" + ko, "tables": ["en-ueb-g2.ctb", "ko-2024-g2.ctb"],
           "segments": [{"table": "en-ueb-g2.ctb", "text": text1, "braille": line1},
                        {"table": "ko-2024-g2.ctb", "text": KOREAN, "braille": ko}], "seg_join": "\n"}
    ex = render(rec, hinted=False)

    # --- panel (a): code given per segment
    man = json.load(open(os.path.join(DATA, "manifest.json")))
    base, over = man["engine_spec"], man.get("engine_spec_overrides") or {}
    jobs = [("en-ueb-g2.ctb", base, [pre, post]), ("ko-2024-g2.ctb", over.get("ko-2024-g2.ctb", base), [ko])]
    with mp.get_context("spawn").Pool(2) as pool:
        bt = dict(pool.map(LB._bt_table, jobs))
    a8m = M._a8m()
    a8m_latex, a8m_err = a8m(span)
    rule = {"prose_before": bt["en-ueb-g2.ctb"][0], "prose_after": bt["en-ueb-g2.ctb"][1], "korean": bt["ko-2024-g2.ctb"][0],
            "math_access8math": a8m_latex, "math_access8math_error": a8m_err,
            "math_canonical_ref": M.canon_latex(LATEX), "math_canonical_access8math": M.canon_latex(a8m_latex) if a8m_latex else None}

    # --- panel (b): code inferred, verified best-of-20 on CPU
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    model.eval()
    ids = torch.tensor([tok(ex["prompt"], add_special_tokens=True)["input_ids"]])
    n_c = len(tok(ex["completion"], add_special_tokens=False)["input_ids"])
    cands = []
    with torch.no_grad():
        for sample, n in ((False, 1), (True, 19)):
            torch.manual_seed(a.seed)
            out = model.generate(ids, max_new_tokens=2 * n_c + 32, do_sample=sample, temperature=0.8 if sample else None,
                                 top_p=0.95 if sample else None, num_return_sequences=n, eos_token_id=tok.eos_token_id,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id, output_scores=True,
                                 return_dict_in_generate=True)
            sc = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            for j in range(n):
                gen = out.sequences[j, ids.shape[1]:]
                keep = gen != (tok.pad_token_id or tok.eos_token_id)
                text = tok.decode(gen[keep], skip_special_tokens=True)
                lp = float(sc[j][keep[: sc.shape[1]]].sum())
                cands.append({"text": text, "lp": lp, "greedy": not sample})
    import bon_gen as E  # noqa: PLC0415
    row = {"id": rec["id"], "prompt": ex["prompt"], "completion": ex["completion"], "tables": rec["tables"]}
    feas = E._feasibility([row], {rec["id"]: {"cands": cands}}, DATA)
    f = [feas[(rec["id"], i)] for i in range(len(cands))]
    F = [i for i in range(len(cands)) if f[i][0]]
    D = [i for i in range(len(cands)) if f[i][1] is not None]
    v = (max(F, key=lambda i: cands[i]["lp"]) if F else
         max(D, key=lambda i: cands[i]["lp"] - 2.0 * f[i][1]) if D else 0)
    out = {"model": a.model, "document": {"line1_text": text1, "line1_cells": line1, "line2_text": KOREAN, "line2_cells": ko,
                        "prose_cells": [pre, post], "nemeth_span_cells": span, "prompt": ex["prompt"],
                        "completion": ex["completion"]},
           "panel_a": rule,
           "panel_b": {"greedy": cands[0]["text"], "greedy_consistent": bool(f[0][0]), "verified": cands[v]["text"],
                       "verified_consistent": bool(f[v][0]), "n_consistent": len(F), "candidates": cands,
                       "feasibility": f}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("document", "panel_a")}, ensure_ascii=False, indent=1))
    print("greedy  :", cands[0]["text"], "| consistent", bool(f[0][0]))
    print("verified:", cands[v]["text"], "| consistent", bool(f[v][0]), "|", len(F), "of 20 consistent")


if __name__ == "__main__":
    main()
