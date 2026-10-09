"""Write the numbers of a braille-edition run into the paper's appendix "The Braille Edition of This Paper".

Rewrites the section prose (from its \\label to the \\clearpage before the pages) from D/summary.json and
D/unit_results.jsonl, written by scripts/braille_edition/braille_edition.py score.

  python scripts/braille_edition/braille_edition_update.py --run D --paper <paper repo> [--dry-run]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re

WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def n(x: int) -> str:
    return f"{x:,}".replace(",", "{,}")


def word_diffs(ref: str, hyp: str) -> list[tuple[str, str]]:
    """(reference words, output words) for each differing word span."""
    import difflib  # noqa: PLC0415
    a, b = ref.split(), hyp.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag != "equal":
            out.append((" ".join(a[i1:i2]), " ".join(b[j1:j2])))
    return out


def tex(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
            .replace("_", r"\_").replace("$", r"\$").replace("{", r"\{").replace("}", r"\}"))


def red_clause(su: dict, units: list[dict]) -> str:
    err, outside = su["err_cells"], su["err_cells_outside_uncertified"]
    if err == 0:
        return "No cell was decoded wrongly."
    if outside == 0:
        return "Every red cell lies inside a yellow span."
    pairs = collections.Counter()
    for u in units:
        if u["certified"] and not u["exact"]:
            for a, b in word_diffs(u["text"], u["hyp"]):
                pairs[(a.strip(".,;:()"), b.strip(".,;:()"))] += 1
    head = f"Of the ${n(err)}$ red cells, ${n(err - outside)}$ lie inside a yellow span; the other ${n(outside)}$ are "
    if not pairs:
        return head + "in verified units, readings that re-transcribe to the same cells."
    parts = [f"{WORDS.get(c, str(c)) + ' readings' if c > 1 else 'a reading'} of \\emph{{{tex(a)}}} as \\emph{{{tex(b)}}}"
             for (a, b), c in pairs.most_common(3)]
    more = "" if len(pairs) <= 3 else f" and {len(pairs) - 3} other readings"
    return head + (", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]) + more + \
        ", which re-transcribe to the same cells."


def paragraph(summary: dict, units: list[dict]) -> str:
    ed, su, sc = summary["edition"], summary["units"], summary["sentence_check"]
    cells = su["cells"]
    region = su["unc_cells"] + (su["err_cells"] - su["err_cells_outside_uncertified"])
    exact_clause = (f"all ${n(sc['exact'])}$ are exact" if sc["exact"] == sc["verified"] else f"${n(sc['exact'])}$ of them are exact")
    rr = sc.get("rerun") or {}
    k, j = rr.get("sentences", 0), rr.get("now_verified", 0)
    if k == 0:
        rerun = ""
    elif j == 0:
        rerun = (", and re-running the unverified one at $n{=}100$ does not verify it" if k == 1 else
                 f", and re-running the unverified {WORDS.get(k, str(k))} at $n{{=}}100$ verifies none of them")
    else:
        rerun = f", and re-running the unverified {WORDS.get(k, str(k))} at $n{{=}}100$ verifies {WORDS.get(j, str(j))} more"
    return (
        "The prose of the main sections, stripped of mathematics, cross-references, and markup, is transcribed to UEB "
        f"grade-2 by a deterministic transcriber. ${n(ed['text_chars'])}$ characters become ${n(cells)}$ cells over "
        f"${n(ed['lines'])}$ lines, or ${n(ed['pages'])}$ embosser pages at 40 cells by 25 lines. The\n"
        "\\texttt{.brf} file accompanies this supplement. We then read the edition back\n"
        f"with the UBT system and score it per cell as bCER does. Of ${n(su['units'])}$ decoding units, "
        f"${n(su['certified'])}$ are verified and\n"
        f"${n(su['certified_and_exact'])}$ of them are exact; at the cell level, ${n(su['err_cells'])}$ of ${n(cells)}$ "
        f"cells (${su['err_cells_pct']:.2f}\\%$)\n"
        f"carry a character error. A sentence-level version of the same check (${n(sc['sentences'])}$\n"
        "markup-free sentences from the main text, code given) behaves the same way:\n"
        f"under best-of-50, ${n(sc['verified'])}$ of ${n(sc['sentences'])}$ sentences are verified and {exact_clause}\n"
        f"(${sc['cer_verified_selection']:.2f}\\%$ CER against ${sc['cer_greedy']:.2f}\\%$ for greedy){rerun}.\n\n"
        f"The pages that follow reproduce all ${n(ed['pages'])}$ embossed pages, with the inkprint each\n"
        "run of cells encodes set beneath it. Cells shaded\n"
        "\\textcolor{hlunc}{\\rule{6pt}{6pt}} yellow belong to a unit the verifier\n"
        "declined to verify; cells shaded \\textcolor{hlerr}{\\rule{6pt}{6pt}} red were\n"
        f"decoded wrongly, with their inkprint set in red. {red_clause(su, units)} The inconsistent region covers "
        f"${n(region)}$ cells (${100.0 * region / max(1, cells):.1f}\\%$ of the edition), since a single unrecovered "
        "character condemns its whole unit.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--paper", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    summary = json.load(open(os.path.join(a.run, "summary.json"), encoding="utf-8"))
    units = [json.loads(x) for x in open(os.path.join(a.run, "unit_results.jsonl"), encoding="utf-8")]
    new = paragraph(summary, units)
    p = os.path.join(a.paper, "_sections", "appendix.tex")
    s = open(p, encoding="utf-8", newline="").read()
    m = re.search(r"(\\label\{app:braille-edition\}\n)(.*?)(\n\\clearpage\n(?:%[^\n]*\n)*\\input\{figures/braille_edition/pages\})", s,
                  flags=re.S)   # comment lines may sit between \clearpage and \input
    assert m, "braille-edition section not found"
    stamp = f"% generated by scripts/braille_edition/braille_edition_update.py (code repo) from {a.run}\n"
    s2 = s[:m.start(2)] + "\n" + stamp + new + s[m.end(2):]
    print(new)
    if not a.dry_run:
        open(p, "w", encoding="utf-8", newline="").write(s2)
        print("->", p)


if __name__ == "__main__":
    main()
