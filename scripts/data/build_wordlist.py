#!/usr/bin/env python3
"""Build the bundled English term and named-entity wordlist for the intra-switch splicer.

Frequent common words plus capitalized tokens and bigrams from the train wiki shard (wiki_txt_A/en.txt); eval shard B
is never read. Output: src/ubt/data/en_terms.txt, one term or phrase per line.
"""

from __future__ import annotations

import collections
import os
import re
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from ubt.paths import DATASETS  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
SRC = f"{DATASETS}/data/wiki_txt_A/en.txt"
OUT = os.path.join(ROOT, "src", "ubt", "data", "en_terms.txt")

_STOP = set("""the of and to in a is was for on as by with at from it his her that
this be are were an or which he she they their its has had not but also into
after first one two new all during when where who more most other some such
only over between about than then them there these those out up no can may
will would could should have been him our your my we you i if so what while
being before both under through against each because until since within
without along across behind beyond above below near off down still very much
many any own same just how now again once here even ever never do does did
done made make like get got go went said say says see seen use used using per
including included part called known became become well back years year time
day days man men way among however several three four five six seven eight
nine ten had""".split())

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,14}")
_CAP = re.compile(r"\b([A-Z][a-z]{2,14})\b")
_CAP_BIGRAM = re.compile(r"\b([A-Z][a-z]{2,14} [A-Z][a-z]{2,14})\b")

N_COMMON, N_NE, N_PHRASE = 900, 700, 400


def main() -> None:
    common: collections.Counter = collections.Counter()
    caps: collections.Counter = collections.Counter()
    bigrams: collections.Counter = collections.Counter()
    with open(SRC, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= 120_000:
                break
            for m in _WORD.finditer(line):
                w = m.group(0)
                if w.islower() and w not in _STOP:
                    common[w] += 1
            for m in _CAP.finditer(line[1:]):  # skip sentence-initial caps
                if m.group(1).lower() not in _STOP:
                    caps[m.group(1)] += 1
            for m in _CAP_BIGRAM.finditer(line[1:]):
                bigrams[m.group(1)] += 1

    terms: list[str] = []
    seen: set[str] = set()
    for pool, n in ((common, N_COMMON), (caps, N_NE), (bigrams, N_PHRASE)):
        taken = 0
        for w, _ in pool.most_common(n * 3):
            if taken >= n:
                break
            if w.lower() in seen:
                continue
            seen.add(w.lower())
            terms.append(w)
            taken += 1

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(terms) + "\n")
    n_phrases = sum(1 for t in terms if " " in t)
    print(f"wrote {len(terms)} entries ({n_phrases} phrases) -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
