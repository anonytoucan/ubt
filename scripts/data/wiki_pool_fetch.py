"""Wikipedia sentence pools for the top-up run, for languages whose only training text was FLORES dev.

One sentence per line in article order, NFC, deduplicated, markup residue dropped; the top-up builder enforces
train/eval separation. Source: Hugging Face `wikimedia/wikipedia`, config 20231101.<wiki code> (CC BY-SA 4.0 / GFDL).

  python scripts/data/wiki_pool_fetch.py --out datasets/data/wiki_txt_C --langs da,af,nb,hu,ms,fil,ur,my
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import unicodedata

WIKI_CODE = {"da": "da", "af": "af", "nb": "no", "hu": "hu", "ms": "ms", "fil": "tl", "ur": "ur", "my": "my"}
# sentence ends: Latin . ! ?, Urdu ۔ ؟, Burmese ။ (followed by space or end)
SPLIT = re.compile(r"(?<=[.!?۔؟။])\s+")
END = re.compile(r"[.!?۔؟။][\"'”’)\]]?$")
BAD = re.compile(r"https?://|www\.|[{}|=<>_\[\]#@\\]|\.\.\.|\s{2,}")


SCRIPT = {"my": (0x1000, 0x109F), "ur": (0x0600, 0x06FF)}


def sentences(text: str, lang: str = ""):
    for para in text.split("\n"):
        para = para.strip()
        if len(para) < 40:                        # headings, list stubs
            continue
        for s in SPLIT.split(para):
            s = unicodedata.normalize("NFC", s.strip())
            if 25 <= len(s) <= 300 and END.search(s) and not BAD.search(s):
                letters = [ch for ch in s if unicodedata.category(ch)[0] in "LM"]    # marks count (Burmese vowels)
                if len(letters) < 0.6 * len(s):
                    continue
                lo_hi = SCRIPT.get(lang)
                if lo_hi and sum(lo_hi[0] <= ord(ch) <= lo_hi[1] for ch in letters) < 0.8 * len(letters):
                    continue                      # off-script passages, e.g. English in the Burmese Wikipedia
                yield s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", required=True)
    ap.add_argument("--per-lang", type=int, default=30000)
    ap.add_argument("--snapshot", default="20231101")
    a = ap.parse_args()
    from datasets import load_dataset  # noqa: PLC0415
    os.makedirs(a.out, exist_ok=True)
    meta = {}
    for lang in a.langs.split(","):
        t0 = time.time()
        cfg = f"{a.snapshot}.{WIKI_CODE[lang]}"
        ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
        seen, n_art, lines = set(), 0, []
        for art in ds:
            n_art += 1
            for s in sentences(art["text"], lang):
                h = hashlib.sha1(s.encode("utf-8")).digest()[:8]
                if h not in seen:
                    seen.add(h)
                    lines.append(s)
            if len(lines) >= a.per_lang:
                break
        path = os.path.join(a.out, f"{lang}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines[: a.per_lang]) + "\n")
        meta[lang] = {"hf_dataset": "wikimedia/wikipedia", "config": cfg, "articles_read": n_art,
                      "sentences": min(len(lines), a.per_lang), "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                      "sec": round(time.time() - t0)}
        print(lang, json.dumps(meta[lang]), flush=True)
    json.dump(meta, open(os.path.join(a.out, "SOURCE.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
