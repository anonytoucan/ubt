"""Fixed data for the tokenizer comparison: keep docs that fit max_len under the original Qwen2.5 tokenizer.

They then also fit the shorter braille-extended encoding, so both tokenizers train on the same documents.
Writes <out>/train.jsonl, <out>/dev.jsonl (first dev_n fitting docs) and fit_report.json (fit counts).
usage: python scripts/data/fixed_split.py <data dir> <out dir> <max_len> <dev_n>"""
import json, multiprocessing as mp, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
TOK = None
def init(name):
    global TOK
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(name)
def length(line):
    from ubt.task_format import render
    ex = render(json.loads(line), hinted=False)
    return len(TOK(ex["prompt"])["input_ids"]) + len(TOK(ex["completion"], add_special_tokens=False)["input_ids"]) + 1
if __name__ == "__main__":
    data, out, max_len, dev_n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    os.makedirs(out, exist_ok=True)
    rep = {}
    for split, cap in (("train", None), ("dev", dev_n)):
        lines = [l for l in open(os.path.join(data, f"{split}.jsonl"), encoding="utf-8") if l.strip()]
        with mp.get_context("fork").Pool(96, initializer=init, initargs=("Qwen/Qwen2.5-7B",)) as p:
            lens = p.map(length, lines, chunksize=500)
        keep = [l for l, n in zip(lines, lens) if n <= max_len]
        if cap:
            keep = keep[:cap]
        with open(os.path.join(out, f"{split}.jsonl"), "w", encoding="utf-8") as fh:
            fh.writelines(keep)
        rep[split] = {"n_in": len(lines), "n_fit_orig_tokenizer": sum(n <= max_len for n in lens),
                      "n_written": len(keep), "n_over": sum(n > max_len for n in lens)}
    import shutil
    shutil.copy2(os.path.join(data, "manifest.json"), os.path.join(out, "manifest.json"))
    json.dump({"source": os.path.abspath(data), "max_len": max_len, "tokenizer": "Qwen/Qwen2.5-7B", **rep},
              open(os.path.join(out, "fit_report.json"), "w"), indent=1)
    print(json.dumps(rep))
