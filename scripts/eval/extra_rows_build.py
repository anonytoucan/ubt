"""Rows for scripts/eval/u1_extra_metrics.py (sim / judge) for every system in the paper's tables, from the scored pools.

Each row: id, hyp_text (segment texts joined with '\\n', as CER uses them), group (Table 1B family), set (error kind of
the selected output), code. Systems: best-of-20 and greedy (code inferred), best-of-20 (code given), LibLouis, GPT-5.5
and the real corpora (group = corpus).

  python scripts/eval/extra_rows_build.py --model rlvr --out work/evals/rlvr/extra
  python scripts/eval/extra_rows_build.py --model byt5 --out work/evals/byt5/extra --pools-only
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path[:0] = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "*", "")))
from ubt.paths import DATASETS, WORK  # noqa: E402
R = WORK
INV = f"{DATASETS}/data_u1_v2/inventory.json"


def family(inv, t):
    b = inv[t]["base_lang"]
    if b == "ja": return "Japanese"
    if t.startswith("zh"): return "Chinese"
    if b == "ko": return "Korean"
    if t in ("kok.tbl", "sah.utb"): return "Konkani+Yakut"
    if inv[t]["contraction"] == "full": return "contracted"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rlvr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pools-only", action="store_true", help="only the code-inferred best-of-20 and greedy rows")
    a = ap.parse_args()
    from ubt.wandb_utils import parse_target
    inv = {t["table_id"]: t for t in json.load(open(INV))["tables"]}
    join = lambda t: "\n".join(x for _, x in (parse_target(t) or []))
    os.makedirs(a.out, exist_ok=True)
    def pool(d, stem, sl, pols):
        pd = {}
        for line in open(os.path.join(d, "per_doc.jsonl"), encoding="utf-8"):
            x = json.loads(line)
            if x["slice"] == sl and x.get("set") == "core":
                pd[x["id"]] = x
        out = {p: [] for p in pols}
        for p in glob.glob(os.path.join(d, f"{stem}.*.jsonl")):
            for line in open(p, encoding="utf-8"):
                x = json.loads(line)
                if x["id"] in pd:
                    doc = pd[x["id"]]
                    for pol in pols:
                        c = x["cands"][doc["pol"][pol]["ci"]]["text"]
                        out[pol].append({"id": x["id"], "hyp_text": join(c), "group": family(inv, doc["table"]),
                                         "set": doc["pol"][pol]["kind"], "code": doc["table"]})
        return out
    m = a.model
    main_pool = bool(glob.glob(f"{R}/evals/{m}/bon/gen_main.*.jsonl"))     # else a core-only pool (gen_core.*)
    agn = pool(f"{R}/evals/{m}/bon", *(("gen_main", "main") if main_pool else ("gen_core", "core")), ["V", "G"])
    files = {f"{m}_V_agn": agn["V"], f"{m}_G_agn": agn["G"]}
    if a.pools_only:
        for name, rows in files.items():
            with open(os.path.join(a.out, f"rows_{name}.jsonl"), "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(name, len(rows))
        return
    giv = pool(f"{R}/evals/{m}/prefill", "gen_core", "core", ["V"])
    files[f"{m}_V_given"] = giv["V"]
    ll = []
    for line in open(f"{R}/liblouis_bwd/per_doc.jsonl", encoding="utf-8"):
        x = json.loads(line)
        if x.get("set") == "core":
            kind = "correct" if x["exact"] else ("consistent_misreading" if x["consistent"] else "inconsistent")
            ll.append({"id": x["id"], "hyp_text": join(x["hyp"]), "group": family(inv, x["table"]), "set": kind, "code": x["table"]})
    files["liblouis"] = ll
    import frontier_fewshot as F
    res = {}
    for p in glob.glob(f"{R}/frontier/batch_output.part*.jsonl"):
        for line in open(p, encoding="utf-8"):
            x = json.loads(line)
            t = F._text_of((x.get("response") or {}).get("body") or {})
            t = re.sub(r"^\s*(text\s*:|the text is\s*:?|transcription\s*:)\s*", "", t, flags=re.I).strip().strip('"').strip()
            res[x["custom_id"]] = t.replace("\n", " ")
    files["gpt55"] = [{"id": x["id"], "hyp_text": res.get(x["id"], ""), "group": family(inv, x["table"]), "set": x["kind"], "code": x["table"]}
                      for x in map(json.loads, open(f"{R}/frontier/per_doc.jsonl", encoding="utf-8"))]
    for cond in ("given", "inferred"):
        rows = []
        for line in open(f"{R}/evals/{m}/real/per_doc.jsonl", encoding="utf-8"):
            x = json.loads(line)
            if x["cond"] == cond:
                rows.append({"id": x["id"], "hyp_text": join(x["pol"]["V"]["text"]), "group": x["group"], "set": x["pol"]["V"]["kind"], "code": x["table"]})
        files[f"{m}_real_{cond}"] = rows
    for name, rows in files.items():
        with open(os.path.join(a.out, f"rows_{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rows))


if __name__ == "__main__":
    main()
