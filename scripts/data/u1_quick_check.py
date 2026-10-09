"""Quick checks for a data-u1 dir made from a fully verified one by row removal only:
manifest hashes and counts, id uniqueness and the S1b one-table/one-stratum rule
(same unit functions as verify_unified)."""
import glob
import json, os, sys, hashlib, multiprocessing as mp
from collections import defaultdict, Counter
sys.path[:0] = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "*", "")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import verify_unified as VU

def units(line):
    r = json.loads(line)
    if r.get("form") == "S1b":
        mine = set(VU.unit_keys(r["text"]))
        for ss in VU.sub_sentences(r["text"]):
            mine.update(VU.unit_keys(ss))
        return ("S1b", r["id"], r["text"], r["tables"][0], mine)
    if r.get("form") in VU.TRAIN_FORMS:
        return ("other", r["id"], None, None, VU.row_units(r))
    return ("skip", r["id"], None, None, None)

if __name__ == "__main__":
    d = sys.argv[1]
    man = json.load(open(os.path.join(d, "manifest.json")))
    bad = []
    for f, v in man["files"].items():
        p = os.path.join(d, f)
        if not os.path.exists(p): bad.append(f"missing {f}"); continue
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if h != v.get("sha256"): bad.append(f"sha mismatch {f}")
        if v.get("n_docs") is not None and f.endswith(".jsonl"):
            n = sum(1 for l in open(p) if l.strip())
            if n != v["n_docs"]: bad.append(f"n_docs mismatch {f}: {n} vs {v['n_docs']}")
    print("files:", "OK" if not bad else bad, flush=True)
    lines = [l for f in ("train.jsonl", "dev.jsonl") for l in open(os.path.join(d, f)) if l.strip()]
    with mp.get_context("fork").Pool(150) as p:
        res = p.map(units, lines, chunksize=2000)
    ids = Counter(x[1] for x in res)
    dup_ids = [i for i, c in ids.items() if c > 1]
    other = set()
    for k, _i, _t, _tab, u in res:
        if k == "other": other |= u
    tabs = defaultdict(set); viol_stratum = 0; viol_table = 0
    for k, _i, t, tab, u in res:
        if k == "S1b":
            tabs[t].add(tab)
            if u & other: viol_stratum += 1
    viol_table = sum(1 for v in tabs.values() if len(v) > 1)
    forms = Counter(json.loads(l)["form"] for l in lines)
    print(json.dumps({"rows_train_dev": len(lines), "duplicate_ids": len(dup_ids),
                      "s1b_texts": len(tabs), "s1b_reused_by_other_stratum": viol_stratum,
                      "s1b_text_in_multiple_tables": viol_table, "forms": dict(forms)}, ensure_ascii=False))
