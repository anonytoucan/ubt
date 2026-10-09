"""Drop S1b rows whose line (or a sub-sentence of it) also occurs inside another train-form row
(verify_unified check `s1b_one_table`). Writes a new data dir; unchanged files are hard-linked."""
import glob
import json, os, sys, hashlib, time, shutil
sys.path[:0] = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "*", "")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import verify_unified as VU
src, out = sys.argv[1], sys.argv[2]
assert not os.path.exists(out)
man = json.load(open(os.path.join(src, "manifest.json")))
rows = {f: [json.loads(l) for l in open(os.path.join(src, f)) if l.strip()] for f in ("train.jsonl", "dev.jsonl")}
other = set()
for f, rs in rows.items():
    for r in rs:
        if r.get("form") != "S1b" and r.get("form") in VU.TRAIN_FORMS:
            other |= VU.row_units(r)
drop = set()
for f, rs in rows.items():
    for r in rs:
        if r.get("form") == "S1b":
            mine = set(VU.unit_keys(r["text"]))
            for ss in VU.sub_sentences(r["text"]):
                mine.update(VU.unit_keys(ss))
            if mine & other:
                drop.add(r["id"])
print("drop", len(drop))
shutil.copytree(src, out, copy_function=os.link)
per = {}
for f, rs in rows.items():
    p = os.path.join(out, f); os.unlink(p)
    kept = [r for r in rs if r["id"] not in drop]
    with open(p, "w", encoding="utf-8") as fh:
        for r in kept: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    man["files"][f] = {"n_docs": len(kept), "bytes": os.path.getsize(p), "sha256": h}
    per[f] = len(rs) - len(kept)
os.unlink(os.path.join(out, "manifest.json"))
man["n_train"] = man["files"]["train.jsonl"]["n_docs"]; man["n_dev"] = man["files"]["dev.jsonl"]["n_docs"]
man["amendments"] = man.get("amendments", []) + [{"kind": "s1b-overlap-drop", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "source_dataset": os.path.abspath(src), "dropped_ids": sorted(drop), "dropped_per_file": per,
    "reason": "verify s1b_one_table: S1b line or a sub-sentence of it reused by another train stratum"}]
json.dump(man, open(os.path.join(out, "manifest.json"), "w"), ensure_ascii=False, indent=1)
print(per)
