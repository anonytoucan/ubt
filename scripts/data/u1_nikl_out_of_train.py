"""data-u1: move every NIKL (S7) row out of train into eval/nikl_dev.jsonl; NIKL is never trained on.

dev keeps its S7 rows (checkpoint selection is evaluation use); the sealed NIKL test stays a
pointer file. Also counts NIKL eval rows sharing a text unit with train. Unchanged files are hard-linked.

  python scripts/data/u1_nikl_out_of_train.py <src data dir> <new data dir>
"""
import glob
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

sys.path[:0] = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "*", "")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import verify_unified as VU  # noqa: E402


def units(line: str) -> set:
    return VU.row_units(json.loads(line))


def sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    assert not os.path.exists(out), out
    man = json.load(open(os.path.join(src, "manifest.json")))
    train_keep, nikl = [], []
    with open(os.path.join(src, "train.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            if '"form": "S7"' in line and json.loads(line).get("form") == "S7":
                r = json.loads(line)
                r["split"] = "eval"
                nikl.append(json.dumps(r, ensure_ascii=False) + "\n")
            else:
                train_keep.append(line)
    with mp.get_context("fork").Pool(150) as p:
        tu = p.map(units, train_keep, chunksize=2000)
        nu = p.map(units, nikl, chunksize=500)
    train_units = set().union(*tu) if tu else set()
    overlap = sum(1 for u in nu if u & train_units)
    print(f"train rows kept {len(train_keep)}, NIKL rows moved {len(nikl)}, "
          f"NIKL eval rows sharing a unit with train: {overlap}", flush=True)
    shutil.copytree(src, out, copy_function=os.link)
    for f in ("train.jsonl", "manifest.json"):
        os.unlink(os.path.join(out, f))
    with open(os.path.join(out, "train.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(train_keep)
    with open(os.path.join(out, "eval", "nikl_dev.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(nikl)
    for f, n in (("train.jsonl", len(train_keep)), ("eval/nikl_dev.jsonl", len(nikl))):
        p = os.path.join(out, f)
        man["files"][f] = {"n_docs": n, "bytes": os.path.getsize(p), "sha256": sha(p)}
    man["n_train"] = len(train_keep)
    man["amendments"] = list(man.get("amendments", [])) + [{
        "kind": "nikl-out-of-train", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dataset": os.path.abspath(src), "moved_rows": len(nikl),
        "moved_to": "eval/nikl_dev.jsonl (form S7, split eval)", "eval_rows_sharing_unit_with_train": overlap,
        "reason": "NIKL is used for test and evaluation only, not for training"}]
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({"n_train": man["n_train"], "nikl_dev_eval": len(nikl), "overlap": overlap}))
