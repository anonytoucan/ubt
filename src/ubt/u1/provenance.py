"""data-u1 provenance: which code and which input files produced a build.

manifest.provenance = {git_head, code_dirty (git status of the code files), code_sha256, code_files {path: sha256},
                       inputs_sha256, inputs {label: sha256}}
Code: every .py under src/ubt, src/ubt/data, and the builder/verifier scripts with the scripts they import.
Inputs: all data a build reads (FLORES, wiki shards, eval_exclusions, NIKL raw, rule examples, word list, configs).
The verifier fails on a changed input and only reports changed code.
"""
from __future__ import annotations

import hashlib
import os
import subprocess

CODE_SCRIPTS = ("scripts/data/build_unified.py", "scripts/data/verify_unified.py",
                "scripts/data/math_grammar.py", "scripts/liblouis/ko_table_nikl_check.py")
CONFIGS = ("configs/unified_u1.yaml", "configs/currency_overrides.yaml")
EXAMPLES = ("resources/korean_braille_2024/examples.jsonl", "resources/korean_braille_2024/examples_math.jsonl")
NIKL_REL = "data_nikl/raw/nikl"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(m: dict[str, str]) -> str:
    h = hashlib.sha256()
    for k in sorted(m):
        h.update(k.encode("utf-8") + b"\0" + m[k].encode() + b"\n")
    return h.hexdigest()


def code_files(repo: str) -> list[str]:
    out = []
    for root in ("src/ubt",):
        for dirpath, dirs, files in os.walk(os.path.join(repo, root)):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for f in sorted(files):
                if f.endswith(".py") or os.path.basename(dirpath) == "data":
                    out.append(os.path.relpath(os.path.join(dirpath, f), repo))
    out += [p for p in CODE_SCRIPTS if os.path.isfile(os.path.join(repo, p))]
    return sorted(out)


def input_files(repo: str, data_dir: str) -> dict[str, str]:
    """label -> absolute path of every data input."""
    out: dict[str, str] = {}
    fl = os.path.join(data_dir, "flores200_dataset")
    for split in ("dev", "devtest"):
        d = os.path.join(fl, split)
        for f in sorted(os.listdir(d)):
            if f.endswith("." + split):
                out[f"flores/{split}/{f}"] = os.path.join(d, f)
    for shard in ("wiki_txt_A", "wiki_txt_B"):
        d = os.path.join(data_dir, shard)
        for f in sorted(os.listdir(d)):
            if f.endswith(".txt"):
                out[f"{shard}/{f}"] = os.path.join(d, f)
    ex = os.path.join(data_dir, "eval_exclusions.txt")
    if os.path.isfile(ex):
        out["eval_exclusions.txt"] = ex
    nk = os.path.join(repo, NIKL_REL)
    for dirpath, dirs, files in os.walk(nk):
        dirs.sort()
        for f in sorted(files):
            if f.endswith(".json"):
                p = os.path.join(dirpath, f)
                out["nikl/" + os.path.relpath(p, nk)] = p
    for rel in EXAMPLES + CONFIGS + ("src/ubt/data/en_terms.txt",):
        p = os.path.join(repo, rel)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"input missing: {p}")
        out[rel] = p
    return out


def _hash_job(p: str) -> str:
    return sha256_file(p)


def hash_inputs(repo: str, data_dir: str, pool=None) -> dict[str, str]:
    files = input_files(repo, data_dir)
    labels = sorted(files)
    paths = [files[k] for k in labels]
    hs = pool.map(_hash_job, paths, chunksize=4) if pool is not None else map(_hash_job, paths)
    return dict(zip(labels, hs))


def git_info(repo: str, files: list[str]) -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                              text=True, check=True).stdout.strip()
        st = subprocess.run(["git", "status", "--porcelain", "--", *files], cwd=repo,
                            capture_output=True, text=True).stdout.strip()
        return {"git_head": head, "code_dirty": bool(st),
                "code_dirty_files": [ln[3:] for ln in st.splitlines()][:50]}
    except Exception as e:  # noqa: BLE001
        return {"git_head": None, "code_dirty": None, "error": str(e)}


def provenance(repo: str, data_dir: str, pool=None) -> dict:
    cf = code_files(repo)
    code = {p: sha256_file(os.path.join(repo, p)) for p in cf}
    inputs = hash_inputs(repo, data_dir, pool)
    return {**git_info(repo, cf), "code_sha256": digest(code), "code_files": code,
            "inputs_sha256": digest(inputs), "inputs": inputs}
