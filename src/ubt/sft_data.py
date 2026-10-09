"""SFT dataset construction from frozen datagen records.

Prompt and completion are tokenized separately, with labels -100 on the
prompt. Overlong examples are dropped, never truncated; the drop count per
split is returned for reporting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ubt.task_format import render

IGNORE_INDEX = -100


def iter_records(path: str):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def encode_example(record: dict, tokenizer, max_len: int, hinted: bool) -> dict | None:
    """One record -> {input_ids, labels}, or None if overlong. The prompt gets
    special tokens (BOS for Gemma-2); the completion gets none, plus an EOS."""
    ex = render(record, hinted=hinted)
    prompt_ids = tokenizer(ex["prompt"], add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]
    completion_ids = completion_ids + [tokenizer.eos_token_id]
    if len(prompt_ids) + len(completion_ids) > max_len:
        return None
    return {
        "input_ids": prompt_ids + completion_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + completion_ids,
    }


_TOK = None
_TOK_NAME = None


def _tok_chunk(args) -> tuple[list[dict], int]:
    """Fork-pool worker: encode a chunk of raw jsonl lines."""
    global _TOK, _TOK_NAME
    lines, tok_name, max_len, hinted = args
    if _TOK is None or _TOK_NAME != tok_name:
        from transformers import AutoTokenizer  # noqa: PLC0415
        _TOK = AutoTokenizer.from_pretrained(tok_name)
        if _TOK.pad_token is None:
            _TOK.pad_token = _TOK.eos_token
        _TOK_NAME = tok_name
    out, dropped = [], 0
    for line in lines:
        enc = encode_example(json.loads(line), _TOK, max_len, hinted)
        if enc is None:
            dropped += 1
        else:
            out.append(enc)
    return out, dropped


# bump when encode_example/render semantics change (invalidates caches)
_FORMAT_VERSION = "v2-bos"


def build_split(path: str, tokenizer, max_len: int, hinted: bool,
                limit: int | None = None,
                workers: int = 24) -> tuple[list[dict], int]:
    """Returns (examples, n_dropped_overlong), tokenized in parallel and cached
    on disk; under an fcntl lock one rank builds the cache, the others load it."""
    import fcntl  # noqa: PLC0415
    import hashlib  # noqa: PLC0415
    import multiprocessing as mp_  # noqa: PLC0415
    import os  # noqa: PLC0415
    import pickle  # noqa: PLC0415

    tok_name = getattr(tokenizer, "name_or_path", "unknown")
    st = os.stat(path)
    key = hashlib.sha1(
        f"{_FORMAT_VERSION}|{path}|{st.st_size}|{int(st.st_mtime)}|"
        f"{max_len}|{hinted}|{limit}|{tok_name}".encode()).hexdigest()[:16]
    cache_dir = os.path.join(os.environ.get("UBT_BASE", "."), "tmp", "tokcache")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"sft_{key}.pkl")

    # Reads need no lock (the writer renames atomically); locking them would
    # serialize the ranks' unpickles and can trip the NCCL watchdog.
    if os.path.isfile(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    with open(cache + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if os.path.isfile(cache):
                with open(cache, "rb") as fh:
                    return pickle.load(fh)
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()
            if limit is not None:
                lines = lines[:limit]
            # Contiguous blocks keep the single-threaded example order, which
            # seeded-shuffle resume replay depends on.
            step = max(1, (len(lines) + workers - 1) // workers)
            chunks = [lines[i : i + step] for i in range(0, len(lines), step)]
            jobs = [(c, tok_name, max_len, hinted) for c in chunks if c]
            if len(jobs) > 1:
                with mp_.get_context("fork").Pool(len(jobs)) as pool:
                    results = pool.map(_tok_chunk, jobs)
            else:
                results = [_tok_chunk(j) for j in jobs]
            out = [ex for part, _ in results for ex in part]
            dropped = sum(d for _, d in results)
            tmp = cache + f".tmp{os.getpid()}"
            with open(tmp, "wb") as fh:
                pickle.dump((out, dropped), fh)
            os.rename(tmp, cache)
            return out, dropped
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@dataclass
class CausalCollator:
    """Left-unpadded batch -> right-padded tensors with label masking."""

    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        import torch  # noqa: PLC0415

        width = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention = [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = width - n
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            labels.append(f["labels"] + [IGNORE_INDEX] * pad)
            attention.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }
