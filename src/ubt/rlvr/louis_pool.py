"""Per-table isolated liblouis translation for RLVR rewards.

liblouis output for a table can depend on which other tables the process compiled before, so the reward uses the
same canonical f as the data (scripts/data/u1_retranscribe.py): one process per table. Each worker is spawned lazily with
its table's engine spec (`engine_spec_overrides` route ko-2024-g2.ctb to the patched liblouis) and is restarted if it
hangs, since liblouis can spin on garbage input.

    pool = LouisPool(manifest["engine_spec"], overrides=manifest.get("engine_spec_overrides"))
    out = pool.translate_many([(table, text), ...])   # -> [braille | None, ...]
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import unicodedata
from collections import defaultdict

DISPLAY = "unicode.dis"


def _norm(text: str) -> str:
    from ubt.u1.sources import _bengali_fix  # noqa: PLC0415
    return _bengali_fix(unicodedata.normalize("NFC", text))


def _worker(table: str, spec_dict: dict, conn) -> None:
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    from ubt.u1 import engine as EN  # noqa: PLC0415
    spec = EN.EngineSpec.from_dict(spec_dict)
    os.chdir(spec.cwd_dir)
    tr = EN.make_translator(spec)
    lou = tr._louis
    tabs = [DISPLAY, table]
    try:                                    # compile once; an invalid table answers None forever
        lou.translateString(tabs, "a", mode=0)
        ok_table = True
    except Exception:  # noqa: BLE001
        ok_table = False
    conn.send(("ready", ok_table))
    while True:
        msg = conn.recv()
        if msg is None:
            return
        out = []
        for t in msg:
            if not ok_table or not t:
                out.append(None)
                continue
            try:
                out.append(lou.translateString(tabs, _norm(t), mode=0))
            except Exception:  # noqa: BLE001
                out.append(None)
        conn.send(out)


class LouisPool:
    def __init__(self, spec_dict: dict, known_tables: set[str] | None = None,
                 timeout_s: float = 20.0, max_workers: int = 200, overrides: dict | None = None,
                 chunk: int = 256):
        self.spec = spec_dict
        self.overrides = dict(overrides or {})
        self.chunk = chunk          # texts per message; the timeout applies per message, not per batch
        self.known = known_tables
        self.timeout = timeout_s
        self.max_workers = max_workers
        self.ctx = mp.get_context("spawn")
        self.workers: dict[str, tuple] = {}
        self.cache: dict[tuple[str, str], str | None] = {}
        self.stats = defaultdict(int)

    # ------------------------------------------------------------------ workers
    def _start(self, table: str):
        if len(self.workers) >= self.max_workers:     # evict the least recently used
            old = min(self.workers, key=lambda k: self.workers[k][2])
            self._kill(old)
        parent, child = self.ctx.Pipe()
        p = self.ctx.Process(target=_worker, args=(table, self.overrides.get(table, self.spec), child),
                             daemon=True)
        p.start()
        if not parent.poll(120):
            p.kill()
            raise RuntimeError(f"louis worker for {table} did not start")
        _tag, ok = parent.recv()
        self.workers[table] = (p, parent, time.time(), ok)
        self.stats["started"] += 1
        return self.workers[table]

    def _kill(self, table: str) -> None:
        p, conn, _t, _ok = self.workers.pop(table)
        try:
            conn.send(None)
        except Exception:  # noqa: BLE001
            pass
        p.kill()

    def close(self) -> None:
        for t in list(self.workers):
            self._kill(t)

    # ------------------------------------------------------------------ api
    def translate_many(self, items: list[tuple[str, str]]) -> list[str | None]:
        res: list[str | None] = [None] * len(items)
        todo: dict[str, list[int]] = defaultdict(list)
        for i, (tab, text) in enumerate(items):
            if self.known is not None and tab not in self.known:
                self.stats["unknown_table"] += 1
                continue
            key = (tab, text)
            if key in self.cache:
                res[i] = self.cache[key]
                self.stats["cache_hit"] += 1
            else:
                todo[tab].append(i)
        # rounds of <= self.chunk texts per table; within a round all tables run in parallel
        pending = {tab: list(idxs) for tab, idxs in todo.items()}
        while pending:
            sent = []
            for tab in list(pending):
                idxs, pending[tab] = pending[tab][: self.chunk], pending[tab][self.chunk:]
                if not pending[tab]:
                    del pending[tab]
                w = self.workers.get(tab) or self._start(tab)
                p, conn, _t, ok = w
                self.workers[tab] = (p, conn, time.time(), ok)
                texts = [items[i][1] for i in idxs]
                conn.send(texts)
                sent.append((tab, idxs, texts))
            for tab, idxs, texts in sent:
                p, conn, _t, _ok = self.workers[tab]
                if conn.poll(self.timeout):
                    outs = conn.recv()
                else:                              # hung on some input: restart, answer None
                    self.stats["timeouts"] += 1
                    self._kill(tab)
                    outs = [None] * len(texts)
                for i, t, o in zip(idxs, texts, outs):
                    res[i] = o
                    self.cache[(tab, t)] = o
        if len(self.cache) > 2_000_000:
            self.cache.clear()
        return res
