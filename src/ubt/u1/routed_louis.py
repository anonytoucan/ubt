"""Per-table routed, isolated f for scoring data_u1_v2 (eval harness, BoN, bCER, RLVR checks).

A drop-in for the `louis` argument of ubt.eval_harness / ubt.u1.reencode. Each table runs in its own spawned
process under its own engine (manifest `engine_spec` or `engine_spec_overrides[table]`), because liblouis output
depends on previously compiled tables and ko-2024-g2.ctb needs a patched liblouis that changes other tables.

  louis = RoutedLouis("datasets/data_u1_v2")
  r = louis.translate("ko-2024-g2.ctb", "안녕하세요")      # r.ok, r.braille
  louis.close()
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class RoutedResult:
    braille: str
    ok: bool


class RoutedLouis:
    def __init__(self, data_dir: str, timeout_s: float = 20.0, max_workers: int = 200):
        from ubt.rlvr.louis_pool import LouisPool  # noqa: PLC0415
        man = json.load(open(os.path.join(data_dir, "manifest.json")))
        inv = json.load(open(os.path.join(data_dir, "inventory.json")))
        self.engine_id = man.get("engine_id")
        self.overrides = sorted((man.get("engine_spec_overrides") or {}).keys())
        self.pool = LouisPool(man["engine_spec"], known_tables={t["table_id"] for t in inv["tables"]},
                              timeout_s=timeout_s, max_workers=max_workers,
                              overrides=man.get("engine_spec_overrides"))

    def translate(self, table: str, text: str, check_undefined: bool = False) -> RoutedResult:
        b = self.pool.translate_many([(table, text)])[0]
        return RoutedResult(braille=b or "", ok=b is not None)

    def translate_many(self, items: list[tuple[str, str]]) -> list[str | None]:
        return self.pool.translate_many(items)

    def close(self) -> None:
        self.pool.close()
