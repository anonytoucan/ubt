"""Forward-only LibLouis translation.

LOUIS_TABLEPATH must include the system tables directory: once it is set,
liblouis does not fall back to its compiled-in path and `unicode.dis` is lost.
liblouis is not thread-safe; use one ForwardTranslator per process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_SYSTEM_TABLE_CANDIDATES = (
    "/usr/local/share/liblouis/tables",
    "/usr/share/liblouis/tables",
    "/opt/conda/share/liblouis/tables",
)

DISPLAY_TABLE = "unicode.dis"


def discover_system_tables_dir() -> str:
    """Locate the liblouis tables directory (must contain unicode.dis).
    `UBT_LOUIS_TABLES` pins it, e.g. to the 3.38 fork at work/liblouis."""
    pinned = os.environ.get("UBT_LOUIS_TABLES", "").strip()
    if pinned:
        pinned = os.path.expanduser(pinned)
        if not os.path.isfile(os.path.join(pinned, DISPLAY_TABLE)):
            raise RuntimeError(
                f"UBT_LOUIS_TABLES={pinned!r} does not contain {DISPLAY_TABLE}")
        return pinned
    for cand in _SYSTEM_TABLE_CANDIDATES:
        if os.path.isfile(os.path.join(cand, DISPLAY_TABLE)):
            return cand
    raise RuntimeError(
        "Could not find the system liblouis tables directory "
        f"(looked in {_SYSTEM_TABLE_CANDIDATES}); is liblouis installed? "
        "Set UBT_LOUIS_TABLES to pin one."
    )


def louis_lib_path() -> str | None:
    """`UBT_LOUIS_LIB`, else <prefix>/lib/liblouis.so for pinned tables at
    <prefix>/share/liblouis/tables, else None (the binding finds its own)."""
    lib = os.environ.get("UBT_LOUIS_LIB", "").strip()
    if lib:
        return os.path.expanduser(lib)
    tables = os.environ.get("UBT_LOUIS_TABLES", "").strip()
    if tables:
        prefix = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.normpath(os.path.expanduser(tables)))))
        cand = os.path.join(prefix, "lib", "liblouis.so")
        if os.path.isfile(cand):
            return cand
    return None


_PRELOADED: str | None = None


def preload_louis_lib() -> str | None:
    """dlopen the pinned liblouis before `import louis` so the binding uses it
    without LD_LIBRARY_PATH. Idempotent; returns the path used."""
    global _PRELOADED
    lib = louis_lib_path()
    if lib is None or _PRELOADED == lib:
        return _PRELOADED
    import ctypes  # noqa: PLC0415
    import sys  # noqa: PLC0415
    if "louis" in sys.modules and _PRELOADED is None:
        # too late to redirect the binding in this process
        return None
    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
    _PRELOADED = lib
    return lib


FALLBACK_DIRS_ENV = "UBT_TABLE_FALLBACK_DIRS"


def build_table_path(extra_table_dirs: list[str] | None,
                     fallback_dirs: list[str] | None = None,
                     inherit_env: bool = True) -> str:
    """LOUIS_TABLEPATH: extra dirs, the inherited value, the system tables dir, then
    `fallback_dirs` and, with inherit_env, $UBT_TABLE_FALLBACK_DIRS (e.g. <data_u1>/engine/ko2024)."""
    prev = os.environ.get("LOUIS_TABLEPATH", "") if inherit_env else ""
    extra = [os.path.expanduser(d) for d in (extra_table_dirs or []) if d]
    tail = [os.path.expanduser(d) for d in (fallback_dirs or []) if d]
    if inherit_env:
        env_tail = os.environ.get(FALLBACK_DIRS_ENV, "")
        tail += [os.path.expanduser(d) for d in env_tail.replace(os.pathsep, ",").split(",")
                 if d.strip()]
    parts = extra + ([prev] if prev else []) + [discover_system_tables_dir()] + tail
    seen: set[str] = set()
    unique = [p for p in parts if not (p in seen or seen.add(p))]
    return ",".join(unique)


@dataclass(frozen=True)
class TranslationResult:
    ok: bool
    braille: str
    reason: str  # "" | "empty_input" | "exception" | "undefined_char"


class ForwardTranslator:
    """Forward (text -> braille) translation; data generation is forward-only."""

    def __init__(self, extra_table_dirs: list[str] | None = None,
                 fallback_dirs: list[str] | None = None,
                 inherit_env: bool = True,
                 strict_dirs: list[str] | None = None):
        """`strict_dirs`: shadow tables without `undefined` rules for the noUndefined
        pass, since liblouis emits a table's `undefined` dots even in that mode."""
        os.environ["LOUIS_TABLEPATH"] = build_table_path(
            extra_table_dirs, fallback_dirs, inherit_env)
        preload_louis_lib()
        # Import must happen after LOUIS_TABLEPATH is set.
        import louis  # noqa: PLC0415

        self._louis = louis
        self._no_undefined = louis.noUndefined
        self._strict_dirs = [os.path.abspath(d) for d in (strict_dirs or []) if d]
        self._strict_cache: dict[str, str] = {}

    def strict_table(self, table: str) -> str:
        """The table spec used for the noUndefined pass."""
        if not self._strict_dirs:
            return table
        hit = self._strict_cache.get(table)
        if hit is None:
            hit = next((os.path.join(d, table) for d in self._strict_dirs
                        if os.path.isfile(os.path.join(d, table))), None)
            if hit is None:
                raise RuntimeError(f"no strict shadow for table {table!r} in {self._strict_dirs}")
            self._strict_cache[table] = hit
        return hit

    def louis_version(self) -> str:
        return self._louis.version()

    def translate(self, table: str, text: str, check_undefined: bool = True) -> TranslationResult:
        """Translate `text` whole (never segmented) through `table`. With check_undefined,
        a differing noUndefined pass flags characters the table does not define."""
        if not text:
            return TranslationResult(False, "", "empty_input")
        tables = [DISPLAY_TABLE, table]
        try:
            braille = self._louis.translateString(tables, text, mode=0)
        except Exception:
            return TranslationResult(False, "", "exception")
        if check_undefined:
            try:
                strict = self._louis.translateString(
                    [DISPLAY_TABLE, self.strict_table(table)], text, mode=self._no_undefined)
            except Exception:
                return TranslationResult(False, "", "exception")
            if strict != braille:
                return TranslationResult(False, braille, "undefined_char")
        return TranslationResult(True, braille, "")
