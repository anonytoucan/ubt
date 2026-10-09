"""data-u1 engine: one pinned liblouis build plus the Korean 2024 table.

- liblouis library and system tables: UBT_LOUIS_LIB / UBT_LOUIS_TABLES (default work/liblouis).
- Korean 2024 table: KO2024_TABLES_DIR / KO2024_TABLE_FILE, copied at build start into <out>/engine/ko2024 under the
  label ko-2024-g2.ctb, so the label resolves everywhere and later edits cannot change a running build.
- latex2nemeth for S4a; the optional ubt.kmath for S4b.

The strict (undefined-character) pass runs on a shadow copy of the tables with every `undefined` rule commented out,
because liblouis emits the `undefined` fallback even in noUndefined mode. A ko-2024 certification flag is honoured only
with a matching certificate. Workers run from an empty directory and the staged dir is searched after the system dir,
so no file can shadow a table.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from ubt.paths import DATASETS, LIBLOUIS, WORK

KO2024_LABEL = "ko-2024-g2.ctb"
NEMETH_LABEL = "nemeth"          # pseudo-table: latex2nemeth(nu_math(latex))
KMATH_LABEL = "ko-math-2024"     # pseudo-table: ubt.kmath (Korean math braille 2024)
PSEUDO_TABLES = (NEMETH_LABEL, KMATH_LABEL)
EXCLUDED_TABLE_GLOBS = ("ko-2020-*",)   # the fork's ko-2020 family

DEFAULT_LOUIS_TABLES = f"{LIBLOUIS}/share/liblouis/tables"
DEFAULT_KO2024_DIR = f"{WORK}/ko2024_tables"
DEFAULT_KO2024_FILE = "ko-2020-g2.ctb"
DEFAULT_JAR = f"{DATASETS}/tools/latex2nemeth/latex2nemeth.jar"

_INCLUDE_RE = re.compile(r"^\s*include\s+(\S+)")
# an active `undefined` rule, optionally with nofor/noback prefixes
_UNDEFINED_RE = re.compile(r"^\s*(?:(?:nofor|noback)\s+)*undefined(?:\s|$)")
STRICT_TRANSFORM = "u1-strict-v1: every active `[nofor|noback] undefined` line commented out"
STRICT_MARK = "#u1-strict# "
CERT_NAME = "ko2024_certificate.json"
CERT_GATES = ("C1", "C2", "C3")
NIKL_SPLIT_COUNTS = {"dev": 251319, "test": 39160}


def strict_c1_problems(cert: dict) -> list[str]:
    """Problems with C1: it must give raw exact counts on every NIKL row for sentence-only input, as u1 re-encodes
    f(text). Checks consistency only, not the measurements."""
    problems = []
    c1 = cert.get("C1")
    if not isinstance(c1, dict):
        return ["C1 must contain unadjusted exact counts"]
    required = {"n": sum(NIKL_SPLIT_COUNTS.values()),
                "exact": sum(NIKL_SPLIT_COUNTS.values()), "excluded": 0}
    if c1.get("metric") != "exact_raw":
        problems.append("C1 metric must be exact_raw (no exception-adjusted score)")
    for key, expected in required.items():
        if type(c1.get(key)) is not int or c1[key] != expected:
            problems.append(f"C1 {key} must be {expected}")
    splits = c1.get("split_counts")
    if (not isinstance(splits, dict) or splits != NIKL_SPLIT_COUNTS
            or any(type(v) is not int for v in splits.values())):
        problems.append(f"C1 split_counts must be {NIKL_SPLIT_COUNTS}")
    if cert.get("input_contract") != "sentence":
        problems.append("u1 currently re-encodes sentence-only inputs; context/options "
                        "certification requires matching builder and verifier support")
    return problems


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class EngineSpec:
    louis_tables: str
    louis_lib: str | None
    louis_version: str
    ko2024_src_dir: str
    ko2024_src_file: str
    stage_dir: str
    cwd_dir: str
    nemeth_jar: str
    ko2024_certified: bool = False      # declared only; check_certificate() decides
    java: str = "java"
    strict_dir: str | None = None       # <out>/engine/strict (shadow tables)

    def strict_dirs(self) -> list[str]:
        """Shadow dirs in the same order as the real LOUIS_TABLEPATH."""
        if not self.strict_dir:
            return []
        return [os.path.join(self.strict_dir, "sys"), os.path.join(self.strict_dir, "ko2024")]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EngineSpec":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def spec_from_config(cfg: dict, out_dir: str) -> EngineSpec:
    ecfg = cfg.get("engine") or {}
    tables = os.environ.get("UBT_LOUIS_TABLES") or ecfg.get("louis_tables") or DEFAULT_LOUIS_TABLES
    lib = os.environ.get("UBT_LOUIS_LIB") or ecfg.get("louis_lib")
    if not lib:
        prefix = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(tables))))
        cand = os.path.join(prefix, "lib", "liblouis.so")
        lib = cand if os.path.isfile(cand) else None
    ko_dir = os.environ.get("KO2024_TABLES_DIR") or ecfg.get("ko2024_tables_dir") or DEFAULT_KO2024_DIR
    ko_file = os.environ.get("KO2024_TABLE_FILE") or ecfg.get("ko2024_table_file") or DEFAULT_KO2024_FILE
    jar = os.environ.get("UBT_LATEX2NEMETH_JAR") or ecfg.get("nemeth_jar") or DEFAULT_JAR
    certified = os.environ.get("KO2024_CERTIFIED", "").strip() in ("1", "true", "yes") \
        or bool(ecfg.get("ko2024_certified", False))
    eng = os.path.join(os.path.abspath(out_dir), "engine")
    return EngineSpec(
        louis_tables=os.path.abspath(tables), louis_lib=os.path.abspath(lib) if lib else None,
        louis_version=str(ecfg.get("louis_version", "3.38.0")),
        ko2024_src_dir=os.path.abspath(ko_dir), ko2024_src_file=ko_file,
        stage_dir=os.path.join(eng, "ko2024"), cwd_dir=os.path.join(eng, "cwd"),
        nemeth_jar=os.path.abspath(jar), ko2024_certified=certified,
        strict_dir=os.path.join(eng, "strict"),
    )


# --------------------------------------------------------------------------- closure
def include_closure(top_path: str, search_dirs: list[str]) -> list[tuple[str, str]]:
    """[(name, path)] of `top_path` and everything it includes, resolved as liblouis does (including file's dir first,
    then `search_dirs`); unresolvable includes raise."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    stack = [(os.path.basename(top_path), top_path)]
    while stack:
        name, path = stack.pop()
        if path in seen:
            continue
        seen.add(path)
        out.append((name, path))
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = _INCLUDE_RE.match(line)
                if not m:
                    continue
                inc = m.group(1)
                cands = [os.path.join(os.path.dirname(path), inc)] + \
                        [os.path.join(d, inc) for d in search_dirs]
                hit = next((c for c in cands if os.path.isfile(c)), None)
                if hit is None:
                    raise FileNotFoundError(f"{path}: cannot resolve include {inc}")
                stack.append((inc, hit))
    return out


def stage_ko2024(spec: EngineSpec) -> dict:
    """Snapshot the ko-2024 table into spec.stage_dir under KO2024_LABEL and return the staging record.

    Only includes inside the source dir are copied; those taken from the system dir are recorded as system deps."""
    src_top = os.path.join(spec.ko2024_src_dir, spec.ko2024_src_file)
    if not os.path.isfile(src_top):
        raise FileNotFoundError(f"ko-2024 table not found: {src_top}")
    closure = include_closure(src_top, [spec.louis_tables])
    if os.path.isdir(spec.stage_dir):
        shutil.rmtree(spec.stage_dir)
    os.makedirs(spec.stage_dir)
    os.makedirs(spec.cwd_dir, exist_ok=True)
    for f in os.listdir(spec.cwd_dir):   # must stay empty
        raise RuntimeError(f"engine cwd dir {spec.cwd_dir} is not empty ({f})")
    staged, system = {}, {}
    src_dir = os.path.realpath(spec.ko2024_src_dir)
    for name, path in closure:
        if os.path.realpath(os.path.dirname(path)) == src_dir:
            dst_name = KO2024_LABEL if path == src_top else name
            shutil.copy2(path, os.path.join(spec.stage_dir, dst_name))
            staged[dst_name] = {"from": path, "sha256": sha256_file(path)}
        else:
            system[name] = {"path": path, "sha256": sha256_file(path)}
    # a system table with the label would shadow the stage
    if os.path.exists(os.path.join(spec.louis_tables, KO2024_LABEL)):
        raise RuntimeError(f"{KO2024_LABEL} exists in the system tables dir")
    cert_src = os.path.join(spec.ko2024_src_dir, CERT_NAME)
    if os.path.isfile(cert_src):   # travels with the data dir (engine/ko2024_certificate.json)
        shutil.copy2(cert_src, os.path.join(os.path.dirname(spec.stage_dir), CERT_NAME))
    return {"label": KO2024_LABEL, "source_dir": spec.ko2024_src_dir,
            "source_file": spec.ko2024_src_file, "stage_dir": spec.stage_dir,
            "certified_declared": spec.ko2024_certified, "staged_files": staged,
            "system_deps": system}


# --------------------------------------------------------------------------- certificate
def check_certificate(spec: EngineSpec, staging: dict, cert_path: str | None = None) -> dict:
    """Verify the ko-2024 certificate against the staged table files; raise if certification is declared but invalid.

    Certificate JSON:
      {"table_file": "<KO2024_TABLE_FILE>",
       "closure_sha256": {"<source file name>": "<sha256>", ...},   # every staged file
       "input_contract": "sentence",
       "C1": {"pass": true, "metric": "exact_raw", "n": 290479, "exact": 290479, "excluded": 0,
              "split_counts": {"dev": 251319, "test": 39160}},
       "C2": {"pass": true, ...}, "C3": {"pass": true, ...},
       "test_informed_edits": false,      # did any table edit look at NIKL test rows?
       ...free-form provenance...}"""
    p = cert_path or os.path.join(spec.ko2024_src_dir, CERT_NAME)
    out = {"declared": bool(spec.ko2024_certified), "path": p, "present": os.path.isfile(p),
           "valid": False, "certified": False, "problems": [], "test_informed_edits": None,
           "certificate": None}
    if out["present"]:
        try:
            with open(p, encoding="utf-8") as fh:
                cert = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            cert = None
            out["problems"].append(f"unreadable: {e}")
        if cert is not None and not isinstance(cert, dict):
            out["problems"].append("certificate must be a JSON object")
            cert = None
        if cert is not None:
            out["certificate"] = cert
            if cert.get("table_file") != spec.ko2024_src_file:
                out["problems"].append(f"table_file {cert.get('table_file')!r} != {spec.ko2024_src_file!r}")
            want = {os.path.basename(v["from"]): v["sha256"]
                    for v in staging["staged_files"].values()}
            got = cert.get("closure_sha256") or {}
            if got != want:
                out["problems"].append(
                    "closure_sha256 != staged files "
                    f"(missing {sorted(set(want) - set(got))}, extra {sorted(set(got) - set(want))}, "
                    f"differ {sorted(k for k in set(want) & set(got) if want[k] != got[k])})")
            for g in CERT_GATES:
                if not (isinstance(cert.get(g), dict) and cert[g].get("pass") is True):
                    out["problems"].append(f"{g} not passed")
            out["problems"].extend(strict_c1_problems(cert))
            ti = cert.get("test_informed_edits")
            if not isinstance(ti, bool):
                out["problems"].append("test_informed_edits must be true/false")
            out["test_informed_edits"] = ti
    else:
        out["problems"].append("no certificate")
    out["valid"] = out["present"] and not out["problems"]
    out["certified"] = out["declared"] and out["valid"]
    if out["declared"] and not out["valid"]:
        raise RuntimeError(f"ko-2024 certification declared (KO2024_CERTIFIED) but the "
                           f"certificate is not valid: {out['problems']} ({p})")
    return out


# --------------------------------------------------------------------------- strict shadow
def _strict_copy(src: str, dst: str) -> bool:
    """Copy one table file with active `undefined` rules commented out; True if any was."""
    with open(src, "rb") as fh:
        raw = fh.read()
    text = raw.decode("utf-8", errors="surrogateescape")
    changed = False
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if _UNDEFINED_RE.match(ln):
            lines[i] = STRICT_MARK + ln
            changed = True
    with open(dst, "wb") as fh:
        fh.write("\n".join(lines).encode("utf-8", errors="surrogateescape") if changed else raw)
    shutil.copystat(src, dst)
    return changed


def has_active_undefined(path: str) -> bool:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return any(_UNDEFINED_RE.match(ln) for ln in fh)


def build_strict_shadow(spec: EngineSpec, dest: str | None = None) -> dict:
    """Strict copies of the whole system tables tree (<dest>/sys) and of the staged ko-2024 dir (<dest>/ko2024).
    Call before any translator exists, then check_strict_shadow on the used tables."""
    dest = dest or spec.strict_dir
    if not dest:
        raise ValueError("no strict dir")
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    changed: list[str] = []
    for tag, root in (("sys", spec.louis_tables), ("ko2024", spec.stage_dir)):
        for dirpath, _dirs, files in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            os.makedirs(os.path.join(dest, tag, rel), exist_ok=True)
            for f in files:
                if _strict_copy(os.path.join(dirpath, f), os.path.join(dest, tag, rel, f)):
                    changed.append(f"{tag}:{os.path.normpath(os.path.join(rel, f))}")
    return {"transform": STRICT_TRANSFORM, "files_changed": sorted(changed)}


def check_strict_shadow(spec: EngineSpec, table_ids: list[str], dest: str | None = None) -> dict:
    """Assert that each used table's strict closure, resolved as under the real LOUIS_TABLEPATH, has the real include
    structure and reaches no active `undefined` rule; report the used tables whose real closure has one."""
    dest = dest or spec.strict_dir
    search = [spec.louis_tables, spec.stage_dir]
    strict_dirs = [os.path.join(dest, "sys"), os.path.join(dest, "ko2024")]
    with_undef = []
    for tid in sorted(set(table_ids)):
        if tid in PSEUDO_TABLES:
            continue
        real = include_closure(table_path(spec, tid), search)
        top = next(os.path.join(d, tid) for d in strict_dirs if os.path.isfile(os.path.join(d, tid)))
        strict = include_closure(top, search)
        if [n for n, _p in real] != [n for n, _p in strict]:
            raise RuntimeError(f"strict shadow of {tid} resolves a different include structure")
        leak = [p for _n, p in strict if has_active_undefined(p)]
        if leak:
            raise RuntimeError(f"strict closure of {tid} still reaches an active `undefined` "
                               f"rule: {leak}")
        if any(has_active_undefined(p) for _n, p in real):
            with_undef.append(tid)
    return {"used_tables_with_undefined_rule": with_undef}


# --------------------------------------------------------------------------- runtime
def apply_env(spec: EngineSpec) -> None:
    """Pin the liblouis build, tables dir and ubt.kmath's Korean table for this process and its children, so S4b
    prose and S1/S7 Korean share the staged ko-2024 file."""
    os.environ["UBT_LOUIS_TABLES"] = spec.louis_tables
    if spec.louis_lib:
        os.environ["UBT_LOUIS_LIB"] = spec.louis_lib
    os.environ["KMATH_TABLE_DIR"] = spec.stage_dir
    os.environ["KMATH_TABLE_NAME"] = KO2024_LABEL


def _mapped_louis_libs() -> set[str]:
    try:
        with open("/proc/self/maps") as fh:
            return {os.path.realpath(ln.split()[-1]) for ln in fh
                    if "liblouis" in ln and ln.split()[-1].startswith("/")}
    except OSError:
        return set()


def make_translator(spec: EngineSpec):
    """ForwardTranslator bound to this engine, ignoring an inherited LOUIS_TABLEPATH; checks the liblouis version and
    that the mapped library file is the pinned one."""
    apply_env(spec)
    from ubt.translate import ForwardTranslator  # noqa: PLC0415
    sd = spec.strict_dirs()
    if sd and not all(os.path.isdir(d) for d in sd):
        raise RuntimeError(f"strict shadow dirs missing: {sd} (build_strict_shadow first)")
    tr = ForwardTranslator(extra_table_dirs=[], fallback_dirs=[spec.stage_dir],
                           inherit_env=False, strict_dirs=sd)
    v = tr.louis_version()
    if v != spec.louis_version:
        raise RuntimeError(f"liblouis version {v} != pinned {spec.louis_version} "
                           f"(lib={spec.louis_lib})")
    if spec.louis_lib:
        mapped = _mapped_louis_libs()
        want = os.path.realpath(spec.louis_lib)
        if mapped and mapped != {want}:
            raise RuntimeError(f"mapped liblouis {sorted(mapped)} != pinned {want}")
    return tr


def init_worker(spec_dict: dict, quiet: bool = True) -> None:
    """Pool initializer: one translator per process, installed as ubt.worker._TR for ubt.worker.process_plan."""
    spec = EngineSpec.from_dict(spec_dict)
    if quiet:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
    os.chdir(spec.cwd_dir)
    import ubt.worker as w  # noqa: PLC0415
    w._TR = make_translator(spec)


def table_path(spec: EngineSpec, table_id: str) -> str:
    if table_id == KO2024_LABEL:
        return os.path.join(spec.stage_dir, KO2024_LABEL)
    return os.path.join(spec.louis_tables, table_id)


def fingerprint(spec: EngineSpec, table_ids: list[str], extra: dict | None = None) -> tuple[str, dict]:
    """(engine_id, map): sha256 of the liblouis library, the used tables' include closures (keyed by relative path),
    the jar and the math normaliser; engine_id = 'u1e-' + 16 hex of the path-free part."""
    search = [spec.louis_tables, spec.stage_dir]
    cache: dict[str, str] = {}

    def key(p: str) -> str:
        rp = os.path.realpath(p)
        for tag, root in (("ko2024", spec.stage_dir), ("sys", spec.louis_tables)):
            r = os.path.realpath(root)
            if rp.startswith(r + os.sep):
                return f"{tag}:{os.path.relpath(rp, r)}"
        return f"abs:{rp}"

    tables: dict[str, dict[str, str]] = {}
    for tid in sorted(set(table_ids)):
        if tid in PSEUDO_TABLES:
            continue
        clo = include_closure(table_path(spec, tid), search)
        tables[tid] = {}
        for _n, p in clo:
            if p not in cache:
                cache[p] = sha256_file(p)
            tables[tid][key(p)] = cache[p]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ident = {
        "louis_version": spec.louis_version,
        "louis_lib_sha256": sha256_file(os.path.realpath(spec.louis_lib)) if spec.louis_lib else None,
        # every translation uses the display table, so it is pinned on its own
        "display_table_sha256": sha256_file(os.path.join(spec.louis_tables, "unicode.dis")),
        "strict_shadow": STRICT_TRANSFORM,
        "ko2024_label": KO2024_LABEL,
        "ko2024_certified": spec.ko2024_certified,
        "tables": tables,
        "nemeth_jar_sha256": sha256_file(spec.nemeth_jar) if os.path.isfile(spec.nemeth_jar) else None,
        "java_version": java_version(spec.java),
        "nu_math_sha256": sha256_file(os.path.join(here, "math_norm.py")),
        # f_nemeth = normalize_nemeth(latex2nemeth(nu_math(.))) and its embedding: the code is pinned
        "nemeth_module_sha256": sha256_file(os.path.join(here, "u1", "nemeth.py")),
        "nemeth_normaliser": "strip; whitespace-run -> U+2800; U+2800-block only (u1 v1)",
        **(extra or {}),
    }
    blob = json.dumps(ident, sort_keys=True, ensure_ascii=False).encode()
    eid = "u1e-" + hashlib.sha256(blob).hexdigest()[:16]
    return eid, {**ident, "louis_lib": spec.louis_lib, "louis_tables_dir": spec.louis_tables,
                 "nemeth_jar": spec.nemeth_jar}


_JAVA_VERSION: dict[str, str | None] = {}


def java_version(java: str = "java") -> str | None:
    """First line of `java -version` (the JVM that runs latex2nemeth)."""
    if java not in _JAVA_VERSION:
        try:
            r = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=60)
            lines = (r.stderr or r.stdout).strip().splitlines()
            _JAVA_VERSION[java] = " | ".join(ln.strip() for ln in lines[:2]) or None
        except (OSError, subprocess.TimeoutExpired):
            _JAVA_VERSION[java] = None
    return _JAVA_VERSION[java]


def kmath_status() -> dict:
    """{available, error?} of the optional Korean math engine; the error is for the manifest, never the engine id."""
    try:
        import ubt.kmath  # noqa: F401, PLC0415
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": f"{type(e).__name__}: {e}"}
    from ubt.u1.kmath_adapter import load_kmath  # noqa: PLC0415
    km, why = load_kmath()
    return {"available": km is not None, **({"error": why} if km is None else {})}


def kmath_fingerprint() -> dict:
    """Engine-id part of the optional Korean math engine: hashes of the ubt.kmath tree and ubt/u1/kmath_adapter.py,
    or only {"available": False}, so an import error message cannot change the id."""
    if not kmath_status()["available"]:
        return {"available": False}
    import ubt.kmath as km  # noqa: PLC0415
    root = os.path.dirname(os.path.abspath(km.__file__))
    files = sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True))
    h = hashlib.sha256()
    for f in files:
        if os.path.isfile(f) and not f.endswith(".pyc") and "__pycache__" not in f:
            h.update(os.path.relpath(f, root).encode())
            h.update(sha256_file(f).encode())
    here = os.path.dirname(os.path.abspath(__file__))
    return {"available": True, "version": getattr(km, "__version__", None),
            "tree_sha256": h.hexdigest(),
            "adapter_sha256": sha256_file(os.path.join(here, "kmath_adapter.py"))}


def preload_kmath() -> None:
    """Import ubt.kmath and its submodules before the pool forks, so workers never load a tree edited later."""
    if kmath_status()["available"]:
        import importlib  # noqa: PLC0415
        import pkgutil  # noqa: PLC0415
        import ubt.kmath as km  # noqa: PLC0415
        for m in pkgutil.walk_packages(km.__path__, km.__name__ + "."):
            try:
                importlib.import_module(m.name)
            except Exception:  # noqa: BLE001
                pass
