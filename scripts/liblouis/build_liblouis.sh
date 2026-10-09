#!/bin/bash
# Build the patched liblouis 3.38 used by the data builder and the scorer, and the Korean 2024 tables.
#   scripts/liblouis/build_liblouis.sh
# 1. liblouis at $COMMIT + the Korean 2020 tables patch + the five engine patches, in order -> $UBT_WORK/liblouis-src
# 2. build with 4-byte characters and install with the Python binding                      -> $UBT_WORK/liblouis
# 3. the 16 Korean tables with the 22 patches of patches/ko2024/manifest.json, in order    -> $UBT_WORK/ko2024_tables
# Sources, patches and tables are checked against the sha256 values in the manifests.
set -euo pipefail
cd "$(dirname "$0")/../.."
export UBT_WORK=${UBT_WORK:-$PWD/work}
REPO=$PWD
SRC=$UBT_WORK/liblouis-src
PREFIX=$UBT_WORK/liblouis
KO=$UBT_WORK/ko2024_tables
COMMIT=d7501beae07f23a2feb74c249d31166d3b48bb87
ENGINE_PATCHES="no_contract_scope native_roman_scope context_shortform_boundaries notallcaps_match short_mixed_caps"

if [ ! -e "$SRC/configure.ac" ]; then
  mkdir -p "$SRC"
  curl -sSL "https://codeload.github.com/liblouis/liblouis/tar.gz/$COMMIT" | tar -xz -C "$SRC" --strip-components 1
  (cd "$SRC" && git apply --whitespace=nowarn "$REPO/patches/liblouis/korean_2020_tables.patch"
   for p in $ENGINE_PATCHES; do git apply "$REPO/patches/liblouis/engine/$p.patch"; done)
fi

python - "$REPO" "$SRC" <<'PY'
import hashlib, json, sys
repo, src = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
em = json.load(open(f"{repo}/patches/liblouis/engine/manifest.json"))
m = json.load(open(f"{repo}/patches/ko2024/manifest.json"))
bad = [f for f, h in em["final_source_sha256"].items() if sha(f"{src}/{f}") != h]
bad += [f for f, h in m["baseline_table_sha256"].items() if sha(f"{src}/tables/{f}") != h]
bad += [p["file"] for p in m["patches"] if sha(f"{repo}/patches/ko2024/{p['file']}") != p["sha256"]]
sys.exit(f"hash mismatch: {bad}" if bad else None)
PY

(cd "$SRC" && ./autogen.sh && ./configure --enable-ucs4 --prefix="$PREFIX" && make -j8 && make install)
pip install "$SRC/python"

mkdir -p "$KO"
python - "$REPO" "$SRC" "$KO" <<'PY'
import hashlib, json, shutil, subprocess, sys
repo, src, ko = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
m = json.load(open(f"{repo}/patches/ko2024/manifest.json"))
for f in m["baseline_table_sha256"]:
    shutil.copy(f"{src}/tables/{f}", f"{ko}/{f}")
for p in m["patches"]:
    subprocess.run(["patch", "--batch", "--forward", "-s", "-p1", "-d", ko, "-i", f"{repo}/patches/ko2024/{p['file']}"], check=True)
bad = [f for f, h in m["accepted_table_sha256"].items() if sha(f"{ko}/{f}") != h]
sys.exit(f"Korean 2024 table hash mismatch: {bad}" if bad else print(f"{len(m['accepted_table_sha256'])} Korean 2024 tables -> {ko}"))
PY
echo "liblouis: UBT_LOUIS_LIB=$PREFIX/lib/liblouis.so UBT_LOUIS_TABLES=$PREFIX/share/liblouis/tables"
