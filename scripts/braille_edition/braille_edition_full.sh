#!/bin/bash
# Build the paper's braille edition: transcribe the main sections to UEB grade 2, read them back with the final model
# (best-of-50, unverified sentences again at n = 100), render pages, update appendix numbers, compile, copy the .brf.
#   UBT_PAPER=<paper sources> scripts/braille_edition/braille_edition_full.sh
# Output: $UBT_WORK/braille_edition/run_<stamp>/, the paper's figures/braille_edition*, and
# $UBT_WORK/braille_edition/deliver/UBT_braille_edition_<stamp>.brf
set -uo pipefail
cd "$(dirname "$0")/../.."
export UBT_WORK=${UBT_WORK:-$PWD/work} UBT_DATASETS=${UBT_DATASETS:-$PWD/datasets}
PAPER=${UBT_PAPER:-$PWD/paper}
STAMP=$(date -u +%m%d_%H%M)
D=$UBT_WORK/braille_edition/run_$STAMP
DELIVER=$UBT_WORK/braille_edition/deliver
TECTONIC=${TECTONIC:-$UBT_WORK/tools/tectonic}
log() { echo "[$(date -u +%T)] $*"; }
fail() { log "FAILED: $*"; exit 1; }

mkdir -p "$D"
log "build from $PAPER"
python scripts/braille_edition/braille_edition.py build --paper-dir "$PAPER" --out "$D" > "$D/build.log" 2>&1 || fail "build ($D/build.log)"
grep -nE '\\(padded|pdeleted|preplaced|added|gadded|greplaced|pcell|gcell|ref|cite)' "$D/edition.text.txt" > "$D/markup_left.txt" \
  && log "warning: LaTeX markup left in the edition text ($D/markup_left.txt)"

log "read back with the final model (waits for free GPUs)"
D=$D scripts/braille_edition/braille_edition_chain.sh > "$D/chain.log" 2>&1
grep -q "\] done" "$D/chain.log" || fail "chain ($D/chain.log)"

log "pages"
cp "$D/braille_edition_all.tex" "$PAPER/figures/braille_edition_all.tex"
TECTONIC=$TECTONIC python scripts/braille_edition/braille_edition_pages.py --tikz "$PAPER/figures/braille_edition_all.tex" \
    --out "$PAPER/figures/braille_edition" > "$D/pages.log" 2>&1 || fail "pages ($D/pages.log)"

log "appendix numbers"
python scripts/braille_edition/braille_edition_update.py --run "$D" --paper "$PAPER" > "$D/update.log" 2>&1 || fail "update ($D/update.log)"

log "compile check"
B=$D/paper_build
mkdir -p "$B"
(cd "$PAPER" && timeout 1500 "$TECTONIC" -X compile main.tex --outdir "$B" --keep-logs > "$B/tectonic.out" 2>&1) || fail "compile ($B/tectonic.out)"
grep -qE "(Reference|Citation) .* undefined" "$B/main.log" && fail "undefined references ($B/main.log)"

mkdir -p "$DELIVER"
OUT=$DELIVER/UBT_braille_edition_$STAMP.brf
cp "$D/edition.brf" "$OUT"
log "done: $OUT"
