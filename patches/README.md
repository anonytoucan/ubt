# liblouis patches

The data builder, the forward-consistency checks and the scorer use liblouis 3.38 (4-byte characters) built from the
upstream commit `d7501beae07f23a2feb74c249d31166d3b48bb87` with three groups of patches.
`scripts/liblouis/build_liblouis.sh` applies them, installs the library under `work/liblouis`, writes the Korean 2024
tables to `work/ko2024_tables`, and checks every source file, patch and table against the sha256 recorded in the
manifests.

| patches | applied to | content |
|---|---|---|
| `liblouis/korean_2020_tables.patch` | the upstream source tree (`git apply`) | Korean grade-2 tables of the 2020 standard (`ko-2020-*.cti`, `ko-2020-g2.ctb`), the renamed 1997 table, YAML tests |
| `liblouis/engine/*.patch` | the same tree, in the order `no_contract_scope`, `native_roman_scope`, `context_shortform_boundaries`, `notallcaps_match`, `short_mixed_caps` | translator and table-compiler features used by the Korean 2024 tables (below); hashes in `liblouis/engine/manifest.json` |
| `ko2024/*.patch` | a copy of the 16 Korean tables of that tree, in the order of `ko2024/manifest.json` | the rules of the Korean Braille Standard 2024 on top of the 2020 tables |

## Engine features

All are opt-in: a table that does not declare them translates as before.

- `no_contract` typeforms are kept at character scope, separate from the word-level numeric and contraction state.
- Sentence-wide capitals analysis for tables that request it: caseless letters stop capitals words and passages,
  `endcapsphrase afterletters` ends a passage before punctuation, digits or signs, and `capsphrasevar <n>` exposes the
  computed passage (including its end) to context rules.
- Consuming context rules may follow a zero-width context opener; `contractioncasedword` bounds shortforms by the cased
  word.
- The `notallcaps` match modifier skips the marked match rule when every matched character is uppercase.
- `capsminpartialword <minimum> <wordlimit>`: a short uppercase run ending inside a mixed-case word takes individual
  capital indicators only when the whole cased word fits the limit (the Korean table selects `3 3`).

Translation remains one liblouis call on the full sentence. Build from a clean tree (`internal.h` changes).
`liblouis/engine/no_contract_scope.yaml` holds the YAML cases of the first patch (`lou_checkyaml`, with `LOUIS_TABLEPATH`
set to the source tree).

## Korean 2024 tables

`configs/unified_u1.yaml` exposes `work/ko2024_tables/ko-2020-g2.ctb` under the label `ko-2024-g2.ctb`. The tables were
measured on the dev split of the NIKL Korean-Braille parallel corpus (licence-bound, not distributed) and on the example
sentences of the standard (`resources/korean_braille_2024/examples.jsonl`); the results are in `ko2024/manifest.json`
(`measurement`). `scripts/liblouis/ko_table_regress.py` repeats the measurement and
`tests/test_ko2024_table_regressions.py` runs the fixtures of `ko2024/regression_cases.json` (set
`KO2024_TABLES_DIR=work/ko2024_tables`).

`resources/korean_braille_2024/examples.jsonl` and `examples_math.jsonl` hold the example sentences of the Korean
Braille Standard 2024 and of its guide 「(주요 개정 내용을 담은) 한국 점자 규정 안내서」 (National Institute of Korean
Language, KOGL Type 1), transcribed page by page. The data builder registers their texts as evaluation-only.
