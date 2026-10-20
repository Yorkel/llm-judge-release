# Camera-ready release validation

The October 2026 candidate was validated in a newly created CPython 3.12.13 environment on macOS arm64. The unchanged `requirements.txt` was installed from cached packages; dependency consistency passed for all 131 installed distributions. Exact versions are recorded in [validation/environment.json](validation/environment.json).

## Completed checks

| Check | Result |
| --- | --- |
| Frozen inputs | All 31 JSONL files match their SHA-256, byte counts and record counts in `data/manifest.csv`. |
| October record integrity | All 17 configuration-run files contain the same 500 unique prompt IDs; cell IDs, stored verdicts, parse flags and served-model metadata agree. Four unparsed judgments remain preserved. |
| October numerical rebuild | All 13 reference CSVs reproduced offline with relative and absolute tolerance `1e-10`; the complete run took 378.53 seconds. All 13 shipped CSV hashes were unchanged afterward. |
| Original numerical rebuild | All 19 canonical CSVs reproduced using the retained August builder; all shipped CSV hashes were unchanged afterward. |
| October figures and tables | Three figures and six tables generated successfully as PDF and PNG. All nine PDF texts match the source exports. The three figure PNGs and four table PNGs are byte-identical to the source exports; the other two tables differ only in spanning-header rule layout. |
| Original figures | All three figures regenerated successfully. Figures 2 and 3 render identically to the frozen PDFs. Figure 1 has a roughly 0.28-point annotation/layout difference with unchanged text and values; the original three PDF files are retained byte-for-byte. |
| Offline tests | All 11 tests passed: five runner import/CLI/path-protection checks and six builder integrity/reference/export failure-path checks. |
| External preparation helper | Help makes no writes; a temporary fixture produced four sets of 500 derived rows with correct links, leaving upstream source records unchanged. |
| Citation and documentation | `CITATION.cff` passes the official CFF 1.2 schema. Local documentation links resolve; original-code MIT terms and upstream attribution are included. |
| Preservation | The original 14 JSONL files, all 19 canonical CSVs and all three original figure PDFs are unchanged from review commit `4bcdb6243d44148660b26876859336a743e322ea`. |

## Repeat the checks

After installing the pinned requirements as described in the [README](README.md), check the downloaded files before regenerating figures:

```bash
shasum -a 256 -c SHA256SUMS
python analysis/validate_release.py
make test
make verify
```

`make verify` recomputes both waves' numerical outputs. The October component takes about six minutes on the validation machine. To regenerate the paper figures and tables, run `make figures`, or `make all` to rebuild both tables and figures. Set `PYTHON=...` for an explicit interpreter.

`SHA256SUMS` identifies every included release file except itself. PDF metadata and rendering can vary with the runtime, so regenerated figure files need not retain their downloaded hashes. Numerical verification compares the result tables separately.

## Scope of the validation

- No model API requests were made. These checks reproduce calculations from the frozen observations; future model responses may differ.
- The original style-feature extractor is unavailable. Its frozen output remains an input to the original downstream analysis.
- The historical external Arena-Hard style-control estimator was not rerun. Its captured report, separate preparation procedure and interval settings are described in [docs/EXTERNAL_STYLE_CHECK.md](docs/EXTERNAL_STYLE_CHECK.md).
- The full build was tested on the environment above. Cross-platform figure rendering was not tested.
- The camera-ready manuscript is maintained separately and is not modified or included in this repository update.
