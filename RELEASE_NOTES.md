# Release notes

## 1.0.0 — camera-ready candidate

This candidate extends the August review release with the October rerun. A publication date, release tag and archival identifier are not assigned here.

- Adds 17 frozen configuration-run files, including the default configuration's third run, without changing the original 14 data files.
- Adds the October step analyses, a single offline verification/build entry point, 13 reference CSVs, and three figures plus six table PDFs under separate October paths.
- Adds input hashes, an October output manifest, citation metadata, source attribution, an MIT licence for original code, and local-file exclusions.
- Records model version, backend fingerprint and verdict-token log probabilities in the October data. The API runner defaults to a new-data directory and protects frozen inputs.
- Preserves the original canonical builder, all 19 canonical CSVs, exploratory notebooks and August figures.
- Includes the historical external style-control report and a preparation helper with a separate upstream reproduction procedure. The external estimator is outside the offline verification and was not rerun for this candidate.

The October data contain 500 prompt IDs, 8,500 paired records and 17,000 judgments. Four judgments have unparsable verdicts; 496 prompts are complete across all 17 configuration-runs. All judgments record the pinned model `gpt-4.1-2025-04-14`; 37 backend fingerprint values are observed. Sample sizes depend on each analysis's required runs.

Use `make verify` for data integrity and numerical checks, `make all` for both sets of tables and figures, and `make test` for release-interface checks. Reproduction uses the supplied frozen records and does not call model APIs. The original style-feature extractor remains unavailable; upstream leaderboard fitting is outside the offline build.

## August review release

- `8004e0567ad1e8ab9d81588923c75480200c07cf` — initial anonymous code release, authored 29 August and committed 30 August 2026.
- `4bcdb6243d44148660b26876859336a743e322ea` — paper-facing exploratory analyses and Figure 3, 30 August 2026.

Those commits identify the original review artefacts. The October files are an additional experiment wave, with distinct output paths and a separate manifest.
