# Audit Report

## Strict-Grader Summary

The project was audited against every requirement in `Guidance.md` and rebuilt before preparing the submission bundle. The current final package includes a Markdown report, notebook report, HTML export, PDF report, executable code, processed data, and saved outputs.

## What Was Fixed During This Audit

- Fixed a reproducibility bug in the final report builder by replacing the hardcoded rolling cointegration-rank share with a value computed from `output/tables/trading_strategies/stat_arb_signals.csv`.
- Restored the PDF deliverable by validating a working headless Chromium-based export path and by preserving the generated `report/final_report.pdf` in subsequent rebuilds and submission packaging.
- Cleaned the bibliography format in `report/references.md` and moved verification URLs into `report/references_verification.md` so the report reads like a submission rather than an internal research note.
- Verified that the Markdown report, appendix, README, and export notes have no broken local links after the rebuild.
- Prepared a clean `submission_ready/` folder and `submission_ready.zip` that exclude stale legacy artifacts such as the old `outputs/` directory.

## Verification Checks

- Final report source exists: `True`
- Final report PDF exists: `True`
- Final appendices exist: `True`
- Clean processed percent dataset exists: `True`
- Clean processed decimal dataset exists: `True`
- Data dictionary exists: `True`
- Markdown local-link checks run: `6`
- Broken local Markdown links found: `0`
- Main-body word count in `report/final_report.md`: `3030`
- PDF page-object heuristic count for `report/final_report.pdf`: `16`

## Residual Risks

- The final PDF is present and included in the bundle, but browser-driven PDF refresh can still be sensitive to sandbox restrictions on some reruns.
