from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fma4200_project.submission_audit import run_audit_and_prepare_submission


def main() -> None:
    result = run_audit_and_prepare_submission()
    print("Audit and submission preparation completed successfully.")
    print(f"Rubric checklist: {result['rubric_checklist_path']}")
    print(f"Runbook: {result['runbook_path']}")
    print(f"Audit report: {result['audit_report_path']}")
    print(f"Submission folder: {result['submission_ready_dir']}")
    print(f"Submission zip: {result['submission_zip_path']}")
    print(f"Broken local Markdown links: {result['broken_link_count']}")
    print(f"Main-body word count: {result['main_body_word_count']}")
    print(f"PDF page heuristic: {result['pdf_page_heuristic']}")


if __name__ == "__main__":
    main()
