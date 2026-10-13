from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fma4200_project.final_report_builder import build_final_report


def main() -> None:
    result = build_final_report()
    print("Final report build completed successfully.")
    print(f"Markdown report: {result['final_report_md']}")
    print(f"Notebook report: {result['final_report_ipynb']}")
    print(f"HTML report: {result['final_report_html']}")
    print(f"PDF report: {result['final_report_pdf']}")
    print(f"Appendices: {result['final_appendices']}")
    print(f"Export notes: {result['export_notes']}")
    print(f"Automatic PDF supported: {result['pdf_supported']}")


if __name__ == "__main__":
    main()
