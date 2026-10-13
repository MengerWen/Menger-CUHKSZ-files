"""Stage 2 evaluation entry point.

This lists planned metrics as a pandas DataFrame; later stages will pass model
outputs into the metric functions in `src.evaluation.metrics`.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import metric_catalog


def main() -> None:
    print(metric_catalog().to_string(index=False))


if __name__ == "__main__":
    main()
