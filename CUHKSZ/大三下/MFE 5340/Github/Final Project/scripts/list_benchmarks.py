"""List benchmark model specifications for later experiments."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.benchmarks import benchmark_frame


def main() -> None:
    print(benchmark_frame().to_string(index=False))


if __name__ == "__main__":
    main()
