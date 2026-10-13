from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fma4200_project.environment import run_environment_check


def main() -> None:
    package_versions = run_environment_check()
    print("Environment check completed successfully.")
    for name, version in package_versions:
        print(f"- {name}=={version}")


if __name__ == "__main__":
    main()

