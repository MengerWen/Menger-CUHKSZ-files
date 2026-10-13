from __future__ import annotations

import importlib
import platform
import sys
from importlib import metadata

from .config import ENVIRONMENT_LOG_PATH, ENVIRONMENT_USED_PATH, REQUIRED_IMPORTS


def check_required_imports() -> list[tuple[str, str]]:
    package_versions: list[tuple[str, str]] = []
    version_name_map = {"sklearn": "scikit-learn"}
    for package_name in REQUIRED_IMPORTS:
        module = importlib.import_module(package_name)
        distribution_name = version_name_map.get(package_name, package_name)
        try:
            version = metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            version = getattr(module, "__version__", "unknown")
        package_versions.append((package_name, version))
    return package_versions


def write_environment_files(package_versions: list[tuple[str, str]]) -> None:
    environment_lines = [
        "# Environment Used",
        "",
        f"- Interpreter: `{sys.executable}`",
        f"- Python version: `{platform.python_version()}`",
        "- Package policy: no packages were installed, upgraded, or removed.",
        "",
        "## Packages Actually Used",
        "",
    ]
    environment_lines.extend([f"- `{name}=={version}`" for name, version in package_versions])
    environment_lines.extend(
        [
            "",
            "## Standard Library Modules Used",
            "",
            "- `ast`",
            "- `dataclasses`",
            "- `io`",
            "- `logging`",
            "- `warnings`",
            "- `pathlib`",
            "- `sys`",
            "- `platform`",
            "- `importlib.metadata`",
            "- `math`",
            "- `textwrap`",
            "- `subprocess`",
        ]
    )
    ENVIRONMENT_USED_PATH.write_text("\n".join(environment_lines) + "\n", encoding="utf-8")

    log_lines = [
        "Environment check completed successfully.",
        f"Interpreter: {sys.executable}",
        f"Python version: {platform.python_version()}",
        "Verified imports:",
    ]
    log_lines.extend([f"- {name}=={version}" for name, version in package_versions])
    ENVIRONMENT_LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def run_environment_check() -> list[tuple[str, str]]:
    package_versions = check_required_imports()
    write_environment_files(package_versions)
    return package_versions
