from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from funding_arb.utils.config import load_config



def test_load_config_reads_yaml_mapping() -> None:
    fixtures_dir = Path("tests/fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    config_path = fixtures_dir / f"_tmp_{uuid4().hex}.yaml"

    try:
        config_path.write_text("section:\n  value: 42\n", encoding="utf-8")
        loaded = load_config(config_path)
        assert loaded["section"]["value"] == 42
    finally:
        if config_path.exists():
            config_path.unlink()