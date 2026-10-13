from __future__ import annotations

from funding_arb.utils.paths import REPO_ROOT, repo_path



def test_repo_root_contains_agents_file() -> None:
    assert (REPO_ROOT / "AGENTS.md").exists()



def test_repo_path_resolves_from_root() -> None:
    assert repo_path("configs", "data", "default.yaml").exists()