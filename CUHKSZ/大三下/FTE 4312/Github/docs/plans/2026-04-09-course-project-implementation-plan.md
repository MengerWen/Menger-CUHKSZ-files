# Course Project Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a clean course-project monorepo that can grow from initial scaffolding into a reproducible quantitative research pipeline, a cost-aware backtest engine, a Solidity vault prototype, and a lightweight demo frontend.

**Architecture:** The project will stay monorepo-style with one Python package for research and backtesting, one Foundry workspace for Solidity contracts, and one lightweight Vite frontend for presentation. The first milestone is a usable scaffold with configs, docs, CLI entry points, and placeholder implementations; later milestones fill in each module incrementally without introducing backend-heavy infrastructure.

**Tech Stack:** Python 3.11+, pandas, numpy, PyYAML, pytest, Foundry, Solidity 0.8.x, Vite, TypeScript

---

### Task 1: Establish the monorepo scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Modify: `README.md`
- Create: `configs/README.md`
- Create: `docs/README.md`

**Step 1: Add root tooling files**

Create package metadata, dev dependencies, environment placeholders, and ignore rules.

**Step 2: Rewrite the root README**

Document:

- project goal
- stack choices
- repo structure
- quickstart commands
- next development milestones

**Step 3: Verify local Python workflow**

Run: `d:\MG\anaconda3\python.exe -m pip install -e .[dev]`
Expected: local package installs successfully

**Step 4: Smoke-test the Python scaffold**

Run: `d:\MG\anaconda3\python.exe -m pytest`
Expected: tests pass once the starter package and tests are added

### Task 2: Build the Python package foundation

**Files:**
- Create: `src/funding_arb/__init__.py`
- Create: `src/funding_arb/utils/config.py`
- Create: `src/funding_arb/utils/paths.py`
- Create: `src/funding_arb/data/pipeline.py`
- Create: `src/funding_arb/features/pipeline.py`
- Create: `src/funding_arb/labels/generator.py`
- Create: `src/funding_arb/models/baselines.py`
- Create: `src/funding_arb/backtest/engine.py`
- Create: `src/funding_arb/evaluation/metrics.py`

**Step 1: Create reusable utilities**

Add config loading and repository path helpers so scripts and tests share the same assumptions.

**Step 2: Add placeholder module entry points**

Each core module should expose a small, importable function that describes its current role. Avoid empty packages.

**Step 3: Add starter evaluation functions**

Implement at least:

```python
def calculate_total_return(returns): ...
def calculate_sharpe_ratio(returns, periods_per_year=365 * 24): ...
def calculate_max_drawdown(equity_curve): ...
```

**Step 4: Run unit tests**

Run: `d:\MG\anaconda3\python.exe -m pytest tests/unit -v`
Expected: basic config, path, and metric tests pass

### Task 3: Add CLI scripts and baseline configs

**Files:**
- Create: `scripts/data/fetch_market_data.py`
- Create: `scripts/features/build_features.py`
- Create: `scripts/models/train_baseline.py`
- Create: `scripts/backtests/run_backtest.py`
- Create: `configs/data/default.yaml`
- Create: `configs/features/default.yaml`
- Create: `configs/models/baseline.yaml`
- Create: `configs/models/lstm.yaml`
- Create: `configs/backtests/default.yaml`
- Create: `configs/demo/default.yaml`

**Step 1: Wire scripts to config loading**

Each script should accept `--config` and print a deterministic summary of the intended job.

**Step 2: Capture assumptions in config**

Put symbol selection, window sizes, fees, slippage, and split dates in config files rather than in the script bodies.

**Step 3: Smoke-test one script per layer**

Run:

- `d:\MG\anaconda3\python.exe scripts/data/fetch_market_data.py --config configs/data/default.yaml`
- `d:\MG\anaconda3\python.exe scripts/features/build_features.py --config configs/features/default.yaml`
- `d:\MG\anaconda3\python.exe scripts/models/train_baseline.py --config configs/models/baseline.yaml`
- `d:\MG\anaconda3\python.exe scripts/backtests/run_backtest.py --config configs/backtests/default.yaml`

Expected: each script prints a scaffold summary without crashing

### Task 4: Prepare data and notebook conventions

**Files:**
- Create: `data/raw/.gitkeep`
- Create: `data/interim/.gitkeep`
- Create: `data/processed/.gitkeep`
- Create: `data/artifacts/.gitkeep`
- Create: `notebooks/README.md`

**Step 1: Create tracked data-stage directories**

Use gitkeeps so the intended directory structure is visible immediately.

**Step 2: Document notebook rules**

Make it explicit that notebooks are exploratory and reusable logic belongs in `src/` or `scripts/`.

### Task 5: Scaffold the Solidity workspace

**Files:**
- Create: `contracts/foundry.toml`
- Create: `contracts/README.md`
- Create: `contracts/src/MockStablecoin.sol`
- Create: `contracts/src/DeltaNeutralVault.sol`
- Create: `contracts/script/README.md`
- Create: `contracts/test/README.md`

**Step 1: Add the Foundry workspace**

Keep the config minimal and Solidity-first.

**Step 2: Add starter contracts**

Include:

- a mock stablecoin for local demos
- a starter vault with deposit, withdraw, pause, and NAV update hooks

**Step 3: Build contracts**

Run: `cd contracts && forge build`
Expected: starter contracts compile

### Task 6: Scaffold the frontend demo

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/main.ts`
- Create: `frontend/src/style.css`
- Create: `frontend/README.md`

**Step 1: Create a lightweight dashboard shell**

Render static cards for:

- strategy snapshot
- backtest metrics
- vault overview
- next milestones

**Step 2: Document the intended data sources**

Clarify that the UI will later consume exported artifacts and optional local contract data.

**Step 3: Smoke-test the app**

Run: `cd frontend && npm install && npm run dev`
Expected: local dev server starts

### Task 7: Add test coverage for the scaffold

**Files:**
- Create: `tests/unit/test_config_loader.py`
- Create: `tests/unit/test_metrics.py`
- Create: `tests/unit/test_paths.py`
- Create: `tests/integration/.gitkeep`
- Create: `tests/fixtures/.gitkeep`

**Step 1: Add fast unit tests**

Cover:

- YAML config loading
- repository path resolution
- starter evaluation metrics

**Step 2: Run tests**

Run: `d:\MG\anaconda3\python.exe -m pytest`
Expected: green test suite for scaffold-level coverage

### Task 8: Expand documentation for future module work

**Files:**
- Create: `docs/architecture/overview.md`
- Create: `docs/modules/data-pipeline.md`
- Create: `docs/modules/models-and-research.md`
- Create: `docs/modules/backtesting.md`
- Create: `docs/contracts/vault.md`
- Create: `docs/demos/frontend.md`

**Step 1: Document module responsibilities**

Each placeholder doc should state:

- the module purpose
- intended future files
- current caveats

**Step 2: Keep docs aligned with the scaffold**

The docs should match the actual folder structure created in the first milestone.

### Task 9: Execute the first real milestone after scaffolding

**Files:**
- Modify: `src/funding_arb/data/pipeline.py`
- Modify: `scripts/data/fetch_market_data.py`
- Modify: `tests/integration/`
- Modify: `docs/modules/data-pipeline.md`

**Step 1: Implement one real ingestion path**

Start with one symbol and one exchange.

**Step 2: Write a canonical schema**

Standardize timestamps, funding interval handling, and column names.

**Step 3: Add an integration-style validation**

Check that the cleaned dataset has the required columns and sorted timestamps.

**Step 4: Document caveats**

Record source-specific assumptions and missing fields.


