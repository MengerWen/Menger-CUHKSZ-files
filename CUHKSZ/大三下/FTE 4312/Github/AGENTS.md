# AGENTS.md

## Purpose

This repository is for a course-project prototype:
**Deep Learning-Based Delta-Neutral Statistical Arbitrage System on Perpetual Funding Rates**.

Future Codex runs should optimize for a clean, credible, end-to-end prototype rather than production-scale infrastructure. The goal is to show a coherent pipeline from market data to modeling, backtesting, vault accounting, and demo visualization.

## Core Working Rules

1. Always inspect the repository structure before coding.
2. Read the relevant docs first, especially [`Proposal.md`](./Proposal.md), the root `README.md` if present, and any module-level docs.
3. For complex tasks, plan first, then implement.
4. Prefer incremental, testable changes over large rewrites.
5. Keep code modular and production-style, but scoped to a course project prototype.
6. Do not over-engineer infrastructure that is not needed for the demo, evaluation, or presentation.
7. Use clear folder structure and readable naming.
8. Add or update documentation whenever a major module is added or materially changed.
9. Add tests whenever practical, especially for core data logic, backtesting logic, and contract behavior.
10. If assumptions are required, state them explicitly in code comments, config, or docs.
11. Keep configuration centralized in config files rather than scattering constants across scripts.
12. Make scripts runnable from the command line.
13. Prefer reproducibility: deterministic configs, pinned dependencies when reasonable, explicit seeds, and documented data sources.
14. Use d:\MG\anaconda3\python.exe for all Python commands in this repository unless the task explicitly requires a different interpreter.

## Project Scope Guardrails

- This is a hybrid quant + smart-contract + demo system.
- The quantitative pipeline should be implemented in Python.
- Prefer the libraries already available in d:\MG\anaconda3\python.exe such as pandas, NumPy, PyYAML, pytest, scikit-learn, torch, ccxt, and web3 instead of adding lower-level fallback implementations without a clear need.
- The smart contract layer should be implemented in Solidity.
- The frontend should stay lightweight unless the repository already establishes a different stack.
- Favor simple local workflows over distributed systems, microservices, or heavy cloud infrastructure.
- Do not add unnecessary databases, message queues, orchestration layers, or backend services unless the repository clearly evolves to require them.
- Backtests should be realistic enough to include transaction costs, slippage, and funding effects, but they do not need to simulate exchange internals in extreme detail.
- The vault is a prototype for accounting and state management, not a production DeFi deployment.

## Default Development Workflow

### Before coding

1. Inspect the repository tree and existing files.
2. Identify which layer the task affects:
   - data ingestion / cleaning
   - features / labels
   - baseline or ML model
   - backtesting / evaluation
   - Solidity vault
   - frontend / demo
   - docs / scripts / tests
3. For non-trivial tasks, write a short implementation plan before editing.
4. Check whether existing configs, scripts, or docs should be extended instead of creating duplicates.

### While coding

1. Make the smallest useful change that moves the project forward.
2. Prefer reusable modules over notebook-only logic.
3. Keep interfaces explicit: inputs, outputs, assumptions, and units should be easy to understand.
4. Avoid hidden behavior and magic constants.
5. Keep raw data, cleaned data, features, labels, models, and backtest outputs clearly separated.
6. When adding a new major module, also add the minimum docs and tests needed to make it usable by the next contributor.

### After coding

After each major task, summarize:

- files created or modified
- how to run
- remaining caveats or assumptions

If relevant, also mention what was intentionally left out to keep scope appropriate for a prototype.

## Expected Repository Structure

Use this as the default target structure unless the repository evolves in a different direction for a good reason.

```text
.
+-- AGENTS.md
+-- README.md
+-- Proposal.md
+-- configs/
|   +-- data/
|   +-- features/
|   +-- models/
|   +-- backtests/
|   `-- demo/
+-- data/
|   +-- raw/
|   +-- interim/
|   +-- processed/
|   `-- artifacts/
+-- docs/
|   +-- architecture/
|   +-- experiments/
|   +-- contracts/
|   `-- demos/
+-- notebooks/
+-- scripts/
|   +-- data/
|   +-- features/
|   +-- models/
|   +-- backtests/
|   `-- demo/
+-- src/
|   `-- funding_arb/
|       +-- data/
|       +-- features/
|       +-- labels/
|       +-- models/
|       +-- strategies/
|       +-- backtest/
|       +-- evaluation/
|       +-- utils/
|       `-- config/
+-- tests/
|   +-- unit/
|   +-- integration/
|   `-- fixtures/
+-- contracts/
|   +-- src/
|   +-- script/
|   +-- test/
|   `-- README.md
`-- frontend/
    +-- src/
    +-- public/
    `-- README.md
```

Notes:

- `src/funding_arb/` should hold reusable Python package code.
- `scripts/` should provide CLI entry points for common workflows.
- `data/` should not become a dumping ground; keep stage boundaries clear.
- `contracts/` should stay focused on the vault prototype and related mocks/interfaces.
- `frontend/` should remain simple and demo-oriented.

## Repository Conventions

### Naming

- Use descriptive, readable names.
- Prefer `snake_case` for Python files, functions, variables, and config keys.
- Prefer `PascalCase` for Solidity contract names and React component names.
- Prefer filenames that reflect a single purpose, for example:
  - `fetch_binance_funding.py`
  - `build_basis_features.py`
  - `train_lstm.py`
  - `run_backtest.py`
  - `DeltaNeutralVault.sol`

### Module boundaries

- Separate ingestion, cleaning, feature engineering, labeling, modeling, and backtesting concerns.
- Do not mix training logic directly into data ingestion scripts.
- Do not bury backtest assumptions inside model code.
- Keep smart-contract logic separate from frontend and off-chain analytics code.

### Configuration

- Centralize configuration under `configs/`.
- Prefer human-readable formats such as YAML, TOML, or JSON.
- Keep environment-specific secrets out of tracked config files.
- Important experimental assumptions should be in config, not hardcoded in notebooks.

### Scripts

- New operational scripts should be runnable from the command line.
- Scripts should accept explicit input/output/config arguments where practical.
- If a script is important to the workflow, document an example command in the relevant README.

## Coding Style Guidance

### Python

- Prefer typed, modular Python over notebook-only experimentation.
- Use small, testable functions and well-scoped classes.
- Use docstrings for non-obvious modules or public functions.
- Validate time alignment carefully for market data joins.
- Guard against lookahead bias and leakage in feature and label pipelines.
- Keep random seeds explicit for model training and evaluation.
- Prefer `pathlib`, dataclasses, and clear tabular interfaces.
- Minimize hidden side effects; return data rather than mutating global state.

### Quant / modeling specifics

- Preserve raw data exactly as ingested whenever possible.
- Record cleaning assumptions explicitly.
- Use time-based train/validation/test splits.
- Make transaction costs, slippage, holding-period rules, and funding accrual explicit in backtests.
- Compare new models against simple baselines before claiming improvement.
- Favor reproducible experiments over ad hoc tuning.
- Report both performance and caveats.

### Solidity

- Keep the vault contract readable and security-conscious.
- Favor standard patterns for:
  - deposits and withdrawals
  - share accounting
  - access control
  - pausability
  - event logging
- Reuse audited libraries such as OpenZeppelin when appropriate instead of reinventing primitives.
- Document what is mocked or simplified relative to a real deployment.
- Avoid premature protocol complexity such as upgrade frameworks, multi-contract orchestration, or elaborate governance unless clearly required.

### Frontend

- Default to a simple stack unless the repo already establishes one.
- Prefer a lightweight dashboard that can show:
  - backtest outputs
  - strategy state
  - vault balances/shares
  - deposit/withdraw demo flows
- Avoid building a heavy backend just to support the demo.
- Keep components simple, readable, and easy to run locally.

## Testing Expectations

Testing should scale with the importance and risk of the change.

### Python tests

Add tests whenever practical for:

- data parsing and schema validation
- cleaning and resampling logic
- feature generation
- label generation
- backtest accounting and PnL logic
- cost/slippage calculations
- evaluation metric calculations

Prefer fast unit tests first. Add integration tests for end-to-end pipeline steps when useful.

### Solidity tests

Add contract tests for:

- deposits
- withdrawals
- share mint/redeem logic
- access control
- pause behavior
- NAV/PnL update logic
- expected events

### Frontend tests

Only add frontend tests where practical. For a lightweight demo, basic component or smoke tests are sufficient if the UI is small.

### General rule

If a change is hard to test automatically, document how it was validated manually.

## Documentation Expectations

- The root `README.md` should explain what the repository does and how to run the main pieces.
- Each major module should have enough local documentation for another contributor to use it without reverse-engineering the code.
- When a major module is added, create or update the relevant README or docs page in the same task.
- Document:
  - data sources
  - assumptions
  - important configs
  - run commands
  - outputs
  - known limitations
- Keep documentation aligned with the actual code and folder layout.

## Handling TODOs and Assumptions

### Assumptions

- Never leave major assumptions implicit.
- If an assumption materially affects behavior, document it in one of:
  - config
  - module README
  - code comment near the relevant logic
  - experiment note in `docs/`

Examples of assumptions worth documenting:

- exchange/source selection
- symbol universe
- funding interval handling
- fill-price approximation
- slippage model
- gas-cost simplification
- oracle update frequency

### TODOs

- Use TODOs sparingly and make them actionable.
- Prefer concrete TODOs over vague placeholders.
- Good format:
  - `TODO: Add multi-exchange normalization for funding-rate schema differences.`
  - `TODO: Replace mock NAV updater with signed off-chain update flow.`
- If a TODO affects correctness, also mention it in docs or the task summary.
- Do not leave dead stubs or placeholder files without explanation.

## Reproducibility Expectations

- Prefer deterministic seeds for model training where applicable.
- Keep dependency declarations current and minimal.
- Do not commit large generated datasets unless there is a clear reason.
- If sample data or mock artifacts are needed for demos/tests, keep them small and documented.
- Record enough information so another teammate can reproduce a backtest or demo run.

## What "Done" Looks Like

A task is closer to done when:

1. the repository structure still makes sense,
2. the new code fits the module boundaries,
3. configs and scripts are usable,
4. tests were added or updated when practical,
5. docs were updated if the change is significant,
6. assumptions and caveats are explicit,
7. a short handoff summary is provided.

## Default Handoff Format For Major Tasks

Use this structure in final summaries after major work:

### Files changed

- list created/modified files

### How to run

- list the key commands

### Caveats

- list remaining limitations, assumptions, or next steps

## Final Reminder

Build the simplest complete version that demonstrates the idea well.
Favor clarity, reproducibility, and credible methodology over unnecessary complexity.
