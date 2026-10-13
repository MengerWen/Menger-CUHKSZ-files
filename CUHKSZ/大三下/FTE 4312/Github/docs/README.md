# Documentation Guide

This directory is organized so a reviewer can move from high-level motivation to implementation detail without guessing what to read next.

## Recommended Reading Order

### 1. Start here

1. [../README.md](../README.md)
2. [../Proposal.md](../Proposal.md)
3. [review_guide.md](review_guide.md)
4. [architecture.md](architecture.md)
5. [final_checklist.md](final_checklist.md)

These files explain the project goal, scope, prototype boundaries, and the final submission state.

### 2. Understand the runnable demo

1. [demo.md](demo.md)
2. [integration.md](integration.md)
3. [demos/frontend.md](demos/frontend.md)

Read these if you want the cleanest demonstration path from research artifacts to vault state to dashboard.

### 3. Understand the quant pipeline

1. [data_schema.md](data_schema.md)
2. [features.md](features.md)
3. [labels.md](labels.md)
4. [baselines.md](baselines.md)
5. [models.md](models.md)
6. [signals.md](signals.md)
7. [backtest.md](backtest.md)
8. [robustness.md](robustness.md)

This sequence follows the actual data flow in the repository.
The deep-learning comparison bundles and `compare-dl` workflow are documented in [models.md](models.md).
These module docs also explain the degenerate-experiment guardrails now used across labels, models, signals, and backtests.

### 4. Understand the smart-contract layer

1. [contracts.md](contracts.md)
2. [contracts/vault.md](contracts/vault.md)

These explain the vault specification and how the Solidity prototype maps to the course-project scope.

### 5. Supplemental module notes

- [modules/data-pipeline.md](modules/data-pipeline.md)
- [modules/data-quality-reporting.md](modules/data-quality-reporting.md)
- [modules/models-and-research.md](modules/models-and-research.md)
- [modules/backtesting.md](modules/backtesting.md)
- [architecture/overview.md](architecture/overview.md)

These are helpful when you already understand the main story and want narrower implementation notes.

### 6. Historical planning docs

- [plans/2026-04-09-course-project-implementation-plan.md](plans/2026-04-09-course-project-implementation-plan.md)

This is useful for historical context, but it is not the primary source of truth anymore.

## Fast Review Paths

### For a grader or instructor

Read:

1. [review_guide.md](review_guide.md)
2. [architecture.md](architecture.md)
3. [demo.md](demo.md)
4. [final_checklist.md](final_checklist.md)

### For a Python contributor

Read:

1. [architecture.md](architecture.md)
2. [data_schema.md](data_schema.md)
3. [features.md](features.md)
4. [labels.md](labels.md)
5. [signals.md](signals.md)
6. [backtest.md](backtest.md)

### For a Solidity contributor

Read:

1. [contracts.md](contracts.md)
2. [integration.md](integration.md)
3. [demos/frontend.md](demos/frontend.md)

## Source Of Truth

- The root [../README.md](../README.md) is the best single entry point.
- [architecture.md](architecture.md) is the main technical design reference.
- [demo.md](demo.md) is the main operational guide for the end-to-end classroom demo.
- Manifest and report `status` / `reason` fields are the source of truth when a strategy row shows zero trades or a model run abstains from trading.
