# Contracts Workspace

This directory contains the Solidity side of the course-project prototype.

## Tooling

The contract workspace uses `Foundry`.

Why Foundry for this project:

- fast local compile/test loop
- Solidity-first workflow
- smaller surface area than a JS-heavy toolchain
- good fit for a compact vault prototype

## Planned Contracts

- `MockStablecoin.sol`
  Local demo asset used for deposit/withdraw flows
- `DeltaNeutralVault.sol`
  Prototype vault for:
  - deposits
  - withdrawals
  - share accounting
  - owner/operator strategy-state updates
  - owner/operator NAV and PnL updates
  - pause controls
  - event logging
- `interfaces/IERC20Like.sol`
  Minimal ERC20-like interface used by the vault

## Workspace Files

- `src/DeltaNeutralVault.sol`
- `src/MockStablecoin.sol`
- `test/DeltaNeutralVault.t.sol`
- `test/utils/TestBase.sol`
- `script/DeployLocal.s.sol`
- `script/UpdateVaultState.s.sol`

## Compile

```bash
cd contracts
forge build
```

## Test

```bash
cd contracts
forge test -vv
```

## Deploy To A Local/Test Network

Example local deployment to Anvil:

```bash
cd contracts
forge script script/DeployLocal.s.sol:DeployLocal \
  --rpc-url http://127.0.0.1:8545 \
  --broadcast
```

Required environment variables:

- `PRIVATE_KEY`

Optional environment variables:

- `INITIAL_OPERATOR`
- `INITIAL_MINT`

If `INITIAL_OPERATOR` is not set, the script uses the deployer address.
If `INITIAL_MINT` is set, the script mints that amount of mock stablecoin to the deployer for demo funding.

## Update Vault State After Deployment

```bash
cd contracts
forge script script/UpdateVaultState.s.sol:UpdateVaultState \
  --rpc-url http://127.0.0.1:8545 \
  --broadcast
```

Supported environment variables:

- `PRIVATE_KEY`
- `VAULT_ADDRESS`
- `UPDATE_STRATEGY_STATE`
- `UPDATE_NAV`
- `UPDATE_PNL`
- `STRATEGY_STATE`
- `NEW_REPORTED_NAV_ASSETS`
- `PNL_DELTA_ASSETS`
- `SIGNAL_HASH`
- `METADATA_HASH`
- `REPORT_HASH`

## Key Assumptions

- one mock stablecoin asset only
- internal, non-transferable shares
- off-chain NAV and PnL reporting by a trusted owner/operator path
- withdrawals require both enough reported NAV and enough actual token liquidity
- no on-chain exchange execution, oracle verification, fees, or withdrawal queue

## Current Environment Note

The contract workspace has now been validated locally with Foundry:

- `forge test -vv` passes against `test/DeltaNeutralVault.t.sol`
- the scripts remain demo-oriented local/testnet helpers rather than production deployment tooling

## Scope Caveat

This contract workspace is a course-project prototype. It is not a production DeFi protocol and should not be treated as audited or deployment-ready.
