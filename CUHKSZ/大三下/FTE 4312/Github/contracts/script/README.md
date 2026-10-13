# Scripts

This directory now contains the first Foundry scripts for the vault prototype:

- `ScriptBase.sol`
  Minimal Foundry cheatcode interface used by local scripts
- `DeployLocal.s.sol`
  Deploys `MockStablecoin` and `DeltaNeutralVault`
- `UpdateVaultState.s.sol`
  Applies strategy-state, NAV, and/or PnL updates to an existing vault

## Deploy

```bash
cd contracts
forge script script/DeployLocal.s.sol:DeployLocal \
  --rpc-url http://127.0.0.1:8545 \
  --broadcast
```

Environment variables:

- required: `PRIVATE_KEY`
- optional: `INITIAL_OPERATOR`, `INITIAL_MINT`

## Update An Existing Vault

```bash
cd contracts
forge script script/UpdateVaultState.s.sol:UpdateVaultState \
  --rpc-url http://127.0.0.1:8545 \
  --broadcast
```

Common environment variables:

- `PRIVATE_KEY`
- `VAULT_ADDRESS`
- `UPDATE_STRATEGY_STATE=true`
- `UPDATE_NAV=true`
- `UPDATE_PNL=true`
- `STRATEGY_STATE=1`
- `NEW_REPORTED_NAV_ASSETS=1250000000`
- `PNL_DELTA_ASSETS=250000000`
- `SIGNAL_HASH=0x...`
- `METADATA_HASH=0x...`
- `REPORT_HASH=0x...`

The update script is intentionally explicit: each action is toggled separately so demo flows stay readable and predictable.
