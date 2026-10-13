# Tests

The contract workspace now includes the main Foundry unit test file:

- `DeltaNeutralVault.t.sol`
- `utils/TestBase.sol`

## Covered Behavior

The current tests cover:

- constructor state
- first deposit share minting
- proportional share minting after NAV changes
- ceil-rounded withdrawals
- insufficient-share reverts
- insufficient-cash reverts when reported NAV exceeds on-chain liquidity
- pause behavior
- owner-only operator management
- owner/operator strategy-state updates
- owner/operator NAV and PnL updates
- negative PnL bounds
- two-step ownership transfer

## Run

```bash
cd contracts
forge test -vv
```

## Design Note

The tests use a tiny local `TestBase.sol` helper instead of `forge-std/Test.sol` so the workspace stays self-contained and easy to inspect in a course-project setting.
