// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {ScriptBase} from "./ScriptBase.sol";
import {DeltaNeutralVault} from "../src/DeltaNeutralVault.sol";

contract UpdateVaultState is ScriptBase {
    error InvalidStrategyState(uint256 rawState);

    event VaultStateUpdated(
        address indexed vault,
        uint8 strategyState,
        uint256 newReportedNavAssets,
        int256 pnlDeltaAssets,
        bool updatedStrategyState,
        bool updatedNav,
        bool updatedPnl
    );

    function run() external {
        uint256 privateKey = vm.envUint("PRIVATE_KEY");
        DeltaNeutralVault vault = DeltaNeutralVault(vm.envAddress("VAULT_ADDRESS"));

        bool shouldUpdateState = vm.envOr("UPDATE_STRATEGY_STATE", false);
        bool shouldUpdateNav = vm.envOr("UPDATE_NAV", false);
        bool shouldUpdatePnl = vm.envOr("UPDATE_PNL", false);

        uint256 rawState = vm.envOr("STRATEGY_STATE", uint256(0));
        uint256 newReportedNavAssets = vm.envOr("NEW_REPORTED_NAV_ASSETS", uint256(0));
        int256 pnlDeltaAssets = vm.envOr("PNL_DELTA_ASSETS", int256(0));
        bytes32 signalHash = vm.envOr("SIGNAL_HASH", bytes32(0));
        bytes32 metadataHash = vm.envOr("METADATA_HASH", bytes32(0));
        bytes32 reportHash = vm.envOr("REPORT_HASH", bytes32(0));

        if (rawState > uint256(DeltaNeutralVault.StrategyState.Settled)) {
            revert InvalidStrategyState(rawState);
        }

        vm.startBroadcast(privateKey);

        if (shouldUpdateState) {
            vault.updateStrategyState(DeltaNeutralVault.StrategyState(rawState), signalHash, metadataHash);
        }

        if (shouldUpdateNav) {
            vault.updateNav(newReportedNavAssets, reportHash);
        }

        if (shouldUpdatePnl) {
            vault.updatePnl(pnlDeltaAssets, reportHash);
        }

        vm.stopBroadcast();

        emit VaultStateUpdated(
            address(vault),
            uint8(vault.strategyState()),
            vault.reportedNavAssets(),
            pnlDeltaAssets,
            shouldUpdateState,
            shouldUpdateNav,
            shouldUpdatePnl
        );
    }
}
