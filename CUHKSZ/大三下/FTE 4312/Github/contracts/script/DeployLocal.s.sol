// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {ScriptBase} from "./ScriptBase.sol";
import {DeltaNeutralVault} from "../src/DeltaNeutralVault.sol";
import {MockStablecoin} from "../src/MockStablecoin.sol";

contract DeployLocal is ScriptBase {
    event VaultDeployed(
        address indexed deployer,
        address indexed asset,
        address indexed vault,
        address operator,
        uint256 initialMint
    );

    function run() external returns (MockStablecoin asset, DeltaNeutralVault vault) {
        uint256 privateKey = vm.envUint("PRIVATE_KEY");
        address deployer = vm.addr(privateKey);
        address operator = vm.envOr("INITIAL_OPERATOR", deployer);
        uint256 initialMint = vm.envOr("INITIAL_MINT", uint256(0));

        vm.startBroadcast(privateKey);

        asset = new MockStablecoin();
        vault = new DeltaNeutralVault(address(asset), deployer, operator);

        if (initialMint > 0) {
            asset.mint(deployer, initialMint);
        }

        vm.stopBroadcast();

        emit VaultDeployed(deployer, address(asset), address(vault), operator, initialMint);
    }
}
