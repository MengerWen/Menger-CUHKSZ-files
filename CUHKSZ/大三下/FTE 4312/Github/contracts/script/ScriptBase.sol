// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface Vm {
    function addr(uint256 privateKey) external returns (address wallet);
    function envUint(string calldata name) external returns (uint256 value);
    function envInt(string calldata name) external returns (int256 value);
    function envAddress(string calldata name) external returns (address value);
    function envBytes32(string calldata name) external returns (bytes32 value);
    function envOr(string calldata name, uint256 defaultValue) external returns (uint256 value);
    function envOr(string calldata name, int256 defaultValue) external returns (int256 value);
    function envOr(string calldata name, address defaultValue) external returns (address value);
    function envOr(string calldata name, bytes32 defaultValue) external returns (bytes32 value);
    function envOr(string calldata name, bool defaultValue) external returns (bool value);
    function startBroadcast(uint256 privateKey) external;
    function stopBroadcast() external;
}

abstract contract ScriptBase {
    Vm internal constant vm = Vm(address(uint160(uint256(keccak256("hevm cheat code")))));
}
