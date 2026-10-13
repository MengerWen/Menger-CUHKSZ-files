// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface Vm {
    function prank(address caller) external;
    function startPrank(address caller) external;
    function stopPrank() external;
    function expectRevert(bytes calldata revertData) external;
    function expectRevert(bytes4 revertData) external;
    function warp(uint256 newTimestamp) external;
    function expectEmit(bool checkTopic1, bool checkTopic2, bool checkTopic3, bool checkData) external;
    function expectEmit(bool checkTopic1, bool checkTopic2, bool checkTopic3, bool checkData, address emitter)
        external;
}

abstract contract TestBase {
    error AssertionFailed(string message);

    Vm internal constant vm = Vm(address(uint160(uint256(keccak256("hevm cheat code")))));

    function assertTrue(bool condition, string memory message) internal pure {
        if (!condition) {
            revert AssertionFailed(message);
        }
    }

    function assertEq(uint256 actual, uint256 expected, string memory message) internal pure {
        if (actual != expected) {
            revert AssertionFailed(message);
        }
    }

    function assertEq(int256 actual, int256 expected, string memory message) internal pure {
        if (actual != expected) {
            revert AssertionFailed(message);
        }
    }

    function assertEq(address actual, address expected, string memory message) internal pure {
        if (actual != expected) {
            revert AssertionFailed(message);
        }
    }

    function assertEq(bytes32 actual, bytes32 expected, string memory message) internal pure {
        if (actual != expected) {
            revert AssertionFailed(message);
        }
    }

    function assertEq(bool actual, bool expected, string memory message) internal pure {
        if (actual != expected) {
            revert AssertionFailed(message);
        }
    }
}
