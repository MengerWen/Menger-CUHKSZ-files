// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {DeltaNeutralVault, Ownable2StepLite, PausableLite} from "../src/DeltaNeutralVault.sol";
import {MockStablecoin} from "../src/MockStablecoin.sol";
import {TestBase} from "./utils/TestBase.sol";

contract DeltaNeutralVaultTest is TestBase {
    event Deposit(address indexed caller, address indexed receiver, uint256 assets, uint256 shares);

    uint256 private constant ONE_UNIT = 1e6;

    MockStablecoin internal asset;
    DeltaNeutralVault internal vault;

    address internal constant OWNER = address(0xA11CE);
    address internal constant OPERATOR = address(0xB0B);
    address internal constant ALICE = address(0xCAFE);
    address internal constant BOB = address(0xBEEF);
    address internal constant CAROL = address(0xCA11);

    function setUp() public {
        asset = new MockStablecoin();
        vault = new DeltaNeutralVault(address(asset), OWNER, OPERATOR);

        asset.mint(ALICE, 10_000 * ONE_UNIT);
        asset.mint(BOB, 10_000 * ONE_UNIT);

        vm.prank(ALICE);
        asset.approve(address(vault), type(uint256).max);

        vm.prank(BOB);
        asset.approve(address(vault), type(uint256).max);
    }

    function testConstructorSetsCoreState() public view {
        assertEq(address(vault.asset()), address(asset), "vault asset should match mock stablecoin");
        assertEq(vault.owner(), OWNER, "owner should be initialized");
        assertEq(vault.operator(), OPERATOR, "operator should be initialized");
        assertEq(uint256(vault.strategyState()), uint256(DeltaNeutralVault.StrategyState.Idle), "initial state");
        assertEq(vault.totalShares(), 0, "total shares should start at zero");
        assertEq(vault.reportedNavAssets(), 0, "reported nav should start at zero");
        assertEq(vault.paused(), false, "vault should start unpaused");
    }

    function testFirstDepositMintsOneToOneShares() public {
        uint256 depositAssets = 1_000 * ONE_UNIT;

        vm.expectEmit(true, true, false, true, address(vault));
        emit Deposit(ALICE, ALICE, depositAssets, depositAssets);

        vm.prank(ALICE);
        uint256 shares = vault.deposit(depositAssets, ALICE);

        assertEq(shares, depositAssets, "first deposit should mint one-to-one shares");
        assertEq(vault.shareBalance(ALICE), depositAssets, "alice share balance");
        assertEq(vault.totalShares(), depositAssets, "total shares should increase");
        assertEq(vault.reportedNavAssets(), depositAssets, "nav should increase by deposit amount");
        assertEq(asset.balanceOf(address(vault)), depositAssets, "cash balance should hold deposited assets");
        assertEq(vault.depositCount(), 1, "deposit count should increment");
    }

    function testSecondDepositUsesExistingNavForSharePricing() public {
        uint256 aliceDeposit = 1_000 * ONE_UNIT;
        uint256 bobDeposit = 1_000 * ONE_UNIT;

        vm.prank(ALICE);
        vault.deposit(aliceDeposit, ALICE);

        vm.prank(OPERATOR);
        vault.updatePnl(int256(500 * ONE_UNIT), keccak256("pnl-up"));

        vm.prank(BOB);
        uint256 bobShares = vault.deposit(bobDeposit, BOB);

        assertEq(bobShares, 666_666_666, "bob should receive floor-rounded proportional shares");
        assertEq(vault.shareBalance(BOB), 666_666_666, "bob share balance");
        assertEq(vault.totalShares(), 1_666_666_666, "total shares after second deposit");
        assertEq(vault.reportedNavAssets(), 2_500 * ONE_UNIT, "reported nav should include pnl and deposit");
    }

    function testWithdrawBurnsCeilRoundedShares() public {
        uint256 depositAssets = 1_000 * ONE_UNIT;

        vm.prank(ALICE);
        vault.deposit(depositAssets, ALICE);

        vm.prank(OPERATOR);
        vault.updateNav(1_500 * ONE_UNIT, keccak256("nav-up"));

        vm.prank(ALICE);
        uint256 sharesBurned = vault.withdraw(1_000 * ONE_UNIT, ALICE);

        assertEq(sharesBurned, 666_666_667, "withdraw should burn ceil-rounded shares");
        assertEq(vault.shareBalance(ALICE), 333_333_333, "remaining shares");
        assertEq(vault.totalShares(), 333_333_333, "total shares after withdrawal");
        assertEq(vault.reportedNavAssets(), 500 * ONE_UNIT, "reported nav after withdrawal");
        assertEq(asset.balanceOf(ALICE), 10_000 * ONE_UNIT, "alice should receive withdrawn assets");
    }

    function testWithdrawRevertsForInsufficientShares() public {
        vm.prank(ALICE);
        vault.deposit(1_000 * ONE_UNIT, ALICE);

        vm.expectRevert(
            abi.encodeWithSelector(DeltaNeutralVault.InsufficientShares.selector, uint256(0), uint256(100 * ONE_UNIT))
        );
        vm.prank(BOB);
        vault.withdraw(100 * ONE_UNIT, BOB);
    }

    function testWithdrawRevertsWhenReportedNavExceedsCashBalance() public {
        vm.prank(ALICE);
        vault.deposit(1_000 * ONE_UNIT, ALICE);

        vm.prank(OPERATOR);
        vault.updateNav(1_500 * ONE_UNIT, keccak256("nav-up"));

        vm.expectRevert(
            abi.encodeWithSelector(
                DeltaNeutralVault.InsufficientCash.selector, uint256(1_000 * ONE_UNIT), uint256(1_200 * ONE_UNIT)
            )
        );
        vm.prank(ALICE);
        vault.withdraw(1_200 * ONE_UNIT, ALICE);
    }

    function testDepositAndWithdrawRevertWhilePaused() public {
        vm.prank(ALICE);
        vault.deposit(1_000 * ONE_UNIT, ALICE);

        vm.prank(OWNER);
        vault.pause();

        vm.expectRevert(PausableLite.EnforcedPause.selector);
        vm.prank(ALICE);
        vault.deposit(100 * ONE_UNIT, ALICE);

        vm.expectRevert(PausableLite.EnforcedPause.selector);
        vm.prank(ALICE);
        vault.withdraw(100 * ONE_UNIT, ALICE);
    }

    function testOnlyOwnerCanSetOperator() public {
        vm.expectRevert(abi.encodeWithSelector(Ownable2StepLite.OwnableUnauthorizedAccount.selector, ALICE));
        vm.prank(ALICE);
        vault.setOperator(CAROL);

        vm.prank(OWNER);
        vault.setOperator(CAROL);

        assertEq(vault.operator(), CAROL, "owner should be able to replace operator");
    }

    function testOwnerOrOperatorCanUpdateStrategyState() public {
        bytes32 signalHash = keccak256("signal");
        bytes32 metadataHash = keccak256("meta");

        vm.prank(OPERATOR);
        vault.updateStrategyState(DeltaNeutralVault.StrategyState.Active, signalHash, metadataHash);

        assertEq(
            uint256(vault.strategyState()),
            uint256(DeltaNeutralVault.StrategyState.Active),
            "operator should update state"
        );
        assertEq(vault.lastSignalHash(), signalHash, "signal hash should update");
        assertEq(vault.lastMetadataHash(), metadataHash, "metadata hash should update");

        vm.expectRevert(abi.encodeWithSelector(DeltaNeutralVault.UnauthorizedUpdater.selector, ALICE));
        vm.prank(ALICE);
        vault.updateStrategyState(DeltaNeutralVault.StrategyState.Emergency, bytes32(0), bytes32(0));
    }

    function testUpdateNavTracksAbsoluteDeltaAndTimestamp() public {
        vm.prank(ALICE);
        vault.deposit(1_000 * ONE_UNIT, ALICE);

        vm.warp(100);
        vm.prank(OPERATOR);
        vault.updateNav(1_250 * ONE_UNIT, keccak256("nav-1"));

        assertEq(vault.reportedNavAssets(), 1_250 * ONE_UNIT, "reported nav should update");
        assertEq(vault.cumulativePnlAssets(), int256(250 * ONE_UNIT), "cumulative pnl should reflect nav delta");
        assertEq(vault.lastNavUpdateAt(), 100, "nav timestamp should track latest update");
        assertEq(vault.lastReportHash(), keccak256("nav-1"), "report hash should update");
        assertEq(vault.navUpdateCount(), 1, "nav update count should increment");
    }

    function testOnlyOwnerOrOperatorCanUpdateNavAndPnl() public {
        vm.expectRevert(abi.encodeWithSelector(DeltaNeutralVault.UnauthorizedUpdater.selector, ALICE));
        vm.prank(ALICE);
        vault.updateNav(100 * ONE_UNIT, keccak256("nav-unauthorized"));

        vm.expectRevert(abi.encodeWithSelector(DeltaNeutralVault.UnauthorizedUpdater.selector, ALICE));
        vm.prank(ALICE);
        vault.updatePnl(int256(10 * ONE_UNIT), keccak256("pnl-unauthorized"));

        vm.prank(OWNER);
        vault.updateNav(200 * ONE_UNIT, keccak256("nav-owner"));

        vm.prank(OPERATOR);
        vault.updatePnl(int256(25 * ONE_UNIT), keccak256("pnl-operator"));

        assertEq(vault.reportedNavAssets(), 225 * ONE_UNIT, "authorized updates should change reported nav");
        assertEq(vault.cumulativePnlAssets(), int256(225 * ONE_UNIT), "authorized updates should change pnl");
    }

    function testUpdatePnlUpdatesNavAndCumulativePnl() public {
        vm.prank(ALICE);
        vault.deposit(1_000 * ONE_UNIT, ALICE);

        vm.warp(250);
        vm.prank(OPERATOR);
        vault.updatePnl(int256(125 * ONE_UNIT), keccak256("pnl-up"));

        assertEq(vault.reportedNavAssets(), 1_125 * ONE_UNIT, "nav should include pnl delta");
        assertEq(vault.cumulativePnlAssets(), int256(125 * ONE_UNIT), "cumulative pnl should increase");
        assertEq(vault.lastNavUpdateAt(), 250, "nav timestamp should update");
        assertEq(vault.lastPnlUpdateAt(), 250, "pnl timestamp should update");
        assertEq(vault.lastReportHash(), keccak256("pnl-up"), "report hash should update");
        assertEq(vault.navUpdateCount(), 1, "nav update count should increment");
    }

    function testUpdatePnlRevertsIfLossWouldPushNavBelowZero() public {
        vm.prank(ALICE);
        vault.deposit(100 * ONE_UNIT, ALICE);

        vm.expectRevert(
            abi.encodeWithSelector(
                DeltaNeutralVault.NegativeNavDeltaTooLarge.selector,
                uint256(100 * ONE_UNIT),
                -int256(101 * ONE_UNIT)
            )
        );
        vm.prank(OPERATOR);
        vault.updatePnl(-int256(101 * ONE_UNIT), keccak256("too-much-loss"));
    }

    function testTwoStepOwnershipTransfer() public {
        vm.prank(OWNER);
        vault.transferOwnership(CAROL);

        assertEq(vault.pendingOwner(), CAROL, "pending owner should be recorded");

        vm.expectRevert(abi.encodeWithSelector(Ownable2StepLite.OwnableUnauthorizedAccount.selector, ALICE));
        vm.prank(ALICE);
        vault.acceptOwnership();

        vm.prank(CAROL);
        vault.acceptOwnership();

        assertEq(vault.owner(), CAROL, "carol should become owner");
        assertEq(vault.pendingOwner(), address(0), "pending owner should clear");

        vm.expectRevert(abi.encodeWithSelector(Ownable2StepLite.OwnableUnauthorizedAccount.selector, OWNER));
        vm.prank(OWNER);
        vault.pause();

        vm.prank(CAROL);
        vault.pause();
        assertTrue(vault.paused(), "new owner should control pause");
    }

    function testOwnerCanClearOperatorAndStillUpdateVaultState() public {
        vm.prank(OWNER);
        vault.setOperator(address(0));

        assertEq(vault.operator(), address(0), "operator should be clearable");

        vm.prank(OWNER);
        vault.updateStrategyState(
            DeltaNeutralVault.StrategyState.Settled, keccak256("signal-final"), keccak256("metadata-final")
        );

        assertEq(
            uint256(vault.strategyState()),
            uint256(DeltaNeutralVault.StrategyState.Settled),
            "owner should still update state without operator"
        );
    }
}
