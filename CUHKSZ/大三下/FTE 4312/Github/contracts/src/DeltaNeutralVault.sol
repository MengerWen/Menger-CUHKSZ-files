// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IERC20Like} from "./interfaces/IERC20Like.sol";

abstract contract Ownable2StepLite {
    error OwnableUnauthorizedAccount(address account);
    error OwnableInvalidOwner(address owner);

    address public owner;
    address public pendingOwner;

    event OwnershipTransferStarted(address indexed previousOwner, address indexed newOwner);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    modifier onlyOwner() {
        if (msg.sender != owner) {
            revert OwnableUnauthorizedAccount(msg.sender);
        }
        _;
    }

    function _initializeOwner(address initialOwner) internal {
        if (initialOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }

        owner = initialOwner;
        emit OwnershipTransferred(address(0), initialOwner);
    }

    function transferOwnership(address newOwner) external onlyOwner {
        if (newOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }

        pendingOwner = newOwner;
        emit OwnershipTransferStarted(owner, newOwner);
    }

    function acceptOwnership() external {
        address pending = pendingOwner;
        if (msg.sender != pending) {
            revert OwnableUnauthorizedAccount(msg.sender);
        }

        address previousOwner = owner;
        owner = pending;
        pendingOwner = address(0);

        emit OwnershipTransferred(previousOwner, pending);
    }
}

abstract contract PausableLite {
    error EnforcedPause();
    error ExpectedPause();

    bool public paused;

    event Paused(address indexed account);
    event Unpaused(address indexed account);

    modifier whenNotPaused() {
        if (paused) {
            revert EnforcedPause();
        }
        _;
    }

    function _pause() internal {
        if (paused) {
            revert EnforcedPause();
        }

        paused = true;
        emit Paused(msg.sender);
    }

    function _unpause() internal {
        if (!paused) {
            revert ExpectedPause();
        }

        paused = false;
        emit Unpaused(msg.sender);
    }
}

abstract contract ReentrancyGuardLite {
    error ReentrancyGuardReentrantCall();

    uint256 private constant NOT_ENTERED = 1;
    uint256 private constant ENTERED = 2;

    uint256 private _status = NOT_ENTERED;

    modifier nonReentrant() {
        if (_status == ENTERED) {
            revert ReentrancyGuardReentrantCall();
        }

        _status = ENTERED;
        _;
        _status = NOT_ENTERED;
    }
}

contract DeltaNeutralVault is Ownable2StepLite, PausableLite, ReentrancyGuardLite {
    error AssetTransferFailed();
    error EmptyVault();
    error InsufficientCash(uint256 available, uint256 requested);
    error InsufficientReportedNav(uint256 available, uint256 requested);
    error InsufficientShares(uint256 available, uint256 requested);
    error InvalidAsset();
    error InvalidReceiver();
    error UnauthorizedUpdater(address account);
    error ValueTooLargeForInt256(uint256 value);
    error VaultInsolvent();
    error ZeroAssets();
    error ZeroShares();
    error NegativeNavDeltaTooLarge(uint256 currentNav, int256 requestedDelta);

    string public constant SHARE_NAME = "Delta Neutral Vault Share";
    string public constant SHARE_SYMBOL = "DNVS";

    enum StrategyState {
        Idle,
        Active,
        Emergency,
        Settled
    }

    IERC20Like public immutable asset;

    address public operator;
    StrategyState public strategyState;

    uint256 public totalShares;
    mapping(address => uint256) public shareBalance;

    uint256 public reportedNavAssets;
    int256 public cumulativePnlAssets;
    uint256 public lastNavUpdateAt;
    uint256 public lastPnlUpdateAt;
    bytes32 public lastReportHash;
    bytes32 public lastSignalHash;
    bytes32 public lastMetadataHash;

    uint256 public depositCount;
    uint256 public withdrawCount;
    uint256 public navUpdateCount;

    event Deposit(address indexed caller, address indexed receiver, uint256 assets, uint256 shares);
    event Withdraw(address indexed caller, address indexed receiver, uint256 assets, uint256 shares);
    event StrategyStateUpdated(
        uint8 previousState,
        uint8 newState,
        bytes32 signalHash,
        bytes32 metadataHash,
        address indexed updater
    );
    event NavUpdated(
        uint256 previousNavAssets,
        uint256 newNavAssets,
        int256 navDeltaAssets,
        bytes32 reportHash,
        address indexed updater
    );
    event PnlUpdated(
        int256 pnlDeltaAssets,
        int256 cumulativePnlAssets,
        uint256 newNavAssets,
        bytes32 reportHash,
        address indexed updater
    );
    event OperatorUpdated(address indexed previousOperator, address indexed newOperator);

    modifier onlyOwnerOrOperator() {
        if (msg.sender != owner && msg.sender != operator) {
            revert UnauthorizedUpdater(msg.sender);
        }
        _;
    }

    constructor(address asset_, address initialOwner, address initialOperator) {
        if (asset_ == address(0)) {
            revert InvalidAsset();
        }

        asset = IERC20Like(asset_);
        _initializeOwner(initialOwner);
        strategyState = StrategyState.Idle;
        _setOperator(initialOperator);
    }

    function totalAssets() external view returns (uint256) {
        return reportedNavAssets;
    }

    function reportedCashBalance() external view returns (uint256) {
        return asset.balanceOf(address(this));
    }

    function convertToShares(uint256 assets) public view returns (uint256 shares) {
        if (assets == 0) {
            return 0;
        }
        if (totalShares == 0) {
            return assets;
        }
        if (reportedNavAssets == 0) {
            revert VaultInsolvent();
        }

        return (assets * totalShares) / reportedNavAssets;
    }

    function convertToAssets(uint256 shares) public view returns (uint256 assets) {
        if (shares == 0) {
            return 0;
        }
        if (totalShares == 0) {
            return shares;
        }

        return (shares * reportedNavAssets) / totalShares;
    }

    function previewDeposit(uint256 assets) public view returns (uint256 shares) {
        if (assets == 0) {
            revert ZeroAssets();
        }

        shares = convertToShares(assets);
        if (shares == 0) {
            revert ZeroShares();
        }
    }

    function previewWithdraw(uint256 assets) public view returns (uint256 shares) {
        if (assets == 0) {
            revert ZeroAssets();
        }
        if (totalShares == 0 || reportedNavAssets == 0) {
            revert EmptyVault();
        }

        shares = _ceilDiv(assets * totalShares, reportedNavAssets);
        if (shares == 0) {
            revert ZeroShares();
        }
    }

    function deposit(uint256 assets, address receiver)
        external
        whenNotPaused
        nonReentrant
        returns (uint256 shares)
    {
        if (receiver == address(0)) {
            revert InvalidReceiver();
        }

        shares = previewDeposit(assets);

        if (!asset.transferFrom(msg.sender, address(this), assets)) {
            revert AssetTransferFailed();
        }

        shareBalance[receiver] += shares;
        totalShares += shares;
        reportedNavAssets += assets;
        depositCount += 1;

        emit Deposit(msg.sender, receiver, assets, shares);
    }

    function withdraw(uint256 assets, address receiver)
        external
        whenNotPaused
        nonReentrant
        returns (uint256 sharesBurned)
    {
        if (receiver == address(0)) {
            revert InvalidReceiver();
        }

        sharesBurned = previewWithdraw(assets);

        uint256 availableShares = shareBalance[msg.sender];
        if (availableShares < sharesBurned) {
            revert InsufficientShares(availableShares, sharesBurned);
        }
        if (reportedNavAssets < assets) {
            revert InsufficientReportedNav(reportedNavAssets, assets);
        }

        uint256 cashBalance = asset.balanceOf(address(this));
        if (cashBalance < assets) {
            revert InsufficientCash(cashBalance, assets);
        }

        shareBalance[msg.sender] = availableShares - sharesBurned;
        totalShares -= sharesBurned;
        reportedNavAssets -= assets;
        withdrawCount += 1;

        if (!asset.transfer(receiver, assets)) {
            revert AssetTransferFailed();
        }

        emit Withdraw(msg.sender, receiver, assets, sharesBurned);
    }

    function setOperator(address newOperator) external onlyOwner {
        _setOperator(newOperator);
    }

    function updateStrategyState(StrategyState newState, bytes32 signalHash, bytes32 metadataHash)
        external
        onlyOwnerOrOperator
    {
        StrategyState previousState = strategyState;
        strategyState = newState;
        lastSignalHash = signalHash;
        lastMetadataHash = metadataHash;

        emit StrategyStateUpdated(
            uint8(previousState),
            uint8(newState),
            signalHash,
            metadataHash,
            msg.sender
        );
    }

    function updateNav(uint256 newReportedNavAssets, bytes32 reportHash) external onlyOwnerOrOperator {
        uint256 previousNavAssets = reportedNavAssets;
        int256 navDeltaAssets = _computeNavDelta(previousNavAssets, newReportedNavAssets);

        reportedNavAssets = newReportedNavAssets;
        cumulativePnlAssets += navDeltaAssets;
        lastNavUpdateAt = block.timestamp;
        lastReportHash = reportHash;
        navUpdateCount += 1;

        emit NavUpdated(previousNavAssets, newReportedNavAssets, navDeltaAssets, reportHash, msg.sender);
    }

    function updatePnl(int256 pnlDeltaAssets, bytes32 reportHash) external onlyOwnerOrOperator {
        uint256 newReportedNavAssets = _applyPnlDelta(reportedNavAssets, pnlDeltaAssets);

        reportedNavAssets = newReportedNavAssets;
        cumulativePnlAssets += pnlDeltaAssets;
        lastNavUpdateAt = block.timestamp;
        lastPnlUpdateAt = block.timestamp;
        lastReportHash = reportHash;
        navUpdateCount += 1;

        emit PnlUpdated(pnlDeltaAssets, cumulativePnlAssets, newReportedNavAssets, reportHash, msg.sender);
    }

    function pause() external onlyOwner {
        _pause();
    }

    function unpause() external onlyOwner {
        _unpause();
    }

    function _setOperator(address newOperator) internal {
        address previousOperator = operator;
        operator = newOperator;
        emit OperatorUpdated(previousOperator, newOperator);
    }

    function _computeNavDelta(uint256 previousNavAssets, uint256 newReportedNavAssets)
        internal
        pure
        returns (int256 navDeltaAssets)
    {
        if (newReportedNavAssets >= previousNavAssets) {
            navDeltaAssets = _toInt256(newReportedNavAssets - previousNavAssets);
        } else {
            navDeltaAssets = -_toInt256(previousNavAssets - newReportedNavAssets);
        }
    }

    function _applyPnlDelta(uint256 currentNavAssets, int256 pnlDeltaAssets) internal pure returns (uint256 newNavAssets) {
        if (pnlDeltaAssets >= 0) {
            return currentNavAssets + uint256(pnlDeltaAssets);
        }

        uint256 negativeMagnitude = _negativeMagnitude(pnlDeltaAssets);
        if (negativeMagnitude > currentNavAssets) {
            revert NegativeNavDeltaTooLarge(currentNavAssets, pnlDeltaAssets);
        }

        return currentNavAssets - negativeMagnitude;
    }

    function _negativeMagnitude(int256 value) internal pure returns (uint256 magnitude) {
        unchecked {
            magnitude = uint256(-(value + 1)) + 1;
        }
    }

    function _toInt256(uint256 value) internal pure returns (int256) {
        if (value > uint256(type(int256).max)) {
            revert ValueTooLargeForInt256(value);
        }

        return int256(value);
    }

    function _ceilDiv(uint256 a, uint256 b) internal pure returns (uint256) {
        return a == 0 ? 0 : ((a - 1) / b) + 1;
    }
}
