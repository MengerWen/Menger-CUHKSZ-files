// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract MockStablecoin {
    error AllowanceExceeded(uint256 allowed, uint256 requested);
    error InsufficientBalance(uint256 available, uint256 requested);
    error InvalidRecipient();
    error InvalidSpender();

    string public constant name = "Mock USD Coin";
    string public constant symbol = "mUSDC";
    uint8 public constant decimals = 6;

    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    function mint(address to, uint256 amount) external {
        if (to == address(0)) {
            revert InvalidRecipient();
        }

        totalSupply += amount;
        balanceOf[to] += amount;

        emit Transfer(address(0), to, amount);
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        if (spender == address(0)) {
            revert InvalidSpender();
        }

        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        _transfer(msg.sender, to, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        uint256 allowed = allowance[from][msg.sender];
        if (allowed < amount) {
            revert AllowanceExceeded(allowed, amount);
        }

        allowance[from][msg.sender] = allowed - amount;
        emit Approval(from, msg.sender, allowance[from][msg.sender]);

        _transfer(from, to, amount);
        return true;
    }

    function _transfer(address from, address to, uint256 amount) internal {
        if (to == address(0)) {
            revert InvalidRecipient();
        }

        uint256 balance = balanceOf[from];
        if (balance < amount) {
            revert InsufficientBalance(balance, amount);
        }

        balanceOf[from] = balance - amount;
        balanceOf[to] += amount;

        emit Transfer(from, to, amount);
    }
}
