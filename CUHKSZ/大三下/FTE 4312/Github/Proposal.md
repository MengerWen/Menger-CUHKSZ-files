# Deep Learning-Based Delta-Neutral Statistical Arbitrage on Perpetual Funding Rates

<div align="center" style="display: flex; justify-content: center; gap: 60px; margin-bottom: 40px;">
  <div style="text-align: center;">
    <strong>文杰</strong><br>
    123090612
  </div>
  <div style="text-align: center;">
    <strong>韩启航</strong><br>
    123090149
  </div>
  <div style="text-align: center;">
    <strong>潘鸿锐</strong><br>
    123090439
  </div>
</div>

## 1. Project Title and Background

**Project Title:**  
**Deep Learning-Based Delta-Neutral Statistical Arbitrage on Perpetual Funding Rates**

Perpetual futures are one of the most important instruments in crypto derivatives markets. Unlike traditional futures, perpetual contracts do not expire; instead, they rely on a **funding rate** mechanism to keep contract prices aligned with the underlying spot price. On dYdX, for example, funding is charged hourly and is explicitly designed to help perpetual prices track the underlying market. ([dYdX Help Center](https://help.dydx.trade/en/articles/166992-default-funding-rates-on-dydx?utm_source=chatgpt.com "Default funding rates on dYdX")) Recent research also shows that perpetual futures are the dominant crypto derivatives product, with daily trading volume regularly exceeding $100 billion. ([arXiv](https://arxiv.org/html/2212.06888v5?utm_source=chatgpt.com "Fundamentals of Perpetual FuturesWe are grateful to Lin ..."))

However, funding rates and basis spreads often fluctuate significantly across time and market conditions. During periods of strong directional sentiment, leverage imbalance, or short-term price dislocation, funding rates may become abnormally high, low, or even switch sign. These dynamics create opportunities for **delta-neutral arbitrage** and **statistical trading strategies**, especially when funding-rate signals are combined with price spread and basis information. At the same time, these opportunities are difficult for ordinary users to capture systematically because they require data analysis, timing decisions, transaction-cost control, and risk management.

This project aims to build a **prototype DeFi strategy system** that combines off-chain quantitative modeling with an on-chain smart contract vault. The system will use historical market data to identify and predict funding-rate arbitrage opportunities, then simulate a delta-neutral strategy whose signals are reflected in a Solidity-based vault contract for asset accounting, profit distribution, and parameter management.

This project aims to build a prototype DeFi strategy system that combines off-chain quantitative modeling with an on-chain smart contract vault. The system will use historical market data to identify and predict funding-rate arbitrage opportunities, then simulate a delta-neutral strategy whose signals are reflected in a Solidity-based vault contract for asset accounting, profit distribution, and parameter management.

---

## 2. Problem Statement

Although funding-rate arbitrage is widely discussed in crypto trading, there are still several practical challenges.

First, **funding-rate mispricing is not constant or easy to exploit**. A high funding rate does not automatically imply profitable arbitrage, because profitability depends on whether the spread converges, how long the position must be held, and whether the expected return remains positive after trading fees, slippage, and gas costs.

Second, **most retail users lack an automated and explainable framework** for identifying such opportunities. Existing discussions of perpetual arbitrage are often strategy-oriented, but do not provide a complete system that links market-data analysis, signal generation, position logic, and on-chain asset management.

Third, **smart contracts cannot directly access external market data or run complex machine learning models**. This is a classic oracle problem: blockchains are isolated from off-chain information, so any system that depends on external prices, funding rates, or computed strategy signals must use some form of oracle or off-chain computation layer. ([Chainlink](https://chain.link/education-hub/oracle-problem?utm_source=chatgpt.com "The Blockchain Oracle Problem"))

Therefore, the core problem of this project is:

> **How can we design a hybrid DeFi prototype that identifies funding-rate arbitrage opportunities with statistical and deep learning models, while using a smart contract vault to manage user deposits, strategy state, and simulated profit distribution in a transparent and risk-aware manner?**

---

## 3. Proposed Solution and Key Features

We propose a **hybrid on-chain/off-chain arbitrage prototype** with two tightly connected components:

### (1) Off-chain Quantitative and Deep Learning Module

This module will collect and process historical perpetual market data, including:

- perpetual futures prices,
    
- spot or index prices,
    
- funding-rate history,
    
- trading volume,
    
- short-term volatility,
    
- basis spreads and related indicators.
    

Perpetual Protocol’s developer documentation, for instance, exposes market and funding-rate data interfaces that are suitable for this kind of research pipeline. ([Perpetual Protocol](https://docs.perp.com/docs/guides/data-source/?utm_source=chatgpt.com "Data Source"))

Based on these data, we will construct predictive signals such as:

- funding spread,
    
- basis deviation,
    
- rolling z-score,
    
- funding sign reversal,
    
- volatility and liquidity indicators.
    

We will then compare two categories of methods:

- **Statistical baselines**, such as threshold rules, rolling mean reversion, spread regression, or ARIMA-style forecasting;
    
- **Deep learning models**, such as **LSTM** or **Transformer** networks, to predict whether funding-rate dislocations are likely to persist, reverse, or generate sufficient post-cost returns.
    

### (2) On-chain Solidity Vault Prototype

Instead of building a production-grade trading protocol, we will implement a **strategy vault prototype** in Solidity. The contract will not directly connect to multiple real exchanges for live execution; rather, it will serve as the on-chain asset-management layer of the system.

Its main features will include:

- user deposits and withdrawals using a mock stablecoin setup,
    
- share accounting for vault participants,
    
- storage of strategy parameters or signal summaries,
    
- profit-and-loss updates based on simulated strategy outcomes,
    
- event logging for transparency,
    
- basic access control and emergency pause mechanisms.
    

This design reflects a realistic **hybrid smart contract** architecture: data processing and prediction happen off-chain, while asset accounting and state management happen on-chain. Chainlink’s oracle documentation explicitly frames this type of design as a response to the inability of blockchains to natively access off-chain data and computation. ([Chainlink](https://chain.link/education-hub/oracle-problem?utm_source=chatgpt.com "The Blockchain Oracle Problem"))

---

## 4. Technical Approach and Preliminary System Design

Our system will contain three layers: **off-chain analytics**, **on-chain contract logic**, and **front-end visualization**.

### Layer A: Data and Modeling Pipeline

We plan to choose one major perpetual market, such as **BTC** or **ETH** perpetuals, and build a dataset containing:

- perpetual prices,
    
- spot/index prices,
    
- funding rates,
    
- volatility and liquidity features.
    

Using these data, we will:

1. define an arbitrage signal construction framework,
    
2. build baseline trading rules,
    
3. train deep learning models for signal prediction,
    
4. evaluate strategy performance under realistic transaction-cost assumptions.
    

The target is not just directional prediction, but **timing and quality of arbitrage entry/exit decisions**.

### Layer B: Strategy Simulation and Evaluation

We will simulate a **delta-neutral strategy**, where long and short exposures are structured to reduce market-direction risk while harvesting funding-rate or basis convergence effects. Strategy evaluation will include:

- cumulative return,
    
- annualized return,
    
- Sharpe ratio,
    
- maximum drawdown,
    
- win rate,
    
- sensitivity to transaction costs and slippage.
    

This part is essential because a strategy that appears profitable before costs may become unprofitable after realistic execution frictions.

### Layer C: Solidity Vault Contract

The smart contract will implement:

- deposit and withdrawal functions,
    
- share minting and redemption logic,
    
- update functions for strategy parameters or oracle-fed results,
    
- NAV or PnL bookkeeping,
    
- event emission for state changes,
    
- simple security patterns such as access control and pausable behavior.
    

Rather than claiming to fully automate multi-venue execution, the contract will serve as the **on-chain representation of the strategy vault**, demonstrating how DeFi infrastructure could manage user participation in a quantitative arbitrage system.

### Layer D: Front-End / Demo Interface

A lightweight front end will visualize:

- historical backtest results,
    
- current strategy state,
    
- vault share balances,
    
- mock deposit/withdraw flows,
    
- signal updates and profit allocation results.
    

This will make the project easier to demonstrate in the final presentation and show the full pipeline from data to strategy to smart contract.

---

## 5. Expected Contributions or Innovations

This project is expected to contribute in the following ways.

First, it brings together **crypto derivatives research and smart contract implementation** in one unified system. Many trading studies stop at backtesting, while many Solidity projects focus only on contract mechanics. Our project combines both.

Second, it applies **deep learning to a practically meaningful DeFi problem**. Instead of predicting simple price direction, we focus on predicting the behavior of **funding-rate dislocations and basis convergence**, which is closer to how quantitative strategies are designed in real derivative markets.

Third, it demonstrates a **hybrid smart contract architecture** for DeFi strategy systems. Because smart contracts cannot natively access off-chain data, a robust design must combine off-chain computation with on-chain settlement and accounting. Our prototype directly reflects this architecture. ([Chainlink](https://chain.link/education-hub/oracle-problem?utm_source=chatgpt.com "The Blockchain Oracle Problem"))

Fourth, it remains realistic in scope. Rather than overpromising a production-grade arbitrage engine, we focus on delivering a **working prototype** that clearly shows:

- a real and relevant DeFi problem,
    
- a quantitative modeling pipeline,
    
- a Solidity contract with meaningful functionality,
    
- and an end-to-end system demonstration.
    

---

## 6. Preliminary Division of Work

Given our team structure of **two Quantitative Finance students and one FinTech student**, the task division is naturally aligned with the project design.

- **Quantitative Finance members**
    
    - data collection and preprocessing,
        
    - feature engineering,
        
    - baseline strategy design,
        
    - deep learning model construction,
        
    - backtesting and evaluation.
        
- **FinTech member**
    
    - Solidity vault contract development,
        
    - on-chain state management,
        
    - contract events and permissions,
        
    - front-end integration and demo support.
        

This division allows the team to leverage both financial modeling and blockchain development skills efficiently.

Github link: [https://github.com/MengerWen/Deep-Learning-Based-Delta-Neutral-Statistical-Arbitrage-on-Perpetual-Funding-Rates](https://github.com/MengerWen/Deep-Learning-Based-Delta-Neutral-Statistical-Arbitrage-on-Perpetual-Funding-Rates/tree/main)