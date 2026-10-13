# FTE 4312 Final Presentation Script

Project: **Deep Learning-Based Delta-Neutral Statistical Arbitrage on Perpetual Funding Rates**

Target duration: **15 minutes**

Presentation page:

```text
http://127.0.0.1:5174/pre/index.html?ppt=1
```

Interactive dashboard, if needed:

```text
http://127.0.0.1:5174/?mode=demo_showcase
```

## Speaker Assignment And Timing

| Speaker | Time | PPT Sections | Main Responsibility |
| --- | ---: | --- | --- |
| Jie Wen | 2:00 | Hero, Executive Outline, Blockchain Scope | Opening, project framing, high-level architecture |
| Hongrui Pan | 6:30 | Problem, System Boundary, Off-Chain Deep Dive, Off-Chain Intelligence | Motivation, off-chain analytics, modeling, arbitrage evidence |
| Qihang Han | 3:55 | On-Chain Deep Dive, Boundary Handoff, Implementation Status, Blockchain Evidence, Live Vault Console, Chain Demo Flow, Caveats, Next Steps | Vault design, chain-facing implementation, live demo, closing |

## Important Presentation Rule

Only discuss the **demo showcase** results in class. Do not compare against the separate real experiment artifacts. Phrase results as "in our demo showcase" or "in this demonstration bundle" when necessary.

## Part 1: Jie Wen - Opening And Overall Framing

### Slide: Title / Hero

**Time: 0:00 - 0:45**

Hello everyone. We are presenting our FTE 4312 course project, **Deep Learning-Based Delta-Neutral Statistical Arbitrage on Perpetual Funding Rates**.

The key idea is to combine two worlds. On one side, we use off-chain quantitative research to analyze perpetual funding-rate opportunities. On the other side, we use an on-chain vault to represent user deposits, share accounting, strategy state, and NAV or PnL synchronization.

So the project is not only a trading model, and it is not only a smart contract. It is a hybrid prototype: **off-chain intelligence, on-chain state machine**.

### Slide: Executive Outline

**Time: 0:45 - 1:25**

Our presentation follows four parts.

First, we explain the project idea: why funding-rate arbitrage creates a natural DeFi strategy problem.

Second, we explain the technical design: what stays off-chain and what moves on-chain.

Third, we show the implementation: the data pipeline, model pipeline, backtest, Solidity vault, scripts, tests, and frontend bridge.

Finally, we demonstrate the system: especially the vault console, where we can operate the vault during the presentation and show state changes on screen.

### Slide: Blockchain Scope

**Time: 1:25 - 2:00**

The most important framing is this: the blockchain contribution is not that we run deep learning on-chain. That would be expensive and unrealistic.

Instead, the blockchain contribution is the vault boundary. The smart contract records pooled capital, internal shares, strategy state, NAV, PnL updates, and event history. The heavy analytics happen off-chain, and the compact result is synchronized to the vault.

I will now pass to Hongrui, who will explain the problem and the off-chain strategy pipeline.

## Part 2: Hongrui Pan - Problem, Architecture Boundary, And Off-Chain Analytics

### Slide: Problem And Motivation

**Time: 2:00 - 3:00**

The motivation starts from perpetual futures. Unlike traditional futures, perpetual contracts do not expire. Exchanges use the funding-rate mechanism to keep the perpetual price close to the spot or index price.

When the market is strongly long, funding can become positive, meaning longs pay shorts. In a simplified delta-neutral trade, we can short the perpetual contract and hedge with a long spot position. The goal is to reduce directional BTC exposure and earn from funding or basis convergence.

But this is not automatically profitable. A high funding rate alone is not enough. We still need to consider basis risk, transaction fees, slippage, holding period, and whether the funding dislocation persists long enough.

That is why this is a good hybrid system problem. The strategy logic requires heavy off-chain research, while pooled capital and accounting benefit from transparent on-chain bookkeeping.

### Slide: System Boundary - What Stays Off-Chain / What Moves On-Chain

**Time: 3:00 - 4:05**

Our system boundary is very deliberate.

The off-chain side handles data-heavy and computation-heavy work. This includes market data processing, feature engineering, label construction, model training, signal generation, backtesting, and report generation.

The on-chain side handles deterministic state. This includes deposits, withdrawals, internal shares, strategy state, role-restricted updates, NAV synchronization, and event logs.

This separation is important because smart contracts cannot natively access exchange data or run complex machine learning models. Instead of forcing everything on-chain, we use a realistic oracle-style architecture: compute off-chain, then publish a small and auditable update on-chain.

### Slide: Off-Chain Deep Dive

**Time: 4:05 - 5:45**

Before the contract is touched, the off-chain pipeline does four things.

First, it builds the research context. The Python pipeline fetches and aligns market data, then creates features such as funding rate, annualized funding, basis spread, z-score, volatility, and short-horizon market indicators.

Second, it builds supervised learning targets. Instead of predicting simple price direction, the label is aligned with the actual strategy objective: whether a potential delta-neutral trade has positive expected value after costs.

Third, it trains and compares models. The demo includes rule-based baselines, simple statistical or machine learning baselines, and deep learning models such as LSTM, GRU, and Transformer-style sequence models.

Fourth, it produces evidence. The output is not just one prediction number. We generate leaderboards, strategy metrics, equity curves, drawdowns, and charts. These artifacts justify whether the strategy view is strong enough to be reflected into the vault.

### Slide: Off-Chain Intelligence

**Time: 5:45 - 7:25**

This slide shows why the vault is not arbitrary. It is supported by off-chain research evidence.

The funding-rate chart shows that the market environment changes over time. Funding can stay calm, spike, reverse, or remain elevated. This matters because strategy timing is central to funding-rate arbitrage.

The spread chart shows the relationship between the perpetual price and spot reference. The basis can widen and mean-revert, which creates potential statistical arbitrage opportunities.

The model comparison chart shows how different signal families behave in the demo. We compare rule-based logic, baseline machine learning, and deep learning. The purpose is not to claim that deep learning is always best, but to show a structured benchmark process.

The equity and drawdown charts are important because we care about post-cost strategy behavior, not just raw prediction accuracy. A signal should be evaluated through trading outcomes, risk, and drawdown.

### Transition To Qihang

**Time: 7:25 - 8:30**

So the off-chain side creates a strategy view: what model is selected, what direction is suggested, what NAV or PnL update should be reported, and what evidence supports that update.

But the user does not interact directly with a Python notebook. The user interacts with a vault. That vault is the chain-facing part of our system.

Qihang will now explain what the Solidity layer stores and how the off-chain result becomes on-chain state.

## Part 3: Qihang Han - On-Chain Vault, Live Demo, And Closing

### Slide: On-Chain Deep Dive

**Time: 8:30 - 9:05**

The Solidity side is centered around two contracts.

`MockStablecoin` gives us a local demo asset, so we can show mint, approve, deposit, and withdraw flows without relying on a live token.

`DeltaNeutralVault` is the main contract. It handles deposits, withdrawals, internal share accounting, strategy-state updates, NAV synchronization, PnL updates, pause control, operator management, and events.

The key point is that we are not executing the full strategy on-chain. The vault is the accounting and state-management layer for an off-chain strategy engine.

### Slide: Boundary Handoff

**Time: 9:05 - 9:35**

The next question is what actually crosses from off-chain analytics into on-chain state.

We keep the payload small. The contract does not receive the full dataset, model, or backtest. It receives only the fields needed for state:

strategy state, reported NAV, PnL delta, signal hash, metadata hash, and report hash.

The hashes anchor the on-chain update to off-chain evidence. The vault stores compact state, while the detailed reasoning stays in generated artifacts and reports.

This keeps the design lightweight, makes the trust boundary explicit, and gives the blockchain a meaningful role without forcing unsuitable computation onto the EVM.

### Slide: Implementation Status

**Time: 9:35 - 10:00**

In terms of implementation, the main demo layers are already built.

Off-chain, we already have data, features, labels, models, signals, backtest, evaluation, reporting, demo, and integration modules.

On-chain, we already have the mock stablecoin, the delta-neutral vault, update functions, events, pause logic, ownership mechanics, and operator control.

Around that, we also have local deployment and update scripts, Foundry tests, and the frontend bridge that packages everything into the presentation page and dashboard.

### Slide: Blockchain Evidence

**Time: 10:00 - 10:25**

This slide answers a simple question: how do we know the chain side is more than a diagram?

The answer is that the vault exposes concrete capabilities: deposit and withdraw, owner and operator separation, strategy-state updates, NAV and PnL synchronization, pause control, and event logging.

These are not just labels on a slide. A deposit mints shares, a NAV update changes reported assets, and a strategy update records state plus hashes. Events provide the audit trail.

Now I will use the live vault console to show the state transitions directly.

### Slide: Live Vault Console

**Time: 10:25 - 11:40**

This is a local presentation simulator of the vault workflow. It mirrors our Solidity vault logic, but it is stable for classroom use because it does not depend on a wallet or live RPC connection.

I will move through four actions quickly.

**Click: Deploy local vault**

This deploys `MockStablecoin` and `DeltaNeutralVault`, and we can see the vault address and event record appear.

**Click: Deposit 10,000 mUSDC**

This represents user funding. Wallet balance decreases, vault cash increases, reported NAV increases, and shares are minted.

**Click: Sync strategy state**

This pushes the selected strategy state from off-chain analytics into the contract-facing state.

**Click: Sync NAV / PnL**

This synchronizes the valuation result. Reported NAV, cumulative PnL, and the report hash all update.

**Click: Withdraw 2,500 mUSDC**

Finally, withdrawal burns shares and returns assets to the user. The main point of this demo is accountability: user-facing state, vault accounting, and event-style logs all change step by step.

### Slide: Chain Demo Flow / Vault Demo Story

**Time: 11:40 - 12:00**

This slide summarizes the same process at the system level.

The flow is deployment, user funding and approval, deposit, strategy-state update, NAV or PnL synchronization, and withdrawal.

The vault stores the parts that should be transparent to users: shares, reported NAV, strategy state, timestamps, and hash references to off-chain artifacts.

So the frontend is not just showing charts. It connects research output to a contract-facing state machine.

### Slide: Caveats

**Time: 12:00 - 12:15**

There are clear limitations. This is a course-project prototype, not an audited DeFi protocol. The vault still uses a trusted owner or operator path, and execution remains off-chain.

These limitations are intentional. The goal is to demonstrate a credible hybrid architecture, not to overclaim production readiness.

### Slide: Next Steps / Closing

**Time: 12:15 - 12:25**

To conclude, our project contributes four things.

First, an end-to-end off-chain research pipeline for funding-rate arbitrage. Second, a cost-aware strategy evaluation workflow. Third, a Solidity vault that represents pooled capital, shares, strategy state, NAV, and PnL. Fourth, a frontend demo that connects the research system to the vault interface.

Future work would include signed or oracle-mediated updates, richer vault lifecycle controls, multi-asset support, and a cleaner operator dashboard.

Thank you. We are happy to take questions.

## Backup Short Version If Time Is Running Out

If the team is behind schedule by more than one minute, shorten as follows:

- Jie Wen: keep the opening, skip detailed outline bullets.
- Hongrui Pan: shorten model explanation to one sentence: "We compare rule-based, baseline ML, and deep-learning signals using post-cost backtesting."
- Qihang Han: combine Implementation Status and Blockchain Evidence into one sentence, and in the live console keep only deploy, deposit, and NAV or PnL sync if necessary.

## Suggested Q&A Answers

### Why not run the model on-chain?

Because model training and backtesting are data-heavy and iterative. Running them on-chain would be expensive and unrealistic. The smart contract should store compact, auditable state: deposits, shares, NAV, PnL, strategy state, and hashes.

### What makes this a blockchain project rather than only a quant project?

The blockchain layer is the vault accounting boundary. Users deposit into a contract, receive internal shares, and observe strategy-state and NAV updates through role-restricted functions and events.

### Is the live vault console a real mainnet transaction?

No. It is a local presentation simulator for stability during class. It mirrors the implemented Solidity vault workflow: deploy, deposit, update strategy, update NAV, withdraw, and event logging. The actual contracts and Foundry tests are in the `contracts/` workspace.

### What is delta-neutral in this project?

It means the strategy tries to reduce directional BTC exposure by pairing opposite legs, such as short perpetual and long spot. The target is not to bet on BTC price direction, but to capture funding-rate carry or basis convergence after costs.

### What is the most important limitation?

The vault currently trusts an owner or operator to synchronize off-chain results. A more production-ready system would need signed updates, oracle verification, stronger operational controls, and security review.
