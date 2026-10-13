# Trading Rules

## Cointegration Statistical Arbitrage

- Use a 240-month rolling training window on log wealth levels.
- Re-estimate the Johansen cointegration relation every 12 months.
- Use the first cointegrating vector when the Johansen trace test rejects at least one rank at the 5% level.
- Compute a spread z-score from the last 60 in-sample spread observations.
- Enter when the z-score exceeds +/-1.5; exit when it reverts inside +/-0.5.
- Convert the cointegration vector into a dollar-neutral weight vector by de-meaning the coefficients and normalizing gross exposure to one.

## Mean-Variance Strategies

- Use a 120-month rolling estimation window for expected returns and covariance matrices.
- The plug-in mean-variance strategy uses sample mean and sample covariance with full investment and no short sales.
- The improved plug-in strategy uses Ledoit-Wolf shrinkage covariance, no short sales, an upper weight bound of 0.35, and an L1 turnover penalty in the monthly optimization.

## Transaction Costs and Turnover

- Transaction costs are set to 10 basis points per one-way turnover unit.
- Turnover is measured as one-half of the L1 distance between current post-return weights and next target weights.
- All signals and weights are formed using information available up to the previous month-end, so the backtests avoid look-ahead bias.
