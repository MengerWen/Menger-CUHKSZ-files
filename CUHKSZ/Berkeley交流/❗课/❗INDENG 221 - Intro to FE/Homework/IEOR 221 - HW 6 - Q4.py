import math

# Given parameters
mu = 0.16  # expected return per annum
sigma = 0.30  # volatility per annum
S0 = 50.0  # current stock price
trading_days_per_year = 250
t = 1 / trading_days_per_year  # time in years for one day

# Part (a): Expected stock price at the end of the next day
expected_st = S0 * math.exp(mu * t)
print(f"(a) Expected stock price: {expected_st:.4f}")

# Part (b): Standard deviation of the stock price at the end of the next day
var_st = S0**2 * math.exp(2 * mu * t) * (math.exp(sigma**2 * t) - 1)
std_dev_st = math.sqrt(var_st)
print(f"(b) Standard deviation: {std_dev_st:.4f}")

# Part (c): 95% confidence limits
# Mean and SD of ln(St)
mean_ln_st = math.log(S0) + (mu - 0.5 * sigma**2) * t
sd_ln_st = sigma * math.sqrt(t)

# Z-score for 95% confidence (two-tailed)
z = 1.96

# Lower and upper bounds for ln(St)
lower_ln = mean_ln_st - z * sd_ln_st
upper_ln = mean_ln_st + z * sd_ln_st

# Exponentiate to get bounds for St
lower_st = math.exp(lower_ln)
upper_st = math.exp(upper_ln)

print(f"(c) 95% confidence limits: {lower_st:.4f} to {upper_st:.4f}")