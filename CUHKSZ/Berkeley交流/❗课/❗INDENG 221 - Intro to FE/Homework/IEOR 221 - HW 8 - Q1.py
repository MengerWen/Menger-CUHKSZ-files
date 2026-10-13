import numpy as np
from math import erf

S0 = 100.0
K = 100.0
T = 1.0
r = 0.06
q = 0.06
sigma = 0.35
n_steps = 100
n_paths = 4000
seed = 102925

rng = np.random.default_rng(seed)
dt = T / n_steps
nudt = (r - q - 0.5 * sigma**2) * dt
sigsdt = sigma * np.sqrt(dt)
disc = np.exp(-r * T)

# Plain MC
Z_plain = rng.standard_normal((n_paths, n_steps))
logS_plain = np.log(S0) + np.cumsum(nudt + sigsdt * Z_plain, axis=1)
ST_plain = np.exp(logS_plain[:, -1])
payoff_plain = np.maximum(ST_plain - K, 0.0)
price_plain = disc * payoff_plain.mean()
se_plain = disc * payoff_plain.std(ddof=1) / np.sqrt(n_paths)

# Antithetic
Z = rng.standard_normal((n_paths // 2, n_steps))
logS_pos = np.log(S0) + np.cumsum(nudt + sigsdt * Z, axis=1)
logS_neg = np.log(S0) + np.cumsum(nudt - sigsdt * Z, axis=1)
ST_pos = np.exp(logS_pos[:, -1])
ST_neg = np.exp(logS_neg[:, -1])
pay_pos = np.maximum(ST_pos - K, 0.0)
pay_neg = np.maximum(ST_neg - K, 0.0)
pair_avg = 0.5 * (pay_pos + pay_neg)
price_anti = disc * pair_avg.mean()
se_anti = disc * pair_avg.std(ddof=1) / np.sqrt(pair_avg.size)

# Control variate using S_T
g = payoff_plain
f = ST_plain
mu_f = S0 * np.exp((r - q) * T)
cov_gf = np.cov(g, f, ddof=1)[0,1]
var_f = np.var(f, ddof=1)
beta_hat = cov_gf / var_f if var_f > 0 else 0.0
h = g - beta_hat * (f - mu_f)
price_cv = disc * h.mean()
se_cv = disc * h.std(ddof=1) / np.sqrt(h.size)

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2)))
d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
bsm_price = S0 * np.exp(-q * T) * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)

print(f"Plain MC        : price = {price_plain:.2f}, SE = {se_plain:.2f}")
print(f"Antithetic      : price = {price_anti:.2f}, SE = {se_anti:.2f}")
print(f"Control Variate : price = {price_cv:.2f}, SE = {se_cv:.2f}, beta = {beta_hat:.4f}")
print(f"BSM (reference) : price = {bsm_price:.2f}")