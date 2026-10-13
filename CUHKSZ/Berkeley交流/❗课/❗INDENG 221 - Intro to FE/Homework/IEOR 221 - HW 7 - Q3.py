from math import log, sqrt, exp, erf

def N(x: float) -> float:
    """Standard normal CDF via error function.
    Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
    """
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def d1_d2(S: float, K: float, r: float, q: float, sigma: float, T: float):
    """Return (d1, d2) for Black–Scholes with continuous dividend yield."""
    if sigma <= 0 or T <= 0:
        raise ValueError("sigma and T must be positive")
    v = sigma * sqrt(T)
    mu = (r - q + 0.5 * sigma * sigma) * T
    d1 = (log(S / K) + mu) / v
    d2 = d1 - v
    return d1, d2


def bs_call_put_div(S: float, K: float, r: float, q: float, sigma: float, T: float):
    """Black–Scholes (European) prices with continuous dividend yield.
    Returns (call, put).
    """
    d1, d2 = d1_d2(S, K, r, q, sigma, T)
    disc_q = exp(-q * T)
    disc_r = exp(-r * T)
    call = S * disc_q * N(d1) - K * disc_r * N(d2)
    put  = -S * disc_q * N(-d1) + K * disc_r * N(-d2)
    return call, put

# (a) Dividend yield from parity

def parity_dividend_yield(S: float, K: float, r: float, T: float, call: float, put: float) -> float:
    """Solve for continuous dividend yield q from put–call parity with dividends:
       c - p = S * e^{-qT} - K * e^{-rT}  =>  e^{-qT} = (c - p + K e^{-rT}) / S
       Return q (per annum, continuous compounding).
    """
    e_minus_qT = (call - put + K * exp(-r * T)) / S
    if e_minus_qT <= 0:
        raise ValueError("Inputs imply nonpositive e^{-qT}; check prices.")
    q = - (1.0 / T) * log(e_minus_qT)
    return q

# (b) Implied volatility solver

def implied_vol_call_div(
    S: float, K: float, r: float, q: float, T: float, call_price: float,
    tol: float = 1e-8, max_iter: int = 100, lo: float = 1e-8, hi: float = 5.0
) -> float:
    """Implied vol for a call with continuous dividend yield using bisection.
       Robust and dependency-free.
    """
    # No-arbitrage bounds for a European call with dividends:
    disc_q = exp(-q * T)
    disc_r = exp(-r * T)
    lower_bound = max(0.0, S * disc_q - K * disc_r)
    upper_bound = S * disc_q  # loose but safe
    if not (lower_bound - 1e-12 <= call_price <= upper_bound + 1e-12):
        raise ValueError("Call price outside no-arbitrage bounds given S, K, r, q, T.")

    f_lo = bs_call_put_div(S, K, r, q, lo, T)[0] - call_price
    f_hi = bs_call_put_div(S, K, r, q, hi, T)[0] - call_price

    # Ensure the bracket has opposite signs; expand if needed
    expand = 0
    while f_lo * f_hi > 0 and expand < 20:
        # expand hi until we bracket a root
        hi *= 2.0
        f_hi = bs_call_put_div(S, K, r, q, hi, T)[0] - call_price
        expand += 1
    if f_lo * f_hi > 0:
        raise RuntimeError("Failed to bracket implied volatility root.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = bs_call_put_div(S, K, r, q, mid, T)[0] - call_price
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        # Bisection step
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    # If not converged, return midpoint
    return 0.5 * (lo + hi)

if __name__ == "__main__":
    S0 = 1000.0
    K  = 950.0
    r  = 0.04      # continuous compounding
    T  = 0.25      # 3 months
    c_obs = 78.0
    p_obs = 26.0

    # (a) Dividend yield from parity
    q_hat = parity_dividend_yield(S0, K, r, T, c_obs, p_obs)
    print(f"(a) Dividend yield (cont.): {100*q_hat:.4f}%")

    # (b) Implied vol from call price with that q
    sigma_hat = implied_vol_call_div(S0, K, r, q_hat, T, c_obs)
    print(f"(b) Implied volatility: {100*sigma_hat:.4f}%")