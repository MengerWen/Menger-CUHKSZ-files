import math
from typing import List, Tuple

def exact_pi_star(y0: float, h: float, gamma: float) -> float:
    """
    Exact indifference probability for lottery (h, -h, pi) at wealth y0
    under CRRA utility.

    Uses a numerically stable reformulation to avoid subtracting 1 inside CRRA utility.

    For gamma != 1:
        u(y) = (y^(1-gamma) - 1)/(1-gamma) = (1 - y^(1-gamma))/(gamma-1)

    Plugging into:
        pi* = (u(y0) - u(y0-h)) / (u(y0+h) - u(y0-h))

    The constant (gamma-1) cancels, yielding:
        pi* = ((y0-h)^(1-g) - y0^(1-g)) / ((y0-h)^(1-g) - (y0+h)^(1-g))

    For gamma == 1 (log utility):
        u(y)=ln y, so:
        pi* = (ln y0 - ln(y0-h)) / (ln(y0+h) - ln(y0-h))
    """
    if y0 <= 0:
        raise ValueError("y0 must be positive.")
    if h <= 0:
        raise ValueError("h must be positive.")
    if y0 - h <= 0:
        raise ValueError("y0 - h must be positive (wealth must stay positive).")

    # Log utility special case
    if abs(gamma - 1.0) < 1e-12:
        u0 = math.log(y0)
        u_minus = math.log(y0 - h)
        u_plus = math.log(y0 + h)
        denom = u_plus - u_minus
        if denom == 0.0:
            raise ZeroDivisionError("Denominator is exactly zero; check parameters.")
        return (u0 - u_minus) / denom

    # Stable CRRA-based expression
    p = 1.0 - gamma  # exponent
    a = (y0 - h) ** p
    b = (y0) ** p
    c = (y0 + h) ** p

    denom = a - c
    if denom == 0.0:
        raise ZeroDivisionError("Denominator is exactly zero; check parameters.")
    return (a - b) / denom

def approx_pi_star(y0: float, h: float, gamma: float) -> float:
    """
    Approximation from class:
        pi* ≈ 1/2 + 1/4 * RRA(y0) * k
    For CRRA, RRA(y)=gamma and k=h/y0, so:
        pi* ≈ 1/2 + 1/4 * gamma * (h/y0)
    """
    k = h / y0
    return 0.5 + 0.25 * gamma * k

def compute_table(
    gammas: List[float],
    y0: float = 10.0,
    h: float = 0.1
) -> List[Tuple[float, float, float, float]]:
    """
    Returns rows of:
        (gamma, exact_pi, approx_pi, exact_minus_approx)
    """
    rows = []
    for g in gammas:
        pi_exact = exact_pi_star(y0, h, g)
        pi_approx = approx_pi_star(y0, h, g)
        rows.append((g, pi_exact, pi_approx, pi_exact - pi_approx))
    return rows

def format_rows_markdown(rows: List[Tuple[float, float, float, float]]) -> str:
    """
    Formats rows into a Markdown table.
    """
    lines = []
    lines.append("| gamma | exact pi* | approx pi* | exact - approx |")
    lines.append("|---:|---:|---:|---:|")
    for g, pe, pa, diff in rows:
        lines.append(f"| {g:g} | {pe:.10f} | {pa:.10f} | {diff:.10e} |")
    return "\n".join(lines)

if __name__ == "__main__":
    gammas = [0.5, 2, 3, 10, 20]
    rows = compute_table(gammas, y0=10.0, h=0.1)
    print(format_rows_markdown(rows))
