"""
OP-6: Option Greeks

Analytical Black-Scholes Greeks (Delta, Vega, Gamma, Theta, Rho, Vanna, Volga)
and Monte Carlo finite-difference Greeks for the Heston model.

Heston Greeks use pathwise / bump-and-reprice with the same seed so that
MC noise partially cancels in the finite difference.
"""

import numpy as np
from scipy.stats import norm

from options_pricing.heston import price_european_call as heston_call


# ---------------------------------------------------------------------------
# Black-Scholes Greeks (closed-form)
# ---------------------------------------------------------------------------

def _d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_delta(S, K, T, r, sigma, option_type="call"):
    """dV/dS. Call in [0,1], put in [-1,0]."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def bs_vega(S, K, T, r, sigma):
    """dV/d(sigma). Same for calls and puts. Scaled by 1% (per vol point)."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100.0


def bs_gamma(S, K, T, r, sigma):
    """d^2V/dS^2. Same for calls and puts."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_theta(S, K, T, r, sigma, option_type="call"):
    """
    dV/dt — time decay per calendar day (negative = loses value each day).
    Divided by 365 to convert annual theta to daily.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    decay = -S * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    if option_type == "call":
        return (decay - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    return (decay + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0


def bs_rho(S, K, T, r, sigma, option_type="call"):
    """
    dV/dr — sensitivity to interest rate.
    Scaled by 1/100 (per 1 basis-point move in r).
    """
    _, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0


def bs_vanna(S, K, T, r, sigma):
    """
    d(Delta)/d(sigma) = d(Vega)/dS.

    Measures how delta changes as volatility moves — important for
    vol-of-vol risk and barrier option hedging.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return -norm.pdf(d1) * d2 / sigma


def bs_volga(S, K, T, r, sigma):
    """
    d^2V/d(sigma)^2 — curvature of price with respect to vol.

    Also called Vomma. Positive for long options: they gain convexity
    as vol rises. Key input for vol-of-vol hedging.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    vega_unscaled = S * norm.pdf(d1) * np.sqrt(T)
    return vega_unscaled * d1 * d2 / sigma


def bs_charm(S, K, T, r, sigma, option_type="call"):
    """
    d(Delta)/dt — rate of change of delta with respect to time.

    Also called delta decay. Useful for delta-hedging near expiry.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    term = norm.pdf(d1) * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
    if option_type == "call":
        return -term / 365.0
    return -term / 365.0  # same formula; sign convention via put-call parity


# ---------------------------------------------------------------------------
# Heston Greeks (Monte Carlo finite difference)
# ---------------------------------------------------------------------------

def heston_greek_fd(
    S, K, T, r, v0, kappa, theta, xi, rho,
    greek,
    h=1e-2,
    n_paths=8000,
    n_steps=252,
    seed=42,
):
    """
    Central finite difference estimate of a Heston Greek.

    Uses identical seeds for bump and base prices so that MC noise
    partially cancels in the difference — common trick to improve
    finite-difference accuracy for stochastic pricers.

    Parameters
    ----------
    greek : str
        One of 'delta', 'vega', 'gamma'.
    h : float
        Bump size. For delta/gamma bump S; for vega bump v0.
    """
    def price(S_=S, v0_=v0):
        return heston_call(
            S0=S_, K=K, T=T, r=r,
            v0=v0_, kappa=kappa, theta=theta, xi=xi, rho=rho,
            n_paths=n_paths, n_steps=n_steps, seed=seed,
        )

    if greek == "delta":
        return (price(S_ = S + h) - price(S_ = S - h)) / (2 * h)

    elif greek == "vega":
        v0_up   = v0 + h
        v0_down = max(1e-6, v0 - h)
        return (price(v0_=v0_up) - price(v0_=v0_down)) / (v0_up - v0_down)

    elif greek == "gamma":
        p_up   = price(S_ = S + h)
        p_mid  = price(S_ = S)
        p_down = price(S_ = S - h)
        return (p_up - 2 * p_mid + p_down) / h**2

    else:
        raise ValueError("greek must be 'delta', 'vega', or 'gamma'")


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def calculate_all_bs_greeks(S, K, T, r, sigma, dS=0.01):
    """
    Return all analytical BS Greeks as a dict.

    Includes: delta, vega (per 1% vol), gamma, theta (daily), rho,
    vanna, volga (vomma), charm (delta decay).
    """
    return {
        "delta": bs_delta(S, K, T, r, sigma),
        "vega":  bs_vega(S, K, T, r, sigma),
        "gamma": bs_gamma(S, K, T, r, sigma),
        "theta": bs_theta(S, K, T, r, sigma),
        "rho":   bs_rho(S, K, T, r, sigma),
        "vanna": bs_vanna(S, K, T, r, sigma),
        "volga": bs_volga(S, K, T, r, sigma),
        "charm": bs_charm(S, K, T, r, sigma),
    }


def calculate_all_greeks(S, K, T, r, params, dS=0.01, n_paths=8000, n_steps=252, seed=42):
    """Return Heston Greeks (delta, vega, gamma) via MC finite difference."""
    v0    = params["v0"]
    kappa = params["kappa"]
    theta = params["theta"]
    xi    = params["xi"]
    rho   = params["rho"]

    delta = heston_greek_fd(S, K, T, r, v0, kappa, theta, xi, rho, "delta",
                            h=dS, n_paths=n_paths, n_steps=n_steps, seed=seed)
    vega  = heston_greek_fd(S, K, T, r, v0, kappa, theta, xi, rho, "vega",
                            h=dS, n_paths=n_paths, n_steps=n_steps, seed=seed)
    gamma = heston_greek_fd(S, K, T, r, v0, kappa, theta, xi, rho, "gamma",
                            h=dS, n_paths=n_paths, n_steps=n_steps, seed=seed)
    return {"delta": delta, "vega": vega, "gamma": gamma}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    S     = 100
    T     = 1.0
    r     = 0.05
    sigma = 0.20
    v0    = 0.0638
    kappa = 0.9334
    theta = 0.0258
    xi    = 0.2530
    rho   = 0.0693
    strikes = np.arange(80, 121, 10)

    print("\nOP-6: Greeks Comparison Table\n")
    header = (
        f"{'Strike':>8} {'BS_Δ':>10} {'H_Δ':>12}"
        f" {'BS_V':>10} {'H_V':>12}"
        f" {'BS_Γ':>10} {'H_Γ':>12}"
        f" {'BS_Θ(day)':>12} {'BS_Vanna':>12} {'BS_Volga':>12}"
    )
    print(header)
    print("-" * len(header))

    for K in strikes:
        bs_d = bs_delta(S, K, T, r, sigma)
        bs_v = bs_vega(S, K, T, r, sigma)
        bs_g = bs_gamma(S, K, T, r, sigma)
        bs_t = bs_theta(S, K, T, r, sigma)
        bs_va = bs_vanna(S, K, T, r, sigma)
        bs_vg = bs_volga(S, K, T, r, sigma)

        h_d = heston_greek_fd(S, K, T, r, v0, kappa, theta, xi, rho, "delta")
        h_v = heston_greek_fd(S, K, T, r, v0, kappa, theta, xi, rho, "vega")
        h_g = heston_greek_fd(S, K, T, r, v0, kappa, theta, xi, rho, "gamma")

        print(
            f"{K:8.2f} {bs_d:10.4f} {h_d:12.4f}"
            f" {bs_v:10.4f} {h_v:12.4f}"
            f" {bs_g:10.4f} {h_g:12.4f}"
            f" {bs_t:12.5f} {bs_va:12.4f} {bs_vg:12.4f}"
        )

    print("\nNote: Heston Greeks are MC finite-difference estimates.")
    print("BS Vega and Rho are per 1%; Theta is per calendar day.")
