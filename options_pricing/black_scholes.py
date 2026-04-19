import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def _validate_inputs(S, K, T, sigma):
    """Basic input validation for Black-Scholes calculations."""
    if S <= 0:
        raise ValueError("Stock price S must be > 0.")
    if K <= 0:
        raise ValueError("Strike price K must be > 0.")
    if T <= 0:
        raise ValueError("Time to expiry T must be > 0.")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be > 0.")


def bs_call_price(S, K, T, r, sigma):
    """
    Black-Scholes price for a European call option.

    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiry in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility

    Returns
    -------
    float
        Call option price
    """
    _validate_inputs(S, K, T, sigma)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price


def bs_put_price(S, K, T, r, sigma):
    """
    Black-Scholes price for a European put option.
    """
    _validate_inputs(S, K, T, sigma)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


def _bs_vega_raw(S, K, T, r, sigma):
    """Unnormalized vega used internally by the IV solver."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    from scipy.stats import norm as _norm
    return S * _norm.pdf(d1) * np.sqrt(T)


def implied_volatility(
    market_price,
    S,
    K,
    T,
    r,
    option_type="call",
    tol=1e-6,
    max_iter=100,
):
    """
    Implied volatility via Newton-Raphson with bisection fallback.

    Newton-Raphson converges quadratically near the solution (typically
    < 10 iterations).  Bisection is used as a safe fallback when the
    Newton step would leave the valid range.

    Parameters
    ----------
    market_price : float
        Observed market option price
    S, K, T, r : float
        Black-Scholes inputs
    option_type : str
        'call' or 'put'
    tol : float
        Convergence tolerance on |model_price - market_price|.
        Default 1e-6 (much tighter than the original 0.01).
    max_iter : int
        Maximum iterations before raising a warning.

    Returns
    -------
    float
        Implied volatility estimate

    Raises
    ------
    ValueError
        If market_price <= 0 or option_type is invalid.
    """
    if market_price <= 0:
        raise ValueError("Market price must be > 0.")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    def price(sigma):
        if option_type == "call":
            return bs_call_price(S, K, T, r, sigma)
        return bs_put_price(S, K, T, r, sigma)

    low_sigma  = 1e-4
    high_sigma = 10.0

    # Bracket check
    if price(high_sigma) < market_price:
        import warnings
        warnings.warn(
            f"Market price {market_price:.4f} exceeds theoretical maximum at "
            f"sigma={high_sigma}. Returning sigma={high_sigma}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return high_sigma

    # Newton-Raphson with bisection fallback
    sigma = 0.20  # initial guess
    for _ in range(max_iter):
        p = price(sigma)
        error = p - market_price

        if abs(error) < tol:
            return sigma

        vega = _bs_vega_raw(S, K, T, r, sigma)

        if abs(vega) > 1e-10:
            sigma_new = sigma - error / vega
        else:
            sigma_new = None  # vega too small → fall through to bisection

        if sigma_new is not None and low_sigma < sigma_new < high_sigma:
            sigma = sigma_new
        else:
            # Bisection step to stay in bracket
            mid = 0.5 * (low_sigma + high_sigma)
            if error > 0:
                high_sigma = sigma
            else:
                low_sigma = sigma
            sigma = mid

    import warnings
    warnings.warn(
        f"implied_volatility did not converge after {max_iter} iterations "
        f"(residual={abs(price(sigma) - market_price):.2e}). "
        "Result may be inaccurate.",
        RuntimeWarning,
        stacklevel=2,
    )
    return sigma


def plot_vol_smile(market_prices, strikes, S, T, r, option_type="call"):
    """
    Plot implied volatility vs strike.

    Parameters
    ----------
    market_prices : list or array
        Market option prices
    strikes : list or array
        Strike prices
    S : float
        Current stock price
    T : float
        Time to expiry
    r : float
        Risk-free rate
    option_type : str
        'call' or 'put'
    """
    if len(market_prices) != len(strikes):
        raise ValueError("market_prices and strikes must have the same length.")

    implied_vols = []
    valid_strikes = []

    for price, K in zip(market_prices, strikes):
        try:
            iv = implied_volatility(price, S, K, T, r, option_type=option_type)
            implied_vols.append(iv)
            valid_strikes.append(K)
        except Exception:
            # Skip invalid points rather than crashing the full smile plot
            continue

    plt.figure(figsize=(8, 5))
    plt.plot(valid_strikes, implied_vols, marker="o")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.title(f"Volatility Smile ({option_type.capitalize()} Options)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Test inputs
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20

    call = bs_call_price(S, K, T, r, sigma)
    put = bs_put_price(S, K, T, r, sigma)

    parity_lhs = call - put
    parity_rhs = S - K * np.exp(-r * T)

    iv = implied_volatility(call, S, K, T, r, option_type="call")

    print("=== Black-Scholes Test ===")
    print(f"Call Price: {call:.4f}")   # ~10.45
    print(f"Put Price:  {put:.4f}")    # ~5.57
    print(f"Put-Call Parity LHS (Call - Put): {parity_lhs:.4f}")
    print(f"Put-Call Parity RHS (S - K*e^(-rT)): {parity_rhs:.4f}")
    print(f"Implied Vol from Call Price: {iv:.4f}")

    # Example smile plot with made-up market prices
    strikes = [80, 90, 100, 110, 120]
    market_prices = [24.0, 16.5, 10.45, 6.8, 4.9]
    plot_vol_smile(market_prices, strikes, S, T, r, option_type="call")
