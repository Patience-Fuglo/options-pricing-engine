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


def implied_volatility(market_price, S, K, T, r, option_type="call", tol=0.01, max_iter=200):
    """
    Estimate implied volatility using the bisection method.

    Parameters
    ----------
    market_price : float
        Observed market option price
    S, K, T, r : floats
        Black-Scholes inputs
    option_type : str
        'call' or 'put'
    tol : float
        Acceptable pricing error
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    float
        Implied volatility estimate
    """
    if market_price <= 0:
        raise ValueError("Market price must be > 0.")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    low_sigma = 0.01
    high_sigma = 5.0

    for _ in range(max_iter):
        mid_sigma = 0.5 * (low_sigma + high_sigma)

        if option_type == "call":
            model_price = bs_call_price(S, K, T, r, mid_sigma)
        else:
            model_price = bs_put_price(S, K, T, r, mid_sigma)

        error = model_price - market_price

        if abs(error) < tol:
            return mid_sigma

        if error > 0:
            high_sigma = mid_sigma
        else:
            low_sigma = mid_sigma

    return mid_sigma


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
