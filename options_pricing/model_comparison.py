import numpy as np
import matplotlib.pyplot as plt

from options_pricing.black_scholes import (
    bs_call_price,
    implied_volatility,
)
from options_pricing.heston import price_european_call


def generate_market_smile(S, strikes, T, r, base_vol=0.20):
    """
    Create synthetic market implied vols with a smile + slight put skew.

    Formula:
        iv = base_vol + 0.0001*(K - S)^2 + 0.002*max(S - K, 0)/S

    Returns
    -------
    market_ivs : np.ndarray
    market_prices : np.ndarray
    """
    strikes = np.asarray(strikes, dtype=float)

    market_ivs = (
        base_vol
        + 0.0001 * (strikes - S) ** 2
        + 0.002 * np.maximum(S - strikes, 0) / S
    )

    market_prices = np.array([
        bs_call_price(S, K, T, r, iv)
        for K, iv in zip(strikes, market_ivs)
    ])

    return market_ivs, market_prices


def bs_fit(S, strikes, T, r, atm_vol):
    """
    Black-Scholes fit using one constant volatility for all strikes.

    Returns
    -------
    bs_prices : np.ndarray
    bs_ivs : np.ndarray
    """
    strikes = np.asarray(strikes, dtype=float)

    bs_prices = np.array([
        bs_call_price(S, K, T, r, atm_vol)
        for K in strikes
    ])

    bs_ivs = np.array([
        implied_volatility(price, S, K, T, r, option_type="call", tol=1e-4)
        for K, price in zip(strikes, bs_prices)
    ])

    return bs_prices, bs_ivs


def heston_fit(S, strikes, T, r, heston_params, n_paths=20000, n_steps=252, seed=42):
    """
    Price calls with Heston, then back out Black-Scholes implied vols.

    heston_params should contain:
        v0, kappa, theta, xi, rho

    Returns
    -------
    heston_prices : np.ndarray
    heston_ivs : np.ndarray
    """
    strikes = np.asarray(strikes, dtype=float)

    v0 = heston_params["v0"]
    kappa = heston_params["kappa"]
    theta = heston_params["theta"]
    xi = heston_params["xi"]
    rho = heston_params["rho"]

    heston_prices = []
    heston_ivs = []

    for i, K in enumerate(strikes):
        price = price_european_call(
            S0=S,
            K=K,
            T=T,
            r=r,
            v0=v0,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed + i,
        )
        heston_prices.append(price)

        iv = implied_volatility(price, S, K, T, r, option_type="call", tol=1e-4)
        heston_ivs.append(iv)

    return np.array(heston_prices), np.array(heston_ivs)


def plot_comparison(strikes, market_ivs, bs_ivs, heston_ivs):
    """
    Plot market, Black-Scholes, and Heston implied volatility curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, market_ivs, "o", label="Market IVs")
    plt.plot(strikes, bs_ivs, "--", linewidth=2, label="Black-Scholes Fit")
    plt.plot(strikes, heston_ivs, "-", linewidth=2, label="Heston Fit")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.title("Volatility Smile: Market vs Black-Scholes vs Heston")
    plt.grid(True)
    plt.legend()
    plt.show()


def fit_error(market_ivs, model_ivs):
    """
    Compute RMSE and max absolute error.
    """
    market_ivs = np.asarray(market_ivs, dtype=float)
    model_ivs = np.asarray(model_ivs, dtype=float)

    errors = market_ivs - model_ivs
    rmse = np.sqrt(np.mean(errors ** 2))
    max_error = np.max(np.abs(errors))

    return rmse, max_error


def print_error_report(model_name, rmse, max_error):
    print(f"{model_name}:")
    print(f"  RMSE:      {rmse:.6f}")
    print(f"  Max Error: {max_error:.6f}")


if __name__ == "__main__":
    # Market / option setup
    S = 100
    T = 1.0
    r = 0.05
    strikes = np.arange(80, 121, 5)

    # Step 1: synthetic market smile
    market_ivs, market_prices = generate_market_smile(S, strikes, T, r, base_vol=0.20)

    # Step 2: Black-Scholes fit using ATM vol only
    atm_index = np.argmin(np.abs(strikes - S))
    atm_vol = market_ivs[atm_index]
    bs_prices, bs_ivs = bs_fit(S, strikes, T, r, atm_vol)

    # Step 3: Heston fit
    heston_params = {
        "v0": 0.04,
        "kappa": 2.0,
        "theta": 0.04,
        "xi": 0.3,
        "rho": -0.7,
    }

    print("\n=== OP-3: MODEL COMPARISON AGAINST MARKET DATA ===\n")
    print(f"Spot Price: {S}")
    print(f"Strikes: {strikes.tolist()}")
    print(f"ATM Vol used by Black-Scholes: {atm_vol:.4f}\n")

    print("Synthetic Market IVs:")
    for K, iv in zip(strikes, market_ivs):
        print(f"  K={K:>3}  IV={iv:.4f}")

    print("\nRunning Heston MC pricing (this may take a moment)...")

    heston_prices, heston_ivs = heston_fit(
        S, strikes, T, r, heston_params,
        n_paths=20000,
        n_steps=252,
        seed=42,
    )

    # Step 4: errors
    bs_rmse, bs_max = fit_error(market_ivs, bs_ivs)
    heston_rmse, heston_max = fit_error(market_ivs, heston_ivs)

    print("\nFit Error Summary:")
    print_error_report("Black-Scholes", bs_rmse, bs_max)
    print_error_report("Heston", heston_rmse, heston_max)

    if heston_rmse < bs_rmse:
        improvement = bs_rmse / heston_rmse
        print(f"\nHeston fits the smile better by {improvement:.2f}x")
    else:
        print("\nHeston did not outperform Black-Scholes with fixed parameters.")
        print("This is expected before calibration (see OP-4).")

    # Step 5: plot
    plot_comparison(strikes, market_ivs, bs_ivs, heston_ivs)
