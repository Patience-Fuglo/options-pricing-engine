"""
Options Pricing Tool - Central Entry Point

Run with: python -m options_pricing.pricing_tool
"""

import numpy as np
from options_pricing.black_scholes import bs_call_price, bs_put_price, implied_volatility
from options_pricing.heston import (
    price_european_call,
    price_european_put,
    feller_condition,
)
from options_pricing.model_comparison import (
    generate_market_smile,
    bs_fit,
    heston_fit,
    fit_error,
)


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run_black_scholes_demo():
    """Demonstrate Black-Scholes pricing (OP-1)."""
    print_header("OP-1: BLACK-SCHOLES MODEL")

    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

    call = bs_call_price(S, K, T, r, sigma)
    put = bs_put_price(S, K, T, r, sigma)

    print(f"\nParameters:")
    print(f"  S = {S}, K = {K}, T = {T}, r = {r}, sigma = {sigma}")

    print(f"\nResults:")
    print(f"  Call Price: {call:.4f}")
    print(f"  Put Price:  {put:.4f}")

    # Put-call parity check
    parity_lhs = call - put
    parity_rhs = S - K * np.exp(-r * T)
    print(f"\nPut-Call Parity:")
    print(f"  Call - Put = {parity_lhs:.4f}")
    print(f"  S - K*exp(-rT) = {parity_rhs:.4f}")

    # Implied volatility
    iv = implied_volatility(call, S, K, T, r, option_type="call")
    print(f"\nImplied Vol Recovery: {iv:.4f}")

    return call, put


def run_heston_demo():
    """Demonstrate Heston Monte Carlo pricing (OP-2)."""
    print_header("OP-2: HESTON STOCHASTIC VOLATILITY MODEL")

    # Parameters
    S0, K = 100, 100
    v0 = 0.04           # Initial variance (sigma_0 = 0.20)
    r = 0.05
    kappa = 2.0         # Mean reversion speed
    theta = 0.04        # Long-run variance
    xi = 0.3            # Vol-of-vol
    rho = -0.7          # Correlation (leverage effect)
    T = 1.0

    print(f"\nParameters:")
    print(f"  S0 = {S0}, K = {K}, T = {T}")
    print(f"  v0 = {v0} (sigma_0 = {np.sqrt(v0):.2f})")
    print(f"  r = {r}, kappa = {kappa}, theta = {theta}")
    print(f"  xi = {xi}, rho = {rho}")

    feller_ok = feller_condition(kappa, theta, xi)
    print(f"\nFeller condition (2*kappa*theta > xi^2): {feller_ok}")

    print(f"\nSimulating 20,000 paths...")

    call = price_european_call(
        S0, K, T, r, v0, kappa, theta, xi, rho,
        n_paths=20000, n_steps=252, seed=42
    )
    put = price_european_put(
        S0, K, T, r, v0, kappa, theta, xi, rho,
        n_paths=20000, n_steps=252, seed=42
    )

    print(f"\nResults:")
    print(f"  Heston Call: {call:.4f}")
    print(f"  Heston Put:  {put:.4f}")

    return call, put


def run_model_comparison():
    """Compare BS vs Heston on volatility smile (OP-3)."""
    print_header("OP-3: MODEL COMPARISON (VOLATILITY SMILE)")

    # Market setup
    S = 100
    T = 1.0
    r = 0.05
    strikes = np.arange(80, 121, 10)  # Fewer strikes for speed

    # Generate synthetic market smile
    market_ivs, market_prices = generate_market_smile(S, strikes, T, r, base_vol=0.20)

    print(f"\nSynthetic Market IVs (with smile/skew):")
    for K, iv in zip(strikes, market_ivs):
        print(f"  K={K:>3}  IV={iv:.4f}")

    # Black-Scholes fit (constant ATM vol)
    atm_index = np.argmin(np.abs(strikes - S))
    atm_vol = market_ivs[atm_index]
    bs_prices, bs_ivs = bs_fit(S, strikes, T, r, atm_vol)

    # Heston fit
    heston_params = {
        "v0": 0.04,
        "kappa": 2.0,
        "theta": 0.04,
        "xi": 0.3,
        "rho": -0.7,
    }

    print(f"\nRunning Heston MC pricing...")
    heston_prices, heston_ivs = heston_fit(
        S, strikes, T, r, heston_params,
        n_paths=10000, n_steps=252, seed=42
    )

    # Errors
    bs_rmse, bs_max = fit_error(market_ivs, bs_ivs)
    heston_rmse, heston_max = fit_error(market_ivs, heston_ivs)

    print(f"\nFit Error Summary:")
    print(f"  Black-Scholes RMSE: {bs_rmse:.6f}")
    print(f"  Heston RMSE:        {heston_rmse:.6f}")

    print(f"\nNote: Heston uses fixed params (not calibrated).")
    print(f"      Run OP-4 to calibrate and improve the fit.")


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("         OPTIONS PRICING TOOL")
    print("    Black-Scholes | Heston | Model Comparison")
    print("=" * 60)

    # Run all demos
    run_black_scholes_demo()
    run_heston_demo()
    run_model_comparison()

    print_header("COMPLETE")
    print("\nAvailable modules:")
    print("  python -m options_pricing.black_scholes    (OP-1)")
    print("  python -m options_pricing.heston           (OP-2)")
    print("  python -m options_pricing.model_comparison (OP-3)")
    print("  python -m options_pricing.pricing_tool     (all)")
    print("\nNext: OP-4 (Calibration)")


if __name__ == "__main__":
    main()
