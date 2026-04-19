"""
Heston Stochastic Volatility Model - Monte Carlo Simulation

The Heston model captures the volatility smile by allowing variance to be stochastic:

    dS_t = r * S_t * dt + sqrt(V_t) * S_t * dW_1
    dV_t = kappa * (theta - V_t) * dt + xi * sqrt(V_t) * dW_2

Where:
    - S_t: Asset price
    - V_t: Instantaneous variance (volatility squared)
    - r: Risk-free rate
    - kappa: Mean reversion speed of variance
    - theta: Long-run variance (V_t reverts to this)
    - xi: Volatility of volatility (vol-of-vol)
    - rho: Correlation between dW_1 and dW_2

The correlation rho < 0 creates the leverage effect (negative skew in equity markets).
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_heston_paths(
    S0,
    V0,
    r,
    kappa,
    theta,
    xi,
    rho,
    T,
    n_steps,
    n_paths,
    seed=None
):
    """
    Simulate asset price paths under the Heston stochastic volatility model.

    Uses the Euler-Maruyama discretization with full truncation scheme
    to prevent negative variance.

    Parameters
    ----------
    S0 : float
        Initial stock price
    V0 : float
        Initial variance (sigma_0^2)
    r : float
        Risk-free interest rate
    kappa : float
        Mean reversion speed of variance
    theta : float
        Long-run variance level
    xi : float
        Volatility of volatility (vol-of-vol)
    rho : float
        Correlation between asset and variance Brownian motions
    T : float
        Time to maturity in years
    n_steps : int
        Number of time steps
    n_paths : int
        Number of Monte Carlo paths
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    S : ndarray of shape (n_steps + 1, n_paths)
        Simulated stock price paths
    V : ndarray of shape (n_steps + 1, n_paths)
        Simulated variance paths
    """
    rng = np.random.default_rng(seed)

    dt = T / n_steps

    # Initialize arrays
    S = np.zeros((n_steps + 1, n_paths))
    V = np.zeros((n_steps + 1, n_paths))

    S[0] = S0
    V[0] = V0

    # Generate correlated Brownian increments
    # dW_1 and dW_2 with correlation rho
    for t in range(1, n_steps + 1):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)

        # Create correlated increments
        dW_S = Z1 * np.sqrt(dt)
        dW_V = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)

        # Full truncation scheme: use max(V, 0) in diffusion terms
        V_plus = np.maximum(V[t - 1], 0)

        # Update variance (CIR process)
        V[t] = (
            V[t - 1]
            + kappa * (theta - V_plus) * dt
            + xi * np.sqrt(V_plus) * dW_V
        )
        # Ensure variance stays non-negative
        V[t] = np.maximum(V[t], 0)

        # Update stock price (log scheme for stability)
        S[t] = S[t - 1] * np.exp(
            (r - 0.5 * V_plus) * dt + np.sqrt(V_plus) * dW_S
        )

    return S, V


def heston_call_price(
    S0,
    K,
    V0,
    r,
    kappa,
    theta,
    xi,
    rho,
    T,
    n_steps=252,
    n_paths=100_000,
    seed=None
):
    """
    Price a European call option using Heston Monte Carlo simulation.

    Parameters
    ----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    V0 : float
        Initial variance
    r : float
        Risk-free rate
    kappa : float
        Mean reversion speed
    theta : float
        Long-run variance
    xi : float
        Vol-of-vol
    rho : float
        Correlation
    T : float
        Time to maturity
    n_steps : int
        Number of time steps (default: 252 for daily)
    n_paths : int
        Number of Monte Carlo paths
    seed : int, optional
        Random seed

    Returns
    -------
    price : float
        Estimated call option price
    std_error : float
        Standard error of the estimate
    """
    S, _ = simulate_heston_paths(
        S0, V0, r, kappa, theta, xi, rho, T, n_steps, n_paths, seed
    )

    # Terminal payoffs
    S_T = S[-1]
    payoffs = np.maximum(S_T - K, 0)

    # Discounted expected payoff
    discount = np.exp(-r * T)
    price = discount * np.mean(payoffs)
    std_error = discount * np.std(payoffs) / np.sqrt(n_paths)

    return price, std_error


def heston_put_price(
    S0,
    K,
    V0,
    r,
    kappa,
    theta,
    xi,
    rho,
    T,
    n_steps=252,
    n_paths=100_000,
    seed=None
):
    """
    Price a European put option using Heston Monte Carlo simulation.

    Parameters
    ----------
    (same as heston_call_price)

    Returns
    -------
    price : float
        Estimated put option price
    std_error : float
        Standard error of the estimate
    """
    S, _ = simulate_heston_paths(
        S0, V0, r, kappa, theta, xi, rho, T, n_steps, n_paths, seed
    )

    # Terminal payoffs
    S_T = S[-1]
    payoffs = np.maximum(K - S_T, 0)

    # Discounted expected payoff
    discount = np.exp(-r * T)
    price = discount * np.mean(payoffs)
    std_error = discount * np.std(payoffs) / np.sqrt(n_paths)

    return price, std_error


def plot_heston_paths(S, V, n_display=10):
    """
    Plot sample paths of stock price and variance.

    Parameters
    ----------
    S : ndarray
        Stock price paths from simulate_heston_paths
    V : ndarray
        Variance paths from simulate_heston_paths
    n_display : int
        Number of paths to display
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    n_steps = S.shape[0] - 1
    time_grid = np.linspace(0, 1, n_steps + 1)

    # Plot stock paths
    for i in range(min(n_display, S.shape[1])):
        axes[0].plot(time_grid, S[:, i], alpha=0.7, linewidth=0.8)
    axes[0].set_ylabel("Stock Price S(t)")
    axes[0].set_title("Heston Model: Sample Paths")
    axes[0].grid(True, alpha=0.3)

    # Plot variance paths
    for i in range(min(n_display, V.shape[1])):
        axes[1].plot(time_grid, V[:, i], alpha=0.7, linewidth=0.8)
    axes[1].set_ylabel("Variance V(t)")
    axes[1].set_xlabel("Time (years)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_terminal_distribution(S, K=None):
    """
    Plot histogram of terminal stock prices.

    Parameters
    ----------
    S : ndarray
        Stock price paths from simulate_heston_paths
    K : float, optional
        Strike price to mark on plot
    """
    S_T = S[-1]

    plt.figure(figsize=(10, 5))
    plt.hist(S_T, bins=100, density=True, alpha=0.7, edgecolor="black")
    plt.xlabel("Terminal Stock Price S(T)")
    plt.ylabel("Density")
    plt.title("Heston Model: Terminal Distribution")

    if K is not None:
        plt.axvline(K, color="red", linestyle="--", label=f"Strike K={K}")
        plt.legend()

    plt.grid(True, alpha=0.3)
    plt.show()


def check_feller_condition(kappa, theta, xi):
    """
    Check if Feller condition is satisfied.

    The Feller condition (2 * kappa * theta > xi^2) ensures that
    the variance process V_t stays strictly positive.

    Parameters
    ----------
    kappa : float
        Mean reversion speed
    theta : float
        Long-run variance
    xi : float
        Vol-of-vol

    Returns
    -------
    bool
        True if Feller condition is satisfied
    """
    feller_value = 2 * kappa * theta
    threshold = xi**2

    satisfied = feller_value > threshold

    print(f"Feller condition check:")
    print(f"  2 * kappa * theta = {feller_value:.4f}")
    print(f"  xi^2 = {threshold:.4f}")
    print(f"  Condition satisfied: {satisfied}")

    if not satisfied:
        print("  ⚠️  Warning: Variance may hit zero. Consider adjusting parameters.")

    return satisfied


if __name__ == "__main__":
    # ========================================
    # Heston Model Parameters (typical values)
    # ========================================
    S0 = 100        # Initial stock price
    K = 100         # Strike price
    V0 = 0.04       # Initial variance (σ₀ = 0.20)
    r = 0.05        # Risk-free rate
    kappa = 2.0     # Mean reversion speed
    theta = 0.04    # Long-run variance (σ_long = 0.20)
    xi = 0.3        # Vol-of-vol
    rho = -0.7      # Correlation (negative = leverage effect)
    T = 1.0         # Time to maturity

    # Simulation parameters
    n_steps = 252   # Daily steps
    n_paths = 100_000

    print("=" * 50)
    print("HESTON MONTE CARLO SIMULATION")
    print("=" * 50)
    print(f"\nModel Parameters:")
    print(f"  S0 = {S0}, K = {K}")
    print(f"  V0 = {V0} (σ₀ = {np.sqrt(V0):.2f})")
    print(f"  r = {r}")
    print(f"  κ (kappa) = {kappa}")
    print(f"  θ (theta) = {theta} (σ_long = {np.sqrt(theta):.2f})")
    print(f"  ξ (xi) = {xi}")
    print(f"  ρ (rho) = {rho}")
    print(f"  T = {T}")
    print(f"\nSimulation: {n_paths:,} paths, {n_steps} steps")

    # Check Feller condition
    print()
    check_feller_condition(kappa, theta, xi)

    # Price options
    print("\n" + "-" * 50)
    print("OPTION PRICING")
    print("-" * 50)

    call_price, call_se = heston_call_price(
        S0, K, V0, r, kappa, theta, xi, rho, T,
        n_steps=n_steps, n_paths=n_paths, seed=42
    )
    put_price, put_se = heston_put_price(
        S0, K, V0, r, kappa, theta, xi, rho, T,
        n_steps=n_steps, n_paths=n_paths, seed=42
    )

    print(f"\nHeston Call Price: {call_price:.4f} ± {call_se:.4f}")
    print(f"Heston Put Price:  {put_price:.4f} ± {put_se:.4f}")

    # Verify put-call parity
    parity_lhs = call_price - put_price
    parity_rhs = S0 - K * np.exp(-r * T)

    print(f"\nPut-Call Parity Check:")
    print(f"  Call - Put = {parity_lhs:.4f}")
    print(f"  S - K*exp(-rT) = {parity_rhs:.4f}")
    print(f"  Difference: {abs(parity_lhs - parity_rhs):.4f} (MC noise)")

    # Compare with Black-Scholes (constant vol)
    print("\n" + "-" * 50)
    print("COMPARISON WITH BLACK-SCHOLES")
    print("-" * 50)

    try:
        from options_pricing.black_scholes import bs_call_price, bs_put_price

        sigma = np.sqrt(V0)  # Use initial vol for BS
        bs_call = bs_call_price(S0, K, T, r, sigma)
        bs_put = bs_put_price(S0, K, T, r, sigma)

        print(f"\nBlack-Scholes (σ = {sigma:.2f}):")
        print(f"  BS Call: {bs_call:.4f}")
        print(f"  BS Put:  {bs_put:.4f}")

        print(f"\nDifference (Heston - BS):")
        print(f"  Call: {call_price - bs_call:+.4f}")
        print(f"  Put:  {put_price - bs_put:+.4f}")

    except ImportError:
        print("\n(Black-Scholes module not found for comparison)")

    # Generate and plot paths
    print("\n" + "-" * 50)
    print("Generating visualization...")
    print("-" * 50)

    S, V = simulate_heston_paths(
        S0, V0, r, kappa, theta, xi, rho, T,
        n_steps=252, n_paths=1000, seed=42
    )

    plot_heston_paths(S, V, n_display=15)
    plot_terminal_distribution(S, K=K)
