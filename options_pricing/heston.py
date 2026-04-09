import numpy as np
import matplotlib.pyplot as plt


def generate_correlated_normals(n_paths, n_steps, rho, seed=None):
    """
    Generate two arrays of standard normal random variables with correlation rho.

    Returns
    -------
    z1, z2 : np.ndarray
        Arrays of shape (n_paths, n_steps)
    """
    if not -1 <= rho <= 1:
        raise ValueError("rho must be between -1 and 1.")

    rng = np.random.default_rng(seed)

    z1 = rng.normal(0.0, 1.0, size=(n_paths, n_steps))
    eps = rng.normal(0.0, 1.0, size=(n_paths, n_steps))
    z2 = rho * z1 + np.sqrt(1 - rho**2) * eps

    return z1, z2


def heston_simulate(
    S0,
    v0,
    kappa,
    theta,
    xi,
    rho,
    r,
    T,
    n_paths=10000,
    n_steps=252,
    seed=None,
):
    """
    Simulate Heston model paths using Euler discretization.

    Heston model:
        dS = r S dt + sqrt(v) S dW2
        dv = kappa (theta - v) dt + xi sqrt(v) dW1

    Returns
    -------
    prices : np.ndarray
        Simulated stock price paths, shape (n_paths, n_steps + 1)
    variances : np.ndarray
        Simulated variance paths, shape (n_paths, n_steps + 1)
    """
    if S0 <= 0:
        raise ValueError("S0 must be > 0.")
    if v0 < 0:
        raise ValueError("v0 must be >= 0.")
    if theta < 0:
        raise ValueError("theta must be >= 0.")
    if xi < 0:
        raise ValueError("xi must be >= 0.")
    if kappa < 0:
        raise ValueError("kappa must be >= 0.")
    if T <= 0:
        raise ValueError("T must be > 0.")
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive integers.")

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    z1, z2 = generate_correlated_normals(n_paths, n_steps, rho, seed=seed)

    prices = np.zeros((n_paths, n_steps + 1))
    variances = np.zeros((n_paths, n_steps + 1))

    prices[:, 0] = S0
    variances[:, 0] = v0

    for t in range(n_steps):
        v_prev = np.maximum(variances[:, t], 0.0)

        v_new = (
            variances[:, t]
            + kappa * (theta - v_prev) * dt
            + xi * np.sqrt(v_prev) * sqrt_dt * z1[:, t]
        )
        v_new = np.maximum(v_new, 0.0)

        S_new = prices[:, t] * np.exp(
            (r - 0.5 * v_prev) * dt
            + np.sqrt(v_prev) * sqrt_dt * z2[:, t]
        )

        variances[:, t + 1] = v_new
        prices[:, t + 1] = S_new

    return prices, variances


def price_european_call(
    S0,
    K,
    T,
    r,
    v0,
    kappa,
    theta,
    xi,
    rho,
    n_paths=10000,
    n_steps=252,
    seed=None,
):
    """
    Price a European call option under the Heston model via Monte Carlo.
    """
    prices, _ = heston_simulate(
        S0=S0,
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        r=r,
        T=T,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )

    terminal_prices = prices[:, -1]
    payoffs = np.maximum(terminal_prices - K, 0.0)
    option_price = np.exp(-r * T) * np.mean(payoffs)

    return option_price


def price_european_put(
    S0,
    K,
    T,
    r,
    v0,
    kappa,
    theta,
    xi,
    rho,
    n_paths=10000,
    n_steps=252,
    seed=None,
):
    """
    Price a European put option under the Heston model via Monte Carlo.
    """
    prices, _ = heston_simulate(
        S0=S0,
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        r=r,
        T=T,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )

    terminal_prices = prices[:, -1]
    payoffs = np.maximum(K - terminal_prices, 0.0)
    option_price = np.exp(-r * T) * np.mean(payoffs)

    return option_price


def plot_paths(paths, n_show=20):
    """
    Plot sample simulated price paths.
    """
    n_show = min(n_show, paths.shape[0])

    plt.figure(figsize=(10, 6))
    for i in range(n_show):
        plt.plot(paths[i], linewidth=1)

    plt.title(f"Heston Monte Carlo Sample Paths ({n_show} shown)")
    plt.xlabel("Time Step")
    plt.ylabel("Stock Price")
    plt.grid(True)
    plt.show()


def feller_condition(kappa, theta, xi):
    """
    Check the Feller condition: 2*kappa*theta > xi^2
    """
    return 2 * kappa * theta > xi**2


if __name__ == "__main__":
    # Test parameters from the task
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05

    v0 = 0.04
    kappa = 2.0
    theta = 0.04
    xi = 0.3
    rho = -0.7

    n_paths = 10000
    n_steps = 252
    seed = 42

    print("=== Heston Monte Carlo Test ===")
    print(f"S0={S0}, K={K}, T={T}, r={r}")
    print(f"v0={v0}, kappa={kappa}, theta={theta}, xi={xi}, rho={rho}")
    print(f"Feller condition satisfied: {feller_condition(kappa, theta, xi)}")

    call_price = price_european_call(
        S0, K, T, r, v0, kappa, theta, xi, rho,
        n_paths=n_paths, n_steps=n_steps, seed=seed
    )

    put_price = price_european_put(
        S0, K, T, r, v0, kappa, theta, xi, rho,
        n_paths=n_paths, n_steps=n_steps, seed=seed
    )

    print(f"Heston Call Price: {call_price:.4f}")
    print(f"Heston Put Price:  {put_price:.4f}")

    paths, variances = heston_simulate(
        S0, v0, kappa, theta, xi, rho, r, T,
        n_paths=200, n_steps=n_steps, seed=seed
    )
    plot_paths(paths, n_show=20)
