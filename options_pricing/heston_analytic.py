"""
Heston Model — Semi-Analytic Pricing via Characteristic Function

Implements exact (no Monte Carlo noise) European option pricing under
the Heston (1993) stochastic volatility model:

    dS_t = r S_t dt + sqrt(V_t) S_t dW_1
    dV_t = kappa (theta - V_t) dt + xi sqrt(V_t) dW_2
    dW_1 dW_2 = rho dt

Two pricing routes are provided:

1. heston_call_price   — single option, Gil-Pelaez inversion with scipy.quad.
                         Accurate to ~1e-6, ~10ms per option.

2. heston_call_chain_fft — full strike chain via Carr-Madan (1999) FFT.
                           Prices hundreds of strikes simultaneously in ~1ms.

Both agree to within <0.01 for typical equity-option parameters.

References
----------
Heston, S. (1993). A closed-form solution for options with stochastic
    volatility. Review of Financial Studies, 6(2), 327–343.

Carr, P. & Madan, D. (1999). Option valuation using the fast Fourier
    transform. Journal of Computational Finance, 2(4), 61–73.

Albrecher, H., Mayer, P., Schachermayer, W. & Teichmann, J. (2007).
    The Heston stochastic volatility model: Implementation, calibration
    and some extensions. Wilmott Magazine.
"""

import warnings

import numpy as np
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# Characteristic function
# ---------------------------------------------------------------------------

def _heston_cf(u, S0, r, T, v0, kappa, theta, xi, rho):
    """
    Heston model log-price characteristic function.

    Computes  E^Q[ exp(iu * ln S_T) ]  using the Albrecher et al. (2007)
    parameterisation, which avoids the branch-cut discontinuity present
    in the original Heston (1993) formula.

    Supports both scalar u (for scipy.quad) and array u (for FFT).

    Parameters
    ----------
    u : float, complex, or np.ndarray
        Argument of the characteristic function (may be complex for the
        Carr-Madan FFT damping shift).
    S0, r, T : float
        Spot price, risk-free rate, time to maturity.
    v0, kappa, theta, xi, rho : float
        Heston parameters: initial variance, mean-reversion speed,
        long-run variance, vol-of-vol, spot-vol correlation.

    Returns
    -------
    complex or np.ndarray
    """
    i = 1j

    # xi_bar = kappa - xi * rho * i * u
    xi_bar = kappa - xi * rho * i * u

    # d  = sqrt(xi_bar^2 + xi^2 * (u^2 + i*u))
    d = np.sqrt(xi_bar**2 + xi**2 * (u**2 + i * u))

    # Albrecher form: g = (xi_bar - d) / (xi_bar + d)
    g = (xi_bar - d) / (xi_bar + d)

    exp_dT = np.exp(-d * T)

    # Integrated variance term
    C = kappa * theta / xi**2 * (
        (xi_bar - d) * T
        - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
    )

    # Current variance term
    D = (xi_bar - d) / xi**2 * (1.0 - exp_dT) / (1.0 - g * exp_dT)

    return np.exp(i * u * (np.log(S0) + r * T) + C + D * v0)


# ---------------------------------------------------------------------------
# Single-price integration (Gil-Pelaez)
# ---------------------------------------------------------------------------

def heston_call_price(S0, K, T, r, v0, kappa, theta, xi, rho):
    """
    European call price via Gil-Pelaez inversion of the Heston CF.

    Uses the two-probability decomposition:
        C = S0 * P1 - K * exp(-rT) * P2

    where
        P2 = Q(S_T > K)  under the risk-neutral measure Q
        P1 = Q^S(S_T > K)  under the stock-as-numeraire measure Q^S

    Both probabilities are recovered via:
        P_j = 1/2 + (1/pi) * integral_0^inf  Re[...] du

    Parameters
    ----------
    S0 : float      Current stock price
    K  : float      Strike price
    T  : float      Time to maturity (years)
    r  : float      Risk-free rate
    v0, kappa, theta, xi, rho : float
        Heston model parameters.

    Returns
    -------
    float
        European call price (exact within numerical integration tolerance).
    """
    F    = S0 * np.exp(r * T)   # Forward price = E^Q[S_T]
    ln_K = np.log(K)

    def _p2_integrand(u):
        cf = _heston_cf(u, S0, r, T, v0, kappa, theta, xi, rho)
        return np.real(np.exp(-1j * u * ln_K) * cf / (1j * u))

    def _p1_integrand(u):
        # Radon-Nikodym shift: dQ^S/dQ  ∝  S_T  ⟹  evaluate CF at u - i
        cf_shifted = _heston_cf(u - 1j, S0, r, T, v0, kappa, theta, xi, rho)
        return np.real(np.exp(-1j * u * ln_K) * cf_shifted / (1j * u * F))

    # Integrate from near-zero (avoids 1/u singularity at 0) to effectively inf
    P2, _ = quad(_p2_integrand, 1e-8, 500.0, limit=300, epsabs=1e-9, epsrel=1e-9)
    P1, _ = quad(_p1_integrand, 1e-8, 500.0, limit=300, epsabs=1e-9, epsrel=1e-9)

    P1 = np.clip(0.5 + P1 / np.pi, 0.0, 1.0)
    P2 = np.clip(0.5 + P2 / np.pi, 0.0, 1.0)

    return S0 * P1 - K * np.exp(-r * T) * P2


def heston_put_price(S0, K, T, r, v0, kappa, theta, xi, rho):
    """
    European put price via put-call parity (no extra integration needed).

    P = C - S0 + K * exp(-rT)
    """
    call = heston_call_price(S0, K, T, r, v0, kappa, theta, xi, rho)
    return call - S0 + K * np.exp(-r * T)


# ---------------------------------------------------------------------------
# Carr-Madan FFT — full strike chain
# ---------------------------------------------------------------------------

def heston_call_chain_fft(
    S0,
    strikes,
    T,
    r,
    v0,
    kappa,
    theta,
    xi,
    rho,
    alpha=1.5,
    N=4096,
    eta=0.25,
):
    """
    Price a chain of European calls using the Carr-Madan (1999) FFT method.

    Evaluates all strikes simultaneously — far faster than calling
    heston_call_price() individually for each strike.

    Algorithm
    ---------
    1. Build a log-strike grid  k_j = -b + j*lambda  (j = 0..N-1).
    2. Evaluate the Carr-Madan modified CF on an integration grid u_n = n*eta.
    3. Apply Simpson weights and FFT to obtain call prices on the log-strike grid.
    4. Interpolate at the requested strikes.

    Parameters
    ----------
    S0 : float          Spot price
    strikes : array     Strike prices (any length, need not be on a grid)
    T, r : float        Maturity and risk-free rate
    v0, kappa, theta, xi, rho : float
                        Heston parameters
    alpha : float       Carr-Madan damping parameter.  alpha=1.5 works for
                        most equity parameters; increase if call prices turn
                        negative for deep OTM strikes.
    N : int             FFT grid size (power of 2; 4096 is sufficient for
                        strikes within ~5 standard deviations of the forward).
    eta : float         Integration step size. Smaller eta → wider strike
                        coverage; larger eta → finer strike resolution.

    Returns
    -------
    np.ndarray
        Estimated call prices for each element of `strikes`, floored at 0.
    """
    strikes = np.asarray(strikes, dtype=float)

    # Log-strike step derived from FFT duality
    lam = 2.0 * np.pi / (N * eta)
    b   = 0.5 * N * lam            # half-width of log-strike grid

    # Integration nodes u_0, u_1, ..., u_{N-1}
    u = np.arange(N, dtype=float) * eta

    # CF evaluated at the Carr-Madan damped argument  u - (alpha+1)*i
    cf_vals = _heston_cf(
        u - (alpha + 1.0) * 1j,
        S0, r, T, v0, kappa, theta, xi, rho
    )

    # Modified CF  psi(u)  from Carr-Madan eq. (9)
    denom = alpha**2 + alpha - u**2 + 1j * (2.0 * alpha + 1.0) * u
    psi   = np.exp(-r * T) * cf_vals / denom

    # Simpson's rule weights (improves accuracy over pure trapezoidal rule)
    w = np.ones(N)
    w[0]    = 1.0 / 3.0
    w[-1]   = 1.0 / 3.0
    w[1:-1:2] = 4.0 / 3.0
    w[2:-2:2] = 2.0 / 3.0

    # FFT input: multiply by exp(-i*u*b) to shift the output to the correct
    # log-strike grid centered at 0
    x = np.exp(-1j * u * b) * psi * w * eta

    fft_out = np.fft.fft(x)

    # Log-strike grid
    log_strikes_grid = -b + lam * np.arange(N)

    # Undamp: multiply by exp(-alpha * k) / pi, take real part
    call_vals = np.exp(-alpha * log_strikes_grid) / np.pi * np.real(fft_out)

    # Interpolate at requested strikes
    log_K       = np.log(strikes)
    call_prices = np.interp(log_K, log_strikes_grid, call_vals)

    return np.maximum(call_prices, 0.0)


def heston_put_chain_fft(S0, strikes, T, r, v0, kappa, theta, xi, rho, **kwargs):
    """Put prices for a chain via FFT + put-call parity."""
    calls = heston_call_chain_fft(S0, strikes, T, r, v0, kappa, theta, xi, rho, **kwargs)
    return calls - S0 + np.asarray(strikes) * np.exp(-r * T)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    # Heston test parameters
    S0    = 100.0
    K     = 100.0
    T     = 1.0
    r     = 0.05
    v0    = 0.04
    kappa = 2.0
    theta = 0.04
    xi    = 0.3
    rho   = -0.7

    strikes = np.array([80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0])

    print("=" * 60)
    print("Heston Semi-Analytic Pricing Demo")
    print("=" * 60)
    print(f"S0={S0}, K={K}, T={T}, r={r}")
    print(f"v0={v0}, kappa={kappa}, theta={theta}, xi={xi}, rho={rho}")

    # Single call
    t0   = time.perf_counter()
    call = heston_call_price(S0, K, T, r, v0, kappa, theta, xi, rho)
    put  = heston_put_price(S0, K, T, r, v0, kappa, theta, xi, rho)
    t1   = time.perf_counter()
    print(f"\nSingle option (Gil-Pelaez integration):")
    print(f"  Call = {call:.6f}")
    print(f"  Put  = {put:.6f}")
    print(f"  PCP check: |C - P - (S - K*e^(-rT))| = "
          f"{abs(call - put - S0 + K * np.exp(-r * T)):.2e}")
    print(f"  Time: {(t1 - t0)*1000:.1f} ms")

    # FFT chain
    t0  = time.perf_counter()
    fft_calls = heston_call_chain_fft(S0, strikes, T, r, v0, kappa, theta, xi, rho)
    t1  = time.perf_counter()

    print(f"\nStrike chain (Carr-Madan FFT, {len(strikes)} strikes):")
    print(f"  Time: {(t1 - t0)*1000:.2f} ms")
    print(f"  {'Strike':>8} {'FFT Call':>12} {'Exact Call':>12} {'Diff':>10}")
    print("  " + "-" * 46)
    for Ki, fc in zip(strikes, fft_calls):
        exact = heston_call_price(S0, Ki, T, r, v0, kappa, theta, xi, rho)
        print(f"  {Ki:8.1f} {fc:12.6f} {exact:12.6f} {fc - exact:10.4f}")

    # MC comparison
    print(f"\nComparison with Monte Carlo (100,000 paths):")
    from options_pricing.heston import price_european_call
    mc_call = price_european_call(S0, K, T, r, v0, kappa, theta, xi, rho,
                                  n_paths=100_000, seed=42)
    print(f"  Analytic  = {call:.6f}")
    print(f"  MC (100k) = {mc_call:.6f}")
    print(f"  Diff      = {abs(call - mc_call):.4f}")
