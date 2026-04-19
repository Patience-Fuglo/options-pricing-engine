"""
OP-4: Heston Model Calibration (Analytic-CF Version)

Calibrates Heston parameters (v0, kappa, theta, xi, rho) to market
option prices by minimising the sum of squared pricing errors.

Uses the semi-analytic characteristic-function pricer (heston_analytic)
instead of Monte Carlo simulation, giving:
    - ~100x speed improvement (no simulation paths needed)
    - No MC noise in the objective function → smoother gradient landscape
    - Reproducible results (deterministic)

The calibration uses scipy's Nelder-Mead by default (gradient-free,
robust to non-smooth objectives) with a SLSQP polish step at the end.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution

from options_pricing.heston_analytic import heston_call_chain_fft
from options_pricing.heston import feller_condition
from options_pricing.black_scholes import bs_call_price, implied_volatility
from options_pricing.model_comparison import generate_market_smile


# ---------------------------------------------------------------------------
# Bounds and parameter constraints
# ---------------------------------------------------------------------------

_BOUNDS = [
    (1e-4, 1.0),     # v0    initial variance
    (0.1,  15.0),    # kappa mean-reversion speed
    (1e-4, 1.0),     # theta long-run variance
    (0.01, 3.0),     # xi    vol-of-vol
    (-0.99, 0.99),   # rho   spot-vol correlation
]


def _is_valid(v0, kappa, theta, xi, rho):
    return (
        v0 > 0 and kappa > 0 and theta > 0 and xi > 0
        and -1.0 < rho < 1.0
    )


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def objective(params, S, strikes, T, r, market_prices, weights=None):
    """
    Weighted sum of squared pricing errors (analytic prices).

    Parameters
    ----------
    params : array-like
        [v0, kappa, theta, xi, rho]
    weights : np.ndarray or None
        Per-strike weights. If None, equal weights are used.
        Common choice: 1/vega (vega-weighted calibration) which down-weights
        deep OTM options whose bid-ask spreads are large.

    Returns
    -------
    float
        Weighted SSE.
    """
    v0, kappa, theta, xi, rho = params

    if not _is_valid(v0, kappa, theta, xi, rho):
        return 1e10

    try:
        model_prices = heston_call_chain_fft(
            S, strikes, T, r, v0, kappa, theta, xi, rho
        )
    except Exception:
        return 1e10

    errors = model_prices - market_prices

    if weights is None:
        return float(np.sum(errors**2))
    return float(np.sum(weights * errors**2))


# ---------------------------------------------------------------------------
# Main calibration routines
# ---------------------------------------------------------------------------

def calibrate_heston_fast(
    S,
    strikes,
    T,
    r,
    market_prices,
    weights=None,
    method="nelder-mead",
    verbose=True,
):
    """
    Calibrate Heston parameters to market call prices.

    Parameters
    ----------
    S : float               Spot price.
    strikes : array         Strike prices (calls only).
    T : float               Common time to maturity.
    r : float               Risk-free rate.
    market_prices : array   Observed call prices.
    weights : array or None Per-strike calibration weights (see `objective`).
    method : str            'nelder-mead' (default, robust) or
                            'de' (differential evolution, global search,
                            slower but better for difficult surfaces).
    verbose : bool          Print progress.

    Returns
    -------
    dict
        Calibrated parameters and fit quality metrics.
    """
    strikes       = np.asarray(strikes, dtype=float)
    market_prices = np.asarray(market_prices, dtype=float)

    if verbose:
        print(f"Calibrating Heston (analytic CF, method={method})...")

    args = (S, strikes, T, r, market_prices, weights)
    x0   = np.array([0.04, 2.0, 0.04, 0.3, -0.5])

    if method == "nelder-mead":
        result = minimize(
            objective,
            x0,
            args=args,
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8, "adaptive": True},
        )
        # Polish with gradient-based method inside bounds
        if result.success or result.fun < 1.0:
            bounds_scipy = [(lo, hi) for lo, hi in _BOUNDS]
            result2 = minimize(
                objective,
                result.x,
                args=args,
                method="L-BFGS-B",
                bounds=bounds_scipy,
                options={"ftol": 1e-12, "gtol": 1e-8},
            )
            if result2.fun < result.fun:
                result = result2

    elif method == "de":
        result = differential_evolution(
            objective,
            bounds=_BOUNDS,
            args=args,
            maxiter=500,
            popsize=15,
            tol=1e-8,
            polish=True,
            seed=42,
            disp=verbose,
        )
    else:
        raise ValueError("method must be 'nelder-mead' or 'de'")

    v0, kappa, theta, xi, rho = result.x

    # Clip to valid range (in case L-BFGS-B drifted slightly outside)
    v0    = np.clip(v0,    _BOUNDS[0][0], _BOUNDS[0][1])
    kappa = np.clip(kappa, _BOUNDS[1][0], _BOUNDS[1][1])
    theta = np.clip(theta, _BOUNDS[2][0], _BOUNDS[2][1])
    xi    = np.clip(xi,    _BOUNDS[3][0], _BOUNDS[3][1])
    rho   = np.clip(rho,   _BOUNDS[4][0], _BOUNDS[4][1])

    final_prices = heston_call_chain_fft(S, strikes, T, r, v0, kappa, theta, xi, rho)
    errors       = final_prices - market_prices
    rmse         = float(np.sqrt(np.mean(errors**2)))
    mae          = float(np.mean(np.abs(errors)))

    return {
        "v0":    v0,
        "kappa": kappa,
        "theta": theta,
        "xi":    xi,
        "rho":   rho,
        "sse":   float(result.fun),
        "rmse":  rmse,
        "mae":   mae,
        "success": result.success,
        "message": result.message,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_results(params):
    """Print calibration results with qualitative interpretations."""
    print("\n" + "=" * 55)
    print("CALIBRATED HESTON PARAMETERS")
    print("=" * 55)
    print(f"  v0    = {params['v0']:.6f}  (initial vol = {np.sqrt(params['v0']):.2%})")
    print(f"  kappa = {params['kappa']:.6f}  (mean-reversion speed)")
    print(f"  theta = {params['theta']:.6f}  (long-run vol = {np.sqrt(params['theta']):.2%})")
    print(f"  xi    = {params['xi']:.6f}  (vol-of-vol)")
    print(f"  rho   = {params['rho']:.6f}  (spot-vol correlation)")
    print(f"\nFeller condition: {feller_condition(params['kappa'], params['theta'], params['xi'])}")
    print(f"  2*kappa*theta = {2*params['kappa']*params['theta']:.4f}")
    print(f"  xi^2          = {params['xi']**2:.4f}")
    print(f"\nFit quality: RMSE={params['rmse']:.6f}  MAE={params['mae']:.6f}")
    print(f"Convergence: {params['message']}")


def compute_fit_quality(S, strikes, T, r, market_prices, params):
    """Compute per-strike pricing errors with calibrated parameters."""
    strikes       = np.asarray(strikes, dtype=float)
    market_prices = np.asarray(market_prices, dtype=float)

    model_prices = heston_call_chain_fft(
        S, strikes, T, r,
        params["v0"], params["kappa"], params["theta"], params["xi"], params["rho"]
    )

    print("\n" + "=" * 60)
    print("PRICING ERROR REPORT")
    print("=" * 60)
    print(f"{'Strike':>8} {'Market':>10} {'Model':>10} {'Error':>10} {'% Error':>10}")
    print("-" * 52)

    errors = model_prices - market_prices
    for K, mkt, mdl, err in zip(strikes, market_prices, model_prices, errors):
        pct = 100 * err / mkt if mkt > 0 else 0.0
        print(f"{K:8.0f} {mkt:10.4f} {mdl:10.4f} {err:+10.4f} {pct:+9.2f}%")

    rmse = float(np.sqrt(np.mean(errors**2)))
    mae  = float(np.mean(np.abs(errors)))
    print("-" * 52)
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")

    return rmse, mae


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("OP-4: HESTON CALIBRATION (ANALYTIC CF)")
    print("=" * 55)

    S       = 100.0
    T       = 1.0
    r       = 0.05
    strikes = np.array([80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0])

    market_ivs, market_prices = generate_market_smile(S, strikes, T, r, base_vol=0.20)

    print(f"\nMarket Setup: S={S}, T={T}, r={r}")
    print(f"Strikes: {strikes.tolist()}")
    print("\nSynthetic Market Data:")
    for K, price, iv in zip(strikes, market_prices, market_ivs):
        print(f"  K={K:>5.0f}  Price={price:.4f}  IV={iv:.4f}")

    params = calibrate_heston_fast(S, strikes, T, r, market_prices, verbose=True)
    print_results(params)

    rmse, mae = compute_fit_quality(S, strikes, T, r, market_prices, params)

    # Compare with Black-Scholes flat vol
    print("\n" + "=" * 55)
    print("COMPARISON: Heston vs Black-Scholes (flat ATM vol)")
    print("=" * 55)
    atm_vol  = market_ivs[np.argmin(np.abs(strikes - S))]
    bs_errors = np.array([bs_call_price(S, K, T, r, atm_vol) for K in strikes]) - market_prices
    bs_rmse  = float(np.sqrt(np.mean(bs_errors**2)))
    print(f"Black-Scholes RMSE (flat vol={atm_vol:.2%}): {bs_rmse:.6f}")
    print(f"Calibrated Heston RMSE:                     {rmse:.6f}")
    if rmse < bs_rmse:
        print(f"=> Heston is {bs_rmse/rmse:.2f}x more accurate than flat-vol BS")
    else:
        print("=> Increase calibration iterations for a tighter fit")
