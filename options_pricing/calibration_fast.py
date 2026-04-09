"""
OP-4: Heston Model Calibration (Fast Version)

Calibrate Heston parameters to fit market implied volatilities.
Uses minimal MC paths for speed during calibration.
"""

import numpy as np
from scipy.optimize import minimize

from options_pricing.heston import price_european_call, feller_condition
from options_pricing.black_scholes import bs_call_price, implied_volatility
from options_pricing.model_comparison import generate_market_smile


def heston_price_fast(S, K, T, r, v0, kappa, theta, xi, rho, n_paths=500, seed=42):
    """Fast Heston pricing with minimal paths."""
    try:
        return price_european_call(
            S0=S, K=K, T=T, r=r,
            v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
            n_paths=n_paths, n_steps=50, seed=seed
        )
    except:
        return np.nan


def objective(params, S, strikes, T, r, market_prices):
    """Sum of squared errors between model and market prices."""
    v0, kappa, theta, xi, rho = params
    
    # Bounds check
    if v0 <= 0.001 or kappa <= 0 or theta <= 0.001 or xi <= 0:
        return 1e10
    if rho < -0.99 or rho > 0.99:
        return 1e10
    
    sse = 0.0
    for i, (K, mkt_price) in enumerate(zip(strikes, market_prices)):
        model_price = heston_price_fast(S, K, T, r, v0, kappa, theta, xi, rho, 
                                        n_paths=500, seed=42+i)
        if np.isnan(model_price):
            return 1e10
        sse += (model_price - mkt_price) ** 2
    
    return sse


def calibrate_heston_fast(S, strikes, T, r, market_prices, verbose=True):
    """
    Fast Heston calibration using Nelder-Mead.
    """
    # Initial guess: (v0, kappa, theta, xi, rho)
    x0 = np.array([0.04, 2.0, 0.04, 0.3, -0.5])
    
    if verbose:
        print("Calibrating Heston (fast mode)...")
    
    result = minimize(
        objective,
        x0,
        args=(S, strikes, T, r, market_prices),
        method='Nelder-Mead',
        options={'maxiter': 100, 'xatol': 0.01, 'fatol': 0.1}
    )
    
    v0, kappa, theta, xi, rho = result.x
    
    return {
        'v0': max(0.001, v0),
        'kappa': max(0.1, kappa),
        'theta': max(0.001, theta),
        'xi': max(0.01, xi),
        'rho': np.clip(rho, -0.99, 0.99),
        'sse': result.fun,
        'success': result.success
    }


def print_results(params):
    """Print calibration results."""
    print("\n" + "=" * 50)
    print("CALIBRATED HESTON PARAMETERS")
    print("=" * 50)
    print(f"  v0    = {params['v0']:.4f}  (initial vol = {np.sqrt(params['v0']):.2%})")
    print(f"  kappa = {params['kappa']:.4f}  (mean reversion speed)")
    print(f"  theta = {params['theta']:.4f}  (long-run vol = {np.sqrt(params['theta']):.2%})")
    print(f"  xi    = {params['xi']:.4f}  (vol of vol)")
    print(f"  rho   = {params['rho']:.4f}  (correlation)")
    print(f"\nFeller condition: {feller_condition(params['kappa'], params['theta'], params['xi'])}")
    print(f"SSE: {params['sse']:.4f}")


def compute_fit_quality(S, strikes, T, r, market_prices, params, n_paths=2000):
    """Compute pricing errors with calibrated params."""
    print("\n" + "=" * 50)
    print("PRICING ERROR REPORT")
    print("=" * 50)
    print(f"{'Strike':>8} {'Market':>10} {'Model':>10} {'Error':>10}")
    print("-" * 40)
    
    errors = []
    for i, (K, mkt) in enumerate(zip(strikes, market_prices)):
        model = heston_price_fast(
            S, K, T, r,
            params['v0'], params['kappa'], params['theta'], params['xi'], params['rho'],
            n_paths=n_paths, seed=999+i
        )
        err = model - mkt
        errors.append(err)
        print(f"{K:8.0f} {mkt:10.4f} {model:10.4f} {err:+10.4f}")
    
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    mae = np.mean(np.abs(errors))
    
    print("-" * 40)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    
    return rmse, mae


if __name__ == "__main__":
    print("=" * 50)
    print("OP-4: HESTON CALIBRATION (FAST MODE)")
    print("=" * 50)
    
    # Setup
    S = 100
    T = 1.0
    r = 0.05
    strikes = np.array([90, 100, 110])  # Just 3 strikes for speed
    
    # Generate synthetic market data
    market_ivs, market_prices = generate_market_smile(S, strikes, T, r, base_vol=0.20)
    
    print(f"\nMarket Setup: S={S}, T={T}, r={r}")
    print(f"Strikes: {strikes.tolist()}")
    print("\nSynthetic Market Data:")
    for K, price, iv in zip(strikes, market_prices, market_ivs):
        print(f"  K={K:>3}  Price={price:.4f}  IV={iv:.4f}")
    
    # Calibrate
    print()
    params = calibrate_heston_fast(S, strikes, T, r, market_prices)
    print_results(params)
    
    # Fit quality
    rmse, mae = compute_fit_quality(S, strikes, T, r, market_prices, params)
    
    # Compare with BS
    print("\n" + "=" * 50)
    print("COMPARISON WITH BLACK-SCHOLES")
    print("=" * 50)
    
    atm_vol = market_ivs[np.argmin(np.abs(strikes - S))]
    bs_errors = []
    for K, mkt in zip(strikes, market_prices):
        bs_price = bs_call_price(S, K, T, r, atm_vol)
        bs_errors.append(bs_price - mkt)
    
    bs_rmse = np.sqrt(np.mean(np.array(bs_errors)**2))
    print(f"Black-Scholes RMSE (flat vol): {bs_rmse:.4f}")
    print(f"Calibrated Heston RMSE:        {rmse:.4f}")
    
    if rmse < bs_rmse:
        print(f"\nHeston beats BS by {bs_rmse/rmse:.2f}x!")
    else:
        print("\nHeston did not beat BS (MC noise or need more iterations)")
    
    print("\n✅ OP-4 Complete!")
