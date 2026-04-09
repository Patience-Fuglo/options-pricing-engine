import numpy as np
from scipy.optimize import differential_evolution

from options_pricing.heston import price_european_call
from options_pricing.model_comparison import generate_market_smile


def objective(params, market_data, S, T, r, n_paths=5000, n_steps=252, seed=123):
    """
    Objective function for Heston calibration.

    Parameters
    ----------
    params : list or np.ndarray
        [v0, kappa, theta, xi, rho]
    market_data : list of dict
        Each dict should contain:
            {"strike": K, "market_price": price}
    S : float
        Spot price
    T : float
        Time to expiry
    r : float
        Risk-free rate

    Returns
    -------
    float
        Sum of squared pricing errors
    """
    v0, kappa, theta, xi, rho = params

    total_error = 0.0

    for i, option in enumerate(market_data):
        K = option["strike"]
        market_price = option["market_price"]

        try:
            model_price = price_european_call(
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
        except Exception:
            return 1e10

        error = model_price - market_price
        total_error += error ** 2

    return total_error


def get_bounds():
    """
    Parameter bounds for Heston calibration.
    """
    return [
        (0.01, 1.0),    # v0
        (0.1, 10.0),    # kappa
        (0.01, 1.0),    # theta
        (0.01, 2.0),    # xi
        (-0.99, 0.99),  # rho
    ]


def calibrate(
    market_data,
    S,
    T,
    r,
    maxiter=100,
    popsize=10,
    n_paths=5000,
    n_steps=252,
    seed=123,
):
    """
    Calibrate Heston parameters to market prices using differential evolution.
    """
    bounds = get_bounds()

    result = differential_evolution(
        objective,
        bounds=bounds,
        args=(market_data, S, T, r, n_paths, n_steps, seed),
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        polish=True,
        disp=True,
    )

    return result


def interpret_kappa(kappa):
    if kappa < 1.0:
        return "volatility reverts slowly to its long-run mean"
    if kappa < 3.0:
        return "volatility reverts to its mean at a moderate pace"
    return "volatility reverts to its mean fairly quickly"


def interpret_theta(theta):
    sigma_long_run = np.sqrt(theta)
    return f"long-run volatility is about {sigma_long_run:.2%}"


def interpret_xi(xi):
    if xi < 0.2:
        return "volatility itself is fairly stable"
    if xi < 0.6:
        return "volatility shows moderate variation over time"
    return "volatility is highly unstable and changes aggressively"


def interpret_rho(rho):
    if rho < -0.5:
        return "strong negative correlation — volatility tends to spike when price falls"
    if rho < 0:
        return "mild negative correlation between price and volatility"
    if rho < 0.5:
        return "mild positive correlation between price and volatility"
    return "strong positive correlation between price and volatility"


def print_calibrated_params(result):
    """
    Print calibrated parameters with plain-English interpretations.
    """
    v0, kappa, theta, xi, rho = result.x

    print("\n=== CALIBRATED HESTON PARAMETERS ===")
    print(f"v0    = {v0:.4f}  (initial variance; initial vol = {np.sqrt(v0):.2%})")
    print(f"kappa = {kappa:.4f}  ({interpret_kappa(kappa)})")
    print(f"theta = {theta:.4f}  ({interpret_theta(theta)})")
    print(f"xi    = {xi:.4f}  ({interpret_xi(xi)})")
    print(f"rho   = {rho:.4f}  ({interpret_rho(rho)})")
    print(f"\nObjective value (SSE): {result.fun:.6f}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")


def pricing_error_report(market_data, calibrated_params, S, T, r, n_paths=10000, n_steps=252, seed=999):
    """
    Reprice all options with calibrated params and print error report.
    """
    v0, kappa, theta, xi, rho = calibrated_params

    print("\n=== PRICING ERROR REPORT ===")
    print(f"{'Strike':>8} {'Market':>12} {'Model':>12} {'Error':>12} {'% Error':>12}")
    print("-" * 60)

    sq_errors = []
    abs_errors = []

    for i, option in enumerate(market_data):
        K = option["strike"]
        market_price = option["market_price"]

        model_price = price_european_call(
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

        error = model_price - market_price
        pct_error = (error / market_price) * 100 if market_price != 0 else 0.0

        sq_errors.append(error ** 2)
        abs_errors.append(abs(error))

        print(f"{K:8.2f} {market_price:12.4f} {model_price:12.4f} {error:12.4f} {pct_error:11.2f}%")

    rmse = np.sqrt(np.mean(sq_errors))
    mae = np.mean(abs_errors)

    print("-" * 60)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

    return rmse, mae


def build_synthetic_market_data(S, strikes, T, r, base_vol=0.20):
    """
    Build synthetic market data from OP-3 smile.
    """
    market_ivs, market_prices = generate_market_smile(S, strikes, T, r, base_vol=base_vol)

    market_data = []
    for K, price, iv in zip(strikes, market_prices, market_ivs):
        market_data.append({
            "strike": float(K),
            "market_price": float(price),
            "market_iv": float(iv),
        })

    return market_data, market_ivs, market_prices


if __name__ == "__main__":
    # Synthetic market setup from OP-3
    S = 100
    T = 1.0
    r = 0.05
    strikes = np.array([85, 95, 100, 105, 115])  # Fewer strikes for speed

    market_data, market_ivs, market_prices = build_synthetic_market_data(
        S=S,
        strikes=strikes,
        T=T,
        r=r,
        base_vol=0.20,
    )

    print("=== OP-4: HESTON CALIBRATION ===")
    print(f"Spot: {S}, T: {T}, r: {r}")
    print(f"Strikes: {strikes.tolist()}")
    print("\nSynthetic Market Prices:")
    for row in market_data:
        print(
            f"  K={int(row['strike']):>3}  "
            f"Price={row['market_price']:.4f}  "
            f"IV={row['market_iv']:.4f}"
        )

    print("\nStarting calibration (fast mode)...")
    print("Note: Using reduced paths/iterations for speed.\n")

    result = calibrate(
        market_data=market_data,
        S=S,
        T=T,
        r=r,
        maxiter=10,     # reduced for speed
        popsize=5,      # reduced for speed
        n_paths=1000,   # reduced for speed
        n_steps=50,     # reduced for speed
        seed=123,
    )

    print_calibrated_params(result)

    rmse, mae = pricing_error_report(
        market_data=market_data,
        calibrated_params=result.x,
        S=S,
        T=T,
        r=r,
        n_paths=3000,   # reduced for speed
        n_steps=100,    # reduced for speed
        seed=999,
    )

    print("\n=== SUMMARY ===")
    print(f"Final RMSE: {rmse:.4f}")
    print(f"Final MAE:  {mae:.4f}")

    if mae < 0.50:
        print("Average pricing error is comfortably below $0.50.")
    else:
        print("Average pricing error is above $0.50. Increase calibration iterations/paths for a tighter fit.")
