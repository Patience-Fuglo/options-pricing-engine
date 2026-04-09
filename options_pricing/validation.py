import numpy as np

from options_pricing.black_scholes import implied_volatility
from options_pricing.heston import price_european_call
from options_pricing.calibration import calibrate, build_synthetic_market_data


def split_options(market_data, calibration_strikes, validation_strikes):
    """
    Split market data into calibration and validation sets by strike.
    """
    calibration_set = [
        row for row in market_data
        if row["strike"] in calibration_strikes
    ]

    validation_set = [
        row for row in market_data
        if row["strike"] in validation_strikes
    ]

    return calibration_set, validation_set


def price_validation_set(validation_data, params, S, T, r, n_paths=10000, n_steps=252, seed=1000):
    """
    Price each validation option using calibrated Heston parameters.
    """
    v0, kappa, theta, xi, rho = params

    model_prices = []
    for i, row in enumerate(validation_data):
        K = row["strike"]

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
        model_prices.append(model_price)

    return np.array(model_prices)


def calculate_errors(market_prices, model_prices, strikes, S, T, r):
    """
    Calculate absolute error, percentage error, and implied vol error.
    """
    results = []

    for K, market_price, model_price in zip(strikes, market_prices, model_prices):
        abs_error = abs(model_price - market_price)
        pct_error = (abs_error / market_price) * 100 if market_price != 0 else 0.0

        market_iv = implied_volatility(
            market_price, S, K, T, r, option_type="call", tol=1e-4
        )
        model_iv = implied_volatility(
            model_price, S, K, T, r, option_type="call", tol=1e-4
        )
        iv_error = abs(model_iv - market_iv)

        results.append({
            "strike": K,
            "market_price": market_price,
            "model_price": model_price,
            "abs_error": abs_error,
            "pct_error": pct_error,
            "market_iv": market_iv,
            "model_iv": model_iv,
            "iv_error": iv_error,
        })

    return results


def classify_moneyness(K, S):
    """
    ITM: K < S
    ATM: within ±5% of spot
    OTM: K > S
    """
    lower_atm = 0.95 * S
    upper_atm = 1.05 * S

    if lower_atm <= K <= upper_atm:
        return "ATM"
    elif K < S:
        return "ITM"
    else:
        return "OTM"


def error_by_moneyness(errors, strikes, S):
    """
    Group errors by moneyness and compute average metrics.
    """
    grouped = {"ITM": [], "ATM": [], "OTM": []}

    for err, K in zip(errors, strikes):
        bucket = classify_moneyness(K, S)
        grouped[bucket].append(err)

    summary = {}
    for bucket, items in grouped.items():
        if len(items) == 0:
            summary[bucket] = {
                "count": 0,
                "rmse": np.nan,
                "mae": np.nan,
                "avg_iv_error": np.nan,
            }
            continue

        abs_errors = np.array([x["abs_error"] for x in items])
        iv_errors = np.array([x["iv_error"] for x in items])

        summary[bucket] = {
            "count": len(items),
            "rmse": np.sqrt(np.mean(abs_errors ** 2)),
            "mae": np.mean(abs_errors),
            "avg_iv_error": np.mean(iv_errors),
        }

    return summary


def print_validation_report(overall_errors, moneyness_errors):
    """
    Print overall validation report and pass/fail assessment.
    """
    abs_errors = np.array([x["abs_error"] for x in overall_errors])
    pct_errors = np.array([x["pct_error"] for x in overall_errors])
    iv_errors = np.array([x["iv_error"] for x in overall_errors])

    rmse = np.sqrt(np.mean(abs_errors ** 2))
    mae = np.mean(abs_errors)
    avg_pct_error = np.mean(pct_errors)
    avg_iv_error = np.mean(iv_errors)

    largest = max(overall_errors, key=lambda x: x["abs_error"])

    print("\n" + "=" * 58)
    print("OP-5: OUT-OF-SAMPLE VALIDATION REPORT")
    print("=" * 58)

    print("\nPer-Strike Validation Errors:")
    print(f"{'Strike':>8} {'Market':>10} {'Model':>10} {'Abs Err':>10} {'% Err':>10} {'IV Err':>10}")
    print("-" * 58)
    for row in overall_errors:
        print(
            f"{row['strike']:8.2f} "
            f"{row['market_price']:10.4f} "
            f"{row['model_price']:10.4f} "
            f"{row['abs_error']:10.4f} "
            f"{row['pct_error']:9.2f}% "
            f"{row['iv_error']:10.4f}"
        )

    print("\nOverall Metrics:")
    print(f"  RMSE:         {rmse:.4f}")
    print(f"  MAE:          {mae:.4f}")
    print(f"  Avg % Error:  {avg_pct_error:.2f}%")
    print(f"  Avg IV Error: {avg_iv_error:.4f}")

    print("\nBy Moneyness:")
    for bucket, stats in moneyness_errors.items():
        print(
            f"  {bucket}: count={stats['count']}, "
            f"RMSE={stats['rmse']:.4f}, "
            f"MAE={stats['mae']:.4f}, "
            f"Avg IV Error={stats['avg_iv_error']:.4f}"
        )

    print("\nLargest Single Error:")
    print(
        f"  Strike {largest['strike']:.2f}: "
        f"Abs Error = {largest['abs_error']:.4f}, "
        f"% Error = {largest['pct_error']:.2f}%"
    )

    print("\nAssessment:")
    if rmse < 0.50:
        print("  PASS: Out-of-sample RMSE < $0.50, which is a good fit.")
    else:
        print("  FAIL: Out-of-sample RMSE >= $0.50, fit is weaker than desired.")

    return {
        "rmse": rmse,
        "mae": mae,
        "avg_pct_error": avg_pct_error,
        "avg_iv_error": avg_iv_error,
        "largest_error": largest["abs_error"],
    }


if __name__ == "__main__":
    S = 100
    T = 1.0
    r = 0.05
    strikes = np.arange(80, 121, 5)

    market_data, _, _ = build_synthetic_market_data(
        S=S,
        strikes=strikes,
        T=T,
        r=r,
        base_vol=0.20,
    )

    calibration_strikes = [80, 90, 100, 110, 120]
    validation_strikes = [85, 95, 105, 115]

    calibration_data, validation_data = split_options(
        market_data,
        calibration_strikes,
        validation_strikes,
    )

    print("Calibration strikes:", calibration_strikes)
    print("Validation strikes:", validation_strikes)
    print("\nCalibrating on half the strikes...\n")

    result = calibrate(
        market_data=calibration_data,
        S=S,
        T=T,
        r=r,
        maxiter=20,
        popsize=8,
        n_paths=3000,
        n_steps=252,
        seed=123,
    )

    params = result.x

    print("Calibrated parameters:")
    print(f"  v0    = {params[0]:.4f}")
    print(f"  kappa = {params[1]:.4f}")
    print(f"  theta = {params[2]:.4f}")
    print(f"  xi    = {params[3]:.4f}")
    print(f"  rho   = {params[4]:.4f}")

    market_prices = np.array([row["market_price"] for row in validation_data])
    validation_strikes_arr = np.array([row["strike"] for row in validation_data])

    model_prices = price_validation_set(
        validation_data,
        params,
        S,
        T,
        r,
        n_paths=8000,
        n_steps=252,
        seed=1000,
    )

    overall_errors = calculate_errors(
        market_prices,
        model_prices,
        validation_strikes_arr,
        S,
        T,
        r,
    )

    moneyness_errors = error_by_moneyness(
        overall_errors,
        validation_strikes_arr,
        S,
    )

    summary = print_validation_report(overall_errors, moneyness_errors)

    print("\nInterpretation:")
    print("  Compare this out-of-sample RMSE to your in-sample calibration RMSE.")
    print("  If it is much worse, the model may be overfitting specific strikes.")
