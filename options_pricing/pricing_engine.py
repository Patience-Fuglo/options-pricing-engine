import numpy as np
import pandas as pd

from options_pricing.black_scholes import (
    bs_call_price,
    bs_put_price,
    implied_volatility,
)
from options_pricing.heston import (
    price_european_call,
    price_european_put,
)
from options_pricing.calibration import calibrate  # Only for BS, not Heston
from options_pricing.calibration_fast import calibrate_heston_fast
from options_pricing.greeks import (
    calculate_all_greeks,
    calculate_all_bs_greeks,
)


class PricingEngine:
    def __init__(self, model_type="bs"):
        if model_type not in {"bs", "heston"}:
            raise ValueError("model_type must be 'bs' or 'heston'.")

        self.model_type = model_type
        self.calibrated_params = None
        self.cache = {}

    def calibrate(self, market_data, S, T, r):
        """
        Calibrate the selected model to market data.

        For BS:
            extract ATM implied vol from the nearest strike to spot.

        For Heston:
            run Heston calibration and store parameters.
        """
        if self.model_type == "bs":
            atm_option = min(market_data, key=lambda x: abs(x["strike"] - S))
            atm_price = atm_option["market_price"]

            atm_iv = implied_volatility(
                atm_price,
                S,
                atm_option["strike"],
                T,
                r,
                option_type="call",
            )

            self.calibrated_params = {"sigma": atm_iv}

            print("\n=== BS CALIBRATION COMPLETE ===")
            print(f"ATM Strike: {atm_option['strike']}")
            print(f"ATM Implied Vol: {atm_iv:.4f}")

        elif self.model_type == "heston":
            print("\n=== HESTON CALIBRATION (FAST) START ===")
            from options_pricing.calibration_fast import calibrate_heston_fast
            strikes = np.array([row["strike"] for row in market_data])
            market_prices = np.array([row["market_price"] for row in market_data])
            result = calibrate_heston_fast(S, strikes, T, r, market_prices, verbose=True)
            self.calibrated_params = {
                "v0": result['v0'],
                "kappa": result['kappa'],
                "theta": result['theta'],
                "xi": result['xi'],
                "rho": result['rho'],
            }
            print("\n=== HESTON CALIBRATION (FAST) COMPLETE ===")
            for key, value in self.calibrated_params.items():
                print(f"{key:>6} = {value:.4f}")

        self.cache = {}

    def _cache_key(self, S, K, T, r, option_type):
        return (
            self.model_type,
            round(float(S), 8),
            round(float(K), 8),
            round(float(T), 8),
            round(float(r), 8),
            option_type,
        )

    def price(self, S, K, T, r, option_type="call"):
        """
        Price a single option using the selected model.
        Checks cache first.
        """
        if option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'.")

        key = self._cache_key(S, K, T, r, option_type)
        if key in self.cache:
            return self.cache[key]

        if self.model_type == "bs":
            if self.calibrated_params is None:
                raise ValueError("BS model not calibrated. Run calibrate() first.")

            sigma = self.calibrated_params["sigma"]

            if option_type == "call":
                price = bs_call_price(S, K, T, r, sigma)
            else:
                price = bs_put_price(S, K, T, r, sigma)

        else:
            if self.calibrated_params is None:
                raise ValueError("Heston model not calibrated. Run calibrate() first.")

            params = self.calibrated_params

            if option_type == "call":
                price = price_european_call(
                    S0=S,
                    K=K,
                    T=T,
                    r=r,
                    v0=params["v0"],
                    kappa=params["kappa"],
                    theta=params["theta"],
                    xi=params["xi"],
                    rho=params["rho"],
                    n_paths=12000,
                    n_steps=252,
                    seed=42,
                )
            else:
                price = price_european_put(
                    S0=S,
                    K=K,
                    T=T,
                    r=r,
                    v0=params["v0"],
                    kappa=params["kappa"],
                    theta=params["theta"],
                    xi=params["xi"],
                    rho=params["rho"],
                    n_paths=12000,
                    n_steps=252,
                    seed=42,
                )

        self.cache[key] = price
        return price

    def _call_delta(self, S, K, T, r):
        if self.model_type == "bs":
            sigma = self.calibrated_params["sigma"]
            greeks = calculate_all_bs_greeks(S, K, T, r, sigma, dS=0.05)
            return greeks["delta"]
        else:
            greeks = calculate_all_greeks(
                S,
                K,
                T,
                r,
                self.calibrated_params,
                dS=0.05,
                n_paths=100,
                n_steps=10,
                seed=42,
            )
            return greeks["delta"]

    def price_chain(self, S, strikes, T, r):
        """
        Price a full option chain and return a DataFrame with:
        Strike, Call Price, Put Price, Call IV, Put IV, Call Delta, Put Delta
        """
        rows = []

        for K in strikes:
            call_price = self.price(S, K, T, r, option_type="call")
            put_price = self.price(S, K, T, r, option_type="put")

            call_iv = implied_volatility(call_price, S, K, T, r, option_type="call")
            put_iv = implied_volatility(put_price, S, K, T, r, option_type="put")

            call_delta = self._call_delta(S, K, T, r)
            put_delta = call_delta - 1.0

            rows.append({
                "Strike": float(K),
                "Call Price": float(call_price),
                "Put Price": float(put_price),
                "Call IV": float(call_iv),
                "Put IV": float(put_iv),
                "Call Delta": float(call_delta),
                "Put Delta": float(put_delta),
            })

        return pd.DataFrame(rows)

    def verify_parity(self, S, K, T, r):
        """
        Verify put-call parity:
            Call - Put ?= S - K*exp(-r*T)
        """
        call_price = self.price(S, K, T, r, option_type="call")
        put_price = self.price(S, K, T, r, option_type="put")

        lhs = call_price - put_price
        rhs = S - K * np.exp(-r * T)
        diff = abs(lhs - rhs)

        print(
            f"K={K:>6.2f} | "
            f"Call-Put={lhs:>10.4f} | "
            f"S-Ke^(-rT)={rhs:>10.4f} | "
            f"Diff={diff:>10.6f}"
        )

        if diff > 0.01:
            print("  Warning: parity difference exceeds $0.01")

        return diff

    def export(self, chain_df, filepath):
        chain_df.to_csv(filepath, index=False)
        print(f"\nChain exported to: {filepath}")

    def print_chain(self, chain_df):
        formatted = chain_df.copy()
        for col in ["Call Price", "Put Price", "Call IV", "Put IV", "Call Delta", "Put Delta"]:
            formatted[col] = formatted[col].map(lambda x: f"{x:.4f}")
        print("\n=== OPTION CHAIN ===")
        print(formatted.to_string(index=False))


if __name__ == "__main__":
    from options_pricing.calibration import build_synthetic_market_data

    S = 100
    T = 1.0
    r = 0.05
    # Ultra-fast: only 2 strikes
    strikes = np.array([90, 100])
    full_chain_strikes = np.array([90, 100])

    market_data, _, _ = build_synthetic_market_data(
        S=S,
        strikes=strikes,
        T=T,
        r=r,
        base_vol=0.20,
    )

    engine = PricingEngine(model_type="heston")
    engine.calibrate(market_data, S, T, r)

    chain_df = engine.price_chain(S, full_chain_strikes, T, r)
    engine.print_chain(chain_df)

    print("\n=== PUT-CALL PARITY CHECK ===")
    for K in full_chain_strikes:
        engine.verify_parity(S, K, T, r)

    engine.export(chain_df, "option_chain_heston.csv")
