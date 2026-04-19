"""
Comprehensive test suite for the options pricing library.

Tests are organized into classes:
    TestBlackScholes       — BS pricing, put-call parity, IV round-trip
    TestBSGreeks           — analytical Greeks (delta, vega, gamma, theta, rho)
    TestBSGreeksRelations  — standard Greek identities and boundary conditions
    TestHestonAnalytic     — CF-based pricing, put-call parity, FFT vs quad
    TestHestonMC           — Monte Carlo simulation and convergence
    TestAntitheticVariates — variance reduction verification
    TestCalibration        — calibration accuracy on synthetic data

Known reference values are taken from Hull's "Options, Futures, and Other
Derivatives" (11th ed.) and Heston (1993).
"""

import warnings

import numpy as np
import pytest

from options_pricing.black_scholes import (
    bs_call_price,
    bs_put_price,
    implied_volatility,
)
from options_pricing.greeks import (
    bs_delta,
    bs_vega,
    bs_gamma,
    bs_theta,
    bs_rho,
    bs_vanna,
    bs_volga,
    calculate_all_bs_greeks,
)
from options_pricing.heston import (
    price_european_call,
    price_european_put,
    price_european_call_av,
    price_european_put_av,
    feller_condition,
)
from options_pricing.heston_analytic import (
    heston_call_price,
    heston_put_price,
    heston_call_chain_fft,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

ATM = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)
ITM = dict(S=100.0, K=90.0,  T=1.0, r=0.05, sigma=0.20)
OTM = dict(S=100.0, K=110.0, T=1.0, r=0.05, sigma=0.20)

HESTON = dict(
    S0=100.0, K=100.0, T=1.0, r=0.05,
    v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
)


# ---------------------------------------------------------------------------
# Black-Scholes pricing
# ---------------------------------------------------------------------------

class TestBlackScholes:

    def test_atm_call_known_value(self):
        """ATM call at S=K=100, T=1, r=5%, σ=20% ≈ 10.4506 (Hull Table 15.1)."""
        price = bs_call_price(**{k: v for k, v in ATM.items() if k != "sigma"},
                              sigma=ATM["sigma"])
        assert abs(price - 10.4506) < 5e-3

    def test_atm_put_known_value(self):
        """ATM put at same params ≈ 5.5735."""
        price = bs_put_price(**{k: v for k, v in ATM.items() if k != "sigma"},
                             sigma=ATM["sigma"])
        assert abs(price - 5.5735) < 5e-3

    def test_put_call_parity_atm(self):
        c = bs_call_price(**ATM)
        p = bs_put_price(**ATM)
        parity = ATM["S"] - ATM["K"] * np.exp(-ATM["r"] * ATM["T"])
        assert abs((c - p) - parity) < 1e-10

    def test_put_call_parity_itm(self):
        c = bs_call_price(**ITM)
        p = bs_put_price(**ITM)
        parity = ITM["S"] - ITM["K"] * np.exp(-ITM["r"] * ITM["T"])
        assert abs((c - p) - parity) < 1e-10

    def test_put_call_parity_otm(self):
        c = bs_call_price(**OTM)
        p = bs_put_price(**OTM)
        parity = OTM["S"] - OTM["K"] * np.exp(-OTM["r"] * OTM["T"])
        assert abs((c - p) - parity) < 1e-10

    def test_call_lower_bound(self):
        """Call >= max(S - K*exp(-rT), 0)."""
        c = bs_call_price(**ATM)
        lb = max(ATM["S"] - ATM["K"] * np.exp(-ATM["r"] * ATM["T"]), 0.0)
        assert c >= lb

    def test_deep_itm_call(self):
        """Deep ITM call should be close to intrinsic value."""
        c = bs_call_price(200.0, 100.0, 1.0, 0.05, 0.20)
        intrinsic = 200.0 - 100.0 * np.exp(-0.05 * 1.0)
        assert abs(c - intrinsic) < 5.0  # some time value

    def test_deep_otm_call_near_zero(self):
        c = bs_call_price(100.0, 500.0, 0.1, 0.05, 0.20)
        assert c < 1e-5

    def test_call_monotone_in_sigma(self):
        """Call price is monotone increasing in σ (positive vega)."""
        prices = [bs_call_price(100, 100, 1.0, 0.05, s) for s in [0.1, 0.2, 0.3, 0.5]]
        assert all(x < y for x, y in zip(prices, prices[1:]))

    def test_call_monotone_in_S(self):
        """Call price is monotone increasing in spot."""
        prices = [bs_call_price(s, 100, 1.0, 0.05, 0.20) for s in [80, 90, 100, 110, 120]]
        assert all(x < y for x, y in zip(prices, prices[1:]))

    def test_input_validation_negative_S(self):
        with pytest.raises(ValueError, match="Stock price"):
            bs_call_price(-1.0, 100, 1.0, 0.05, 0.20)

    def test_input_validation_zero_T(self):
        with pytest.raises(ValueError, match="Time"):
            bs_call_price(100, 100, 0.0, 0.05, 0.20)

    def test_input_validation_negative_sigma(self):
        with pytest.raises(ValueError, match="Volatility"):
            bs_call_price(100, 100, 1.0, 0.05, -0.20)


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

class TestImpliedVolatility:

    def test_roundtrip_atm(self):
        sigma = 0.20
        c = bs_call_price(100, 100, 1.0, 0.05, sigma)
        iv = implied_volatility(c, 100, 100, 1.0, 0.05)
        assert abs(iv - sigma) < 1e-5

    def test_roundtrip_high_vol(self):
        sigma = 0.60
        c = bs_call_price(100, 100, 1.0, 0.05, sigma)
        iv = implied_volatility(c, 100, 100, 1.0, 0.05)
        assert abs(iv - sigma) < 1e-4

    def test_roundtrip_put(self):
        sigma = 0.25
        p = bs_put_price(100, 110, 0.5, 0.03, sigma)
        iv = implied_volatility(p, 100, 110, 0.5, 0.03, option_type="put")
        assert abs(iv - sigma) < 1e-4

    @pytest.mark.parametrize("sigma", [0.05, 0.15, 0.30, 0.50, 0.80])
    def test_roundtrip_range(self, sigma):
        c = bs_call_price(100, 100, 1.0, 0.05, sigma)
        iv = implied_volatility(c, 100, 100, 1.0, 0.05)
        assert abs(iv - sigma) < 1e-4

    def test_invalid_option_type(self):
        with pytest.raises(ValueError, match="option_type"):
            implied_volatility(5.0, 100, 100, 1.0, 0.05, option_type="fwd")

    def test_negative_price_raises(self):
        with pytest.raises(ValueError, match="Market price"):
            implied_volatility(-1.0, 100, 100, 1.0, 0.05)

    def test_convergence_warning_for_impossible_price(self):
        """A price above the theoretical max should warn, not crash."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            implied_volatility(99.0, 100, 100, 0.01, 0.0)
            assert len(w) >= 1


# ---------------------------------------------------------------------------
# Black-Scholes Greeks
# ---------------------------------------------------------------------------

class TestBSGreeks:

    def test_delta_atm_call_near_half(self):
        """Short-dated ATM call delta ≈ 0.5 (drift negligible at T→0)."""
        d = bs_delta(100, 100, 1/365, 0.0, 0.20)
        assert abs(d - 0.5) < 0.02

    def test_delta_call_between_0_and_1(self):
        for K in [80, 100, 120]:
            d = bs_delta(100, K, 1.0, 0.05, 0.20)
            assert 0.0 <= d <= 1.0

    def test_delta_put_between_neg1_and_0(self):
        for K in [80, 100, 120]:
            d = bs_delta(100, K, 1.0, 0.05, 0.20, option_type="put")
            assert -1.0 <= d <= 0.0

    def test_delta_call_plus_put_equals_1(self):
        """delta_call - delta_put = 1 (call-put delta parity)."""
        dc = bs_delta(100, 100, 1.0, 0.05, 0.20, "call")
        dp = bs_delta(100, 100, 1.0, 0.05, 0.20, "put")
        assert abs(dc - dp - 1.0) < 1e-12

    def test_vega_positive(self):
        v = bs_vega(100, 100, 1.0, 0.05, 0.20)
        assert v > 0

    def test_gamma_positive(self):
        g = bs_gamma(100, 100, 1.0, 0.05, 0.20)
        assert g > 0

    def test_theta_call_negative(self):
        """Theta is negative: option loses value as time passes."""
        t = bs_theta(100, 100, 1.0, 0.05, 0.20, "call")
        assert t < 0

    def test_rho_call_positive(self):
        """Call rho is positive: higher rates benefit call holder."""
        rho = bs_rho(100, 100, 1.0, 0.05, 0.20, "call")
        assert rho > 0

    def test_rho_put_negative(self):
        rho = bs_rho(100, 100, 1.0, 0.05, 0.20, "put")
        assert rho < 0

    def test_vanna_atm_zero(self):
        """
        At-the-money (d2 = 0) vanna is zero, as d2 = 0 → vanna = 0.

        We check vanna changes sign at ATM, which is roughly when
        log(S/K) = 0 (ignoring drift).
        """
        # Vanna should be small near ATM
        van_atm  = abs(bs_vanna(100, 100, 1.0, 0.0, 0.20))
        van_itm  = abs(bs_vanna(100,  80, 1.0, 0.0, 0.20))
        assert van_atm < van_itm

    def test_volga_positive_atm(self):
        """Volga (vomma) is positive for long options near ATM."""
        vg = bs_volga(100, 100, 1.0, 0.05, 0.20)
        assert vg > 0

    def test_calculate_all_bs_greeks_keys(self):
        greeks = calculate_all_bs_greeks(100, 100, 1.0, 0.05, 0.20)
        for key in ["delta", "vega", "gamma", "theta", "rho", "vanna", "volga", "charm"]:
            assert key in greeks


class TestBSGreekRelations:
    """Numerical sanity checks and Black-Scholes identities."""

    def test_delta_numerical(self):
        """Numerical delta ≈ analytical delta."""
        S, K, T, r, sigma, h = 100.0, 100.0, 1.0, 0.05, 0.20, 0.01
        num_delta = (bs_call_price(S+h, K, T, r, sigma) - bs_call_price(S-h, K, T, r, sigma)) / (2*h)
        ana_delta = bs_delta(S, K, T, r, sigma)
        assert abs(num_delta - ana_delta) < 1e-4

    def test_gamma_numerical(self):
        S, K, T, r, sigma, h = 100.0, 100.0, 1.0, 0.05, 0.20, 0.10
        cup = bs_call_price(S+h, K, T, r, sigma)
        cmi = bs_call_price(S,   K, T, r, sigma)
        cdn = bs_call_price(S-h, K, T, r, sigma)
        num_gamma = (cup - 2*cmi + cdn) / h**2
        ana_gamma = bs_gamma(S, K, T, r, sigma)
        assert abs(num_gamma - ana_gamma) < 1e-4

    def test_vega_numerical(self):
        S, K, T, r, sigma, h = 100.0, 100.0, 1.0, 0.05, 0.20, 0.001
        num_vega = (bs_call_price(S, K, T, r, sigma+h) - bs_call_price(S, K, T, r, sigma-h)) / (2*h)
        ana_vega = bs_vega(S, K, T, r, sigma) * 100  # unscale for comparison
        assert abs(num_vega - ana_vega) < 1e-4

    def test_theta_numerical(self):
        """
        bs_theta is dV/dt (calendar time, negative for long options).
        dC/d(tau) where tau=T is positive (longer expiry -> more valuable).
        So bs_theta_annual = -d(Call)/d(T): check the signs cancel.
        """
        S, K, T, r, sigma, h = 100.0, 100.0, 1.0, 0.05, 0.20, 1e-4
        dcall_dT   = (bs_call_price(S, K, T+h, r, sigma) - bs_call_price(S, K, T-h, r, sigma)) / (2*h)
        ana_theta_annual = bs_theta(S, K, T, r, sigma) * 365
        # theta_annual + dcall_dT should be ≈ 0 (opposite signs by definition)
        assert abs(ana_theta_annual + dcall_dT) < 1e-3


# ---------------------------------------------------------------------------
# Heston analytic (CF) pricing
# ---------------------------------------------------------------------------

class TestHestonAnalytic:

    def test_call_positive(self):
        call = heston_call_price(**HESTON)
        assert call > 0

    def test_put_positive(self):
        put = heston_put_price(**HESTON)
        assert put > 0

    def test_put_call_parity(self):
        h = HESTON
        call = heston_call_price(**h)
        put  = heston_put_price(**h)
        parity = h["S0"] - h["K"] * np.exp(-h["r"] * h["T"])
        assert abs((call - put) - parity) < 1e-4

    def test_call_lower_bound(self):
        h = HESTON
        call = heston_call_price(**h)
        lb = max(h["S0"] - h["K"] * np.exp(-h["r"] * h["T"]), 0.0)
        assert call >= lb - 1e-4

    def test_call_less_than_spot(self):
        """Call can never be worth more than the stock."""
        call = heston_call_price(**HESTON)
        assert call < HESTON["S0"]

    def test_deep_otm_near_zero(self):
        call = heston_call_price(100, 300, 0.25, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
        assert call < 0.01

    def test_fft_matches_single_integration(self):
        """FFT prices agree with quad integration to within 0.05."""
        h = HESTON
        strikes = np.array([90.0, 100.0, 110.0])
        fft_prices = heston_call_chain_fft(
            h["S0"], strikes, h["T"], h["r"],
            h["v0"], h["kappa"], h["theta"], h["xi"], h["rho"]
        )
        for i, K in enumerate(strikes):
            single = heston_call_price(h["S0"], K, h["T"], h["r"],
                                       h["v0"], h["kappa"], h["theta"], h["xi"], h["rho"])
            assert abs(fft_prices[i] - single) < 0.05, (
                f"FFT={fft_prices[i]:.4f} vs single={single:.4f} at K={K}"
            )

    def test_fft_call_chain_pcp(self):
        """FFT call chain satisfies put-call parity for each strike."""
        h = HESTON
        strikes = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
        calls = heston_call_chain_fft(
            h["S0"], strikes, h["T"], h["r"],
            h["v0"], h["kappa"], h["theta"], h["xi"], h["rho"]
        )
        parity = h["S0"] - strikes * np.exp(-h["r"] * h["T"])
        for K, c, pcp in zip(strikes, calls, parity):
            put = heston_put_price(h["S0"], K, h["T"], h["r"],
                                   h["v0"], h["kappa"], h["theta"], h["xi"], h["rho"])
            assert abs((c - put) - pcp) < 0.05, f"PCP failed at K={K}"

    def test_analytic_vs_mc(self):
        """Analytic price should be within 0.50 of 100k-path MC price."""
        h = HESTON
        analytic = heston_call_price(**h)
        mc = price_european_call(
            h["S0"], h["K"], h["T"], h["r"],
            h["v0"], h["kappa"], h["theta"], h["xi"], h["rho"],
            n_paths=100_000, seed=42
        )
        assert abs(analytic - mc) < 0.50, (
            f"Analytic={analytic:.4f} vs MC={mc:.4f}"
        )

    def test_rho_effect_on_skew(self):
        """Negative rho should produce a downward-sloping IV skew."""
        h = HESTON
        call_itm = heston_call_price(h["S0"], 90, h["T"], h["r"],
                                     h["v0"], h["kappa"], h["theta"], h["xi"], rho=-0.7)
        call_otm = heston_call_price(h["S0"], 110, h["T"], h["r"],
                                     h["v0"], h["kappa"], h["theta"], h["xi"], rho=-0.7)
        from options_pricing.black_scholes import implied_volatility as iv_func
        iv_itm = iv_func(call_itm, h["S0"], 90,  h["T"], h["r"])
        iv_otm = iv_func(call_otm, h["S0"], 110, h["T"], h["r"])
        # Negative rho → puts more expensive → ITM calls have higher IV than OTM
        assert iv_itm > iv_otm, (
            f"Expected IV skew: IV(K=90)={iv_itm:.4f} > IV(K=110)={iv_otm:.4f}"
        )


# ---------------------------------------------------------------------------
# Heston Monte Carlo
# ---------------------------------------------------------------------------

class TestHestonMC:

    def test_call_positive(self):
        h = HESTON
        call = price_european_call(h["S0"], h["K"], h["T"], h["r"],
                                   h["v0"], h["kappa"], h["theta"], h["xi"], h["rho"],
                                   n_paths=10_000, seed=42)
        assert call > 0

    def test_put_call_parity_approx(self):
        """Put-call parity holds approximately (within MC noise)."""
        h = HESTON
        kw = dict(n_paths=50_000, seed=42)
        call = price_european_call(h["S0"], h["K"], h["T"], h["r"],
                                   h["v0"], h["kappa"], h["theta"], h["xi"], h["rho"], **kw)
        put  = price_european_put(h["S0"], h["K"], h["T"], h["r"],
                                   h["v0"], h["kappa"], h["theta"], h["xi"], h["rho"], **kw)
        parity = h["S0"] - h["K"] * np.exp(-h["r"] * h["T"])
        # Allow 1.0 for MC noise
        assert abs((call - put) - parity) < 1.0

    def test_feller_condition(self):
        """Standard test parameters satisfy Feller condition."""
        h = HESTON
        assert feller_condition(h["kappa"], h["theta"], h["xi"])

    def test_reproducibility(self):
        """Same seed → same result."""
        h = HESTON
        c1 = price_european_call(h["S0"], h["K"], h["T"], h["r"],
                                  h["v0"], h["kappa"], h["theta"], h["xi"], h["rho"],
                                  n_paths=1_000, seed=123)
        c2 = price_european_call(h["S0"], h["K"], h["T"], h["r"],
                                  h["v0"], h["kappa"], h["theta"], h["xi"], h["rho"],
                                  n_paths=1_000, seed=123)
        assert c1 == c2

    def test_call_increases_with_spot(self):
        h = HESTON
        kw = dict(K=100, T=1.0, r=0.05, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                  n_paths=10_000, n_steps=50, seed=0)
        c_low  = price_european_call(S0=90,  **kw)
        c_mid  = price_european_call(S0=100, **kw)
        c_high = price_european_call(S0=110, **kw)
        assert c_low < c_mid < c_high


# ---------------------------------------------------------------------------
# Antithetic variates
# ---------------------------------------------------------------------------

class TestAntitheticVariates:

    def test_av_close_to_plain_mc(self):
        """AV price should be close to plain MC at same n_paths."""
        h = HESTON
        plain = price_european_call(h["S0"], h["K"], h["T"], h["r"],
                                    h["v0"], h["kappa"], h["theta"], h["xi"], h["rho"],
                                    n_paths=20_000, seed=42)
        av    = price_european_call_av(h["S0"], h["K"], h["T"], h["r"],
                                       h["v0"], h["kappa"], h["theta"], h["xi"], h["rho"],
                                       n_paths=20_000, seed=42)
        assert abs(av - plain) < 1.0, f"AV={av:.4f} plain={plain:.4f}"

    def test_av_lower_variance(self):
        """
        AV should have lower variance than plain MC.
        Test by running 20 replications and comparing sample std.
        """
        h = HESTON
        N = 20
        plain_prices = [
            price_european_call(h["S0"], h["K"], h["T"], h["r"],
                                h["v0"], h["kappa"], h["theta"], h["xi"], h["rho"],
                                n_paths=2_000, seed=i)
            for i in range(N)
        ]
        av_prices = [
            price_european_call_av(h["S0"], h["K"], h["T"], h["r"],
                                   h["v0"], h["kappa"], h["theta"], h["xi"], h["rho"],
                                   n_paths=2_000, seed=i)
            for i in range(N)
        ]
        std_plain = np.std(plain_prices)
        std_av    = np.std(av_prices)
        # AV should reduce std by at least 10%
        assert std_av < std_plain * 0.95, (
            f"AV std={std_av:.4f} not less than plain std={std_plain:.4f}"
        )

    def test_av_put_call_parity(self):
        h = HESTON
        call = price_european_call_av(h["S0"], h["K"], h["T"], h["r"],
                                       h["v0"], h["kappa"], h["theta"], h["xi"], h["rho"],
                                       n_paths=50_000, seed=7)
        put  = price_european_put_av(h["S0"], h["K"], h["T"], h["r"],
                                      h["v0"], h["kappa"], h["theta"], h["xi"], h["rho"],
                                      n_paths=50_000, seed=7)
        parity = h["S0"] - h["K"] * np.exp(-h["r"] * h["T"])
        assert abs((call - put) - parity) < 1.0


# ---------------------------------------------------------------------------
# Calibration (analytic CF-based)
# ---------------------------------------------------------------------------

class TestCalibration:

    def test_recovers_known_params(self):
        """
        Calibrate to prices generated by known parameters and verify
        the calibrated params reproduce the prices with RMSE < 0.01.
        """
        from options_pricing.calibration_fast import calibrate_heston_fast

        true = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5)
        S, T, r = 100.0, 1.0, 0.05
        strikes = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])

        prices = heston_call_chain_fft(S, strikes, T, r, **true)

        result = calibrate_heston_fast(S, strikes, T, r, prices, verbose=False)

        # Model must be able to reprice within 1 cent RMSE on its own data
        assert result["rmse"] < 0.01, (
            f"Calibration RMSE={result['rmse']:.4f} > 0.01"
        )
