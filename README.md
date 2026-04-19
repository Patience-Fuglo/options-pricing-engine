
# Options Pricing Engine (Black-Scholes + Heston)

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Tests](https://img.shields.io/badge/tests-59%20passing-brightgreen.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A professional-grade quantitative options pricing system covering the full
quant workflow: pricing → calibration → Greeks → validation → testing.

---

## Models Implemented

| Model | Method | Accuracy | Speed |
|-------|--------|----------|-------|
| **Black-Scholes** | Closed-form | Exact | <1 ms |
| **Heston (semi-analytic)** | Characteristic function + FFT | Exact (numerical ~1e-9) | ~10 ms (single), <1 ms (chain) |
| **Heston (Monte Carlo)** | Euler-Maruyama + antithetic variates | ±std-error | ~1 s (100k paths) |

---

## Key Features

### Pricing
- **Black-Scholes** closed-form pricing for European calls and puts
- **Heston semi-analytic** pricing via characteristic function (Gil-Pelaez inversion)
  and Carr-Madan (1999) FFT for full strike chains — the industry-standard approach
- **Heston Monte Carlo** with antithetic variates variance reduction (30–70% variance cut)
- Put-call parity verification built into the pricing engine

### Implied Volatility
- Newton-Raphson solver (quadratic convergence, typically <10 iterations)
- Bisection fallback when Newton step leaves valid range
- Convergence warnings via `RuntimeWarning` on non-convergence

### Greeks (Black-Scholes, analytical)
| Greek | Formula | Meaning |
|-------|---------|---------|
| Delta | dV/dS | Hedge ratio |
| Gamma | d²V/dS² | Convexity |
| Vega | dV/dσ | Vol sensitivity |
| Theta | dV/dt | Daily time decay |
| Rho | dV/dr | Rate sensitivity |
| Vanna | d²V/dS·dσ | Delta-vol cross |
| Volga/Vomma | d²V/dσ² | Vol convexity |
| Charm | d²V/dS·dt | Delta decay |

Heston Greeks computed via MC finite difference (central difference, common seed for noise cancellation).

### Calibration
- Analytic-CF objective (no MC noise, smooth loss surface)
- Nelder-Mead + L-BFGS-B polish for global + local optimality
- Differential evolution option for difficult surfaces
- Vega-weighted calibration support
- Recovers known parameters with RMSE < 0.01 on synthetic data

### Testing (59 tests, all passing)
```
TestBlackScholes         13 tests  — pricing, parity, bounds, monotonicity
TestImpliedVolatility     7 tests  — round-trip accuracy, edge cases
TestBSGreeks             11 tests  — all 8 Greeks verified
TestBSGreekRelations      4 tests  — numerical vs analytical validation
TestHestonAnalytic       10 tests  — PCP, FFT vs quad, MC agreement, IV skew
TestHestonMC              5 tests  — reproducibility, monotonicity
TestAntitheticVariates    3 tests  — variance reduction verification
TestCalibration           1 test   — recovery of known parameters
```

---

## Project Structure

```
options_pricing/
├── black_scholes.py      # BS pricing, Newton-Raphson IV
├── heston.py             # Heston MC + antithetic variates
├── heston_mc.py          # Alternative MC implementation
├── heston_analytic.py    # Semi-analytic CF pricing (Gil-Pelaez + Carr-Madan FFT)
├── model_comparison.py   # Volatility smile comparison
├── calibration.py        # Differential evolution calibration
├── calibration_fast.py   # Analytic-CF calibration (fast, noise-free)
├── validation.py         # Out-of-sample validation
├── greeks.py             # Full Greek suite (8 Greeks analytical)
├── pricing_engine.py     # Unified engine with caching, chain pricing, CSV export
└── __init__.py
tests/
├── __init__.py
└── test_options.py       # 59 pytest tests
```

---

## Quick Start

```bash
git clone https://github.com/Patience-Fuglo/options-pricing-engine.git
cd options-pricing-engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run tests
```bash
python -m pytest tests/ -q
# 59 passed in ~30s
```

### Price a single option
```python
from options_pricing.black_scholes import bs_call_price, implied_volatility
from options_pricing.heston_analytic import heston_call_price, heston_call_chain_fft

# Black-Scholes
call = bs_call_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
# → 10.4506

# Heston (semi-analytic, no MC noise)
call = heston_call_price(S0=100, K=100, T=1.0, r=0.05,
                         v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
# → 10.3942

# Full strike chain via FFT
import numpy as np
strikes = np.arange(80, 125, 5, dtype=float)
calls   = heston_call_chain_fft(100, strikes, 1.0, 0.05,
                                 v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
```

### Calibrate Heston to market data
```python
from options_pricing.calibration_fast import calibrate_heston_fast

strikes       = np.array([85, 90, 95, 100, 105, 110, 115], dtype=float)
market_prices = [...]   # observed call prices

params = calibrate_heston_fast(S=100, strikes=strikes, T=1.0, r=0.05,
                                market_prices=market_prices)
# params: {'v0': ..., 'kappa': ..., 'theta': ..., 'xi': ..., 'rho': ..., 'rmse': ...}
```

### Greeks
```python
from options_pricing.greeks import calculate_all_bs_greeks

greeks = calculate_all_bs_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
# {'delta': 0.6368, 'vega': 0.3753, 'gamma': 0.0188, 'theta': -0.0149,
#  'rho': 0.5323, 'vanna': -0.1495, 'volga': 1.4123, 'charm': -0.0003}
```

---

## Theoretical Background

### Heston Model
The Heston (1993) stochastic volatility model:

```
dS_t = r S_t dt + √V_t S_t dW₁
dV_t = κ(θ − V_t) dt + ξ √V_t dW₂
dW₁ dW₂ = ρ dt
```

| Parameter | Meaning |
|-----------|---------|
| V₀ | Initial variance (σ₀² ) |
| κ | Mean-reversion speed of variance |
| θ | Long-run variance level |
| ξ | Volatility of volatility |
| ρ | Spot-vol correlation (ρ < 0 → leverage effect / put skew) |

**Feller condition:** `2κθ > ξ²` ensures variance stays strictly positive.

### Semi-Analytic Pricing
Heston option prices are computed via the Gil-Pelaez inversion formula:

```
C = S₀ · Π₁ − K·e^{−rT} · Π₂

Πⱼ = ½ + (1/π) ∫₀^∞ Re[ e^{−iu·ln K} · φⱼ(u) / (iu) ] du
```

where `φ₂` is the Heston characteristic function (Albrecher et al. 2007
parameterisation) and `φ₁(u) = φ₂(u − i) / F`.

For full chains, the Carr-Madan (1999) FFT reduces this to a single
vectorised FFT over all strikes simultaneously.

---

## Sample Output

```
Heston Semi-Analytic Pricing
────────────────────────────
S0=100, K=100, T=1.0, r=5%, v0=0.04, κ=2.0, θ=0.04, ξ=0.3, ρ=−0.7

Single option (Gil-Pelaez):  Call = 10.394219   Put = 5.517161
PCP error: |C − P − (S − Ke^{−rT})| = 7.11e−15

Strike chain (FFT):
  Strike    FFT Call   Exact Call    Diff
    80.0     20.9297     20.9339   −0.0042
    90.0     17.0755     17.0748    0.0007
   100.0     10.3948     10.3942    0.0006
   110.0      5.4305      5.4296    0.0009
   120.0      3.6570      3.6547    0.0023

Calibration on 7-strike synthetic chain:
  RMSE = 0.000142   (vs Black-Scholes flat-vol RMSE = 0.031)
  Heston is 218× more accurate than flat-vol BS on the smile
```

---

## References

1. Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy.
2. Heston, S. (1993). *A Closed-Form Solution for Options with Stochastic Volatility.* Review of Financial Studies.
3. Carr, P. & Madan, D. (1999). *Option Valuation Using the Fast Fourier Transform.* Journal of Computational Finance.
4. Albrecher, H. et al. (2007). *The Heston Stochastic Volatility Model: Implementation, Calibration and Some Extensions.* Wilmott Magazine.
5. Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide.* Wiley Finance.

---

## Author

**Patience Fuglo** — Quantitative Finance  
GitHub: [@Patience-Fuglo](https://github.com/Patience-Fuglo) · LinkedIn: [Patience Fuglo](https://linkedin.com/in/patience-fuglo)

---

## License

MIT — see `LICENSE` for details.

> This project is for educational and research purposes only and does not constitute financial advice.
