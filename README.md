
# 📈 Options Pricing Engine (Black-Scholes + Heston)

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

A robust, production-style options pricing system implementing:

- Black-Scholes (closed-form)
- Heston stochastic volatility (Monte Carlo)
- Model calibration to market data
- Volatility smile analysis
- Out-of-sample validation
- Greeks calculation (Delta, Gamma, Vega, Theta)
- Full pricing engine with caching and CSV export

---

## 🎯 Overview

This project replicates core components of real-world quantitative trading systems:

- **Pricing Models:** Black-Scholes, Heston (MC)
- **Calibration:** Fit model to synthetic/market data
- **Smile Analysis:** Volatility smile/skew
- **Validation:** Out-of-sample error, generalization
- **Greeks:** Delta, Gamma, Vega, Theta (risk metrics)
- **Engine:** Price chains, verify put-call parity, export

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/options-pricing-engine.git
cd options-pricing-engine
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Run full pricing engine:

```bash
python -m options_pricing.pricing_engine
```

### Run individual modules:

```bash
python -m options_pricing.black_scholes
python -m options_pricing.heston
python -m options_pricing.model_comparison
python -m options_pricing.validation
python -m options_pricing.greeks
```

---

## 🧪 Testing

To verify installation and core functionality, run:

```bash
python -m options_pricing.pricing_engine
```
This should print the option chain and calibration results without errors.

---

## 🖨️ Sample Output

```
=== HESTON CALIBRATION (FAST) COMPLETE ===
		v0 = 0.0334
 kappa = 1.2439
 theta = 0.0461
		xi = 0.3038
	 rho = -0.9900

=== OPTION CHAIN ===
 Strike Call Price Put Price Call IV Put IV Call Delta Put Delta
	 90.0    17.1789    2.9131  0.2171 0.2214     0.8395   -0.1605
	100.0    10.2778    5.5242  0.1952 0.1988     0.7269   -0.2731
```

---

## 📂 Project Structure

```
options_pricing/
├── black_scholes.py      # BS pricing + implied vol
├── heston.py             # Heston Monte Carlo simulation
├── model_comparison.py   # Volatility smile comparison
├── calibration.py        # Heston parameter calibration
├── validation.py         # Out-of-sample testing
├── greeks.py             # Greeks calculations
├── pricing_engine.py     # Final pricing system
└── __init__.py
```

---

## 📊 Key Features

| Feature | Description |
|---------|-------------|
| **Black-Scholes** | Closed-form pricing, implied vol |
| **Heston (MC)** | Stochastic volatility, MC simulation |
| **Calibration** | Fit model to market/synthetic data |
| **Smile Analysis** | Volatility smile/skew, model comparison |
| **Validation** | Out-of-sample error, generalization |
| **Greeks** | Delta, Gamma, Vega, Theta (risk metrics) |
| **Engine** | Price chains, verify parity, export CSV |

---

## 📈 Results Summary

- Heston model significantly improves fit over Black-Scholes
- Out-of-sample validation confirms generalization (not overfitting)
- Greeks differ meaningfully under stochastic volatility
- Volatility smile captured by Heston, not BS

---

## ⚠️ Notes

- Heston Monte Carlo introduces noise → small pricing differences expected
- Put-call parity may slightly deviate under MC due to simulation noise
- Calibration parameters may not always be economically realistic (depends on data)

---

## 🔥 Future Improvements

- Use real market data (Polygon, Bloomberg, etc.)
- Speed up Monte Carlo with vectorization / GPU
- Implement analytic Heston pricing (Fourier methods)
- Add American options (LSM method)
- Add volatility surface fitting
- Build API or web dashboard

---

## 👤 Author

**Patience Fuglo**  
Quantitative Finance | Machine Learning | AI Systems
- GitHub: [@Patience-Fuglo](https://github.com/Patience-Fuglo)
- LinkedIn: [Patience Fuglo](https://linkedin.com/in/patience-fuglo)

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## 📌 License

MIT License

---

## ⚠️ Disclaimer

This project is for educational and research purposes only.  
It does not constitute financial advice or a production trading system.

---

<p align="center">
	⭐ Star this repo if you find it useful!
</p>
