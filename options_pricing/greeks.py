def calculate_all_bs_greeks(S, K, T, r, sigma, dS=0.01):
    """
    Return dict of BS greeks: delta, vega, gamma
    """
    delta = bs_delta(S, K, T, r, sigma)
    vega = bs_vega(S, K, T, r, sigma)
    gamma = bs_gamma(S, K, T, r, sigma)
    return {"delta": delta, "vega": vega, "gamma": gamma}


def calculate_all_greeks(S, K, T, r, params, dS=0.01, n_paths=8000, n_steps=252, seed=42):
    """
    Return dict of Heston greeks: delta, vega, gamma (MC finite diff)
    """
    v0 = params["v0"]
    kappa = params["kappa"]
    theta = params["theta"]
    xi = params["xi"]
    rho = params["rho"]
    delta = heston_greek_fd(S, K, T, r, v0, kappa, theta, xi, rho, "delta", h=dS, n_paths=n_paths, n_steps=n_steps, seed=seed)
    vega = heston_greek_fd(S, K, T, r, v0, kappa, theta, xi, rho, "vega", h=dS, n_paths=n_paths, n_steps=n_steps, seed=seed)
    gamma = heston_greek_fd(S, K, T, r, v0, kappa, theta, xi, rho, "gamma", h=dS, n_paths=n_paths, n_steps=n_steps, seed=seed)
    return {"delta": delta, "vega": vega, "gamma": gamma}
import numpy as np
from options_pricing.black_scholes import bs_call_price as bs_call, implied_volatility
from options_pricing.heston import price_european_call as heston_call

# --- Black-Scholes Greeks ---
def bs_delta(S, K, T, r, sigma):
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def bs_vega(S, K, T, r, sigma):
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def bs_gamma(S, K, T, r, sigma):
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# --- Heston Greeks (MC finite diff) ---
def heston_greek_fd(S, K, T, r, v0, kappa, theta, xi, rho, greek, h=1e-2, n_paths=8000, n_steps=252, seed=42):
    if greek == "delta":
        price_up = heston_call(S0=S+h, K=K, T=T, r=r, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, n_paths=n_paths, n_steps=n_steps, seed=seed)
        price_down = heston_call(S0=S-h, K=K, T=T, r=r, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, n_paths=n_paths, n_steps=n_steps, seed=seed)
        return (price_up - price_down) / (2*h)
    elif greek == "vega":
        price_up = heston_call(S0=S, K=K, T=T, r=r, v0=v0+h, kappa=kappa, theta=theta, xi=xi, rho=rho, n_paths=n_paths, n_steps=n_steps, seed=seed)
        price_down = heston_call(S0=S, K=K, T=T, r=r, v0=max(1e-6, v0-h), kappa=kappa, theta=theta, xi=xi, rho=rho, n_paths=n_paths, n_steps=n_steps, seed=seed)
        return (price_up - price_down) / (2*h)
    elif greek == "gamma":
        price_up = heston_call(S0=S+h, K=K, T=T, r=r, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, n_paths=n_paths, n_steps=n_steps, seed=seed)
        price = heston_call(S0=S, K=K, T=T, r=r, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, n_paths=n_paths, n_steps=n_steps, seed=seed)
        price_down = heston_call(S0=S-h, K=K, T=T, r=r, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, n_paths=n_paths, n_steps=n_steps, seed=seed)
        return (price_up - 2*price + price_down) / (h**2)
    else:
        raise ValueError("greek must be 'delta', 'vega', or 'gamma'")

if __name__ == "__main__":
    S = 100
    T = 1.0
    r = 0.05
    sigma = 0.20
    v0 = 0.0638
    kappa = 0.9334
    theta = 0.0258
    xi = 0.2530
    rho = 0.0693
    strikes = np.arange(80, 121, 10)

    print("\nOP-6: Greeks Comparison Table\n")
    print(f"{'Strike':>8} {'BS_Delta':>10} {'Heston_Delta':>14} {'BS_Vega':>10} {'Heston_Vega':>14} {'BS_Gamma':>10} {'Heston_Gamma':>14}")
    print("-"*80)
    for K in strikes:
        bs_price = bs_call(S, K, T, r, sigma)
        bs_d = bs_delta(S, K, T, r, sigma)
        bs_v = bs_vega(S, K, T, r, sigma)
        bs_g = bs_gamma(S, K, T, r, sigma)
        h_d = heston_greek_fd(S, K, T, r, v0, kappa, theta, xi, rho, "delta")
        h_v = heston_greek_fd(S, K, T, r, v0, kappa, theta, xi, rho, "vega")
        h_g = heston_greek_fd(S, K, T, r, v0, kappa, theta, xi, rho, "gamma")
        print(f"{K:8.2f} {bs_d:10.4f} {h_d:14.4f} {bs_v:10.4f} {h_v:14.4f} {bs_g:10.4f} {h_g:14.4f}")
    print("\nNote: Heston Greeks are MC finite-difference esti