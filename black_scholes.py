import numpy as np
from scipy.stats import norm

def black_scholes_price(S0, K, r, T, sigma, option_type='call'):
    """
    Computes the Black-Scholes price for a European call or put option.

    Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        r (float): Risk-free rate
        T (float): Time to maturity (in years)
        sigma (float): Volatility
        option_type (str): 'call' or 'put'

    Returns:
        float: Option price
    """
    if T <= 0 or sigma <= 0:
        return max(0.0, S0 - K) if option_type == 'call' else max(0.0, K - S0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
