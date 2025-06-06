import numpy as np

def generate_gbm_paths(S0, r, sigma, T, n_steps, n_sims):
    dt = T / n_steps
    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = S0

    rand_normals = np.random.normal(0, 1, size=(n_sims, n_steps))

    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand_normals[:, t - 1]
        )

    return paths


def price_option_mc(paths, K, r, T, option_type='call'):
    Parameters:
        paths (ndarray): Simulated GBM paths (n_sims x n_steps+1)
        K (float): Strike price
        r (float): Risk-free interest rate
        T (float): Time to maturity
        option_type (str): 'call' or 'put'

    Returns:
        float: Estimated present value of the option
    """
    S_T = paths[:, -1]  

    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - S_T, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted_payoff = np.exp(-r * T) * np.mean(payoffs)
    return discounted_payoff
