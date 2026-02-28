import numpy as np
from scipy.stats import multivariate_t
from sklearn.covariance import LedoitWolf


def run_monte_carlo(
    returns_array,
    num_simulations=5000,
    num_days=252,
    df=6,
    random_state=42
):
    """
    Institutional-grade Monte Carlo simulation.

    Returns:
        simulated_asset_returns
        Shape: (num_simulations, num_days, num_assets)
    """

    np.random.seed(random_state)

    # ===== Estimate covariance using shrinkage =====
    lw = LedoitWolf()
    lw.fit(returns_array)
    cov_matrix = lw.covariance_

    # For risk engine: mean = 0 (separate alpha from risk)
    num_assets = returns_array.shape[1]
    mean = np.zeros(num_assets)

    # ===== Generate Student-t distributed samples =====
    simulated = multivariate_t.rvs(
        loc=mean,
        shape=cov_matrix,
        df=df,
        size=(num_simulations, num_days)
    )

    return simulated