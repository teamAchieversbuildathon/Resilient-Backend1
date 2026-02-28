import numpy as np


def calculate_cvar(returns, alpha=0.95):
    """
    CVaR on terminal returns
    """

    var_threshold = np.percentile(
        returns,
        (1 - alpha) * 100
    )

    cvar = returns[
        returns <= var_threshold
    ].mean()

    return cvar