import numpy as np
from scipy.optimize import minimize


def optimize_cvar(
    simulated_asset_returns,
    alpha=0.95,
    max_weight=0.4
):
    """
    Minimize CVaR on terminal portfolio returns.
    """

    num_simulations, num_days, num_assets = simulated_asset_returns.shape

    def portfolio_terminal_returns(weights):

        # Daily portfolio log returns
        portfolio_daily = np.einsum(
            "sda,a->sd",
            simulated_asset_returns,
            weights
        )

        # Terminal log return
        terminal_log = portfolio_daily.sum(axis=1)

        # Convert to simple return
        terminal_return = np.exp(terminal_log) - 1

        return terminal_return

    def objective(weights):

        terminal_returns = portfolio_terminal_returns(weights)

        var_threshold = np.percentile(
            terminal_returns,
            (1 - alpha) * 100
        )

        cvar = terminal_returns[
            terminal_returns <= var_threshold
        ].mean()

        return -cvar  # minimize CVaR

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    ]

    bounds = [(0, max_weight) for _ in range(num_assets)]

    init = np.ones(num_assets) / num_assets

    result = minimize(
        objective,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300}
    )

    return result.x