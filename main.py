from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
import pandas as pd

from engine.monte_carlo import run_monte_carlo
from engine.cvar_optimization import optimize_cvar
from engine.tail_risk import calculate_cvar

app = FastAPI(title="Resilient Risk Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class InvestorProfile(BaseModel):
    investment_amount: float
    risk_level: str
    time_horizon: int


@app.get("/")
def root():
    return {"status": "Risk Engine Online"}


@app.post("/analyze")
def analyze_portfolio(profile: InvestorProfile):

    # =========================
    # 1. Load real log returns
    # =========================
    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "returns.csv")

    returns = pd.read_csv(
    DATA_PATH,
    index_col=0
    )

    asset_names = returns.columns.tolist()
    returns_array = returns.values

    # =========================
    # 2. Time scaling
    # =========================
    num_days = 252 * profile.time_horizon

    # =========================
    # 3. Risk profile control
    # =========================
    if profile.risk_level.lower() == "conservative":
        max_weight = 0.25
    elif profile.risk_level.lower() == "growth":
        max_weight = 0.6
    else:
        max_weight = 0.4

    # =========================
    # 4. Monte Carlo simulation
    # =========================
    simulated_assets = run_monte_carlo(
        returns_array,
        num_simulations=8000,
        num_days=num_days
    )

    # =========================
    # 5. Optimize CVaR
    # =========================
    weights = optimize_cvar(
        simulated_assets,
        alpha=0.95,
        max_weight=max_weight
    )

    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()

    # =========================
    # 6. Compute terminal returns
    # =========================
    portfolio_daily = np.einsum(
        "sda,a->sd",
        simulated_assets,
        weights
    )

    terminal_log = portfolio_daily.sum(axis=1)
    terminal_returns = np.exp(terminal_log) - 1

    cvar_value = calculate_cvar(
        terminal_returns,
        alpha=0.95
    )

    # =========================
    # 7. Allocation output
    # =========================
    allocation = {
        asset: round(weight * 100, 2)
        for asset, weight in zip(asset_names, weights)
    }

    return {
        "investment_amount": profile.investment_amount,
        "risk_profile": profile.risk_level,
        "time_horizon_years": profile.time_horizon,

        "estimated_worst_case_loss_pct": round(
            abs(cvar_value) * 100,
            2
        ),

        "confidence_level": "95% CVaR",

        "strategic_asset_allocation": allocation,

        "engine": "Student-t Monte Carlo + Shrinkage + CVaR"

    }

