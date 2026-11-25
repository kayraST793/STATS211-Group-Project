"""
MCMC Driver Script for the Bayesian Student-t Model of GLD Daily Log Returns

Runs 4 Metropolis-within-Gibbs chains (see src/mcmc.py) over
(mu, sigma, nu), each started from an overdispersed initial point
built from the sample mean/std of the data, per the plan's Model
Specification "Multiple chains" paragraph:
    - mu0 spread around the sample mean
    - sigma0 at {0.5, 1, 1.5, 2} x sample std
    - nu0 at {4, 8, 15, 30}

Input:
    data/GLD_returns.csv

Outputs:
    results/mcmc/chain_1.csv .. chain_4.csv
        columns: mu, sigma, nu, is_burnin
        (all n_iter draws per chain, including burn-in; is_burnin lets
        downstream consumers drop the first n_burnin rows per chain)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 1. Paths ----------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# Allow running this file directly (e.g. `python3 src/run_mcmc.py`), where
# Python puts this file's own directory on sys.path rather than the repo
# root, by making sure the repo root is importable before the src.* import.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.mcmc import run_multiple_chains  # noqa: E402
RESULTS_DIR = PROJECT_DIR / "results" / "mcmc"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load data -------------------------------------------------------------
returns = pd.read_csv(
    DATA_DIR / "GLD_returns.csv",
    index_col=0,
    parse_dates=True,
)["Log_Return"]
data = returns.to_numpy()

sample_mean = returns.mean()
sample_std = returns.std()

# 3. Model hyperparameters (Model Specification baseline) ------------------
HYPERPARAMS = dict(mu0=0.0, tau0=0.05, s0=0.05, nu_shift=2.0, nu_rate=0.1)

# 4. Overdispersed starting points for 4 chains -----------------------------
MU0_OFFSETS = [-1.0, -0.5, 0.5, 1.0]      # in units of sample_std, around sample_mean
SIGMA0_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0]  # x sample_std
NU0_VALUES = [4.0, 8.0, 15.0, 30.0]

inits = [
    dict(
        mu=sample_mean + offset * sample_std,
        sigma=multiplier * sample_std,
        nu=nu0,
    )
    for offset, multiplier, nu0 in zip(MU0_OFFSETS, SIGMA0_MULTIPLIERS, NU0_VALUES)
]

SEEDS = [1, 2, 3, 4]
N_ITER = 20000
N_BURNIN = 5000
INIT_STEPS = dict(mu=0.005, sigma=0.3, nu=0.3)

# 5. Run chains --------------------------------------------------------------
results = run_multiple_chains(
    data,
    HYPERPARAMS,
    inits,
    n_iter=N_ITER,
    n_burnin=N_BURNIN,
    init_steps=INIT_STEPS,
    seeds=SEEDS,
)

# 6. Write per-chain draws and report diagnostics ----------------------------
is_burnin = np.arange(N_ITER) < N_BURNIN

for i, result in enumerate(results, start=1):
    chain_df = pd.DataFrame({
        "mu": result["mu"],
        "sigma": result["sigma"],
        "nu": result["nu"],
        "is_burnin": is_burnin,
    })
    chain_df.to_csv(RESULTS_DIR / f"chain_{i}.csv", index=False)

    post_burnin = ~is_burnin
    post_mu_mean = result["mu"][post_burnin].mean()
    post_sigma_mean = result["sigma"][post_burnin].mean()
    post_nu_mean = result["nu"][post_burnin].mean()

    print(f"Chain {i} (init={inits[i - 1]}, seed={SEEDS[i - 1]}):")
    print(f"  accept_rate = {result['accept_rate']}")
    print(f"  posterior mean: mu={post_mu_mean:.6f}, "
          f"sigma={post_sigma_mean:.6f}, nu={post_nu_mean:.4f}")
