"""
MCMC Driver Script for the Bayesian Student-t Model of GLD Daily Log Returns

Runs 4 Metropolis-within-Gibbs chains (see src/mcmc.py) over
(mu, sigma, nu), each started from an overdispersed initial point
built from the sample mean/std of the data, per the plan's Model
Specification "Multiple chains" paragraph:
    - mu0 spread around the sample mean
    - sigma0 at {0.5, 1, 1.5, 2} x sample std
    - nu0 at {4, 8, 15, 30}

This script runs that 4-chain experiment three times -- once at the
baseline prior rate nu_rate=lambda=0.1, and once each at the
sensitivity values lambda=0.05 and lambda=0.2 -- to check whether
posterior conclusions about nu (tail-heaviness) are sensitive to this
one prior hyperparameter. Each run is validated with standard MCMC
diagnostics (R-hat, effective sample size) before being compared.

Input:
    data/GLD_returns.csv

Outputs:
    results/mcmc/lambda_{0.10,0.05,0.20}/chain_1.csv .. chain_4.csv
        columns: mu, sigma, nu, is_burnin
        (all n_iter draws per chain, including burn-in; is_burnin lets
        downstream consumers drop the first n_burnin rows per chain)
    figures/mcmc/trace_nu_lambda_{0.10,0.05,0.20}.png
        per-experiment trace plot of nu across all 4 chains
    figures/mcmc/nu_posterior_by_lambda.png
        overlaid pooled post-burn-in nu posterior density by lambda
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from scipy.stats import gaussian_kde
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Paths ----------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# Allow running this file directly (e.g. `python3 src/run_mcmc.py`), where
# Python puts this file's own directory on sys.path rather than the repo
# root, by making sure the repo root is importable before the src.* import.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.mcmc import run_multiple_chains  # noqa: E402
from src.diagnostics import summarize_posterior, trace_plot  # noqa: E402

RESULTS_DIR = PROJECT_DIR / "results" / "mcmc"
FIGURES_DIR = PROJECT_DIR / "figures" / "mcmc"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load data -------------------------------------------------------------
returns = pd.read_csv(
    DATA_DIR / "GLD_returns.csv",
    index_col=0,
    parse_dates=True,
)["Log_Return"]
data = returns.to_numpy()

sample_mean = returns.mean()
sample_std = returns.std()

# 3. Shared experiment configuration (Model Specification baseline) --------
# mu0, tau0, s0, nu_shift, the init strategy, seeds, n_iter, and n_burnin are
# held fixed across all three experiments; only nu_rate varies.
MU0_OFFSETS = [-1.0, -0.5, 0.5, 1.0]      # in units of sample_std, around sample_mean
SIGMA0_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0]  # x sample_std
NU0_VALUES = [4.0, 8.0, 15.0, 30.0]

INITS = [
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


def run_experiment(nu_rate: float, label: str) -> list[dict]:
    """
    Run the 4-chain MCMC experiment at a given nu_rate (prior lambda) and
    write per-chain draws to results/mcmc/{label}/chain_{i}.csv.

    Parameters:
    - nu_rate: the nu_rate (lambda) hyperparameter for the Exponential
      prior shape on nu, shifted by nu_shift
    - label: subdirectory name under results/mcmc/ to write outputs to

    Returns:
    - list of run_chain-shaped result dicts, one per chain (see
      src/mcmc.run_multiple_chains)
    """
    hyperparams = dict(mu0=0.0, tau0=0.05, s0=0.05, nu_shift=2.0, nu_rate=nu_rate)

    results = run_multiple_chains(
        data,
        hyperparams,
        INITS,
        n_iter=N_ITER,
        n_burnin=N_BURNIN,
        init_steps=INIT_STEPS,
        seeds=SEEDS,
    )

    out_dir = RESULTS_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    is_burnin = np.arange(N_ITER) < N_BURNIN
    for i, result in enumerate(results, start=1):
        chain_df = pd.DataFrame({
            "mu": result["mu"],
            "sigma": result["sigma"],
            "nu": result["nu"],
            "is_burnin": is_burnin,
        })
        chain_df.to_csv(out_dir / f"chain_{i}.csv", index=False)

    return results


def _post_burnin_chains(results: list[dict], param: str) -> list[np.ndarray]:
    """Slice burn-in off every chain's draws for one parameter."""
    return [result[param][N_BURNIN:] for result in results]


if __name__ == "__main__":
    experiments = [
        (0.1, "lambda_0.10"),   # baseline
        (0.05, "lambda_0.05"),
        (0.2, "lambda_0.20"),
    ]

    # 4. Run baseline + sensitivity experiments, validate diagnostics --------
    summaries = {}  # label -> {param: summary_dict}
    nu_pooled_draws = {}  # label -> pooled post-burn-in nu draws

    for nu_rate, label in experiments:
        results = run_experiment(nu_rate, label)

        print(f"\n=== {label} (nu_rate={nu_rate}) ===")

        param_chains = {
            param: _post_burnin_chains(results, param)
            for param in ("mu", "sigma", "nu")
        }

        param_summaries = {
            param: summarize_posterior(chains)
            for param, chains in param_chains.items()
        }
        summaries[label] = param_summaries

        for param, summary in param_summaries.items():
            print(f"  {param}: {summary}")

        for i, result in enumerate(results, start=1):
            print(f"  chain {i} accept_rate = {result['accept_rate']}")

        trace_plot(param_chains["nu"], f"nu trace ({label})",
                   FIGURES_DIR / f"trace_nu_{label}.png")

        nu_pooled_draws[label] = np.concatenate(param_chains["nu"])

    # 5. Compare p(nu | data, lambda) across the three lambda values ---------
    all_nu = np.concatenate(list(nu_pooled_draws.values()))
    x_grid = np.linspace(all_nu.min(), all_nu.max(), 500)

    plt.figure()
    for _, label in experiments:
        density = gaussian_kde(nu_pooled_draws[label])(x_grid)
        plt.plot(x_grid, density, label=label)
    plt.xlabel("nu")
    plt.ylabel("density")
    plt.title("Posterior of nu across prior lambda (nu_rate) values")
    plt.legend()
    plt.savefig(FIGURES_DIR / "nu_posterior_by_lambda.png", dpi=150)
    plt.close()

    print("\n=== nu posterior comparison across lambda ===")
    print(f"{'label':<14}{'mean':>10}{'sd':>10}{'ci_2.5':>10}{'ci_97.5':>10}")
    for _, label in experiments:
        nu_summary = summaries[label]["nu"]
        print(f"{label:<14}{nu_summary['mean']:>10.4f}{nu_summary['sd']:>10.4f}"
              f"{nu_summary['ci_2.5']:>10.4f}{nu_summary['ci_97.5']:>10.4f}")
