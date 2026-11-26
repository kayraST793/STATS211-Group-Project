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
    figures/mcmc/prior_vs_posterior.png
        prior vs. posterior density curves for mu, sigma, nu (baseline
        model, lambda=0.10), showing how the observed GLD returns update
        each parameter's initial (prior) beliefs
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from scipy.stats import gaussian_kde, norm, halfnorm, expon
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

FAST_DEBUG = os.environ.get("RUN_MCMC_FAST") == "1"

# Debug runs write to separate directories so they can never overwrite the
# real multi-chain results/figures used for the actual analysis.
if FAST_DEBUG:
    RESULTS_DIR = PROJECT_DIR / "results" / "mcmc_debug"
    FIGURES_DIR = PROJECT_DIR / "figures" / "mcmc_debug"
else:
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

# Fast debug mode (FAST_DEBUG flag set above, before RESULTS_DIR/FIGURES_DIR):
# `RUN_MCMC_FAST=1 python3 src/run_mcmc.py` runs a single chain on the
# baseline experiment only, with far fewer iterations, for quick iteration
# on plots/code. NOT a substitute for the real 4-chain analysis: with 1
# chain, r_hat is undefined (reported as NaN -- see src/diagnostics.py) and
# the "multiple independent chains" requirement is not satisfied. Always do
# a full default run before drawing conclusions.
if FAST_DEBUG:
    INITS = INITS[:1]
    SEEDS = SEEDS[:1]
    N_ITER = 500
    N_BURNIN = 100


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
    if FAST_DEBUG:
        print(
            "*** RUN_MCMC_FAST=1: debug mode -- 1 chain, baseline only, "
            f"n_iter={N_ITER}. NOT the real analysis (r_hat is undefined "
            "with 1 chain). Run without this flag for the full result. ***"
        )
        experiments = [(0.1, "lambda_0.10")]  # baseline only
    else:
        experiments = [
            (0.1, "lambda_0.10"),   # baseline
            (0.05, "lambda_0.05"),
            (0.2, "lambda_0.20"),
        ]

    # 4. Run baseline + sensitivity experiments, validate diagnostics --------
    summaries = {}  # label -> {param: summary_dict}
    nu_pooled_draws = {}  # label -> pooled post-burn-in nu draws
    baseline_param_chains = None  # lambda_0.10's {param: [chain_draws, ...]}

    for nu_rate, label in experiments:
        results = run_experiment(nu_rate, label)

        print(f"\n=== {label} (nu_rate={nu_rate}) ===")

        param_chains = {
            param: _post_burnin_chains(results, param)
            for param in ("mu", "sigma", "nu")
        }
        if label == "lambda_0.10":
            baseline_param_chains = param_chains

        param_summaries = {
            param: summarize_posterior(chains)
            for param, chains in param_chains.items()
        }
        summaries[label] = param_summaries

        for param, summary in param_summaries.items():
            print(f"  {param}: {summary}")

        for i, result in enumerate(results, start=1):
            print(f"  chain {i} accept_rate = {result['accept_rate']}")

        trace_plot(
            param_chains["nu"], "Degrees of freedom (nu)",
            FIGURES_DIR / f"trace_nu_{label}.png",
            title=f"Degrees of Freedom (nu) Trace — lambda = {nu_rate}",
        )

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

    # 6. Prior vs. posterior comparison for mu, sigma, nu (baseline model) ----
    # Shows how the observed GLD returns update each parameter's prior
    # (initial) beliefs into its posterior (updated) beliefs.
    baseline_mu = np.concatenate(baseline_param_chains["mu"])
    baseline_sigma = np.concatenate(baseline_param_chains["sigma"])
    baseline_nu = np.concatenate(baseline_param_chains["nu"])

    # Each prior's full natural range, wide enough to show its true shape
    # (bell/half-normal peak/exponential decay). The posterior is far more
    # concentrated than the prior (that is the point of this plot), so prior
    # and posterior are put on separate y-axes (twinx) -- sharing one y-axis
    # would always flatten the lower-peaked curve to a near-invisible line.
    # (param label, x-grid over the prior's full range, prior pdf,
    #  baseline posterior draws, x-axis label)
    prior_posterior_specs = [
        ("mu", np.linspace(-0.2, 0.2, 1000),
         norm(loc=0.0, scale=0.05).pdf, baseline_mu, "mu"),
        ("sigma", np.linspace(0.0, 0.2, 1000),
         halfnorm(scale=0.05).pdf, baseline_sigma, "sigma"),
        ("nu", np.linspace(2.0, 50.0, 1000),
         expon(loc=2.0, scale=1.0 / 0.1).pdf, baseline_nu,
         "nu (degrees of freedom)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax_prior, (param, grid, prior_pdf, posterior_draws, xlabel) in zip(
            axes, prior_posterior_specs):
        posterior_density = gaussian_kde(posterior_draws)(grid)

        ax_prior.plot(grid, prior_pdf(grid), linestyle="--", color="gray", label="Prior")
        ax_prior.set_xlabel(xlabel)
        ax_prior.set_ylabel("prior density", color="gray")
        ax_prior.tick_params(axis="y", labelcolor="gray")
        ax_prior.set_title(param)
        ax_prior.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))

        ax_post = ax_prior.twinx()
        ax_post.plot(grid, posterior_density, color="C0", label="Posterior")
        ax_post.set_ylabel("posterior density", color="C0")
        ax_post.tick_params(axis="y", labelcolor="C0")

        lines = ax_prior.get_lines() + ax_post.get_lines()
        ax_prior.legend(lines, [line.get_label() for line in lines])

    fig.suptitle(
        "Prior vs. Posterior (baseline model, lambda = 0.10): how the "
        "observed GLD returns update each parameter's beliefs\n"
        "(each parameter's prior shown over its full range; "
        "prior and posterior use separate density scales)"
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "prior_vs_posterior.png", dpi=150)
    plt.close(fig)
