"""
MCMC Driver Script for the Bayesian Student-t Model of GLD Daily Log Returns

Runs a single Metropolis-within-Gibbs chain (see src/mcmc.py) over
(mu, sigma, nu), started from the sample mean/std of the data.

This script runs that single-chain experiment three times -- once at the
baseline prior rate nu_rate=lambda=0.1, and once each at the
sensitivity values lambda=0.05 and lambda=0.2 -- to check whether
posterior conclusions about nu (tail-heaviness) are sensitive to this
one prior hyperparameter.

Note: with a single chain, R-hat (which compares between-chain and
within-chain variance) is undefined and reported as NaN -- see
src/diagnostics.py.

Input:
    data/GLD_returns.csv

Outputs:
    results/mcmc/lambda_{0.10,0.05,0.20}/chain_1.csv
        columns: mu, sigma, nu, is_burnin
        (all n_iter draws, including burn-in; is_burnin lets downstream
        consumers drop the first n_burnin rows)
    figures/mcmc/trace_nu_lambda_{0.10,0.05,0.20}.png
        per-experiment trace plot of nu
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

# RUN_MCMC_FAST=1 runs a quick debug pass (fewer iterations, baseline
# experiment only -- see below). Debug output goes to separate *_debug
# directories so it never overwrites real results.
FAST_DEBUG = os.environ.get("RUN_MCMC_FAST") == "1"

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
# mu0, tau0, s0, nu_shift, the init, seed, n_iter, and n_burnin are held
# fixed across all three experiments; only nu_rate varies.
INITS = [dict(mu=sample_mean, sigma=sample_std, nu=8.0)]
SEEDS = [1]
N_ITER = 20000
N_BURNIN = 5000
INIT_STEPS = dict(mu=0.005, sigma=0.3, nu=0.3)

if FAST_DEBUG:
    N_ITER = 500
    N_BURNIN = 100


def run_experiment(nu_rate: float, label: str) -> list[dict]:
    """
    Run the single-chain MCMC experiment at a given nu_rate (prior lambda)
    and write its draws to results/mcmc/{label}/chain_1.csv.

    Parameters:
    - nu_rate: the nu_rate (lambda) hyperparameter for the Exponential
      prior shape on nu, shifted by nu_shift
    - label: subdirectory name under results/mcmc/ to write outputs to

    Returns:
    - list of run_chain-shaped result dicts, one per chain (see
      src/mcmc.run_multiple_chains)
    """
    hyperparams = dict(mu0=0.0, tau0=10, s0=10, nu_shift=2.0, nu_rate=nu_rate)

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
            f"*** RUN_MCMC_FAST=1: debug mode -- baseline only, n_iter={N_ITER}. "
            "Run without this flag for the full result. ***"
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

    # The prior is now extremely diffuse (tau0=s0=10) relative to the
    # posterior (which stays concentrated at the data's ~0.01 scale) -- a
    # ~1000x mismatch in both location and width. twinx() alone only fixes a
    # y-scale mismatch (it still shares the x-axis), so a single shared
    # x-window can't show both curves meaningfully. Instead: top row shows
    # each prior's true shape over its own full range (with a dashed red
    # line marking the posterior mean -- the posterior itself is too narrow
    # to render as a curve there); bottom row zooms into the posterior's own
    # range, where prior and posterior share an x-window and only need
    # separate y-axes (twinx).
    # (param label, x-grid over the prior's full range, prior pdf,
    #  baseline posterior draws, x-axis label)
    prior_posterior_specs = [
        ("mu", np.linspace(-30.0, 30.0, 1000),
         norm(loc=0.0, scale=10).pdf, baseline_mu, "mu"),
        ("sigma", np.linspace(0.0, 30.0, 1000),
         halfnorm(scale=10).pdf, baseline_sigma, "sigma"),
        ("nu", np.linspace(2.0, 50.0, 1000),
         expon(loc=2.0, scale=1.0 / 0.1).pdf, baseline_nu,
         "nu (degrees of freedom)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, (param, prior_grid, prior_pdf, posterior_draws, xlabel) in \
            enumerate(prior_posterior_specs):
        post_pad = 0.25 * (posterior_draws.max() - posterior_draws.min())
        posterior_grid = np.linspace(
            posterior_draws.min() - post_pad, posterior_draws.max() + post_pad, 1000
        )
        posterior_density = gaussian_kde(posterior_draws)(posterior_grid)
        posterior_mean = posterior_draws.mean()

        ax_full = axes[0, col]
        ax_full.plot(prior_grid, prior_pdf(prior_grid), linestyle="--",
                     color="gray", label="Prior")
        ax_full.axvline(posterior_mean, color="C3", linestyle=":",
                        label="Posterior mean")
        ax_full.set_title(f"{param} — prior's full range")
        ax_full.set_xlabel(xlabel)
        ax_full.set_ylabel("prior density")
        ax_full.legend()

        ax_zoom = axes[1, col]
        ax_zoom.plot(posterior_grid, prior_pdf(posterior_grid), linestyle="--",
                     color="gray", label="Prior")
        ax_zoom.set_xlabel(xlabel)
        ax_zoom.set_ylabel("prior density", color="gray")
        ax_zoom.tick_params(axis="y", labelcolor="gray")
        ax_zoom.set_title(f"{param} — zoomed to posterior")
        ax_zoom.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
        ax_zoom.ticklabel_format(axis="y", useOffset=False)

        ax_zoom_post = ax_zoom.twinx()
        ax_zoom_post.plot(posterior_grid, posterior_density, color="C0", label="Posterior")
        ax_zoom_post.set_ylabel("posterior density", color="C0")
        ax_zoom_post.tick_params(axis="y", labelcolor="C0")

        lines = ax_zoom.get_lines() + ax_zoom_post.get_lines()
        ax_zoom.legend(lines, [line.get_label() for line in lines])

    fig.suptitle(
        "Prior vs. Posterior (baseline model, lambda = 0.10): how the "
        "observed GLD returns update each parameter's beliefs\n"
        "(top: prior's true shape over its full range; "
        "bottom: zoomed to the posterior, separate density scales)"
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "prior_vs_posterior.png", dpi=150)
    plt.close(fig)
