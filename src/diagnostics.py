"""
MCMC Convergence Diagnostics

Standard diagnostics for assessing mixing/convergence of multiple
Metropolis-within-Gibbs chains (see src/mcmc.py) and for summarizing
their pooled post-burn-in posterior draws.

Functions:
    r_hat                 -- Gelman-Rubin potential scale reduction factor
    effective_sample_size -- ESS via Geyer's initial monotone sequence
    trace_plot             -- overlay of per-chain traces, saved to disk
    summarize_posterior    -- pooled mean/sd/CI plus r_hat/ess for one param
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def r_hat(chains: list[np.ndarray]) -> float:
    """
    Classic Gelman-Rubin potential scale reduction factor across
    `len(chains)` equal-length chains of post-burn-in draws.

    Parameters:
    - chains: list of 1-D arrays, all the same length n

    Returns:
    - sqrt(((n-1)/n * W + B/n) / W), or NaN if fewer than 2 chains are
      given (R-hat is undefined with a single chain -- there is no
      between-chain variance to compare against)
    """
    m = len(chains)
    if m < 2:
        return float("nan")

    n = len(chains[0])

    chain_means = np.array([chain.mean() for chain in chains])
    chain_vars = np.array([chain.var(ddof=1) for chain in chains])

    overall_mean = chain_means.mean()

    B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2)
    W = chain_vars.mean()

    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def effective_sample_size(chain: np.ndarray) -> float:
    """
    Effective sample size via the standard autocorrelation-sum formula,
    summing lag-t autocorrelations in consecutive pairs and stopping at
    the first pair whose sum is negative (Geyer's initial monotone
    sequence).

    Parameters:
    - chain: 1-D array of draws

    Returns:
    - n / (1 + 2 * sum_of_autocorrelations_up_to_cutoff)
    """
    n = len(chain)
    x = chain - chain.mean()

    # Autocorrelation at all lags 1..n-1, normalized by lag-0 autocovariance.
    var = np.dot(x, x) / n
    autocorr = np.empty(n - 1)
    for lag in range(1, n):
        autocorr[lag - 1] = np.dot(x[:-lag], x[lag:]) / (n * var)

    # Sum consecutive pairs (rho_1+rho_2, rho_3+rho_4, ...) and stop at the
    # first negative pair-sum.
    acf_sum = 0.0
    for i in range(0, len(autocorr) - 1, 2):
        pair_sum = autocorr[i] + autocorr[i + 1]
        if pair_sum < 0:
            break
        acf_sum += pair_sum

    ess = n / (1 + 2 * acf_sum)
    return float(ess)


def trace_plot(chains: list[np.ndarray], param_name: str, out_path: Path,
                title: str | None = None) -> None:
    """
    Plot one line per chain over iteration index and save to out_path.

    Parameters:
    - chains: list of 1-D arrays, one per chain
    - param_name: the sampled quantity being traced; used as the y-axis
      label and, if `title` is not given, as the plot title
    - out_path: file path to save the figure to
    - title: optional plot title; defaults to f"Trace plot: {param_name}"
    """
    plt.figure()
    for i, chain in enumerate(chains):
        plt.plot(chain, linewidth=0.5, label=f"chain {i}")
    plt.xlabel("MCMC iteration (sweep)")
    plt.ylabel(param_name)
    plt.legend()
    plt.title(title if title is not None else f"Trace plot: {param_name}")
    plt.savefig(out_path, dpi=150)
    plt.close()


def summarize_posterior(chains: list[np.ndarray]) -> dict:
    """
    Summarize pooled post-burn-in draws for one parameter across chains.

    Parameters:
    - chains: list of 1-D arrays, all the same length, post-burn-in draws
      for one parameter, one array per chain

    Returns:
    - dict with keys "mean", "sd", "ci_2.5", "ci_97.5", "r_hat", "ess"
      (mean/sd/CI pool all chains' draws; ess is the average of
      effective_sample_size over the individual chains)
    """
    pooled = np.concatenate(chains)

    ess_values = [effective_sample_size(chain) for chain in chains]

    return {
        "mean": float(pooled.mean()),
        "sd": float(pooled.std(ddof=1)),
        "ci_2.5": float(np.percentile(pooled, 2.5)),
        "ci_97.5": float(np.percentile(pooled, 97.5)),
        "r_hat": r_hat(chains),
        "ess": float(np.mean(ess_values)),
    }
