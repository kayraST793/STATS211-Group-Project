"""
MCMC Convergence Diagnostics

Standard diagnostics for assessing mixing of Metropolis-within-Gibbs
chains (see src/mcmc.py) and for summarizing their pooled post-burn-in
posterior draws.

Functions:
    autocorrelation       -- sample ACF of a chain at lags 0..max_lag
    effective_sample_size -- ESS via Geyer's initial monotone sequence
    trace_plot             -- overlay of per-chain traces, saved to disk
    acf_plot               -- bar plot of a chain's autocorrelation function
    burnin_trace_plot      -- stacked full-chain traces marking burn-in
    summarize_posterior    -- pooled mean/sd/CI plus ess for one param
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def autocorrelation(chain: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Sample autocorrelation function (ACF) of a chain at lags 0..max_lag.

    Parameters:
    - chain: 1-D array of draws
    - max_lag: highest lag to compute (inclusive)

    Returns:
    - array of length max_lag + 1; acf[0] == 1.0 by definition, acf[k] is
      the lag-k autocorrelation (normalized by the lag-0 autocovariance)
    """
    n = len(chain)
    x = chain - chain.mean()
    var = np.dot(x, x) / n

    acf = np.empty(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.dot(x[:-lag], x[lag:]) / (n * var)
    return acf


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
    autocorr = autocorrelation(chain, n - 1)[1:]  # drop lag 0 (always 1.0)

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


def acf_plot(chain: np.ndarray, param_name: str, out_path: Path,
             max_lag: int = 50) -> None:
    """
    Bar plot of a chain's autocorrelation function up to max_lag, with a
    dashed +-1.96/sqrt(n) band marking approximate statistical
    insignificance (a common rule of thumb for "has the chain decorrelated
    by this lag").

    Parameters:
    - chain: 1-D array of draws
    - param_name: used as the plot title and y-axis label
    - out_path: file path to save the figure to
    - max_lag: highest lag to plot (default 50)
    """
    acf = autocorrelation(chain, max_lag)
    band = 1.96 / np.sqrt(len(chain))

    plt.figure()
    plt.bar(range(max_lag + 1), acf, width=0.6)
    plt.axhline(band, color="gray", linestyle="--", linewidth=0.8)
    plt.axhline(-band, color="gray", linestyle="--", linewidth=0.8)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("lag")
    plt.ylabel(f"autocorrelation ({param_name})")
    plt.title(f"Autocorrelation: {param_name}")
    plt.savefig(out_path, dpi=150)
    plt.close()


def burnin_trace_plot(draws: dict[str, np.ndarray], n_burnin: int,
                      out_path: Path, title: str | None = None,
                      n_show: int | None = None,
                      settled_values: dict[str, float] | None = None) -> None:
    """
    Stacked trace plots of the start of the chain, marking burn-in.

    One vertically stacked subplot per entry in `draws`, all sharing the
    iteration axis. Burn-in iterations are shaded and drawn in a separate
    colour from the retained posterior-sampling region, with a dashed line
    at the boundary. Samples are plotted exactly as drawn -- no smoothing,
    subsampling, or transformation.

    Parameters:
    - draws: {display label: 1-D array of ALL iterations (burn-in first)},
      in the order the subplots should appear
    - n_burnin: number of leading iterations treated as burn-in
    - out_path: file path to save the figure to
    - title: optional figure title
    - n_show: display only the first n_show iterations. Use this to zoom
      into the transient when n_burnin is far longer than the chain takes
      to settle, which would otherwise render as a flat band. Defaults to
      the whole chain.
    - settled_values: {display label: value} drawn as a horizontal
      reference line, so convergence toward it is visible
    """
    burnin_color = "#c44e52"
    posterior_color = "#2d6da4"

    n_total = len(next(iter(draws.values())))
    n_show = min(n_show or n_total, n_total)
    # How much of the visible window is burn-in (all of it, when zoomed in
    # further than the burn-in boundary).
    n_burnin_visible = min(n_burnin, n_show)

    fig, axes = plt.subplots(len(draws), 1, figsize=(12, 9), sharex=True)

    for ax, (label, chain) in zip(np.atleast_1d(axes), draws.items()):
        chain = chain[:n_show]
        iterations = np.arange(1, n_show + 1)

        ax.axvspan(1, n_burnin_visible, color=burnin_color, alpha=0.12,
                   zorder=0, label=f"Burn-in (1–{n_burnin:,})")
        ax.plot(iterations[:n_burnin_visible], chain[:n_burnin_visible],
                color=burnin_color, linewidth=0.9, zorder=1)

        if n_show > n_burnin:
            ax.plot(iterations[n_burnin:], chain[n_burnin:],
                    color=posterior_color, linewidth=0.9, zorder=1,
                    label=f"Posterior sampling ({n_burnin + 1:,}–{n_total:,})")
            ax.axvline(n_burnin, color="black", linestyle="--", linewidth=1.2,
                       zorder=3,
                       label=f"End of burn-in (iteration {n_burnin:,})")

        if settled_values is not None and label in settled_values:
            ax.axhline(settled_values[label], color="black", linestyle=":",
                       linewidth=1.3, zorder=2, label="Posterior mean")

        ax.set_ylabel(label)
        ax.margins(x=0)
        ax.grid(alpha=0.2, linewidth=0.5)

    axes[-1].set_xlabel("MCMC iteration")
    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))

    # One legend for the whole figure: every subplot shares these regions.
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                        frameon=True, bbox_to_anchor=(0.5, 0.955))
    for line in legend.get_lines():
        line.set_linewidth(2.0)  # thin trace lines are illegible in a legend

    if title is not None:
        fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def summarize_posterior(chains: list[np.ndarray]) -> dict:
    """
    Summarize pooled post-burn-in draws for one parameter across chains.

    Parameters:
    - chains: list of 1-D arrays, all the same length, post-burn-in draws
      for one parameter, one array per chain

    Returns:
    - dict with keys "mean", "sd", "ci_2.5", "ci_97.5", "ess"
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
        "ess": float(np.mean(ess_values)),
    }
