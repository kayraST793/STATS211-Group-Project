import numpy as np

from src.diagnostics import (autocorrelation, effective_sample_size,
                             summarize_posterior)


def test_autocorrelation_lag_zero_is_one():
    rng = np.random.default_rng(0)
    chain = rng.normal(0, 1, size=500)
    acf = autocorrelation(chain, max_lag=10)
    assert acf.shape == (11,)
    assert np.isclose(acf[0], 1.0)


def test_autocorrelation_near_zero_for_iid_draws():
    rng = np.random.default_rng(0)
    chain = rng.normal(0, 1, size=5000)
    acf = autocorrelation(chain, max_lag=20)
    assert np.all(np.abs(acf[1:]) < 0.1)  # iid draws: negligible correlation past lag 0


def test_autocorrelation_stays_high_for_random_walk():
    chain = np.cumsum(np.random.default_rng(0).normal(0, 1, size=5000))
    acf = autocorrelation(chain, max_lag=20)
    assert np.all(acf[1:] > 0.5)  # random walk: slow-decaying, strongly positive ACF


def test_effective_sample_size_near_n_for_iid_draws():
    rng = np.random.default_rng(0)
    chain = rng.normal(0, 1, size=5000)
    ess = effective_sample_size(chain)
    assert 3000 < ess <= 5000  # i.i.d. draws should have ESS close to n


def test_effective_sample_size_small_for_highly_autocorrelated_chain():
    n = 5000
    chain = np.cumsum(np.random.default_rng(0).normal(0, 1, size=n))  # random walk: highly autocorrelated
    assert effective_sample_size(chain) < 200


def test_summarize_posterior_keys_and_sane_values():
    rng = np.random.default_rng(0)
    chains = [rng.normal(5, 1, size=1000) for _ in range(4)]
    summary = summarize_posterior(chains)
    assert set(summary) == {"mean", "sd", "ci_2.5", "ci_97.5", "ess"}
    assert 4.5 < summary["mean"] < 5.5
    assert summary["ci_2.5"] < summary["mean"] < summary["ci_97.5"]
