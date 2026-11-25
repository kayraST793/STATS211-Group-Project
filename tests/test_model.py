import numpy as np
from scipy import stats
from src.model import (log_likelihood, log_prior_mu, log_prior_sigma,
                       log_prior_nu, log_posterior)


def test_log_likelihood_matches_scipy_student_t():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, size=20)
    mu, sigma, nu = 0.3, 1.5, 7.0
    expected = np.sum(stats.t.logpdf(data, df=nu, loc=mu, scale=sigma))
    assert np.isclose(log_likelihood(mu, sigma, nu, data), expected, rtol=1e-8)


def test_log_likelihood_finite_for_large_nu():
    data = np.array([0.01, -0.02, 0.0, 0.05])
    assert np.isfinite(log_likelihood(0.0, 0.02, 1e4, data))


def test_log_likelihood_finite_for_extreme_residual():
    data = np.array([50.0])  # extreme outlier relative to sigma
    assert np.isfinite(log_likelihood(0.0, 0.01, 5.0, data))


def test_log_prior_mu_matches_normal_logpdf():
    from scipy.stats import norm
    assert np.isclose(log_prior_mu(0.02, 0.0, 0.05), norm.logpdf(0.02, 0.0, 0.05))


def test_log_prior_sigma_matches_halfnorm_logpdf():
    from scipy.stats import halfnorm
    assert np.isclose(log_prior_sigma(0.03, 0.05), halfnorm.logpdf(0.03, scale=0.05))


def test_log_prior_sigma_rejects_nonpositive():
    assert log_prior_sigma(0.0, 0.05) == -np.inf
    assert log_prior_sigma(-1.0, 0.05) == -np.inf
    assert np.isfinite(log_prior_sigma(0.03, 0.05))


def test_log_prior_nu_rejects_at_or_below_shift():
    assert log_prior_nu(2.0, shift=2.0, rate=0.1) == -np.inf
    assert log_prior_nu(1.0, shift=2.0, rate=0.1) == -np.inf
    assert np.isfinite(log_prior_nu(5.0, shift=2.0, rate=0.1))


def test_log_posterior_sums_likelihood_and_priors():
    data = np.array([0.01, -0.01, 0.02])
    hyper = dict(mu0=0.0, tau0=0.05, s0=0.05, nu_shift=2.0, nu_rate=0.1)
    mu, sigma, nu = 0.001, 0.02, 6.0
    expected = (log_likelihood(mu, sigma, nu, data)
                + log_prior_mu(mu, hyper["mu0"], hyper["tau0"])
                + log_prior_sigma(sigma, hyper["s0"])
                + log_prior_nu(nu, hyper["nu_shift"], hyper["nu_rate"]))
    assert np.isclose(log_posterior(mu, sigma, nu, data, hyper), expected)


def test_log_posterior_is_minus_inf_outside_support():
    data = np.array([0.01, -0.01])
    hyper = dict(mu0=0.0, tau0=0.05, s0=0.05, nu_shift=2.0, nu_rate=0.1)
    assert log_posterior(0.0, -0.01, 6.0, data, hyper) == -np.inf
    assert log_posterior(0.0, 0.02, 1.5, data, hyper) == -np.inf
