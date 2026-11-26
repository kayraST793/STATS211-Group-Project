import numpy as np
from scipy.stats import norm, halfnorm, expon, t


def log_likelihood(mu: float, sigma: float, nu: float,
                    data: np.ndarray) -> float:
    """
    Student-t log-likelihood of data given (mu, sigma, nu).
    """
    if sigma <= 0 or nu <= 2:
        return -np.inf

    return float(np.sum(
        t.logpdf(data, df=nu, loc=mu, scale=sigma)
    ))


def log_prior_mu(mu: float, mu0: float, tau0: float) -> float:
    """
    mu ~ Normal(mu0, tau0^2)
    """
    return float(norm.logpdf(mu, loc=mu0, scale=tau0))


def log_prior_sigma(sigma: float, s0: float) -> float:
    """
    sigma ~ HalfNormal(s0)
    """
    if sigma <= 0:
        return -np.inf

    return float(halfnorm.logpdf(sigma, loc=0, scale=s0))


def log_prior_nu(nu: float, shift: float, rate: float) -> float:
    """
    nu = shift + Exponential(rate)
    """
    if nu <= shift:
        return -np.inf

    return float(expon.logpdf(
        nu,
        loc=shift,
        scale=1 / rate
    ))


def log_posterior(mu: float, sigma: float, nu: float,
                   data: np.ndarray, hyperparams: dict) -> float:
    """
    Log-posterior = log-likelihood + log-priors.
    """
    lp_mu = log_prior_mu(
        mu, hyperparams["mu0"], hyperparams["tau0"]
    )

    lp_sigma = log_prior_sigma(
        sigma, hyperparams["s0"]
    )

    lp_nu = log_prior_nu(
        nu,
        hyperparams["nu_shift"],
        hyperparams["nu_rate"]
    )

    if not np.all(np.isfinite([lp_mu, lp_sigma, lp_nu])):
        return -np.inf

    ll = log_likelihood(mu, sigma, nu, data)

    return float(ll + lp_mu + lp_sigma + lp_nu)
