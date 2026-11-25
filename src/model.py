import numpy as np
from scipy.special import gammaln
from scipy.stats import norm


def log_likelihood(mu: float, sigma: float, nu: float, data: np.ndarray) -> float:
    """
    Log-likelihood of data under Student-t distribution.

    Parameters:
    - mu: location parameter
    - sigma: scale parameter
    - nu: degrees of freedom
    - data: array of observations

    Returns:
    - Log-likelihood value
    """
    residuals = (data - mu) / sigma

    # Student-t log-pdf formula
    log_pdf = (gammaln((nu + 1) / 2) - gammaln(nu / 2)
               - 0.5 * np.log(nu * np.pi * sigma ** 2)
               - (nu + 1) / 2 * np.log(1 + residuals ** 2 / nu))

    return np.sum(log_pdf)


def log_prior_mu(mu: float, mu0: float, tau0: float) -> float:
    """
    Log-prior for mu: Normal distribution N(mu0, tau0^2).

    Parameters:
    - mu: parameter value
    - mu0: mean of the prior
    - tau0: standard deviation of the prior

    Returns:
    - Log-prior value
    """
    return norm.logpdf(mu, loc=mu0, scale=tau0)


def log_prior_sigma(sigma: float, s0: float) -> float:
    """
    Log-prior for sigma: Half-Normal distribution (exponential on positive line).

    Parameters:
    - sigma: scale parameter
    - s0: scale of the half-normal

    Returns:
    - Log-prior value or -inf if sigma <= 0
    """
    if sigma <= 0:
        return -np.inf

    # Half-normal log-pdf: log(2) - 0.5*log(pi*s0^2) - (sigma^2)/(2*s0^2)
    return np.log(2) - 0.5 * np.log(np.pi * s0 ** 2) - (sigma ** 2) / (2 * s0 ** 2)


def log_prior_nu(nu: float, shift: float, rate: float) -> float:
    """
    Log-prior for nu: Shifted exponential distribution.

    Parameters:
    - nu: degrees of freedom
    - shift: shift parameter (minimum value of nu)
    - rate: rate parameter of the exponential

    Returns:
    - Log-prior value or -inf if nu <= shift
    """
    if nu <= shift:
        return -np.inf

    # Shifted exponential log-pdf: log(rate) - rate*(nu - shift)
    return np.log(rate) - rate * (nu - shift)


def log_posterior(mu: float, sigma: float, nu: float, data: np.ndarray,
                  hyperparams: dict) -> float:
    """
    Log-posterior: log-likelihood + log-priors.

    Parameters:
    - mu, sigma, nu: parameters
    - data: observations
    - hyperparams: dict with keys 'mu0', 'tau0', 's0', 'nu_shift', 'nu_rate'

    Returns:
    - Log-posterior value or -inf if parameters are out of support
    """
    # Check if parameters are in support (priors return -inf out of support)
    log_prior_sigma_val = log_prior_sigma(sigma, hyperparams['s0'])
    log_prior_nu_val = log_prior_nu(nu, hyperparams['nu_shift'], hyperparams['nu_rate'])

    if log_prior_sigma_val == -np.inf or log_prior_nu_val == -np.inf:
        return -np.inf

    # Sum all log terms
    return (log_likelihood(mu, sigma, nu, data)
            + log_prior_mu(mu, hyperparams['mu0'], hyperparams['tau0'])
            + log_prior_sigma_val
            + log_prior_nu_val)
