import numpy as np

from src.model import log_posterior


def propose_mu(mu: float, step: float, rng: np.random.Generator) -> float:
    """
    Symmetric random-walk proposal for mu.

    Parameters:
    - mu: current value
    - step: random-walk standard deviation
    - rng: numpy random Generator

    Returns:
    - Proposed mu value
    """
    return mu + rng.normal(0, step)


def propose_sigma(sigma: float, step: float, rng: np.random.Generator) -> tuple[float, float]:
    """
    Random-walk proposal for sigma on the log scale: sigma' = sigma * exp(eps).
    Always strictly positive.

    Parameters:
    - sigma: current value
    - step: random-walk standard deviation (on log scale)
    - rng: numpy random Generator

    Returns:
    - (sigma_proposed, log_jacobian) where log_jacobian = log(sigma_proposed / sigma)
    """
    eps = rng.normal(0, step)
    sigma_prop = sigma * np.exp(eps)
    log_jacobian = np.log(sigma_prop / sigma)
    return sigma_prop, log_jacobian


def propose_nu(nu: float, shift: float, step: float, rng: np.random.Generator) -> tuple[float, float]:
    """
    Random-walk proposal for nu on the shifted-log scale:
    nu' = shift + (nu - shift) * exp(eps). Always strictly above shift.

    Parameters:
    - nu: current value
    - shift: lower bound for nu (nu must stay > shift)
    - step: random-walk standard deviation (on shifted-log scale)
    - rng: numpy random Generator

    Returns:
    - (nu_proposed, log_jacobian) where log_jacobian = log((nu_proposed - shift) / (nu - shift))
    """
    eps = rng.normal(0, step)
    nu_prop = shift + (nu - shift) * np.exp(eps)
    log_jacobian = np.log((nu_prop - shift) / (nu - shift))
    return nu_prop, log_jacobian


def run_chain(data: np.ndarray, hyperparams: dict, init: dict, n_iter: int,
              n_burnin: int, init_steps: dict, rng: np.random.Generator,
              tune_every: int = 50, target_accept: float = 0.44) -> dict:
    """
    Run a single Metropolis-within-Gibbs chain for (mu, sigma, nu).

    Each sweep performs one Metropolis update per block, in order
    mu, sigma, nu, holding the other two parameters fixed at their
    current values. During burn-in, step sizes are adapted every
    `tune_every` sweeps by rescaling log(step) by +-0.1 depending on
    whether the block's empirical acceptance rate over the window was
    above/below `target_accept`. Adaptation stops once burn-in ends.

    Parameters:
    - data: observed returns
    - hyperparams: dict with keys mu0, tau0, s0, nu_shift, nu_rate
    - init: dict with keys mu, sigma, nu (starting values)
    - n_iter: total number of sweeps to run (including burn-in)
    - n_burnin: number of initial sweeps treated as burn-in
    - init_steps: dict with keys mu, sigma, nu (initial random-walk step sizes)
    - rng: numpy random Generator
    - tune_every: adapt step sizes every this many sweeps during burn-in
    - target_accept: target acceptance rate for step-size adaptation

    Returns:
    - dict with keys "mu", "sigma", "nu" (np.ndarray of length n_iter each,
      including burn-in draws), "accept_rate" (dict of post-burn-in
      acceptance rates per block), and "final_steps" (dict of step sizes
      at the end of the run).
    """
    nu_shift = hyperparams["nu_shift"]

    mu = init["mu"]
    sigma = init["sigma"]
    nu = init["nu"]

    steps = dict(init_steps)

    mu_draws = np.empty(n_iter)
    sigma_draws = np.empty(n_iter)
    nu_draws = np.empty(n_iter)

    window_accepts = dict(mu=0, sigma=0, nu=0)
    post_burnin_accepts = dict(mu=0, sigma=0, nu=0)
    n_post_burnin = 0

    current_lp = log_posterior(mu, sigma, nu, data, hyperparams)

    for it in range(n_iter):
        is_post_burnin = it >= n_burnin

        # --- mu block (symmetric random walk, no Jacobian) ---
        mu_prop = propose_mu(mu, steps["mu"], rng)
        prop_lp = log_posterior(mu_prop, sigma, nu, data, hyperparams)
        log_accept_ratio = prop_lp - current_lp
        accepted_mu = np.log(rng.uniform()) < log_accept_ratio
        if accepted_mu:
            mu = mu_prop
            current_lp = prop_lp
            window_accepts["mu"] += 1
            if is_post_burnin:
                post_burnin_accepts["mu"] += 1

        # --- sigma block (log random walk, with Jacobian) ---
        sigma_prop, log_jac_sigma = propose_sigma(sigma, steps["sigma"], rng)
        prop_lp = log_posterior(mu, sigma_prop, nu, data, hyperparams)
        log_accept_ratio = (prop_lp - current_lp) + log_jac_sigma
        accepted_sigma = np.log(rng.uniform()) < log_accept_ratio
        if accepted_sigma:
            sigma = sigma_prop
            current_lp = prop_lp
            window_accepts["sigma"] += 1
            if is_post_burnin:
                post_burnin_accepts["sigma"] += 1

        # --- nu block (shifted-log random walk, with Jacobian) ---
        nu_prop, log_jac_nu = propose_nu(nu, nu_shift, steps["nu"], rng)
        prop_lp = log_posterior(mu, sigma, nu_prop, data, hyperparams)
        log_accept_ratio = (prop_lp - current_lp) + log_jac_nu
        accepted_nu = np.log(rng.uniform()) < log_accept_ratio
        if accepted_nu:
            nu = nu_prop
            current_lp = prop_lp
            window_accepts["nu"] += 1
            if is_post_burnin:
                post_burnin_accepts["nu"] += 1

        mu_draws[it] = mu
        sigma_draws[it] = sigma
        nu_draws[it] = nu

        if is_post_burnin:
            n_post_burnin += 1

        sweep_number = it + 1
        if not is_post_burnin and sweep_number % tune_every == 0:
            for block in ("mu", "sigma", "nu"):
                rate = window_accepts[block] / tune_every
                if rate > target_accept:
                    steps[block] *= np.exp(0.1)
                else:
                    steps[block] *= np.exp(-0.1)
                window_accepts[block] = 0

    accept_rate = {
        block: post_burnin_accepts[block] / n_post_burnin
        for block in ("mu", "sigma", "nu")
    }

    return {
        "mu": mu_draws,
        "sigma": sigma_draws,
        "nu": nu_draws,
        "accept_rate": accept_rate,
        "final_steps": steps,
    }


def run_multiple_chains(data: np.ndarray, hyperparams: dict, inits: list[dict],
                         n_iter: int, n_burnin: int, init_steps: dict,
                         seeds: list[int], tune_every: int = 50,
                         target_accept: float = 0.44) -> list[dict]:
    """
    Run several independent Metropolis-within-Gibbs chains, one per
    (init, seed) pair, each with its own numpy Generator.

    Parameters:
    - data: observed returns
    - hyperparams: dict with keys mu0, tau0, s0, nu_shift, nu_rate
    - inits: list of dicts, each with keys mu, sigma, nu (starting values)
    - n_iter: total number of sweeps to run per chain (including burn-in)
    - n_burnin: number of initial sweeps per chain treated as burn-in
    - init_steps: dict with keys mu, sigma, nu (initial random-walk step sizes)
    - seeds: list of ints, one per chain, used to seed each chain's rng
    - tune_every: adapt step sizes every this many sweeps during burn-in
    - target_accept: target acceptance rate for step-size adaptation

    Returns:
    - list of run_chain-shaped result dicts, one per (init, seed) pair,
      in the same order as inits/seeds.
    """
    if len(inits) != len(seeds):
        raise ValueError(
            f"inits and seeds must have the same length, got "
            f"{len(inits)} inits and {len(seeds)} seeds"
        )

    results = []
    for init, seed in zip(inits, seeds):
        rng = np.random.default_rng(seed)
        result = run_chain(data, hyperparams, init, n_iter, n_burnin,
                            init_steps, rng, tune_every=tune_every,
                            target_accept=target_accept)
        results.append(result)

    return results
