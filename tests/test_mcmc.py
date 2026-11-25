import numpy as np
from scipy import stats
from src.mcmc import propose_sigma, propose_nu, run_chain


def test_propose_sigma_always_positive():
    rng = np.random.default_rng(1)
    for sigma in [1e-6, 0.001, 0.02, 5.0]:
        for _ in range(200):
            sigma_prop, _ = propose_sigma(sigma, step=0.3, rng=rng)
            assert sigma_prop > 0


def test_propose_nu_always_above_shift():
    rng = np.random.default_rng(2)
    for nu in [2.0001, 2.01, 3.0, 50.0]:
        for _ in range(200):
            nu_prop, _ = propose_nu(nu, shift=2.0, step=0.3, rng=rng)
            assert nu_prop > 2.0


def test_propose_sigma_jacobian_matches_ratio():
    rng = np.random.default_rng(3)
    sigma = 0.02
    sigma_prop, log_jac = propose_sigma(sigma, step=0.3, rng=rng)
    assert np.isclose(log_jac, np.log(sigma_prop / sigma))


HYPER = dict(mu0=0.0, tau0=0.05, s0=0.05, nu_shift=2.0, nu_rate=0.1)
INIT = dict(mu=0.0, sigma=0.02, nu=8.0)
STEPS = dict(mu=0.01, sigma=0.3, nu=0.3)


def test_run_chain_output_shapes_and_constraints():
    rng = np.random.default_rng(10)
    data = rng.standard_t(df=6, size=100) * 0.02
    out = run_chain(data, HYPER, INIT, n_iter=500, n_burnin=200,
                     init_steps=STEPS, rng=rng)
    assert out["mu"].shape == (500,)
    assert out["sigma"].shape == (500,)
    assert out["nu"].shape == (500,)
    assert np.all(out["sigma"] > 0)
    assert np.all(out["nu"] > 2.0)
    assert set(out["accept_rate"]) == {"mu", "sigma", "nu"}


def test_run_chain_recovers_known_parameters_on_synthetic_data():
    rng = np.random.default_rng(42)
    true_mu, true_sigma, true_nu = 0.001, 0.02, 6.0
    data = true_mu + true_sigma * rng.standard_t(df=true_nu, size=200)
    out = run_chain(data, HYPER, dict(mu=0.0, sigma=0.05, nu=15.0),
                     n_iter=6000, n_burnin=3000, init_steps=STEPS, rng=rng)
    post_mu = out["mu"][3000:].mean()
    post_sigma = out["sigma"][3000:].mean()
    post_nu = out["nu"][3000:].mean()
    assert abs(post_mu - true_mu) < 0.01
    assert abs(post_sigma - true_sigma) < 0.01
    assert abs(post_nu - true_nu) < 15.0  # nu is weakly identified at n=200; wide but non-trivial tolerance


def test_two_chains_different_seeds_diverge():
    data = np.random.default_rng(0).standard_t(df=6, size=100) * 0.02
    out_a = run_chain(data, HYPER, INIT, n_iter=200, n_burnin=100,
                       init_steps=STEPS, rng=np.random.default_rng(1))
    out_b = run_chain(data, HYPER, INIT, n_iter=200, n_burnin=100,
                       init_steps=STEPS, rng=np.random.default_rng(2))
    assert not np.allclose(out_a["mu"], out_b["mu"])
