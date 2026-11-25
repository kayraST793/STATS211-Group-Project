# Person 2 — Bayesian Student-t Model & MCMC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and document the Bayesian Student-t model of GLD daily log returns: likelihood + priors, conditional-posterior derivation, a tuned Metropolis-within-Gibbs sampler, and multiple independent posterior-draw chains, ready for Person 3 (diagnostics) and Person 4 (VaR/ES) to consume.

**Architecture:** `src/model.py` holds pure math (log-likelihood, log-priors, log-posterior) with no I/O or randomness, so it can be unit-tested against closed-form references (`scipy.stats`). `src/mcmc.py` holds the sampler mechanics (proposal steps, adaptive tuning, single-chain and multi-chain runners) and depends only on `model.py`. `src/run_mcmc.py` is the thin driver script that wires data loading + hyperparameters + output persistence together — this is the file a teammate runs. `docs/person2_methodology.md` documents the math and algorithm in prose for the report.

**Tech Stack:** Python, numpy, scipy (`scipy.special.gammaln`, `scipy.stats` for test oracles), pandas (I/O only), pytest.
Use `/opt/anaconda3/bin/python3` for all commands below — the system `python3` (3.13) has no numpy installed; `/opt/anaconda3/bin/python3` (3.12, via anaconda) has numpy/scipy/pandas/matplotlib already present and is what Person 1's scripts (`src/data_prep.py`, `src/eda.py`) already rely on implicitly.

**Spec:** No separate spec doc exists yet for this sub-project. The model specification below (derived from the user's task list, their prior choices, and the EDA numbers already in the repo) is the spec; it is embedded here rather than in a separate file because this is a single, self-contained sub-project plan.

**Data:** `data/GLD_returns.csv` (column `Log_Return`, indexed by `Date`), produced by Person 1's `src/data_prep.py`. Confirmed present: n=753 daily log returns, mean≈0.0009, std≈0.0139, skewness≈-0.95, excess kurtosis≈6.5 — heavy-tailed and left-skewed, consistent with needing a Student-t model.

## Model Specification

**Likelihood.** Returns `r_1..r_n` are i.i.d. location-scale Student-t given `(μ, σ, ν)`:

```
f(r_i | μ, σ, ν) = Γ((ν+1)/2) / (Γ(ν/2) · sqrt(νπ) · σ) · (1 + (1/ν)·((r_i-μ)/σ)²)^(-(ν+1)/2)
```

```
log L(μ,σ,ν) = n·[lgamma((ν+1)/2) - lgamma(ν/2) - 0.5·log(νπ) - log σ]
               - ((ν+1)/2) · Σ_i log(1 + (1/ν)·((r_i-μ)/σ)²)
```

**Priors** (the three the user already chose), with hyperparameters set weakly-informative relative to the EDA numbers above (sample std 0.0139 ⟹ prior scales of 0.05 are deliberately ~4x wider; `ν` prior mean 12 covers everything from heavy-tailed, ν≈3-6, to near-normal, ν≥30):
- `μ ~ Normal(mu0=0, tau0=0.05)`
- `σ ~ HalfNormal(s0=0.05)`, i.e. `p(σ) ∝ exp(-σ²/(2·s0²))` for `σ>0`, else 0
- `ν ~ ShiftedExponential(shift=2, rate=0.1)`, i.e. `ν = 2 + Exponential(rate=0.1)`, `p(ν) = rate·exp(-rate·(ν-2))` for `ν>2`, else 0 (shift=2 is required for the Student-t distribution to have finite variance)

**Conditional posteriors.** Each full conditional is proportional to `L(μ,σ,ν) · prior` holding the other two parameters fixed:

```
p(μ | σ,ν,data) ∝ L(μ,σ,ν) · Normal_pdf(μ; mu0, tau0)
p(σ | μ,ν,data) ∝ L(μ,σ,ν) · HalfNormal_pdf(σ; s0),            σ>0
p(ν | μ,σ,data) ∝ L(μ,σ,ν) · ShiftedExp_pdf(ν; shift=2, rate),  ν>2
```

None of these is a standard closed-form distribution (the Student-t likelihood's `(1+...)^(-(ν+1)/2)` term is not conjugate to any of the three priors) — this is exactly why each conditional is sampled with a Metropolis step (within an outer Gibbs sweep over the three blocks) rather than direct simulation.

**Sampler (Metropolis-within-Gibbs).** One sweep = one Metropolis update of each of μ, σ, ν in turn, each conditioning on the other two's current values:
- `μ`: symmetric random-walk proposal `μ' = μ + N(0, step_mu²)`.
- `σ`: random walk on the log scale, `σ' = σ·exp(ε)`, `ε ~ N(0, step_log_sigma²)` (keeps `σ'>0`). This proposal is symmetric in `log σ` but not in `σ`, so the Metropolis-Hastings acceptance ratio needs the Jacobian correction `σ'/σ`.
- `ν`: random walk on the shifted log scale, `ν' = 2 + (ν-2)·exp(ε)`, `ε ~ N(0, step_log_nu²)` (keeps `ν'>2`), with Jacobian correction `(ν'-2)/(ν-2)`.
- Each block's acceptance probability is `min(1, exp(log_posterior(proposed) - log_posterior(current)) · jacobian)`.

**Tuning.** During burn-in only, every 50 sweeps compute each block's empirical acceptance rate over that window and rescale its step size: multiply the log-step by `+0.1` if the rate is above the 0.44 target (univariate RW-MH optimal), by `-0.1` if below. Freeze all step sizes once burn-in ends.

**Multiple chains.** Run ≥4 chains, each with its own numpy `default_rng` seed and an overdispersed starting point (e.g. `μ0` spread around the sample mean, `σ0` at {0.5, 1, 1.5, 2}× sample std, `ν0` at {4, 8, 15, 30}), so that later convergence diagnostics (Person 3) are meaningful.

## Global Constraints

- Priors are exactly: Normal(μ), Half-Normal(σ), Shifted-Exponential(ν) — do not substitute a plain Exponential or Gamma for ν.
- `ν` must never be evaluated or proposed at or below 2 (undefined variance / prior support boundary) — every proposal and initial value must respect `ν>2` and `σ>0` exactly, not just "almost always".
- Use `scipy.special.gammaln` for log-Gamma terms (not `math.lgamma` elementwise) so the likelihood vectorizes over all 753 returns in one call.
- All sampler randomness must go through an explicit `numpy.random.Generator` passed in by the caller (no bare `np.random.seed`/global state) so multi-chain runs are reproducible per-chain.
- Run all python/pytest commands with `/opt/anaconda3/bin/python3` per the Tech Stack note above.

## Review Focus

- **ν proposed ≤ 2 or σ proposed ≤ 0**: the shifted-log / log transforms must make this structurally impossible, not just unlikely — a test must try many proposals from an edge-case current state (e.g. `ν` near 2, tiny `σ`) and assert the constraint always holds.
- **`ν` large (near-normal tail) still produces finite, non-NaN log-likelihood**: the `(1 + x²/ν)` term and `lgamma(ν/2)` must not overflow/underflow for `ν` in the thousands; a test should check `log_likelihood` returns a finite value at `ν=1e4`.
- **A single data point with `(r_i-μ)/σ` extremely large** (an outlier relative to current `σ`) must not produce `NaN`/`inf` in the log-likelihood or crash the sampler — test the likelihood at a deliberately extreme residual.
- **Sampler recovers known parameters from synthetic data**: with a fixed seed, simulate ~200 Student-t draws from known `(μ,σ,ν)` and check the post-burn-in posterior mean of each parameter lands within a generous, pre-committed tolerance — this is the one test that would catch a sign error or wrong acceptance ratio that unit tests on isolated functions would miss.
- **Independent chains don't silently collapse to the same RNG stream**: a test must assert that two chains started from different seeds produce different sample paths (catches an accidentally-shared or reseeded RNG).

---

## Task 1: Model math (likelihood, priors, posterior)

**Files:**
- Create: `src/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces:
  - `log_likelihood(mu: float, sigma: float, nu: float, data: np.ndarray) -> float`
  - `log_prior_mu(mu: float, mu0: float, tau0: float) -> float`
  - `log_prior_sigma(sigma: float, s0: float) -> float` (returns `-inf` for `sigma<=0`)
  - `log_prior_nu(nu: float, shift: float, rate: float) -> float` (returns `-inf` for `nu<=shift`)
  - `log_posterior(mu: float, sigma: float, nu: float, data: np.ndarray, hyperparams: dict) -> float`, where `hyperparams` has keys `mu0, tau0, s0, nu_shift, nu_rate`
- Consumes: nothing (first task)

- [ ] **Step 1: Write failing tests for `log_likelihood`**

```python
import numpy as np
from scipy import stats
from src.model import log_likelihood

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
```

- [ ] **Step 2: Run tests, verify they fail** (module doesn't exist yet)

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_model.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError` or `ImportError` for `src.model`

- [ ] **Step 3: Implement `log_likelihood` in `src/model.py`**

Vectorize over `data` with `scipy.special.gammaln`, using the formula in the Model Specification section above. Add a top-of-file module docstring is not required; no comments needed beyond the formula being self-evident from variable names.

- [ ] **Step 4: Run tests, verify they pass**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_model.py -v`
Expected: PASS (3/3)

- [ ] **Step 5: Write failing tests for the priors and `log_posterior`**

```python
from src.model import log_prior_mu, log_prior_sigma, log_prior_nu, log_posterior

def test_log_prior_mu_matches_normal_logpdf():
    from scipy.stats import norm
    assert np.isclose(log_prior_mu(0.02, 0.0, 0.05), norm.logpdf(0.02, 0.0, 0.05))

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
```

- [ ] **Step 6: Run tests, verify they fail**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_model.py -v`
Expected: remaining 5 tests FAIL/ERROR (functions not defined)

- [ ] **Step 7: Implement `log_prior_mu`, `log_prior_sigma`, `log_prior_nu`, `log_posterior` in `src/model.py`**

Use `scipy.stats.norm.logpdf` for `log_prior_mu`. For `log_prior_sigma`/`log_prior_nu`, write the half-normal/shifted-exponential log-densities directly from the Model Specification formulas (guard the out-of-support case with an explicit `if`/`-np.inf` return, evaluated before calling `log_likelihood` in `log_posterior` so an out-of-support proposal short-circuits without evaluating the likelihood).

- [ ] **Step 8: Run tests, verify they pass**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_model.py -v`
Expected: PASS (8/8)

- [ ] **Step 9: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: add Student-t log-likelihood, priors, and log-posterior"
```

---

## Task 2: Metropolis-within-Gibbs single-chain sampler

**Files:**
- Create: `src/mcmc.py`
- Test: `tests/test_mcmc.py`

**Interfaces:**
- Consumes: `log_posterior(mu, sigma, nu, data, hyperparams)` from Task 1 (`src/model.py`)
- Produces:
  - `propose_mu(mu, step, rng) -> float`
  - `propose_sigma(sigma, step, rng) -> tuple[float, float]` returns `(sigma_proposed, log_jacobian)`
  - `propose_nu(nu, shift, step, rng) -> tuple[float, float]` returns `(nu_proposed, log_jacobian)`
  - `run_chain(data: np.ndarray, hyperparams: dict, init: dict, n_iter: int, n_burnin: int, init_steps: dict, rng: np.random.Generator, tune_every: int = 50, target_accept: float = 0.44) -> dict` returning `{"mu": np.ndarray[n_iter], "sigma": np.ndarray[n_iter], "nu": np.ndarray[n_iter], "accept_rate": {"mu": float, "sigma": float, "nu": float}, "final_steps": dict}` where the three draw arrays include burn-in draws at the start (caller slices if it wants post-burn-in only) and `accept_rate` is computed over the post-burn-in draws only. `init` has keys `mu, sigma, nu`; `init_steps` has keys `mu, sigma, nu`.

- [ ] **Step 1: Write failing tests for the proposal functions**

```python
import numpy as np
from src.mcmc import propose_sigma, propose_nu

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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_mcmc.py -v`
Expected: FAIL/ERROR, `src.mcmc` doesn't exist

- [ ] **Step 3: Implement `propose_mu`, `propose_sigma`, `propose_nu` in `src/mcmc.py`**

`propose_sigma`/`propose_nu` draw `eps = rng.normal(0, step)` and apply the shifted-log transform from the Model Specification; `log_jacobian` is `log(sigma_prop/sigma)` (resp. `log((nu_prop-shift)/(nu-shift))`) — this is exactly `eps` algebraically, but compute it from the transformed values (not by returning `eps` directly) so the test above is a real check of the implementation rather than a tautology.

- [ ] **Step 4: Run tests, verify they pass**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_mcmc.py -v`
Expected: PASS (3/3)

- [ ] **Step 5: Write failing tests for `run_chain`**

```python
import numpy as np
from scipy import stats
from src.mcmc import run_chain

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
```

- [ ] **Step 6: Run tests, verify they fail**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_mcmc.py -v`
Expected: FAIL/ERROR, `run_chain` not defined

- [ ] **Step 7: Implement `run_chain` in `src/mcmc.py`**

Loop `n_iter` sweeps; each sweep does one Metropolis update per block in order `mu, sigma, nu`, accepting with `min(1, exp(Δlog_posterior + log_jacobian))` (jacobian is 0/omitted for `mu`); every `tune_every` sweeps while `iter < n_burnin`, compute each block's acceptance rate over that window and rescale `log(step)` by `+0.1`/`-0.1` per the Model Specification's tuning rule, resetting the window's accept counter; stop adapting once `iter >= n_burnin`. Store every sweep's `(mu, sigma, nu)` (including burn-in) into the three output arrays; compute `accept_rate` only over indices `>= n_burnin`.

- [ ] **Step 8: Run tests, verify they pass**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_mcmc.py -v`
Expected: PASS (7/7). If `test_run_chain_recovers_known_parameters_on_synthetic_data` is flaky, widen its tolerances rather than deleting it — it is the one test that catches a wrong acceptance-ratio sign.

- [ ] **Step 9: Commit**

```bash
git add src/mcmc.py tests/test_mcmc.py
git commit -m "feat: implement Metropolis-within-Gibbs sampler with adaptive tuning"
```

---

## Task 3: Multi-chain runner and driver script

**Files:**
- Modify: `src/mcmc.py` (add `run_multiple_chains`)
- Create: `src/run_mcmc.py`
- Test: `tests/test_mcmc.py` (add cases)

**Interfaces:**
- Consumes: `run_chain(...)` from Task 2
- Produces: `run_multiple_chains(data, hyperparams, inits: list[dict], n_iter, n_burnin, init_steps, seeds: list[int], tune_every=50, target_accept=0.44) -> list[dict]`, one `run_chain`-shaped dict per `(init, seed)` pair (`len(inits) == len(seeds)` required; raise `ValueError` otherwise)

- [ ] **Step 1: Write failing test for `run_multiple_chains`**

```python
from src.mcmc import run_multiple_chains

def test_run_multiple_chains_returns_one_result_per_seed():
    data = np.random.default_rng(0).standard_t(df=6, size=100) * 0.02
    inits = [dict(mu=0.0, sigma=0.01, nu=4.0), dict(mu=0.0, sigma=0.04, nu=20.0)]
    seeds = [1, 2]
    results = run_multiple_chains(data, HYPER, inits, n_iter=300, n_burnin=100,
                                   init_steps=STEPS, seeds=seeds)
    assert len(results) == 2
    assert not np.allclose(results[0]["mu"], results[1]["mu"])

def test_run_multiple_chains_rejects_mismatched_lengths():
    import pytest
    data = np.array([0.01, -0.01])
    with pytest.raises(ValueError):
        run_multiple_chains(data, HYPER, [dict(mu=0.0, sigma=0.01, nu=4.0)],
                             n_iter=10, n_burnin=5, init_steps=STEPS, seeds=[1, 2])
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_mcmc.py -v`
Expected: FAIL/ERROR, `run_multiple_chains` not defined

- [ ] **Step 3: Implement `run_multiple_chains` in `src/mcmc.py`**

Validate `len(inits) == len(seeds)`, else `raise ValueError`. For each `(init, seed)` pair, build `rng = np.random.default_rng(seed)` and call `run_chain`, collecting results into a list in input order.

- [ ] **Step 4: Run tests, verify they pass**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_mcmc.py -v`
Expected: PASS (all cases)

- [ ] **Step 5: Implement `src/run_mcmc.py` driver script**

Load `data/GLD_returns.csv` the same way `src/eda.py` does. Define `HYPERPARAMS = dict(mu0=0.0, tau0=0.05, s0=0.05, nu_shift=2.0, nu_rate=0.1)` (matches Model Specification). Build 4 overdispersed inits from the sample mean/std per the Model Specification's "Multiple chains" paragraph, with `init_steps = dict(mu=0.005, sigma=0.3, nu=0.3)`, `seeds=[1,2,3,4]`, `n_iter=20000`, `n_burnin=5000`. Call `run_multiple_chains`, then for each chain write `results/mcmc/chain_{i}.csv` (columns `mu,sigma,nu`, all `n_iter` draws including burn-in, plus enough info for Person 3 to drop burn-in — add an `is_burnin` boolean column) via `pandas.DataFrame.to_csv`, creating `results/mcmc/` with `Path.mkdir(parents=True, exist_ok=True)` first. Print each chain's `accept_rate` and final post-burn-in posterior mean of `mu, sigma, nu` to stdout.

- [ ] **Step 6: Run the driver script end-to-end**

Run: `cd /Users/binderyaenkhtuvshin/STAT211project/STATS211-Group-Project && /opt/anaconda3/bin/python3 src/run_mcmc.py`
Expected: exits 0, prints 4 chains' acceptance rates (each should be roughly 0.2-0.6; if wildly off, revisit `init_steps`) and posterior means, and creates `results/mcmc/chain_1.csv` .. `chain_4.csv`

- [ ] **Step 7: Commit**

```bash
git add src/mcmc.py src/run_mcmc.py tests/test_mcmc.py results/mcmc/
git commit -m "feat: add multi-chain runner and MCMC driver script"
```

---

## Task 4: Methodology documentation

**Files:**
- Create: `docs/person2_methodology.md`

**Interfaces:**
- Consumes: the Model Specification section above (this task transcribes/expands it into report prose) and the actual hyperparameter/tuning values used in `src/run_mcmc.py` from Task 3
- Produces: a markdown document for the group report, no code interface

- [ ] **Step 1: Write `docs/person2_methodology.md`**

Include, in order: (1) the Student-t likelihood formula and why Student-t (tie back to Person 1's EDA kurtosis/skewness numbers already in `src/eda.py`'s comments); (2) the three priors with their exact hyperparameters and a one-line justification for each relative to the EDA summary stats (n=753, std≈0.0139); (3) the three conditional posteriors and an explicit statement that none is conjugate/closed-form, hence Metropolis-within-Gibbs; (4) the algorithm: one Gibbs sweep = three Metropolis updates (mu random walk, sigma log random walk with Jacobian, nu shifted-log random walk with Jacobian), with the acceptance-ratio formula; (5) the tuning rule (target 0.44, adapt every 50 sweeps during burn-in only); (6) the multi-chain setup (4 chains, overdispersed starts, independent seeds) and where the output lives (`results/mcmc/chain_*.csv`) for Person 3/4 to consume.

- [ ] **Step 2: Self-check against the task list**

Re-read the original Person 2 bullet list (specify likelihood/priors; derive conditionals; implement MWG; tune sampler; generate posterior draws; run multiple chains; document methodology) and confirm each bullet has a corresponding section in the doc. Fix any gap found.

- [ ] **Step 3: Commit**

```bash
git add docs/person2_methodology.md
git commit -m "docs: document Bayesian Student-t model and MCMC methodology"
```

---

## Task 5: Baseline validation + prior-sensitivity analysis over λ (`nu_rate`)

The baseline model uses `nu_rate=λ=0.1` (Model Specification). This task validates that baseline run with standard MCMC diagnostics, then re-runs the same 4-chain experiment at `λ=0.05` and `λ=0.2` and compares the resulting posterior `p(ν | data, λ)` across the three values — a direct prior-sensitivity check on the one hyperparameter most likely to affect `ν` (and hence tail-heaviness) conclusions.

**Files:**
- Create: `src/diagnostics.py`
- Modify: `src/run_mcmc.py` (parameterize `nu_rate`; run baseline + two sensitivity experiments; call diagnostics; write comparison outputs)
- Test: `tests/test_diagnostics.py`

**Interfaces:**
- Consumes: `run_multiple_chains(...)` result list from Task 3 (list of `run_chain`-shaped dicts with `mu, sigma, nu` arrays and `accept_rate`)
- Produces:
  - `r_hat(chains: list[np.ndarray]) -> float` — classic Gelman-Rubin potential-scale-reduction factor across `len(chains)` chains of equal length (post-burn-in draws only; caller slices burn-in off before calling)
  - `effective_sample_size(chain: np.ndarray) -> float` — ESS via the standard autocorrelation-sum formula, summing lag-`t` autocorrelations until the first negative pair-sum (Geyer's initial monotone sequence), for one chain
  - `trace_plot(chains: list[np.ndarray], param_name: str, out_path: Path) -> None` — one line per chain over iteration index, saved to `out_path`
  - `summarize_posterior(chains: list[np.ndarray]) -> dict` returning `{"mean": float, "sd": float, "ci_2.5": float, "ci_97.5": float, "r_hat": float, "ess": float}` pooling post-burn-in draws across chains for mean/sd/CI, using `r_hat`/`effective_sample_size` as defined above

- [ ] **Step 1: Write failing tests for `r_hat` and `effective_sample_size`**

```python
import numpy as np
from src.diagnostics import r_hat, effective_sample_size

def test_r_hat_near_one_for_identical_well_mixed_chains():
    rng = np.random.default_rng(0)
    chains = [rng.normal(0, 1, size=2000) for _ in range(4)]
    assert r_hat(chains) < 1.05

def test_r_hat_large_for_diverged_chains():
    rng = np.random.default_rng(0)
    chains = [rng.normal(offset, 0.01, size=500) for offset in [-5, -2, 2, 5]]
    assert r_hat(chains) > 1.5

def test_effective_sample_size_near_n_for_iid_draws():
    rng = np.random.default_rng(0)
    chain = rng.normal(0, 1, size=5000)
    ess = effective_sample_size(chain)
    assert 3000 < ess <= 5000  # i.i.d. draws should have ESS close to n

def test_effective_sample_size_small_for_highly_autocorrelated_chain():
    n = 5000
    chain = np.cumsum(np.random.default_rng(0).normal(0, 1, size=n))  # random walk: highly autocorrelated
    assert effective_sample_size(chain) < 200
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_diagnostics.py -v`
Expected: FAIL/ERROR, `src.diagnostics` doesn't exist

- [ ] **Step 3: Implement `r_hat` and `effective_sample_size` in `src/diagnostics.py`**

`r_hat`: compute between-chain variance `B` and within-chain variance `W` across the `m` equal-length chains per the classic Gelman-Rubin formula, return `sqrt(((n-1)/n*W + B/n) / W)`. `effective_sample_size`: compute the chain's autocorrelation at lags `1..n-1` (via `np.correlate` or a direct loop on the centered series), sum consecutive lag pairs and stop at the first pair whose sum is negative (Geyer), return `n / (1 + 2*sum_of_autocorrelations_up_to_cutoff)`.

- [ ] **Step 4: Run tests, verify they pass**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_diagnostics.py -v`
Expected: PASS (4/4)

- [ ] **Step 5: Write failing test for `summarize_posterior` and implement it**

```python
def test_summarize_posterior_keys_and_sane_values():
    rng = np.random.default_rng(0)
    chains = [rng.normal(5, 1, size=1000) for _ in range(4)]
    summary = summarize_posterior(chains)
    assert set(summary) == {"mean", "sd", "ci_2.5", "ci_97.5", "r_hat", "ess"}
    assert 4.5 < summary["mean"] < 5.5
    assert summary["ci_2.5"] < summary["mean"] < summary["ci_97.5"]
```

Implement by pooling all chains' draws with `np.concatenate` for `mean`/`sd`/percentiles, and calling `r_hat(chains)` / averaging `effective_sample_size` across chains for the other two keys.

Run: `/opt/anaconda3/bin/python3 -m pytest tests/test_diagnostics.py -v`
Expected: PASS (5/5)

- [ ] **Step 6: Implement `trace_plot` in `src/diagnostics.py`**

One `matplotlib` figure, one line per chain (`plt.plot(chain, linewidth=0.5, label=f"chain {i}")`), `plt.legend()`, `plt.title(param_name)`, saved via `plt.savefig(out_path, dpi=150)` then `plt.close()`. No test (plotting side-effects aren't unit-tested here); verified visually in Step 8.

- [ ] **Step 7: Extend `src/run_mcmc.py` to run the baseline + two sensitivity experiments**

Wrap the existing Task-3 driver body in a function `run_experiment(nu_rate: float, label: str) -> list[dict]` that builds `hyperparams` with the given `nu_rate`, calls `run_multiple_chains` with the same 4 overdispersed inits/seeds as Task 3, writes `results/mcmc/{label}/chain_{i}.csv`, and returns the chain-result list. In `if __name__ == "__main__":`, call it three times — `run_experiment(0.1, "lambda_0.10")` (baseline), `run_experiment(0.05, "lambda_0.05")`, `run_experiment(0.2, "lambda_0.20")` — and for each: slice burn-in off every chain's `mu, sigma, nu`, call `summarize_posterior` on each parameter's chain list, call `trace_plot` for `nu` into `figures/mcmc/trace_nu_{label}.png`, and print `label`, the three parameters' `summarize_posterior` dicts, and each chain's `accept_rate`.

- [ ] **Step 8: Run the full experiment end-to-end and inspect diagnostics**

Run: `cd /Users/binderyaenkhtuvshin/STAT211project/STATS211-Group-Project && /opt/anaconda3/bin/python3 src/run_mcmc.py`
Expected: exits 0; for all three `label`s, printed `r_hat` for `mu, sigma, nu` is `< 1.1` and `ess` is a few hundred or more (if not, increase `n_iter` or revisit `init_steps` before proceeding — this is the actual acceptance gate for this step, not just "it ran"); `figures/mcmc/trace_nu_*.png` show all 4 chains overlapping with no visible trend; `results/mcmc/lambda_*/chain_*.csv` exist for all three labels.

- [ ] **Step 9: Compare `p(ν | data, λ)` across the three λ values**

Add to `src/run_mcmc.py`: after running all three experiments, overlay the three labels' pooled post-burn-in `ν` draws as density histograms on one `matplotlib` figure (`plt.hist(..., density=True, alpha=0.5, label=label)` per experiment), saved to `figures/mcmc/nu_posterior_by_lambda.png`, and print a small comparison table (one row per `λ`, columns `mean, sd, ci_2.5, ci_97.5` from `summarize_posterior`) so the write-up can state whether the `ν` conclusion is sensitive to this prior choice.

- [ ] **Step 10: Run full test suite and commit**

Run: `/opt/anaconda3/bin/python3 -m pytest tests/ -v`
Expected: all tests across Tasks 1-5 PASS

```bash
git add src/diagnostics.py src/run_mcmc.py tests/test_diagnostics.py results/mcmc/ figures/mcmc/
git commit -m "feat: add MCMC diagnostics and prior-sensitivity sweep over nu_rate"
```

## Self-Review Notes

1. **Spec coverage:** All 7 Person 2 bullets map to a task — likelihood/priors → Task 1 + Task 4; conditional derivation → Model Specification + Task 4; MWG implementation → Task 2; tuning → Task 2 (adaptive step rescaling); posterior draws → Task 3; multiple chains → Task 3 (`run_multiple_chains`); documentation → Task 4. Task 5 is additive (baseline validation + λ-sensitivity), requested directly by the user on top of the original 7 bullets.
2. **Step scan:** each test step gives concrete asserted values/tolerances; each implementation step names the exact function signature and points at the one formula in the Model Specification (or, for Task 5, the named standard diagnostic formula) that determines its body, rather than re-deriving it in the step.
3. **Type consistency:** `log_posterior(mu, sigma, nu, data, hyperparams)` (Task 1) is called identically inside `run_chain` (Task 2); `run_chain`'s return dict keys (`mu, sigma, nu, accept_rate, final_steps`) are the same keys Task 3's driver script and Task 5's `run_experiment`/`summarize_posterior` calls read.
4. **Review Focus coverage:** constraint violations → Task 2 Step 1 tests + Task 1 Step 5 tests; large-ν/extreme-residual numerical stability → Task 1 Step 1 tests; synthetic recovery → Task 2 Step 5; independent-seed divergence → Task 2 Step 5 + Task 3 Step 1; convergence/mixing (R-hat, ESS) → Task 5 Steps 1-4; prior sensitivity → Task 5 Steps 7-9.
5. **Proportion:** five tasks, each a few functions; no task's code body is pre-written beyond the formulas that task's own signature/tests already pin down.
