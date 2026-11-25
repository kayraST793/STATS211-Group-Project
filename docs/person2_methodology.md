# Methodology: Bayesian Student-t Model and Metropolis-within-Gibbs Sampler

## 1. Student-t Likelihood and Model Motivation

We model daily log returns `r_1, ..., r_n` (n = 753) as independent, identically distributed observations from a location-scale Student-t distribution with parameters `(μ, σ, ν)`:

$$f(r_i \mid \mu, \sigma, \nu) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right) \sqrt{\nu\pi} \sigma} \left(1 + \frac{1}{\nu}\left(\frac{r_i - \mu}{\sigma}\right)^2\right)^{-(\nu+1)/2}$$

The log-likelihood over all observations is:

$$\log L(\mu, \sigma, \nu) = n \left[ \log \Gamma\left(\frac{\nu+1}{2}\right) - \log \Gamma\left(\frac{\nu}{2}\right) - \frac{1}{2}\log(\nu\pi) - \log \sigma \right] - \frac{\nu+1}{2} \sum_{i=1}^{n} \log\left(1 + \frac{1}{\nu}\left(\frac{r_i - \mu}{\sigma}\right)^2\right)$$

**Why Student-t?** Exploratory data analysis (Person 1) revealed that GLD daily log returns exhibit heavy tails and negative skewness inconsistent with a normal distribution: the sample skewness is approximately −0.95 (left-skewed) and excess kurtosis is approximately 6.5 (much heavier than normal). The Normal Q-Q plot showed substantial deviations in both tails, with more severe negative extremes than a normal model would predict. The Student-t distribution, whose tails decay as a power law rather than exponentially, is well-suited to model this heavy-tailed behavior. While the standard symmetric Student-t does not capture the negative skewness, it provides a principled starting point for understanding the return distribution's tail behavior through the degrees-of-freedom parameter `ν`.

## 2. Prior Specifications

We specify weakly-informative priors using the EDA summary statistics as anchors. The sample mean is approximately 0.0009, the sample standard deviation is approximately 0.0139. We deliberately choose prior scales that are roughly 4 times wider than the observed data variability to avoid over-regularization.

**Prior for μ (location):**
$$\mu \sim \text{Normal}(\mu_0 = 0, \tau_0 = 0.05)$$

The mean is centered at zero (the theoretical expected return under risk neutrality) and the standard deviation 0.05 is about 3.6 times the sample standard deviation, allowing flexibility for the posterior to be driven primarily by the likelihood.

**Prior for σ (scale):**
$$\sigma \sim \text{HalfNormal}(s_0 = 0.05), \quad \text{i.e.,} \quad p(\sigma) \propto \exp\left(-\frac{\sigma^2}{2s_0^2}\right) \text{ for } \sigma > 0$$

The half-normal prior on the positive real line reflects that scale is inherently non-negative. The hyperparameter `s_0 = 0.05` is again roughly 3.6 times the sample standard deviation, expressing weak information while remaining concentrated enough to avoid exploring pathological scale values.

**Prior for ν (degrees of freedom):**
$$\nu \sim \text{ShiftedExponential}(\text{shift}=2, \text{rate}=0.1), \quad \text{i.e.,} \quad \nu = 2 + \text{Exponential}(0.1)$$

Equivalently, $p(\nu) = 0.1 \cdot \exp(-0.1(\nu - 2))$ for $\nu > 2$. The shift of 2 is mandatory: the Student-t distribution requires $\nu > 2$ for the variance to be finite. The exponential component has mean 10, so the prior mean for `ν` is 12. This prior mass is spread across a wide range: heavy-tailed regimes ($\nu \approx 3$–$6$) down to near-normal behavior ($\nu \geq 30$), reflecting genuine uncertainty about the tail heaviness while keeping computational burden manageable.

## 3. Conditional Posterior Distributions and Non-Conjugacy

Conditioning on the data and fixing two parameters, each full conditional posterior is proportional to the product of the likelihood and its corresponding prior:

$$p(\mu \mid \sigma, \nu, \text{data}) \propto L(\mu, \sigma, \nu) \cdot \text{Normal}_{\text{pdf}}(\mu; \mu_0, \tau_0)$$

$$p(\sigma \mid \mu, \nu, \text{data}) \propto L(\mu, \sigma, \nu) \cdot \text{HalfNormal}_{\text{pdf}}(\sigma; s_0), \quad \sigma > 0$$

$$p(\nu \mid \mu, \sigma, \text{data}) \propto L(\mu, \sigma, \nu) \cdot \text{ShiftedExp}_{\text{pdf}}(\nu; 2, 0.1), \quad \nu > 2$$

**Non-Conjugacy:** None of these conditionals belongs to a standard, closed-form distribution family. The Student-t likelihood term $(1 + \cdot)^{-(\nu+1)/2}$ appearing in `ν`'s conditional is not conjugate to any of our three priors. The hierarchical dependence of this term on all three parameters, combined with the nonlinearity of the log transformation, precludes direct sampling from these posteriors. This non-conjugacy is precisely why we employ Metropolis-within-Gibbs: each conditional is sampled via a Metropolis-Hastings step rather than direct simulation.

## 4. Metropolis-within-Gibbs Algorithm

The sampler alternates between three Metropolis updates in a Gibbs structure. One **sweep** consists of three sequential updates:

**μ update (symmetric random-walk proposal):**
1. Propose: $\mu' = \mu + \varepsilon_\mu$, where $\varepsilon_\mu \sim N(0, \text{step}_\mu^2)$
2. Acceptance probability: 
$$\alpha = \min\left(1, \exp\left(\log p(\mu' \mid \sigma, \nu, \text{data}) - \log p(\mu \mid \sigma, \nu, \text{data})\right)\right)$$
3. Accept $\mu \gets \mu'$ with probability $\alpha$; otherwise retain $\mu$

The random walk is symmetric in `μ`, so the proposal ratio cancels; there is no Jacobian correction.

**σ update (log-scale random walk with Jacobian):**
1. Propose: $\sigma' = \sigma \cdot \exp(\varepsilon_\sigma)$, where $\varepsilon_\sigma \sim N(0, \text{step}_\sigma^2)$. This keeps $\sigma' > 0$.
2. The proposal is symmetric in $\log \sigma$ but not in $\sigma$. The Metropolis-Hastings ratio includes the Jacobian:
$$\alpha = \min\left(1, \exp\left(\log p(\sigma' \mid \mu, \nu, \text{data}) - \log p(\sigma \mid \mu, \nu, \text{data})\right) \cdot \frac{\sigma'}{\sigma}\right)$$
3. Accept $\sigma \gets \sigma'$ with probability $\alpha$; otherwise retain $\sigma$

**ν update (shifted-log-scale random walk with Jacobian):**
1. Propose: $\nu' = 2 + (\nu - 2) \cdot \exp(\varepsilon_\nu)$, where $\varepsilon_\nu \sim N(0, \text{step}_\nu^2)$. This keeps $\nu' > 2$.
2. The proposal is symmetric in $\log(\nu - 2)$ but not in $\nu$. The acceptance ratio includes the Jacobian:
$$\alpha = \min\left(1, \exp\left(\log p(\nu' \mid \mu, \sigma, \text{data}) - \log p(\nu \mid \mu, \sigma, \text{data})\right) \cdot \frac{\nu' - 2}{\nu - 2}\right)$$
3. Accept $\nu \gets \nu'$ with probability $\alpha$; otherwise retain $\nu$

## 5. Adaptive Tuning During Burn-in

The random-walk step sizes are adapted during the burn-in period to achieve an empirical acceptance rate close to the univariate random-walk Metropolis-Hastings optimal rate of 0.44.

**Tuning procedure:**
- During each burn-in sweep, record whether each block's proposal was accepted
- Every 50 sweeps, compute the acceptance rate for each block over that window
- For each block:
  - If acceptance rate > 0.44: multiply $\text{step} \gets \text{step} \cdot \exp(+0.1)$
  - If acceptance rate ≤ 0.44: multiply $\text{step} \gets \text{step} \cdot \exp(-0.1)$
  - Reset the window counter
- Once burn-in ends (after 5000 sweeps), freeze all step sizes at their current values

This adaptive scheme is a form of optimal scaling that ensures efficient mixing without relying on preliminary runs or manual tuning.

## 6. Multi-Chain Setup and Output

To enable convergence diagnostics (Person 3), we run **4 independent chains** with overdispersed starting points and independent random-number seeds.

**Overdispersed initialization:**

The sample mean and sample standard deviation of the data provide starting anchors:

| Chain | μ₀ offset | σ₀ multiplier | ν₀ |
|-------|-----------|---------------|----|
| 1 | sample_mean − 1.0 × sample_std | 0.5 × sample_std | 4.0 |
| 2 | sample_mean − 0.5 × sample_std | 1.0 × sample_std | 8.0 |
| 3 | sample_mean + 0.5 × sample_std | 1.5 × sample_std | 15.0 |
| 4 | sample_mean + 1.0 × sample_std | 2.0 × sample_std | 30.0 |

Each chain is seeded independently (seeds 1, 2, 3, 4) and uses `np.random.default_rng()` from NumPy, ensuring reproducibility and independence.

**Run parameters:**
- Total iterations per chain: 20000
- Burn-in period: 5000 iterations (discarded; $n_{\text{post-burnin}} = 15000$ per chain)
- Initial step sizes: $\text{step}_\mu = 0.005$, $\text{step}_\sigma = 0.3$, $\text{step}_\nu = 0.3$ (log scale)
- Adaptation window: every 50 sweeps during burn-in; frozen thereafter

**Run result on GLD data (753 daily log returns):**

After running all four chains on the actual GLD data, empirical acceptance rates ranged from 0.385 to 0.516 across the three parameter blocks and four chains—close to the 0.44 target, indicating well-tuned proposals. Post-burn-in posterior means converged to approximately:
- $\hat{\mu} \approx 0.00157$
- $\hat{\sigma} \approx 0.0099$
- $\hat{\nu} \approx 4.0$

The posterior mean for `ν` near 4 reflects the heavy-tailed nature of the data, confirming the motivation for the Student-t model.

**Output files:**

As described in Section 7 below, `src/run_mcmc.py` actually runs this 4-chain experiment three times, once per value of the `ν` prior's rate hyperparameter `λ` (`nu_rate`). Each chain writes its draws to a separate CSV file, organized into one subdirectory per `λ`:

- `results/mcmc/lambda_0.10/chain_1.csv` .. `chain_4.csv` — **baseline model** (`nu_rate = 0.1`, the value used everywhere else in this doc and the code's default hyperparameter)
- `results/mcmc/lambda_0.05/chain_1.csv` .. `chain_4.csv` — sensitivity-analysis alternative (lower rate, heavier-tailed `ν` prior)
- `results/mcmc/lambda_0.20/chain_1.csv` .. `chain_4.csv` — sensitivity-analysis alternative (higher rate, lighter-tailed `ν` prior)

A reader who just wants "the model" (not the sensitivity sweep) should use `results/mcmc/lambda_0.10/`.

Each file has columns `mu, sigma, nu, is_burnin` (20000 rows per chain). The `is_burnin` boolean flag allows downstream consumers (Person 3 and 4) to easily exclude the first 5000 burn-in draws and focus on the post-burn-in posteriors.

## 7. Convergence Diagnostics and Prior-Sensitivity Analysis

**Diagnostics module.** `src/diagnostics.py` implements the convergence diagnostics used to validate each of the three experiments above before comparing them:

- **`r_hat`** — the classic Gelman-Rubin potential scale reduction factor, computed across the 4 chains for a given parameter. Values near 1.0 indicate the chains have converged to the same distribution; the conventional threshold for concern is `r_hat > 1.1`.
- **`effective_sample_size`** — effective sample size for a single chain, computed via Geyer's initial monotone sequence method (summing consecutive pairs of lag-`t` autocorrelations and stopping at the first pair whose sum goes negative). This corrects the nominal draw count for autocorrelation in the chain.
- **`trace_plot`** — overlays the per-chain traces for a parameter and saves the figure, for visual inspection of mixing.
- **`summarize_posterior`** — pools the post-burn-in draws across all 4 chains to report mean/sd/95% CI, alongside `r_hat` and `ess` for that parameter. **Note:** the `ess` value it reports is the *average* of `effective_sample_size` across the 4 individual chains, not a sum across chains — a per-chain figure, not a total pooled ESS.

**Diagnostic results (all three λ values).** Across all three experiments (`λ = 0.05, 0.10, 0.20`), `r_hat` for `μ`, `σ`, and `ν` ranged from approximately 0.9998 to 1.0003 — well under the 1.1 threshold, indicating excellent convergence of all 4 chains in every experiment. Per-chain effective sample size (as computed/averaged by `summarize_posterior`, i.e. per chain, not summed across the 4 chains) ranged roughly from 1200 to 3200 draws per chain, out of 15000 post-burn-in draws per chain — consistent with the moderate autocorrelation expected from a random-walk Metropolis sampler.

**Motivation for the sensitivity analysis.** The prior for `ν` (Section 2) has a rate hyperparameter `λ` that was set to `0.1` somewhat arbitrarily. Because the central scientific conclusion of this model is that GLD returns are heavy-tailed (small `ν`), it's important to check that this conclusion is a feature of the data and likelihood, not an artifact of how we happened to set `λ`. `src/run_mcmc.py` therefore reruns the full 4-chain experiment at `λ = 0.05` and `λ = 0.20` — a 4x range around the baseline — and compares the resulting posteriors for `ν`.

**Sensitivity result.** Pooled post-burn-in posterior summary for `ν`, by `λ`:

| λ (nu_rate) | mean(ν) | sd(ν) | 95% CI(ν) |
|---|---|---|---|
| 0.05 | ≈4.036 | ≈0.654 | ≈[2.99, 5.52] |
| 0.10 (baseline) | ≈4.026 | ≈0.652 | ≈[2.98, 5.51] |
| 0.20 | ≈3.965 | ≈0.629 | ≈[2.96, 5.42] |

**Conclusion:** the posterior for `ν` is essentially insensitive to this prior choice across a 4x range of `λ` — all three 95% CIs overlap almost completely. The heavy-tails conclusion (`ν ≈ 4`, well below a near-normal `ν ≥ 30`) is therefore robust to this prior specification, not an artifact of the particular rate chosen.

**Caveat for Person 4 (VaR/ES):** the posterior 95% CI for `ν` ([2.98, 5.51] at baseline) sits close to the prior's hard support boundary at `ν = 2` (required for finite variance) — the committed MCMC draws reach as low as `ν ≈ 2.31`. This is a genuinely heavy-tailed, finite-but-high-kurtosis regime, not a near-normal one. Person 4 should expect tail risk (VaR/ES) under this model to exceed, likely substantially, what a normal-distribution assumption would predict.

---

**References:**

- **Person 1 (EDA):** Summary statistics and Q-Q plot analysis in `src/eda.py`
- **Model implementation:** Likelihood and priors in `src/model.py`; sampler in `src/mcmc.py`; driver (3-experiment λ sweep) in `src/run_mcmc.py`; convergence diagnostics and posterior summaries (R-hat, effective sample size, trace plots) in `src/diagnostics.py`
