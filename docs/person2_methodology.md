# Methodology: Bayesian Student-t Model and Metropolis-within-Gibbs Sampler

## 1. Student-t Likelihood and Model Motivation

We model daily log returns `r_1, ..., r_n` (n = 753) as independent, identically distributed observations from a location-scale Student-t distribution with parameters `(μ, σ, ν)`:

$$f(r_i \mid \mu, \sigma, \nu) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right) \sqrt{\nu\pi} \sigma} \left(1 + \frac{1}{\nu}\left(\frac{r_i - \mu}{\sigma}\right)^2\right)^{-(\nu+1)/2}$$

The log-likelihood over all observations is:

$$\log L(\mu, \sigma, \nu) = n \left[ \log \Gamma\left(\frac{\nu+1}{2}\right) - \log \Gamma\left(\frac{\nu}{2}\right) - \frac{1}{2}\log(\nu\pi) - \log \sigma \right] - \frac{\nu+1}{2} \sum_{i=1}^{n} \log\left(1 + \frac{1}{\nu}\left(\frac{r_i - \mu}{\sigma}\right)^2\right)$$

**Why Student-t?** Exploratory data analysis (Person 1) revealed that GLD daily log returns exhibit heavy tails and negative skewness inconsistent with a normal distribution: the sample skewness is approximately −0.95 (left-skewed) and excess kurtosis is approximately 6.5 (much heavier than normal). The Normal Q-Q plot showed substantial deviations in both tails, with more severe negative extremes than a normal model would predict. The Student-t distribution, whose tails decay as a power law rather than exponentially, is well-suited to model this heavy-tailed behavior. While the standard symmetric Student-t does not capture the negative skewness, it provides a principled starting point for understanding the return distribution's tail behavior through the degrees-of-freedom parameter `ν`.

## 2. Prior Specifications

We specify weakly-informative priors using the EDA summary statistics as anchors. The sample mean is approximately 0.0009, the sample standard deviation is approximately 0.0139. The prior scales are deliberately set orders of magnitude wider than the observed data variability, so that the posterior is driven essentially entirely by the likelihood.

**Prior for μ (location):**
$$\mu \sim \text{Normal}(\mu_0 = 0, \tau_0 = 10)$$

The mean is centered at zero (the theoretical expected return under risk neutrality). The standard deviation of 10 is roughly 700 times the sample standard deviation — on the scale of daily log returns this is effectively flat, encoding no meaningful prior information about the drift.

**Prior for σ (scale):**
$$\sigma \sim \text{HalfNormal}(s_0 = 10), \quad \text{i.e.,} \quad p(\sigma) \propto \exp\left(-\frac{\sigma^2}{2s_0^2}\right) \text{ for } \sigma > 0$$

The half-normal prior on the positive real line reflects that scale is inherently non-negative. The hyperparameter `s_0 = 10` is likewise vastly wider than the data scale, so the prior contributes essentially no information beyond the positivity constraint.

Figure `figures/mcmc/prior_vs_posterior.png` shows this directly: over the entire range the posterior occupies, each prior density is visually flat, and the posterior is sharply concentrated by comparison. This is the intended behaviour of a vague prior dominated by the likelihood.

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
- Once burn-in ends (after 20,000 sweeps), freeze all step sizes at their current values

This adaptive scheme is a form of optimal scaling that ensures efficient mixing without relying on preliminary runs or manual tuning.

## 6. Chain Setup, Burn-in, Thinning, and Output

We run a **single long chain** per experiment, started from the data's own summary statistics.

**Initialization:**

| Parameter | Starting value |
|-----------|----------------|
| μ₀ | sample_mean (≈ 0.0009) |
| σ₀ | sample_std (≈ 0.0139) |
| ν₀ | 8.0 |

The chain is seeded with `np.random.default_rng(1)`, making every result in this document exactly reproducible.

**Run parameters:**
- Total sweeps: 200,000
- Burn-in: 20,000 sweeps (discarded)
- Thinning: keep every 20th post-burn-in draw
- **Retained posterior sample: (200,000 − 20,000) / 20 = 9,000 draws**
- Initial step sizes: $\text{step}_\mu = 0.005$, $\text{step}_\sigma = 0.3$, $\text{step}_\nu = 0.3$ (log scale)
- Adaptation window: every 50 sweeps during burn-in; frozen thereafter

**Burn-in.** Figure `figures/mcmc/burnin_trace_lambda_0.10.png` plots the opening iterations of the raw chain for all three parameters. The chain settles far faster than the burn-in length: `ν` falls from its start near 8 to its posterior region (≈ 4) within roughly 40 sweeps, and `σ` descends from the sample standard deviation (0.0139) to ≈ 0.0099 by roughly iteration 120. Discarding 20,000 sweeps is therefore a very conservative choice.

**Thinning.** Successive sweeps of a random-walk Metropolis sampler are autocorrelated, so the retained sample keeps only every 20th draw (see Section 7 for the autocorrelation evidence behind this interval). Thinning does not by itself increase the information in a chain — it reduces storage — which is why the total sweep count was raised to 200,000 to compensate.

**Run result on GLD data (753 daily log returns):**

Empirical acceptance rates ranged from 0.417 to 0.510 across the three parameter blocks and the three experiments — close to the 0.44 target, indicating well-tuned proposals. Posterior means for the baseline model:
- $\hat{\mu} \approx 0.00157$
- $\hat{\sigma} \approx 0.00989$
- $\hat{\nu} \approx 4.02$

The posterior mean for `ν` near 4 reflects the heavy-tailed nature of the data, confirming the motivation for the Student-t model.

Two consistency checks worth noting. First, the model-implied standard deviation, $\sigma\sqrt{\nu/(\nu-2)} = 0.01394$, matches the sample standard deviation of 0.01389 to within 0.4% — even though nothing in the fitting procedure targets that quantity directly. Second, the posterior mean for `μ` (0.00157) is *higher* than the arithmetic sample mean (0.00090): the Student-t likelihood downweights the extreme −10.8% outlier that drags the sample mean down. This is robust estimation behaving as intended, and is itself an argument for the Student-t over the normal.

**Output files:**

As described in Section 7 below, `src/run_mcmc.py` runs this single-chain experiment three times, once per value of the `ν` prior's rate hyperparameter `λ` (`nu_rate`), writing to one subdirectory per `λ`:

- `results/mcmc/lambda_0.10/chain_1.csv` — **baseline model** (`nu_rate = 0.1`, the value used everywhere else in this doc and the code's default hyperparameter)
- `results/mcmc/lambda_0.05/chain_1.csv` — sensitivity-analysis alternative (lower rate, heavier-tailed `ν` prior)
- `results/mcmc/lambda_0.20/chain_1.csv` — sensitivity-analysis alternative (higher rate, lighter-tailed `ν` prior)

A reader who just wants "the model" (not the sensitivity sweep) should use `results/mcmc/lambda_0.10/`.

**CSV format — important for Person 3 and Person 4.** Each file has exactly three columns, `mu, sigma, nu`, and 9,000 rows. **Burn-in has already been discarded and the remainder already thinned, so every row is a usable posterior draw — there is no `is_burnin` column to filter on.** Simply read the file and use all of it:

```python
draws = pd.read_csv("results/mcmc/lambda_0.10/chain_1.csv")  # 9,000 ready-to-use draws
```

## 7. Convergence Diagnostics and Prior-Sensitivity Analysis

**Diagnostics module.** `src/diagnostics.py` implements the diagnostics used to validate each of the three experiments above before comparing them:

- **`autocorrelation`** — the sample autocorrelation function (ACF) of a chain at lags 0..`max_lag`, used to assess mixing speed and to justify the thinning interval.
- **`effective_sample_size`** — effective sample size, computed via Geyer's initial monotone sequence method (summing consecutive pairs of lag-`t` autocorrelations and stopping at the first pair whose sum goes negative). This corrects the nominal draw count for residual autocorrelation.
- **`trace_plot`** — plots a parameter's trace and saves the figure, for visual inspection of mixing.
- **`acf_plot`** — bar plot of a parameter's ACF with a ±1.96/√n significance band.
- **`burnin_trace_plot`** — stacked traces of the opening iterations with the burn-in period shaded (the figure referenced in Section 6).
- **`summarize_posterior`** — reports mean/sd/95% CI and `ess` for a parameter.

**Note on R-hat.** The Gelman-Rubin statistic is deliberately *not* reported. R-hat compares between-chain variance to within-chain variance, so it is undefined for the single-chain design used here. Convergence is instead assessed from the burn-in trace (Section 6), the trace plots, the ACF, and the effective sample size.

**Autocorrelation and the thinning interval.** On the raw (unthinned) post-burn-in chain of 180,000 draws, the ACF decays as expected for a random-walk Metropolis sampler, with `μ` mixing fastest and `σ` and `ν` decaying together more slowly:

| Parameter | ACF at lag 20 | First lag inside the 95% band (±0.005) |
|---|---|---|
| μ | ≈ 0.00 | 15 |
| σ | ≈ 0.04 | 29 |
| ν | ≈ 0.05 | 35 |

The thinning interval of 20 does not fully clear the (very tight) significance band for `σ` and `ν`, but the residual correlation at that lag is small enough that the retained draws are nearly independent in practice. The direct evidence is the effective sample size of the retained sample:

| Parameter | ESS | Retained draws | Effectively independent |
|---|---|---|---|
| μ | 9000 | 9000 | 100.0% |
| σ | 8186 | 9000 | 91.0% |
| ν | 7939 | 9000 | 88.2% |

Raising the thinning interval to ≈35 would fully clear the band, at the cost of roughly 40% of the retained draws. See `figures/mcmc/acf_comparison_lambda_0.10.png` for the three decay curves with the thinning interval marked.

**Motivation for the sensitivity analysis.** The prior for `ν` (Section 2) has a rate hyperparameter `λ` that was set to `0.1` somewhat arbitrarily. Because the central scientific conclusion of this model is that GLD returns are heavy-tailed (small `ν`), it's important to check that this conclusion is a feature of the data and likelihood, not an artifact of how we happened to set `λ`. `src/run_mcmc.py` therefore reruns the full experiment at `λ = 0.05` and `λ = 0.20` — a 4x range around the baseline — and compares the resulting posteriors for `ν`.

**Sensitivity result.** Posterior summary for `ν`, by `λ` (9,000 retained draws each):

| λ (nu_rate) | mean(ν) | sd(ν) | 95% CI(ν) |
|---|---|---|---|
| 0.05 | ≈4.032 | ≈0.657 | ≈[2.97, 5.51] |
| 0.10 (baseline) | ≈4.025 | ≈0.651 | ≈[2.98, 5.50] |
| 0.20 | ≈3.976 | ≈0.635 | ≈[2.95, 5.42] |

**Conclusion:** the posterior for `ν` is essentially insensitive to this prior choice across a 4x range of `λ` — all three 95% CIs overlap almost completely. The heavy-tails conclusion (`ν ≈ 4`, well below a near-normal `ν ≥ 30`) is therefore robust to this prior specification, not an artifact of the particular rate chosen.

**Caveat for Person 4 (VaR/ES):** the posterior 95% CI for `ν` ([2.98, 5.50] at baseline) sits close to the prior's hard support boundary at `ν = 2` (required for finite variance) — the committed MCMC draws reach as low as `ν ≈ 2.43`. This is a genuinely heavy-tailed, finite-but-high-kurtosis regime, not a near-normal one. Person 4 should expect tail risk (VaR/ES) under this model to exceed, likely substantially, what a normal-distribution assumption would predict.

---

**References:**

- **Person 1 (EDA):** Summary statistics and Q-Q plot analysis in `src/eda.py`
- **Model implementation:** Likelihood and priors in `src/model.py`; sampler in `src/mcmc.py`; driver (3-experiment λ sweep) in `src/run_mcmc.py`; diagnostics and posterior summaries (autocorrelation, effective sample size, trace/ACF/burn-in plots) in `src/diagnostics.py`

**Figures** (all in `figures/mcmc/`):

| File | Shows |
|---|---|
| `burnin_trace_lambda_0.10.png` | μ/σ/ν traces over the opening iterations, burn-in shaded |
| `trace_nu_lambda_{0.10,0.05,0.20}.png` | ν trace for each λ experiment |
| `acf_{mu,sigma,nu}_lambda_0.10.png` | per-parameter ACF on the raw chain |
| `acf_comparison_lambda_0.10.png` | all three ACF decay curves, thinning interval marked |
| `nu_posterior_by_lambda.png` | ν posterior densities across the three λ values |
| `prior_vs_posterior.png` | prior vs. posterior for each parameter |
