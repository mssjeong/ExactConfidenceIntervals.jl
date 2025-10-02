# ExactCI.jl

A Julia package that calculates the exact confidence intervals of maximum likelihood estimators.

## Installation

```
pkg> add ExactCI
```

or

```
julia> using Pkg
julia> Pkg.add("ExactCI")
```

## List of keyword arguments

### Required

- `generator`: Random sample generator of the model.
- `likelihood`: Likelihood function of the model.

### Optional

- `alpha`: Significance level of the confidence intervals. Defaults to 0.05.
- `appx_order`: Order of the Taylor series. Defaults to 2.
- `bounded`: Vector of indices of the parameters with a constraint $-1\leq\theta_i\leq 1$.
- `format`: Number format for the parameter estimation. Available values are `:BigFloat` and `:Float64`. Defaults to `:Float64`.
- `mle_init`: Vector of initial values of the model parameters for the maximum likelihood estimation. Defaults to a vector of 0.5s.
- `mle_max_iter`: Maximum iteration number for the maximum likelihood estimation of model parameters.
- `num_par`: Number of parameters of the model. This only needs to be specified when the automatic counting fails.
- `num_sim`: Number of simulated random samples used when calculating probabilities under $\mathbb{P}^{\theta_i,\theta_{-i}}$.

## Reference
