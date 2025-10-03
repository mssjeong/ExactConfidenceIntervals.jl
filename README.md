# ExactCI.jl

A Julia package that calculates the exact confidence intervals of maximum likelihood estimators. An implementation of the method proposed in Jeong (2023).

## Installation

```
pkg> add ExactCI
```

or

```
julia> using Pkg
julia> Pkg.add("ExactCI")
```

## Examples

See `examples` folder.

## List of keyword arguments

### Required

- `generator`: Random sample generator of the model. Define the function such that `y::Array = function_name(th::Vector, n::Int)` for models without exogenous variables, and `y::Array = function_name(th::Vector, n::Int, x::Array)` for models with exogenous variables.
- `likelihood`: Likelihood function of the model. Define the function such that `z::Float = function_name(th::Vector, y::Array)` for models without exogenous variables, and `z::Float = function_name(th::Vector, y::Array, x::Array)` for models with exogenous variables. The output variable `z` represents the sum of the log-likelihoods or the product of the likelihoods.

When defining the generator and likelihood functions, the input variable types must not be specified because they are assigned automatically according to the `format` keyword argument.

### Optional

- `alpha`: Significance level of the confidence intervals. Defaults to 0.05.
- `appx_order`: Order of the Taylor series. Defaults to 2.
- `bounded`: Vector of indices of the parameters with a constraint $-1\leq\theta_i\leq 1$.
- `format`: Number format for the parameter estimation. Available values are `:BigFloat` and `:Float64`. Defaults to `:Float64`.
- `mle_init`: Vector of initial values of the model parameters for the maximum likelihood estimation. Defaults to a vector of 0.5s.
- `mle_max_iter`: Maximum iteration number for the maximum likelihood estimation of model parameters.
- `num_par`: Number of parameters of the model. This only needs to be specified when the automatic counting fails.
- `num_sim`: Number of simulated random samples used when calculating probabilities under $\mathbb{P}^{\theta_i,\theta_{-i}}$.
- `opt_max_iter`: Maximum iteration number for the numerical optimization when computing confidence intervals.
- `positive`: Vector of indices of the parameters with a constraint $\theta_i\geq 0$.
- `prob_tol`: Tolerance level for the numerical solution. Defaults to 0.005, roughly indicating that the maximum bias in the coverage probability of the confidence interval is 0.5%p.
- `robust`: Degree of robustness check. Defaults to 3.
- `seed`: Seed for random number generation. Defaults to `Xoshiro(0)`.

## Reference

Jeong M (2023). “A numerical method to obtain exact confidence intervals for likelihoodbased parameter estimators.” Journal of Statistical Planning and Inference, 226, 20–29. doi:10.1016/j.jspi.2022.12.006.
