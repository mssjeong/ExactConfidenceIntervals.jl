# ExactConfidenceIntervals.jl

A Julia package for computing exact confidence intervals of maximum likelihood estimators, implementing the method proposed in [Jeong (2023)](https://doi.org/10.1016/j.jspi.2022.12.006).

## Installation

```julia
pkg> add ExactConfidenceIntervals
```
or
```julia
julia> using Pkg
julia> Pkg.add("ExactConfidenceIntervals")
```

## Basic usage

Models without exogenous variables:
```julia
th, ci = exactci(y, generator=function_name, likelihood=function_name)
```

Models with exogenous variables:
```julia
th, ci = exactci(y, x, generator=function_name, likelihood=function_name)
```

Replace `exactci` with `exactci_left` to obtain left-sided confidence intervals, and with `exactci_right` to obtain right-sided confidence intervals.

## Required keyword arguments

- `generator`: Random sample generator of the model. Define the function such that `y::Array = function_name(theta::Vector, n::Int)` for models without exogenous variables, and `y::Array = function_name(theta::Vector, n::Int, x::Array)` for models with exogenous variables.
- `likelihood`: Likelihood function of the model. Define the function such that `z::Float = function_name(theta::Vector, y::Array)` for models without exogenous variables, and `z::Float = function_name(theta::Vector, y::Array, x::Array)` for models with exogenous variables. When `format=:Float64`, it is recommended to define the likelihood function carefully to prevent numerical overflow or underflow. Utilizing functions from [SpecialFunctions.jl](https://github.com/JuliaMath/SpecialFunctions.jl) or [LogExpFunctions.jl](https://github.com/JuliaStats/LogExpFunctions.jl) would be helpful.

When defining the generator and likelihood functions, do not specify the input variable types because they are assigned automatically according to the `format` keyword argument.

## Examples

See [`examples`](https://github.com/mssjeong/ExactConfidenceIntervals.jl/tree/main/examples) folder.

## Multi-processing support

To utilize multiple workers when computing confidence intervals, load the package on all workers as follows:

```julia
julia> using Distributed
julia> addprocs(Threads.nthreads(); exeflags=`--threads=1`)
julia> println("Number of workers: ", nworkers())
julia> @everywhere using ExactConfidenceIntervals
```

The `generator` and `likelihood` functions must also be defined on all workers. The performance of multi-processing varies depending on the model.

## Optional keyword arguments

- `alpha`: Significance level of the confidence intervals. Defaults to 0.05.
- `appx_order`: Order of the Taylor series to approximate the invariant quantile function. Defaults to 2.
- `bounded`: Vector of indices of the parameters with a constraint $-1\leq\theta_i\leq 1$. Reparameterize the model to impose different range constraints.
- `format`: Number format for the parameter estimation. Available values are `:BigFloat` and `:Float64`. Defaults to `:Float64`.
- `mle_init`: Vector of initial values of the model parameters for the maximum likelihood estimation. Defaults to a vector of 0.5s.
- `mle_max_iter`: Maximum iteration number for the maximum likelihood estimation of model parameters.
- `num_par`: Number of parameters of the model. Only needs to be specified when the automatic counting fails.
- `num_sim`: Number of random samples used when calculating simulated probabilities.
- `opt_max_iter`: Maximum iteration number for the numerical optimization when computing confidence intervals.
- `positive`: Vector of indices of the parameters with a constraint $\theta_i\geq 0$. Reparameterize the model to impose different inequality constraints.
- `prob_tol`: Tolerance level for the numerical solution. Defaults to 0.005, roughly indicating that the maximum bias in the coverage probability of the confidence interval is 0.5%p.
- `robust`: Degree of robustness check. Defaults to 3.
- `seed`: Seed for random number generation. Defaults to `Xoshiro(0)`.

## Reference

Jeong M (2023). “A numerical method to obtain exact confidence intervals for likelihood-based parameter estimators.” Journal of Statistical Planning and Inference, 226, 20–29. [doi:10.1016/j.jspi.2022.12.006](https://doi.org/10.1016/j.jspi.2022.12.006).
