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

## Reference
