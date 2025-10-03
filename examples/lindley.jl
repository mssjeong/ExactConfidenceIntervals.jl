# Lindley distribution
using Pkg
using CSV
using DataFrames

using ExactCI
using Distributions

function gen_lindley(th,n)
    return min.(floatmax(Float64),rand(Lindley(max(10.0^-307,th[1])),n))
end

function lik_lindley(th,y)
    t = max(floatmin(Float64),th[1])
    return sum(max.(-709,log.(t^2 .*(1.0 .+ y)./(1.0 + t))) .- y.*t)
end

df = CSV.read("lindley.csv", DataFrame)
y = df[:,"y"]

th, ci = exactci(y, generator=gen_lindley, likelihood=lik_lindley, positive=[1])
