# Gumbel distribution
using Distributed
using Pkg
using CSV
using DataFrames

addprocs(Threads.nthreads(); exeflags=`--threads=1`)
println("Number of workers: ",nworkers())

@everywhere using ExactCI
@everywhere using Distributions

@everywhere function gen_gumbel(th,n)
    return @views rand(Gumbel(th[1],max(floatmin(Float64),th[2])),n)
end

@everywhere function lik_gumbel(th,y)
    mu = @views th[1]
    sg = @views max(th[2],floatmin(Float64))
    mys = (mu .- y)./sg
    return sum(-exp.(mys) .+ mys .- log(sg))
end

df = CSV.read("gumbel.csv", DataFrame)
y = df[:,"y"]

th, ci = exactci(y, generator=gen_gumbel, likelihood=lik_gumbel, positive=[2])
