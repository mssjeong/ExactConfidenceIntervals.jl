# gamma distribution
using Distributed
using CSV
using DataFrames

addprocs(Threads.nthreads(); exeflags=`--threads=1`)
println("Number of workers: ",nworkers())

@everywhere using ExactConfidenceIntervals
@everywhere using Distributions
@everywhere using SpecialFunctions

@everywhere function gen_gamma(th,n)
    return @views rand(Gamma(max(floatmin(Float64),th[1]),max(floatmin(Float64),th[2])),n)
end

@everywhere function lik_gamma(th,y)
    k = @views max(th[1],floatmin(Float64))
    t = @views max(th[2],floatmin(Float64))
    return sum((k-1).*log.(y) .- min.(floatmax(Float64)/length(y),y./t) .- k*log(t) .- loggamma(k))
end

df = CSV.read("gamma.csv", DataFrame)
y = df[:,"y"]

th, ci = exactci(y, generator=gen_gamma, likelihood=lik_gamma, positive=[1,2])

