# Cauchy distribution
using Distributed
using CSV
using DataFrames

addprocs(Threads.nthreads(); exeflags=`--threads=1`)
println("Number of workers: ",nworkers())

@everywhere using ExactConfidenceIntervals
@everywhere using Distributions

@everywhere function gen_cauchy(th,n)
    return @views rand(Cauchy(th[1],max(floatmin(Float64),th[2])),n)
end

@everywhere function lik_cauchy(th,y)
    s = @views th[2]
    yms = @views min.(floatmax(Float64),(y .- th[1]).^2 ./s)
    return -sum(log.(s .+ yms))
end

df = CSV.read("cauchy.csv", DataFrame)
y = df[:,"y"]

th, ci = exactci(y, generator=gen_cauchy, likelihood=lik_cauchy, positive=[2])

