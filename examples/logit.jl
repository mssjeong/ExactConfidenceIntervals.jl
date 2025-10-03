# logistic regression
using Distributed
using CSV
using DataFrames

addprocs(Threads.nthreads(); exeflags=`--threads=1`)
println("Number of workers: ",nworkers())

@everywhere using ExactConfidenceIntervals
@everywhere using Distributions

@everywhere function gen_logit(th,n,x)
    return Float64.((x*th .+ rand(Logistic(),n)) .> 0)
end

@everywhere function lik_logit(th,y,x)
    xth = x*th

    indp = xth .> 709
    indx = (xth .<= 709) .&& (xth .>= -10)
    indn = xth .< -10

    logexpx = similar(th,length(y))
    expx = @views exp.(xth[indn])
    logexpx[indp] = @views xth[indp]
    logexpx[indx] = @views log.(1.0 .+ exp.(xth[indx]))
    logexpx[indn] = expx .- expx.^2 ./2.0 .+ expx.^3 ./3.0

    return sum(y.*xth .- logexpx)
end

df = CSV.read("logit.csv", DataFrame)
y = df[:,"y"]
x = Array(df[:,["x0","x1","x2"]])

th, ci = exactci(y, x, generator=gen_logit, likelihood=lik_logit)

