# Poisson regression
using Distributed
using CSV
using DataFrames

addprocs(Threads.nthreads(); exeflags=`--threads=1`)
println("Number of workers: ",nworkers())

@everywhere using ExactConfidenceIntervals
@everywhere using Distributions

@everywhere function gen_poisson(th,n,x)
    mu = max.(floatmin(Float64),min.(10.0^31,exp.(x*th)))
    y = zeros(n)
    for i = 1:n
        if mu[i] > 10.0^18
            y[i] = @views round(rand(Normal(mu[i],sqrt(mu[i]))))
        else
            y[i] = @views rand(Poisson(mu[i]))
        end
    end
    return y
end

@everywhere function symlog(x)
    return sign(x)*log(abs(x) + 1.0)
end

@everywhere function lik_poisson(th,y,x)
    xth = x*th
    mxth = maximum(xth)
    if mxth > 709
        return -mxth - log(sum(exp.(xth .- mxth) .- (y.*xth).*exp(-mxth)) +
            exp(-mxth))
    else
        return symlog(sum(y.*xth .- exp.(xth)))
    end
end

df = CSV.read("poisson.csv", DataFrame)
y = df[:,"y"]
x = Array(df[:,["x0","x1","x2"]])

th, ci = exactci(y, x, generator=gen_poisson, likelihood=lik_poisson)

