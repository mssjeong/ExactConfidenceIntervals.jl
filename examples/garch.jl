# GARCH(1,1) process
using Distributed
using CSV
using DataFrames

addprocs(Threads.nthreads(); exeflags=`--threads=1`)
println("Number of workers: ",nworkers())

@everywhere using ExactConfidenceIntervals
@everywhere using Distributions

@everywhere function gen_garch(th,n)
    om = @views th[1]
    ap = @views th[2]
    bt = @views th[3]

    y = zeros(n)
    ep = rand(Normal(),n)
    st = 0.0
    for i = 2:n
        st = @views max(om + ap*y[i-1]^2 + bt*st,floatmin(Float64))
        y[i] = @views sqrt(st)*ep[i]
    end

    return y
end

@everywhere function lik_garch(th,y)
    n = length(y)
    om = @views th[1]
    ap = @views th[2]
    bt = @views th[3]

    lik = 0.0
    st = 0.0
	for i = 2:n
        st = @views min(floatmax(Float64),max(floatmin(Float64),om + ap*y[i-1]^2 + bt*st))
        lik -= @views min(floatmax(Float64)/n,(log(st) + y[i]^2/st)/2.0)
    end

    return lik
end

df = CSV.read("garch.csv", DataFrame)
y = df[:,"y"]

th, ci = exactci(y, generator=gen_garch, likelihood=lik_garch, positive=[1,2,3], appx_order=4)

