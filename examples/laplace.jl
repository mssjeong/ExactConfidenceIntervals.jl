# asymmetric Laplace distribution
using Distributed
using CSV
using DataFrames

addprocs(Threads.nthreads(); exeflags=`--threads=1`)
println("Number of workers: ",nworkers())

@everywhere using ExactConfidenceIntervals
@everywhere using Distributions

@everywhere function gen_laplace(th,n)
    mu = @views th[1]
    ld = @views max(floatmin(Float64),th[2])
    kp = @views max(floatmin(Float64),th[3])
    u = rand(Uniform(-kp,1/kp),n)
    s = sign.(u)
    sks = s.*kp.^s
    return max.(-floatmax(Float64),min.(floatmax(Float64),mu .- 1.0./(ld.*sks).*log.(1.0 .- u.*sks)))
end

@everywhere function lik_laplace(th,y)
    n = length(y)
    mu = @views th[1]
    ld = @views max(floatmin(Float64),th[2])
    kp = @views max(floatmin(Float64),th[3])

    maxv = floatmax(Float64)/n
    
    lik = Vector{Float64}(undef,n)
    indg = findall(x->x>=mu,y)
    indl = findall(x->x<mu,y)
    lg = min(709.8,max(-708.4,log(ld/(kp + 1.0/kp))))
    lik[indg] = @views min.(maxv,max.(-maxv,lg .- ld*kp.*(y[indg] .- mu)))
    lik[indl] = @views min.(maxv,max.(-maxv,lg .+ ld/kp.*(y[indl] .- mu)))
    return sum(lik)
end

df = CSV.read("laplace.csv", DataFrame)
y = df[:,"y"]

th, ci = exactci(y, generator=gen_laplace, likelihood=lik_laplace, positive=[2,3])

