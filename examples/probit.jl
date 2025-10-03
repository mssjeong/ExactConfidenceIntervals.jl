# probit regression
using Distributed
using Pkg
using CSV
using DataFrames

addprocs(Threads.nthreads(); exeflags=`--threads=1`)
println("Number of workers: ",nworkers())

@everywhere using ExactCI
@everywhere using Distributions
@everywhere using SpecialFunctions

@everywhere function gen_probit(th,n,x)
    return Float64.((x*th .+ rand(Normal(),n)) .> 0)
end

@everywhere function lik_probit(th,y,x)
    n = length(y)
    xth = x*th

    indp2 = xth .> 38
    indp1 = (xth .> 7) .&& (xth .<= 38)
    indx = (xth .<= 7) .&& (xth .>= -7)
    indn1 = (xth .< -7) .&& (xth .>= -38)
    indn2 = xth .< -38

    xthp2 = @views xth[indp2]
    xthp1 = @views xth[indp1]
    xthx = @views xth[indx]
    xthn1 = @views xth[indn1]
    xthn2 = @views xth[indn2]

    logcdf = similar(th,n)
    logcdf[indp2] = -erfc.(xthp2./sqrt(2))./2.0
    logcdf[indp1] = -erfc.(xthp1./sqrt(2))./2.0
    logcdf[indx] = log.(erfc.(-xthx./sqrt(2))./2.0)
    logcdf[indn1] = log.(-exp.(-xthn1.^2 ./2.0)./(sqrt(2.0*pi).*xthn1))
    logcdf[indn2] = -xthn2.^2 ./ 2.0 .- log.(-xthn2.*sqrt(2.0*pi))

    log1cdf = similar(th,n)
    log1cdf[indp2] = -xthp2.^2 ./ 2.0 .- log.(xthp2.*sqrt(2.0*pi))
    log1cdf[indp1] = log.(exp.(-xthp1.^2 ./2.0)./(sqrt(2.0*pi).*xthp1))
    log1cdf[indx] = log.(erfc.(xthx./sqrt(2))./2.0)
    log1cdf[indn1] = -erfc.(-xthn1./sqrt(2))./2.0
    log1cdf[indn2] = -erfc.(-xthn2./sqrt(2))./2.0

    return sum(abs.(y).*logcdf .+ abs.(1.0 .- y).*log1cdf)
end

df = CSV.read("probit.csv", DataFrame)
y = df[:,"y"]
x = Array(df[:,["x0","x1","x2"]])

th, ci = exactci(y, x, generator=gen_probit, likelihood=lik_probit)
