# Cox proportional hazards model
using Distributed
using CSV
using DataFrames

addprocs(Threads.nthreads(); exeflags=`--threads=1`)
println("Number of workers: ",nworkers())

@everywhere using ExactConfidenceIntervals
@everywhere using Distributions

@everywhere function gen_hazard(th,n,x)
    return min.(floatmax(Float64),rand.(Exponential.(min.(10.0^307,max.(floatmin(Float64),exp.(-x*th))))))
end

@everywhere function lik_hazard(th,y,x)
    n = size(y,1)
    xth = x*th

    lsth = Vector{Float64}(undef,n)
    for i = 1:n
        sth = @views xth[findall(x->x>=y[i],y)]
        msth = maximum(sth)
        lsth[i] = msth + log(sum(exp.(sth .- msth)))
    end

    return sum(xth .- lsth)
end 

df = CSV.read("hazard.csv", DataFrame)
y = df[:,"y"]
x = Array(df[:,["x1","x2","x3"]])

th, ci = exactci(y, x, generator=gen_hazard, likelihood=lik_hazard)

