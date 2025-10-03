# Heckman selection model
using Distributed
using Pkg
using CSV
using DataFrames

addprocs(Threads.nthreads(); exeflags=`--threads=1`)
println("Number of workers: ",nworkers())

@everywhere using ExactCI
@everywhere using Distributions
@everywhere using SpecialFunctions

@everywhere function gen_heckman(th,n,x)
# selection equation: y1* = gm'*x1 + e1
# observation equation: y2* = bt'*x2 + e2
# adjust indices according to the dimensions of the exogenous variables
    sg2 = @views max(floatmin(Float64),th[1])
    rho = @views min(max(th[2],nextfloat(-1.0)),prevfloat(1.0))
    gm = @views th[3:4]
    bt = @views th[5:6]
    x1 = @views x[:,1:2]
    x2 = @views x[:,3:4]

    sgm = [1.0 rho*sqrt(sg2); rho*sqrt(sg2) sg2]

    e = rand(MvNormal(sgm),n)'
    y1s = @views x1*gm .+ e[:,1]
    y2s = @views x2*bt .+ e[:,2]
    y1 = Float64.(y1s .> 0.0)
    y2 = copy(y2s)
    y2[y1s .<= 0.0] .= 0.0

    return [y1 y2]
end

@everywhere function lik_heckman(th,y,x)
# selection equation: y1* = gm'*x1 + e1
# observation equation: y2* = bt'*x2 + e2
# adjust indices according to the dimensions of the exogenous variables
    n = size(y,1)
    sg2 = @views max(th[1],floatmin(Float64))
    rho = @views min(max(th[2],nextfloat(-1.0)),prevfloat(1.0))
    gm = @views th[3:4]
    bt = @views th[5:6]
    y1 = @views y[:,1]
    y2 = @views y[:,2]
    x1 = @views x[:,1:2]
    x2 = @views x[:,3:4]
    zgm = x1*gm
    xbt = x2*bt

    zindp1 = -zgm .>= 7.0
    zindx = (-zgm .>= -7.0) .&& (-zgm .< 7.0)
    zindn1 = (-zgm .< -7.0) .&& (-zgm .>= -38.0)
    zindn2 = -zgm .< -38.0

    zgmp1 = @views zgm[zindp1]
    zgmx = @views zgm[zindx]
    zgmn1 = @views zgm[zindn1]
    zgmn2 = @views zgm[zindn2]

    logcdfz = zeros(n)
    logcdfz[zindp1] = -erfc.(-zgmp1./sqrt(2))./2.0
    logcdfz[zindx] = log.(erfc.(zgmx./sqrt(2))./2.0)
    logcdfz[zindn1] = log.(-exp.(-zgmn1.^2 ./2.0)./(sqrt(2.0*pi).*(-zgmn1)))
    logcdfz[zindn2] = -zgmn2.^2 ./ 2.0 .- log.(zgmn2.*sqrt(2.0*pi))

    zgmxbt = (zgm .+ rho.*(y2 .- xbt)./sqrt(sg2))./sqrt(1.0-rho^2)
    zgmxbt = min.(floatmax(Float64)/n,max.(-floatmax(Float64)/n,zgmxbt))

    zxindp1 = zgmxbt .>= 7.0
    zxindx = (zgmxbt .>= -7.0) .&& (zgmxbt .< 7.0)
    zxindn1 = (zgmxbt .< -7.0) .&& (zgmxbt .>= -38.0)
    zxindn2 = zgmxbt .< -38.0

    zgmxbtp1 = @views zgmxbt[zxindp1]
    zgmxbtx = @views zgmxbt[zxindx]
    zgmxbtn1 = @views zgmxbt[zxindn1]
    zgmxbtn2 = @views zgmxbt[zxindn2]

    logcdfzx = zeros(n)
    logcdfzx[zxindp1] = -erfc.(zgmxbtp1./sqrt(2))./2.0
    logcdfzx[zxindx] = log.(erfc.(-zgmxbtx./sqrt(2))./2.0)
    logcdfzx[zxindn1] = log.(-exp.(-zgmxbtn1.^2 ./2.0)./(sqrt(2.0*pi).*zgmxbtn1))
    logcdfzx[zxindn2] = -zgmxbtn2.^2 ./ 2.0 .- log.(-zgmxbtn2.*sqrt(2.0*pi))

    yxbs = max.(-floatmax(Float64)/n,min.(floatmax(Float64)/n,(y2.-xbt).^2 ./(2.0*sg2)))

    return sum(y1.*(logcdfzx .- log(sqrt(2.0*pi*sg2)) .- yxbs) .+ (1.0 .- y1).*logcdfz)
end

# selection equation: s* = gm0*z0 + gm1*z1 + e1
# observation equation: y* = bt0*x0 + bt1*x1 + e2
df = CSV.read("heckman.csv", DataFrame)
y = Array(df[:,["s","y"]])
x = Array(df[:,["z0","z1","x0","x1"]])

th, ci = exactci(y, x, generator=gen_heckman, likelihood=lik_heckman, positive=[1], bounded=[2])
