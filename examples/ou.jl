# Ornstein-Uhlenbeck process
using Distributed
using CSV
using DataFrames

addprocs(Threads.nthreads(); exeflags=`--threads=1`)
println("Number of workers: ",nworkers())

@everywhere using ExactConfidenceIntervals
@everywhere using Distributions

@everywhere function gen_ou(th,n)
# dY_t = (a1 + a2*Y_t)*dt + s*dW_t

    # delta = 0.004   # observation frequency of the data

    a1 = @views th[1]
    a2 = @views th[2]
    s = @views th[3]

    y = zeros(n);
    ep = rand(Normal(0,1),n)

    if 2.0*a2*delta > 700
        y[1] = 0.0
        for i = 2:n
           y[i] = @views exp(a2*delta)*(y[i-1] - a1/a2*(exp(-a2*delta)-1.0) + s*sqrt((exp(-2.0*a2*delta)-1.0)/(-2.0*a2))*ep[i])
        end
    elseif abs(2.0*a2*delta) > 2.0
        a = exp(a2*delta)
        b = s*sqrt((1-exp(2.0*a2*delta))/(-2.0*a2))
        y[1] = 0.0
        for i = 2:n
           y[i] = @views y[i-1]*a - a1/a2*(1.0-a) + b*ep[i]
        end
    elseif a2 != 0.0
        a = exp(a2*delta)
        b = s*sqrt(delta+delta^2*a2+(2.0*delta^3*a2^2)/3.0 +
            (delta^4*a2^3)/3.0 + (2.0*delta^5*a2^4)/15.0 + (2.0*delta^6*a2^5)/45.0 +
            (4.0*delta^7*a2^6)/315.0 + (delta^8*a2^7)/315.0 + (2.0*delta^9*a2^8)/2835.0 +
            (2.0*delta^10*a2^9)/14175.0 + (4.0*delta^11*a2^10)/155925.0)
        y[1] = 0.0
        for i = 2:n
            y[i] = @views y[i-1]*a - a1/a2*(1.0-a) + b*ep[i]
        end
    else
        y = [0.0; cumsum(s.*sqrt(delta).*view(ep,2:n),dims=1)]
    end

    return y
end

@everywhere function lik_ou(th,y)
# dY_t = (a1 + a2*Y_t)*dt + s*dW_t

    # delta = 0.004   # observation frequency of the data

    a1 = @views th[1]
    a2 = @views th[2]
    s = @views max(th[3],floatmin(Float64))

    if 2.0*a2*delta > 700
        return @views sum(-log(s) - 2.0*a2*delta/2.0 .- 
            ((sqrt(exp(-2.0*a2*delta)).*y[2:end].-y[1:end-1].+a1*(exp(-a2*delta)-1.0)/a2)./(s*sqrt((exp(-2.0*a2*delta)-1.0)/(-2.0*a2)))).^2 ./2.0)
    elseif abs(2.0*a2*delta) > 2.0
        b = s*sqrt((1.0-exp(2.0*a2*delta))/(-2.0*a2))
        c = (1.0-exp(a2*delta))/a2
        return @views sum(-log(b) .- ((y[2:end].-y[1:end-1].*exp(a2*delta).+a1*c)./b).^2 ./2.0)
    else
        b = s*sqrt(delta+delta^2*a2+(delta^3*a2^2)/1.5 +
            (delta^4*a2^3)/3.0 + (delta^5*a2^4)/7.5 + (delta^6*a2^5)/22.5 +
            (delta^7*a2^6)/78.75 + (delta^8*a2^7)/315.0 + (delta^9*a2^8)/1417.5 +
            (delta^10*a2^9)/7087.5 + (delta^11*a2^10)/38981.25)
        c = -(delta+delta^2*a2/2.0+(delta^3*a2^2)/6.0 +
            (delta^4*a2^3)/24.0 + (delta^5*a2^4)/120.0 + (delta^6*a2^5)/720.0 +
            (delta^7*a2^6)/5040.0 + (delta^8*a2^7)/40320.0 + (delta^9*a2^8)/362880.0 +
            (delta^10*a2^9)/3628800.0 + (delta^11*a2^10)/39916800.0)
        return @views sum(-log(b) .- ((y[2:end].-y[1:end-1].*exp(a2*delta).+a1*c)./b).^2 ./2.0)
    end
end

df = CSV.read("ou.csv", DataFrame)
y = df[:,"y"]

@everywhere const delta = 0.004   # observation frequency of the data

th, ci = exactci(y, generator=gen_ou, likelihood=lik_ou, positive=[3], appx_order=6)

