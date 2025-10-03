module ExactConfidenceIntervals
using Random
using Distributions
using LinearAlgebra
using Optim
using LineSearches
using FixedPointAcceleration
using PRIMA
using Distributed
using DistributedArrays
using OffsetArrays
export exactci_left, exactci_right, exactci
 
function count_th(likelihoodc::Function,y::Array{Float64};external::Array{Float64}=[0.0])
    for i = 1:1000
        if i == 1000
            println("[Error] Specify the number of parameters manually. Errors in the likelihood function can also cause this.")
            error()
        end

        th = ones(i).*0.5
        try
            if external==[0.0]
                a = likelihoodc(th,y)
            else
                a = likelihoodc(th,y,external)
            end
        catch e
            continue
        end

        th = ones(i+1).*0.5
        th2 = copy(th)
        th2[i+1] = 0.25
        try
            if external==[0.0]
                a = likelihoodc(th,y)
                a2 = likelihood(th2,y)
                if a == a2
                    return i
                end
            else
                a = likelihoodc(th,y,external)
                a2 = likelihoodc(th2,y,external)
                if a == a2
                    return i
                end
            end
        catch e
            return i
        end
        println("[Error] Specify the number of parameters manually. Errors in the likelihood function can also cause this.")
        error()
    end
end

function numbeta(k::Int64,d::Int64)
    return Int64(factorial(big(k+d-1))/(factorial(big(k-1))*factorial(d)))
end

# index of even product functions
function evenp(grid_model::Int64,nth::Int64)
    aa = Vector{Tuple}(undef,nth::Int64-1)
    for i = 1:(nth-1)
        bb = zeros(grid_model+1)
        bb[1:2:end] .= 1.0
        aa[i] = tuple(bb...)
    end
    cc = vec(prod.(slicearray(collect(Iterators.product(aa...)))))
    return findall(x->x>0,cc)
end

# reorder coordinates
function reorderc(iqfunc_big::Function,coord_med::Array{Float64},evenind::Vector{Int64},scalei::Array{Float64},anmedi::Array{Float64})
    k = size(coord_med,1)
    final_ind = ones(Int64,k)

    fv = zeros(k,k)
    for i = 1:k
        bti = zeros(k)
        bti[i] = 1.0
        fv[:,i] = iqfunc_big(BigFloat.(coord_med),BigFloat.(bti),BigFloat.(scalei),BigFloat.(anmedi))
    end

    noteven = setdiff(collect(1:k),evenind)

    ko = length(noteven)
    gapne = zeros(ko)
    majp = zeros(Bool,k,ko)
    for i = 1:ko
        j = noteven[i]
        gapne[i] = maximum(abs.(fv[:,j]))
        signp = fv[:,j] .>= 0
        signn = fv[:,j] .< 0
        if sum(fv[signp,j]) >= abs(sum(fv[signn,j]))
            majp[:,i] = signp
        else
            majp[:,i] = signn
        end
    end
    gapind = sortperm(gapne,rev=true)

    mask = ones(Int64,k)
    mask[1] = -1
    for i = 1:ko
        j = noteven[gapind[i]]
        dd = abs.(fv[:,j])
        md = findall(x->x==0,majp[:,i])
        dd[md] .= tanh.(abs.(fv[md,j])) .- 1.0
        dd[findall(x->x<0,mask)] .= -1.0
        heind = sortperm(dd,rev=true)
        final_ind[j] = heind[1]
        mask[heind[1]] = -1
    end

    ke = length(evenind)

    gapev = zeros(ke)
    for i = 1:ke
        j = evenind[i]
        gapev[i] = maximum(abs.(fv[:,j]))
    end
    gapind = sortperm(gapev,rev=true)

    for i = 1:ke
        j = evenind[gapind[i]]
        dd = abs.(fv[:,j])
        dd[findall(x->x<0,mask)] .= -1.0
        heind = sortperm(dd,rev=true)
        final_ind[j] = heind[1]
        mask[heind[1]] = -1
    end

    return final_ind
end

# check whether off boundary
function pseudo_sep(empd::Array{Float64},coord::Vector{Float64},gridq::Float64)
    nc = size(empd,1)
    emp_c = empd .- coord'
    chsum = zeros(Bool,nc)
    for i = 1:nc
        chsum[i] = sum(emp_c*emp_c[i,:] .<= 0)/nc .>= (1-gridq)/2 - 0.003
    end
    return !prod(chsum)
end

function pseudo_sepm(empd::Array{Float64},coord::Vector{Float64},gridq::Float64)
    nc = size(empd,1)
    emp_c = empd .- coord'
    chsum = zeros(Bool,nc)
    Threads.@threads for i = 1:nc
        chsum[i] = sum(emp_c*emp_c[i,:] .<= 0)/nc .>= (1-gridq)/2 - 0.003
    end
    return !prod(chsum)
end

# check whether at center
function pseudo_incm(empd::Array{Float64},coord::Vector{Float64},gridq::Float64)
    k = length(coord)
    ch = zeros(Bool,k)
    for i = 1:k
        lb = quantile(empd[:,i],(1-gridq)/2)
        ub = quantile(empd[:,i],gridq + (1-gridq)/2)
        ch[i] = coord[i] >= lb && coord[i] <= ub
    end
    return !prod(ch)
end

# generate boundary coordinates
function bcube(empd::Array{Float64},gridq::Float64)
    k = size(empd,2)
    if k > 1
        bcoor = zeros(2*k,k)
        for i = 1:k
            qi = quantile(empd[:,i],[(1-gridq)/2,1-(1-gridq)/2])
            empdi = empd[:,i]
            empdc = empd[:,filter(x->x!=i,1:k)]
            mli = vec(median(empdc[empdi .<= qi[1],:],dims=1))
            mui = vec(median(empdc[empdi .>= qi[2],:],dims=1))
            insert!(mli,i,qi[1])
            insert!(mui,i,qi[2])
            bcoor[2*i-1,:] = mli
            bcoor[2*i,:] = mui
        end
    else
        bcoor = quantile(empd,[(1-gridq)/2,1-(1-gridq)/2])
    end
    return bcoor
end

function fixcoord(empo::Array{Float64},initm::Vector{Float64},iqfunc_big::Function,iqfunc_big_proc::Function,med_gen::Int64,qt::Float64,empd::Array{Float64},demp::Array{Float64},empdd::Array{Float64},gridq::Float64,seed::Xoshiro,seed_fx::Xoshiro,k::Int64,numc::Int64,jj::Int64,th_hatc::Vector{Float64},qtleft1::Float64,scalei::Array{Float64},anmedi::Array{Float64},nth::Int64,grid_model::Int64,bt_cor::Array{Float64},n::Int64,mb::Int64,format::Symbol,external::Array{Float64},mle_max_iter::Int64,bounded::Vector{Int64},positive::Vector{Int64},genera::Function,likeli::Function)
    seed1 = copy(seed_fx)
    num_bt = numbeta(grid_model+1,nth-1)

    tcord = zeros(numc,k)
    tcord_med = similar(tcord)
    nk = size(demp,1)

    # first coordinate
    seed2 = copy(seed1)
    cmedt, seed1 = tip_match(seed2,1,initm,empd,th_hatc,qtleft1,jj,1-qt,qt,nth,n,mb,med_gen,format,external,mle_max_iter,bounded,positive,genera,likeli)
    tcord[1,:] = cmedt[filter(x->x!=jj,1:nth)]
    seed2 = copy(seed1)
    tcord_med[1,:] = mediantip(seed2,qt,tcord[1,:],th_hatc,qtleft1,jj,n,med_gen,nth,format,external,mle_max_iter,bounded,positive,genera,likeli)

    # the other coordinates
    for i = 2:num_bt
        bti = zeros(num_bt)
        bti[i] = 1.0

        mmc = 4
        mmc = Int64(ceil(pi^((nth-1)/2)*(mmc/2)^(nth-1)/gamma((nth-1)/2+1)*sqrt(det(bt_cor))))   # volume of hyperellipsoid
        mmblock = zeros(mmc,2*nth-1)
        for j = 1:mmc
            ncloc = 1
            nmedc = 1
            @label cgen2
            mmblock[j,1:(nth-1)] = demp[rand(seed,1:nk),:]
            if pseudo_sepm(empdd,mmblock[j,1:(nth-1)],min(0.999,gridq+0.05)) == true
               @goto cgen2
            end

            seed2 = copy(seed1)
            mmblock[j,nth:(2*nth-2)] = mediantip(seed2,qt,mmblock[j,1:(nth-1)],th_hatc,qtleft1,jj,n,med_gen,nth,format,external,mle_max_iter,bounded,positive,genera,likeli)
            if pseudo_sepm(empd,mmblock[j,nth:(2*nth-2)],gridq) == true
                if nmedc > nk*2
                    return nothing, nothing, nothing
                end
                nmedc = nmedc + 1
                @goto cgen2
            end

            if abs(iqfunc_big(BigFloat.(Array(th_hatc')),BigFloat.(bti),BigFloat.(scalei),BigFloat.(anmedi))[1] - 
                    iqfunc_big(BigFloat.(Array(mmblock[j,nth:(2*nth-2)]')),BigFloat.(bti),BigFloat.(scalei),BigFloat.(anmedi))[1]) < 
                    1.0
                if ncloc > nk*2
                    return nothing, nothing, nothing
                end
                ncloc = ncloc + 1
                @goto cgen2
            end
            distv = zeros(i-1)
            for jj = 1:(i-1)
                distv[jj] = sum(((tcord_med[jj,:] .- mmblock[j,nth:(2*nth-2)])./scalei').^2)
            end
            mmblock[j,2*nth-1] = minimum(distv)
        end

        indm = findfirst(x->x==maximum(mmblock[:,2*nth-1]),mmblock[:,2*nth-1])
        tcord[i,:] = mmblock[indm,1:(nth-1)]
        tcord_med[i,:] = mmblock[indm,nth:(2*nth-2)]
    end

    return tcord, tcord_med, seed1
end

function robcoord(iqfunc_big::Function,med_gen::Int64,qt::Float64,empd::Array{Float64},demp::Array{Float64},empdd::Array{Float64},gridq::Float64,seed::Xoshiro,seed_rb::Xoshiro,tcord_fix::Array{Float64},tcord_med_fix::Array{Float64},k::Int64,numc::Int64,jj::Int64,th_hatc::Vector{Float64},qtleft1::Float64,scalei::Array{Float64},anmedi::Array{Float64},nth::Int64,grid_model::Int64,bt_cor::Array{Float64},n::Int64,mb::Int64,format::Symbol,external::Array{Float64},mle_max_iter::Int64,bounded::Vector{Int64},positive::Vector{Int64},genera::Function,likeli::Function)
    num_bt = numbeta(grid_model+1,nth-1)
    nfix = size(tcord_fix,1)

    tcord = zeros(numc,k)
    tcord_med = similar(tcord)
    nk = size(demp,1)

    seed1 = copy(seed_rb)

    for i = 1:numc
        j = rand(seed,2:num_bt)

        bti = zeros(num_bt)
        bti[j] = 1.0

        mmc = 4
        mmc = Int64(ceil(pi^((nth-1)/2)*(mmc/2)^(nth-1)/gamma((nth-1)/2+1)*sqrt(det(bt_cor))))   # volume of hyperellipsoid

        mmblock = zeros(mmc,2*nth-1)
        for j = 1:mmc
            @label cgen2
            mmblock[j,1:(nth-1)] = demp[rand(seed,1:nk),:]
            if pseudo_sepm(empdd,mmblock[j,1:(nth-1)],min(0.999,gridq+0.05)) == true
               @goto cgen2
            end                

            seed2 = copy(seed1)
            mmblock[j,nth:(2*nth-2)] = mediantip(seed2,qt,mmblock[j,1:(nth-1)],th_hatc,qtleft1,jj,n,med_gen,nth,format,external,mle_max_iter,bounded,positive,genera,likeli)
            if pseudo_sepm(empd,mmblock[j,nth:(2*nth-2)],gridq) == true
                @goto cgen2
            end

            if abs(iqfunc_big(BigFloat.(Array(th_hatc')),BigFloat.(bti),BigFloat.(scalei),BigFloat.(anmedi))[1] - 
                    iqfunc_big(BigFloat.(Array(mmblock[j,nth:(2*nth-2)]')),BigFloat.(bti),BigFloat.(scalei),BigFloat.(anmedi))[1]) < 
                    1.0
                @goto cgen2
            end

            if i==1
                tcord_med_all = copy(tcord_med_fix)
            else
                tcord_med_all = [tcord_med_fix;tcord_med[1:i-1,:]]
            end
            distv = zeros(nfix+i-1)
            for jj = 1:(nfix+i-1)
                distv[jj] = sum(((tcord_med_all[jj,:] .- mmblock[j,nth:(2*nth-2)])./scalei').^2)
            end
            mmblock[j,2*nth-1] = minimum(distv)
        end

        indm = findfirst(x->x==maximum(mmblock[:,2*nth-1]),mmblock[:,2*nth-1])
        tcord[i,:] = mmblock[indm,1:(nth-1)]
        tcord_med[i,:] = mmblock[indm,nth:(2*nth-2)]
    end

    return tcord, tcord_med
end

function randcube(seed::Xoshiro,k::Int64,n::Int64)
    return rand(seed,Uniform(-1.0,1.0),n,k)
end

function adjcube(coord::Array{Float64},ithlb::Vector{Float64},ithub::Vector{Float64},bt_cor::Array{Float64})
    tcord = copy(coord)
    nth = length(ithlb)

    ithav = (ithlb .+ ithub)./2
    for jj = 1:nth
        tcord[:,jj] = tcord[:,jj].*(ithub[jj]-ithav[jj]) .+ ithav[jj]
    end
    return tcord
end

function randsphere(seed::Xoshiro,k::Int64,n::Int64)
    u = rand(seed,Float64,n)

    x = randn(seed,n,k)
    mag = sqrt.(sum(x.^2,dims=2))
    x = x./mag.*u.^(1/k)

    return x
end

function adjsphere(coord::Array{Float64},ithlb::Vector{Float64},ithub::Vector{Float64},bt_cor::Array{Float64})
    tcord = copy(coord)
    nth = length(ithlb)

    ithav = (ithlb .+ ithub)./2
    for jj = 1:nth
        tcord[:,jj] = tcord[:,jj].*(ithub[jj]-ithav[jj]) .+ ithav[jj]
    end
    return tcord
end

function mgetcoordz(iqfunc_big::Function,num_bt::Int64,tcord_med::Array{Float64},th_hati::Vector{Float64},scalei::Array{Float64},anmedi::Array{Float64})
    tcord_med2 = tcord_med[2:end,:]
    num_cd = size(tcord_med,1)
    coordz = ones(num_cd)
    for i = 2:num_cd
        bti = zeros(num_bt)
        bti[i] = 1.0

        x2 = (tcord_med2 .- th_hati')./scalei
        dth = sqrt.(sum(x2.^2,dims=2))
        x1 = x2./dth
        f2 = abs.(iqfunc_big(BigFloat.(Array(tcord_med2)),BigFloat.(bti),BigFloat.(scalei),BigFloat.(anmedi)))
        f1 = abs.(iqfunc_big(BigFloat.(Array(x1.*scalei .+ th_hati')),BigFloat.(bti),BigFloat.(scalei),BigFloat.(anmedi)))
        pp = log.(f2./f1)./log.(dth)
        ss = f1.*((dth.-1).^pp .+ (dth.+1).^pp .+ dth.*((dth.+1).^pp .- (dth.-1).^pp))./(2.0.*(pp.+1))   # -1,1
        ss[findall(x->x<1,dth)] .= f2[findall(x->x<1,dth)]
        ss = max.(1.0,ss)
        ss = ss.*max.(1.0,pp).^3
        coordz[i] = maximum(ss)
    end
    return coordz
end

function mgetcoordz_f(iqfunc_big::Function,num_bt::Int64,tcord_med::Array{Float64},th_hati::Vector{Float64},scalei::Array{Float64},anmedi::Array{Float64})
    tcord_med2 = tcord_med[2:end,:]
    coordz = ones(num_bt)
    for i = 2:num_bt
        bti = zeros(num_bt)
        bti[i] = 1.0

        x2 = (tcord_med2 .- th_hati')./scalei
        dth = sqrt.(sum(x2.^2,dims=2))
        x1 = x2./dth
        f2 = abs.(iqfunc_big(BigFloat.(Array(tcord_med2)),BigFloat.(bti),BigFloat.(scalei),BigFloat.(anmedi)))
        f1 = abs.(iqfunc_big(BigFloat.(Array(x1.*scalei .+ th_hati')),BigFloat.(bti),BigFloat.(scalei),BigFloat.(anmedi)))
        pp = log.(f2./f1)./log.(dth)
        ss = f1.*((dth.-1).^pp .+ (dth.+1).^pp .+ dth.*((dth.+1).^pp .- (dth.-1).^pp))./(2.0.*(pp.+1))
        ss[findall(x->x<1,dth)] .= f2[findall(x->x<1,dth)]
        ss = max.(1.0,ss)
        ss = ss.*max.(1.0,pp).^3
        coordz[i] = maximum(ss)
    end
    return coordz
end

function mediantip(seed::Xoshiro,qt::Float64,tcord::Vector{Float64},th_hatc::Vector{Float64},qtleft1::Float64,jj::Int64,n::Int64,mb::Int64,nth::Int64,format::Symbol,external::Array{Float64},mle_max_iter::Int64,bounded::Vector{Int64},positive::Vector{Int64},genera::Function,likeli::Function)
    if format==:BigFloat
        th_cord = BigFloat.(copy(tcord))
        externalv = BigFloat.(copy(external))
    else
        th_cord = Float64.(copy(tcord))
        externalv = Float64.(copy(external))
    end

    dseed = Xoshiro.(rand(seed,0:UInt128(2)^128-1,mb))
    insert!(th_cord,jj,qtleft1)

    workid = workers()
    if external==[0.0]
        mblock = DArray((mb,nth),workid,[Int(min(nworkers(),mb)),1]) do inds
            arr = zeros(inds[1],nth)
            for j in inds[1]
                copy!(Random.default_rng(),dseed[j])
                mcheck = zeros(Int64,1)
                arr[j,:] = mle_est(genera(th_cord,n),th_cord,mle_max_iter,bounded,positive,mcheck,likelihoodm=likeli)
            end
            parent(arr)
        end
        boot = convert(Array,mblock)
        close(mblock)
    else
        mblock = DArray((mb,nth),workid,[Int(min(nworkers(),mb)),1]) do inds
            arr = zeros(inds[1],nth)
            for j in inds[1]
                copy!(Random.default_rng(),dseed[j])
                mcheck = zeros(Int64,1)
                arr[j,:] = mle_estx(genera(th_cord,n,externalv),th_cord,mle_max_iter,bounded,positive,mcheck,external=externalv,likelihoodm=likeli)
            end
            parent(arr)
        end
        boot = convert(Array,mblock)
        close(mblock)
    end
    boot = boot[.!vec(any(isnan,boot;dims=2)),:]
    booti = boot[:,jj]
    bootc = boot[:,filter(x->x!=jj,1:nth)]

    if qt >= 0.5
        bqt = quantile(booti,qt)
        tcord_med = vec(median(bootc[findall(x->x>=bqt,booti),:],dims=1))
    else
        bqt = quantile(booti,qt)
        tcord_med = vec(median(bootc[findall(x->x<=bqt,booti),:],dims=1))
    end

    return tcord_med
end

function trimarray(a::SubArray,p::Int64)
    d = ndims(a)
    if d==2
        ret = a[1:end-p,1:end-p]
    else
        k = size(a)[1]
        ta = selectdim(a,1,1)
        ret = trimarray(ta,p)
        for i=2:k-p
            ta = selectdim(a,1,i)
            ta = trimarray(ta,p)
            ret = cat(ret,ta;dims=d)
        end
    end
    return ret
end

function slicearray(a::Array)
    sz = size(a)
    d = length(sz)
    k = sz[1]
    if d == 1
        ret = reshape(a,length(a),1)
    elseif d == 2
        ret = a[reverse(tril!(trues(k,k)),dims=1)]
    else
        ret = []
        for i = 1:k
            tra = selectdim(a,1,i)
            tra = trimarray(tra,i-1)
            ret = [ret; slicearray(tra)]
        end
    end
    return ret
end

function powerfl(x::SubArray{Float64},p::Float64)
    if p < (1.0 + 10^(-15))
        xr = Vector{Float64}(undef,length(x))
        i = findall(x->x>=0,x)
        j = findall(x->x<0,x)
        xr[i] = @views x[i].^p
        xr[j] = @views -abs.(x[j]).^p
        return xr
    elseif isodd(Int(floor(p-10^(-15))))
        return abs.(x).^p
    else
        xr = Vector{Float64}(undef,length(x))
        i = findall(x->x>=0,x)
        j = findall(x->x<0,x)
        xr[i] = @views x[i].^p
        xr[j] = @views -abs.(x[j]).^p
        return xr
    end
end
function powerfl(x::SubArray{BigFloat},p::Float64)
    if p < (1.0 + 10^(-15))
        xr = Vector{BigFloat}(undef,length(x))
        i = findall(x->x>=0,x)
        j = findall(x->x<0,x)
        xr[i] = @views x[i].^p
        xr[j] = @views -abs.(x[j]).^p
        return xr
    elseif isodd(Int(floor(p-10^(-15))))
        return abs.(x).^p
    else
        xr = Vector{BigFloat}(undef,length(x))
        i = findall(x->x>=0,x)
        j = findall(x->x<0,x)
        xr[i] = @views x[i].^p
        xr[j] = @views -abs.(x[j]).^p
        return xr
    end
end

function decscale!(x::Array{Float64},anmed::Array{Float64},scalevi::Array{Float64})
    x .-= anmed
    x ./= scalevi
end
function decscale!(x::Array{BigFloat},anmed::Array{BigFloat},scalevi::Array{BigFloat})
    x .-= anmed
    x ./= scalevi
end

function fixednorm(x::Vector{Float64})
    return sqrt(mean(x.^2))
end

function fixedsign(iqfunc_big::Function,num_bt::Int64,th_hati::Array{Float64},coord::Array{Float64},scalei::Array{Float64},anmedi::Array{Float64})
    k = copy(num_bt)
    signv = zeros(Int64,k)
    signv[1] = 1.0
    for i = 2:k
        bti = zeros(k)
        bti[i] = 1.0
        if iqfunc_big(BigFloat.(th_hati),BigFloat.(bti),BigFloat.(scalei),BigFloat.(anmedi)) <= iqfunc_big(BigFloat.(Array(coord[i,:]')),BigFloat.(bti),BigFloat.(scalei),BigFloat.(anmedi))
            signv[i] = -1.0
        else
            signv[i] = 1.0
        end
    end
    return signv
end

function genfvs(nth::Int64,grid_model::Int64,grid_sub::Int64)
    aa = Vector{Tuple}(undef,nth::Int64-1)
    for i = 1:(nth::Int64-1)
        bb = Vector{Any}(undef,grid_model::Int64+1)
        bb[1] = :(1.0)
        for j = 1:(grid_model::Int64)
            bb[j+1] = :(x[:,$i].^$j)
        end
        aa[i] = tuple(bb...)
    end
    cc = slicearray(collect(Iterators.product(aa...)))
    kk = length(cc)

    ddv = Vector{Expr}(undef,kk+2)
    ddv[1] = :(decscale!(x,anmed,scalevi))
    ddv[2] = :(ret = fill(bt[1],size(x,1)))
    for i = 2:kk
        eev = Vector{Any}(undef,nth::Int64)
        eev[1] = :(bt[$i])
        for j = 1:(nth::Int64-1)
            eev[j+1] = cc[i][j]
        end
        temp = Expr(:call,:.*,eev...)
        ddv[i+1] = :(@views ret .+= $temp)
    end
    ddv[kk+2] = :(return ret)

    dd = Expr(:block,ddv...)

    return dd
end

function genfun(expr::Expr,args::Array,gs=gensym())
    eval(Expr(:function,Expr(:call,gs,args...),expr))
    return (a::Array{Float64},b::Vector{Float64},c::Array{Float64},d::Array{Float64})->Base.invokelatest(eval(gs),a,b,c,d)
end

function genfun_big(expr::Expr,args::Array,gs=gensym())
    eval(Expr(:function,Expr(:call,gs,args...),expr))
    return (a::Array{BigFloat},b::Vector{BigFloat},c::Array{BigFloat},d::Array{BigFloat})->Base.invokelatest(eval(gs),a,b,c,d)
end

# parameter restriction functions

function trianglewave(x::Float64)
    return 4.0*abs((x + 1.0)/4.0 - floor((x + 1.0)/4.0 + 0.5)) - 1.0
end
function trianglewave(x::BigFloat)
    return 4.0*abs((x + 1.0)/4.0 - floor((x + 1.0)/4.0 + 0.5)) - 1.0
end

function trns_one(th0::Float64,ii::Int64,th_hat::Vector{Float64},scalev::Vector{Float64},bounded::Vector{Int64},positive::Vector{Int64})
    ths = th0/scalev[ii]/5 + th_hat[ii]

    idx = copy(positive::Vector{Int64})
    if ii in idx
        ths = abs(ths)
    end

    idx = copy(bounded)
    if ii in idx
        ths = trianglewave(ths)
    end

    return ths
end

function itrns_one(th0::Float64,ii::Int64,th_hat::Vector{Float64},scalev::Vector{Float64},bounded::Vector{Int64},positive::Vector{Int64})
    ths = copy(th0)
    return (ths - th_hat[ii])*scalev[ii]*5
end

function trns_full_body(th0,bounded,positive)
    th = copy(th0)

    aa = 20.0

    # th>0 restriction
    ths = @views abs.(th[positive]) .- aa/2.0
    jj = findall(x->x<aa/2.0,ths)
    ths[jj] = (view(ths,jj) .+ aa/2.0).^2 ./(aa*2.0)
    th[positive] = ths

    # -1<th<1 restriction
    th[bounded] = sin.(view(th,bounded)./20.0)

    return th
end
function trns_full(th0::Vector{Float64},bounded::Vector{Int64},positive::Vector{Int64})
    return trns_full_body(th0,bounded,positive)
end
function trns_full(th0::Vector{BigFloat},bounded::Vector{Int64},positive::Vector{Int64})
    return trns_full_body(th0,bounded,positive)
end

function itrns_full_body(th0,bounded,positive)
    th = copy(th0)

    aa = 20.0

    # th>0 restriction
    ths = th[positive]
    ii = findall(x->x>=aa/2,ths)
    jj = findall(x->x<aa/2,ths)
    ths[ii] = ths[ii] .+ aa/2.0
    bb = ths[jj]
    bb[bb.<0] .= 0.0
    ths[jj] = sqrt.(bb./(aa*2.0)).*(aa*2.0)
    th[positive] = ths

    # -1<th<1 restriction
    th[bounded] = asin.(th[bounded]).*20.0

    return th
end
function itrns_full(th0::Vector{Float64},bounded::Vector{Int64},positive::Vector{Int64})
    return itrns_full_body(th0,bounded,positive)
end
function itrns_full(th0::Vector{BigFloat},bounded::Vector{Int64},positive::Vector{Int64})
    return itrns_full_body(th0,bounded,positive)
end

# parameter estimation component

function mle_est_body(yy,init::T,mle_max_iter,bounded,positive,mcheck!;likelihoodm=nothing) where {T}
    lm_lk_full(th::T) = likelihoodm(trns_full(th,bounded,positive),yy)
    result = optimize(lm_lk_full,itrns_full(init,bounded,positive),BFGS(linesearch=LineSearches.BackTracking(),
                   alphaguess=LineSearches.InitialStatic(alpha=1)),
                   Optim.Options(iterations=mle_max_iter))
    if result.iterations==mle_max_iter
        mcheck![1] = 1
    end
    return trns_full(result.minimizer,bounded,positive)
end
function mle_est(yy::Array{Float64},init::Vector{Float64},mle_max_iter::Int64,bounded::Vector{Int64},positive::Vector{Int64},mcheck::Vector{Int64};likelihoodm::Function=nothing)
    return mle_est_body(yy,init,mle_max_iter,bounded,positive,mcheck;likelihoodm=likelihoodm)
end
function mle_est(yy::Array{BigFloat},init::Vector{BigFloat},mle_max_iter::Int64,bounded::Vector{Int64},positive::Vector{Int64},mcheck::Vector{Int64};likelihoodm::Function=nothing)
    return mle_est_body(yy,init,mle_max_iter,bounded,positive,mcheck;likelihoodm=likelihoodm)
end

function mle_estx_body(yy,init::T,mle_max_iter,bounded,positive,mcheck!;external=[0.0],likelihoodm=nothing) where {T}
    lm_lk_full(th::T) = likelihoodm(trns_full(th,bounded,positive),yy,external)
    result = optimize(lm_lk_full,itrns_full(init,bounded,positive),BFGS(linesearch=LineSearches.BackTracking(),
                   alphaguess=LineSearches.InitialStatic(alpha=1)),
                   Optim.Options(iterations=mle_max_iter))
    if result.iterations==mle_max_iter
        mcheck![1] = 1
    end
    return trns_full(result.minimizer,bounded,positive)
end
function mle_estx(yy::Array{Float64},init::Vector{Float64},mle_max_iter::Int64,bounded::Vector{Int64},positive::Vector{Int64},mcheck::Vector{Int64};external::Array{Float64}=[0.0],likelihoodm::Function=nothing)
    return mle_estx_body(yy,init,mle_max_iter,bounded,positive,mcheck;external=external,likelihoodm=likelihoodm)
end
function mle_estx(yy::Array{BigFloat},init::Vector{BigFloat},mle_max_iter::Int64,bounded::Vector{Int64},positive::Vector{Int64},mcheck::Vector{Int64};external::Array{BigFloat}=[0.0],likelihoodm::Function=nothing)
    return mle_estx_body(yy,init,mle_max_iter,bounded,positive,mcheck;external=external,likelihoodm=likelihoodm)
end

function qsortil(parv::Array{Float64},i::Int64,p1::Float64,p2::Float64,p3::Float64)
    m = size(parv,1)
    nth = size(parv,2)
    sparv = parv[sortperm(parv[:,i]),:]
    i1 = Int(ceil(p1*m))
    return sparv[i1,i], sparv[Int(ceil(p3*m)),i], median(sparv[i1:Int(ceil(p2*m)),filter(x->x!=i,1:nth)],dims=1)
end

function qsortiu(parv::Array{Float64},i::Int64,p1::Float64,p2::Float64,p3::Float64)
    m = size(parv,1)
    nth = size(parv,2)
    sparv = parv[sortperm(parv[:,i]),:]
    i3 = Int(floor(p3*m + 1))
    return sparv[Int(floor(p1*m + 1)),i], sparv[i3,i], median(sparv[Int(floor(p2*m + 1)):i3,filter(x->x!=i,1:nth)],dims=1)
end

function uqsortil(parv::Vector{Float64},p1::Float64,p3::Float64)
    m = length(parv)
    sparv = parv[sortperm(parv)]
    i1 = Int(ceil(p1*m))
    return sparv[i1], sparv[Int(ceil(p3*m))]
end

function uqsortiu(parv::Vector{Float64},p1::Float64,p3::Float64)
    m = length(parv)
    sparv = parv[sortperm(parv)]
    i3 = Int(floor(p3*m + 1))
    return sparv[Int(floor(p1*m + 1))], sparv[i3]
end

function simprob(bt::Vector{Float64},thq0::Vector{Float64},qt::Float64,ii::Int64,iqfunc::Function,n::Int64,m::Int64,nth::Int64,mle_max_iter::Int64,bounded::Vector{Int64},positive::Vector{Int64},scalev::Vector{Float64},anmed::Vector{Float64};external::Array{Float64}=[0.0],generatort::Function=nothing,likelihoodt::Function=nothing,format::Symbol=:Float64)
    if format==:BigFloat
        thq = BigFloat.(copy(thq0))
        externalv = BigFloat.(copy(external))
    else
        thq = Float64.(copy(thq0))
        externalv = Float64.(copy(external))
    end

    parv = zeros(m::Int64,nth::Int64)
    mmcheck = 0
    if external==[0.0]
        for i = 1:m::Int64
            mcheck! = zeros(Int64,1)
		    parv[i,:] = mle_est(generatort(thq,n),thq,mle_max_iter,bounded::Vector{Int64},positive::Vector{Int64},mcheck!,likelihoodm=likelihoodt)
            if mcheck![1] == 1
                mmcheck += 1
            end
        end
    else
        for i = 1:m::Int64
            mcheck! = zeros(Int64,1)
		    parv[i,:] = mle_estx(generatort(thq,n,externalv),thq,mle_max_iter,bounded::Vector{Int64},positive::Vector{Int64},mcheck!,
				external=externalv,likelihoodm=likelihoodt)
            if mcheck![1] == 1
                mmcheck += 1
            end
        end
    end

    if mmcheck > m - 0.1
        print("\n[Warning] Estimation taking too long. Increasing mle_max_iter or appx_order may help.")
    end

    return sum(view(parv,:,ii) .<= iqfunc(Array(view(parv,:,filter(x->x!=ii,1:nth::Int64))),bt,Array(scalev[filter(x->x!=ii,1:nth::Int64)]'),Array(anmed[filter(x->x!=ii,1:nth::Int64)]')))/m::Int64, parv
end

function simprobm(seed::UInt128,bt::Vector{Float64},thq0::Vector{Float64},qt::Float64,ii::Int64,iqfunc::Function,n::Int64,m::Int64,nth::Int64,mle_max_iter::Int64,bounded::Vector{Int64},positive::Vector{Int64},scalev::Vector{Float64},anmed::Vector{Float64};external::Array{Float64}=[0.0],generatort::Function=nothing,likelihoodt::Function=nothing,format::Symbol=:Float64)
    if format==:BigFloat
        thq = BigFloat.(copy(thq0))
        externalv = BigFloat.(copy(external))
    else
        thq = Float64.(copy(thq0))
        externalv = Float64.(copy(external))
    end
    dseed = Xoshiro.(rand(MersenneTwister(seed),0:UInt128(2)^128-1,m))
    parv = zeros(m,nth)
    workid = workers()
    if external==[0.0]
        mblock = DArray((m,nth),workid,[Int(min(nworkers(),m)),1]) do inds
            arr = zeros(inds[1],nth)
            for j in inds[1]
                copy!(Random.default_rng(),dseed[j])
                mcheck = zeros(Int64,1)
                arr[j,:] = mle_est(generatort(thq,n),thq,mle_max_iter,bounded::Vector{Int64},positive::Vector{Int64},mcheck,likelihoodm=likelihoodt)
            end
            parent(arr)
        end
        parv = convert(Array,mblock)
        close(mblock)
    else
        mblock = DArray((m,nth),workid,[Int(min(nworkers(),m)),1]) do inds
            arr = zeros(inds[1],nth)
            for j in inds[1]
                copy!(Random.default_rng(),dseed[j])
                mcheck = zeros(Int64,1)
                arr[j,:] = mle_estx(generatort(thq,n,externalv),thq,mle_max_iter,bounded::Vector{Int64},positive::Vector{Int64},mcheck,
				    external=externalv,likelihoodm=likelihoodt)
            end
            parent(arr)
        end
        parv = convert(Array,mblock)
        close(mblock)
    end

    return sum(view(parv,:,ii) .<= iqfunc(Array(view(parv,:,filter(x->x!=ii,1:nth::Int64))),bt,Array(scalev[filter(x->x!=ii,1:nth::Int64)]'),Array(anmed[filter(x->x!=ii,1:nth::Int64)]')))/m::Int64, parv
end

# for test inversion
function simprob_ti(seed::UInt128,bt::Float64,thq0::Float64,qt::Float64,ii::Int64,n::Int64,m::Int64,nth::Int64,mle_max_iter::Int64,bounded::Vector{Int64},positive::Vector{Int64},scalev::Vector{Float64},anmed::Vector{Float64};external::Array{Float64}=[0.0],generatort::Function=nothing,likelihoodt::Function=nothing,format::Symbol=:Float64)
    if format==:BigFloat
        thq = BigFloat.(copy(thq0))
        externalv = BigFloat.(copy(external))
    else
        thq = Float64.(copy(thq0))
        externalv = Float64.(copy(external))
    end

    Random.seed!(rand(MersenneTwister(seed),0:UInt128(2)^128-1))

    parv = zeros(m::Int64)
    mmcheck = 0
    if external==[0.0]
        for i = 1:m::Int64
            mcheck! = zeros(Int64,1)
		    parv[i] = mle_est(generatort([thq],n),[thq],mle_max_iter,bounded::Vector{Int64},positive::Vector{Int64},mcheck!,likelihoodm=likelihoodt)[1]
            if mcheck![1] == 1
                mmcheck += 1
            end
        end
    else
        for i = 1:m::Int64
            mcheck! = zeros(Int64,1)
		    parv[i] = mle_estx(generatort([thq],n,externalv),[thq],mle_max_iter,bounded::Vector{Int64},positive::Vector{Int64},mcheck!,
				external=externalv,likelihoodm=likelihoodt)[1]
            if mcheck![1] == 1
                mmcheck += 1
            end
        end
    end

    if mmcheck > m - 0.1
        print("\n[Warning] Estimation taking too long. Increasing mle_max_iter or appx_order may help.")
    end

    if qt > 0.5
        return sum(parv .< bt - eps(bt))/m, parv
    else
        return sum(parv .<= bt + eps(bt))/m, parv
    end
end

function lincrop(x0::Vector,grid_model::Int64,grid_sub::Int64,nth::Int64)
    pnv = ones(Int64,nth)
    pnv[2] = 2
    for i = 3:nth
        pnv[i] = numbeta(grid_model+1,i-2) + 1
    end

    x1 = x0[pnv]
    return x1
end

function ilincrop(x0::Vector,grid_model::Int64,grid_sub::Int64,nth::Int64)
    pnv = ones(Int64,nth)
    pnv[2] = 2
    for i = 3:nth
        pnv[i] = numbeta(grid_model+1,i-2) + 1
    end

    x1 = zeros(numbeta(grid_model+1,nth-1))
    x1[pnv] = x0
    return x1
end

function scale_bt(bt0::Vector{Float64},sratio::Float64,qt::Float64,grid_model::Int64,grid_sub::Int64,nth::Int64,ii::Int64,scalev::Vector{Float64},coordz::Vector{Float64},fftol::Float64)
    bt = copy(bt0)
    bt = bt.*(0.1).*sratio

    bt = bt.*scalev[ii]

    bt = bt./coordz

    return bt
end

function iscale_bt(bt0::Vector{Float64},sratio::Float64,qt::Float64,grid_model::Int64,grid_sub::Int64,nth::Int64,ii::Int64,scalev::Vector{Float64},coordz::Vector{Float64},fftol::Float64)
    bt = copy(bt0)

    bt = bt.*coordz

    bt = bt./scalev[ii]

    bt = bt./(0.1)./sratio

    return bt
end

function pos_inc(x::Float64)
    if x >= 0
        return x + 1.0
    elseif x >= log(1.0/200.0)
        return 2.0/(1 + exp(-2*x))
    else
        a = 1.0/40000.0
        b = 4.798304866548037
        return -a/(b + x)
    end
end

function ipos_inc(x::BigFloat)
    if x >= 1
        return x - 1.0
    elseif x >= 2.0/40001.0
        return log(-x/(x - 2.0))/2.0
    else
        a = 1.0/40000.0
        b = 4.798304866548037
        return -(a + b*x)/x
    end
end

function rng_inc(x::Float64)
    x = x/1.0
    if abs(x) <= -log(1.0/200.0)
        return 2.0/(1 + exp(-2*x)) - 1.0
    elseif x < 0
        a = 1.0/40000.0
        b = 4.798304866548037
        return -a/(b + x) - 1.0
    else
        a = 1.0/40000.0
        b = 4.798304866548037
        return a/(x - b) + 1.0
    end
end

function irng_inc(x::BigFloat)
    if abs(x) <= 1.0 - 2.0/40001.0
        return log(-(x + 1.0)/(x - 1.0))/2.0
    elseif x < 0
        a = 1.0/40000.0
        b = 4.798304866548037
        return -(a + b*(x + 1.0))/(x + 1.0)
    else
        a = 1.0/40000.0
        b = 4.798304866548037
        return (a + b*(x - 1.0))/(x - 1.0)
    end
    x = x*1.0
end

function trns_bt(iqfunc_big::Function,x0::Vector{Float64},sratio::Float64,thc::Vector{Float64},ii::Int64,bounded::Vector{Int64},positive::Vector{Int64},grid_model::Int64,grid_sub::Int64,nth::Int64,coord::Array{Float64},coord_med::Array{Float64},coordz::Vector{Float64},scalev::Vector{Float64},anmed::Vector{Float64},fftol::Float64,qt::Float64)
    x = copy(x0)

    x = fixedsign(iqfunc_big,length(x),Array(deleteat!(copy(thc),ii)'),coord_med,Array(scalev[filter(x->x!=ii,1:end)]'),Array(anmed[filter(x->x!=ii,1:end)]')).*x

    x = scale_bt(x,sratio,qt,grid_model,grid_sub,nth,ii,scalev,coordz,fftol)

    idx = copy(positive::Vector{Int64})
    if ii in idx
        cc = thc[ii] - (iqfunc_big(BigFloat.(Array(deleteat!(copy(thc),ii)')),BigFloat.(x),BigFloat.(Array(scalev[filter(x->x!=ii,1:end)]')),BigFloat.(Array(anmed[filter(x->x!=ii,1:end)]')))[1] - x[1])

        x[1] = pos_inc(x[1]) + cc
    end

    idx = copy(bounded)
    if ii in idx
        cc = thc[ii] - (iqfunc_big(BigFloat.(Array(deleteat!(copy(thc),ii)')),BigFloat.(x),BigFloat.(Array(scalev[filter(x->x!=ii,1:end)]')),BigFloat.(Array(anmed[filter(x->x!=ii,1:end)]')))[1] - x[1])

        x[1] = rng_inc(x[1]) + cc
    end  
    
    return x
end

function itrns_bt(iqfunc_big::Function,x0::Vector{Float64},sratio::Float64,thc::Vector{Float64},ii::Int64,bounded::Vector{Int64},positive::Vector{Int64},grid_model::Int64,grid_sub::Int64,nth::Int64,coord::Array{Float64},coord_med::Array{Float64},coordz::Vector{Float64},scalev::Vector{Float64},anmed::Vector{Float64},fftol::Float64,qt::Float64)
    x = copy(x0)

    idx = copy(bounded)
    if ii in idx
        cc = thc[ii] - (iqfunc_big(BigFloat.(Array(deleteat!(copy(thc),ii)')),BigFloat.(x),BigFloat.(Array(scalev[filter(x->x!=ii,1:end)]')),BigFloat.(Array(anmed[filter(x->x!=ii,1:end)]')))[1] - x[1])

        x[1] = irng_inc(x[1] - cc)
    end

    idx = copy(positive::Vector{Int64})
    if ii in idx
        cc = thc[ii] - (iqfunc_big(BigFloat.(Array(deleteat!(copy(thc),ii)')),BigFloat.(x),BigFloat.(Array(scalev[filter(x->x!=ii,1:end)]')),BigFloat.(Array(anmed[filter(x->x!=ii,1:end)]')))[1] - x[1])

        x[1] = ipos_inc(x[1] - cc)
    end

    x = iscale_bt(x,sratio,qt,grid_model,grid_sub,nth,ii,scalev,coordz,fftol)

    x = fixedsign(iqfunc_big,length(x),Array(deleteat!(copy(thc),ii)'),coord_med,Array(scalev[filter(x->x!=ii,1:end)]'),Array(anmed[filter(x->x!=ii,1:end)]')).*x

    return x
end

function phi_con(iqfunc::Function,x::Vector{Float64},i::Int64,qt::Float64,th_hat::Vector{Float64},n::Int64,m::Int64,nth::Int64,grid_model::Int64,mle_max_iter::Int64,seed::UInt128,bounded::Vector{Int64},positive::Vector{Int64},scalev::Vector{Float64},anmed::Vector{Float64};external::Array{Float64}=[0.0],generatorp::Function=nothing,likelihoodp::Function=nothing,format::Symbol=:Float64)
    bt = zeros(numbeta(grid_model::Int64+1,nth::Int64-1))
    bt[1] = th_hat[i]
    pv = zeros(nth::Int64)
    pv[i] = trns_one(x[1],i,Float64.(th_hat),scalev,bounded::Vector{Int64},positive::Vector{Int64})
    pv[filter(x->x!=i,1:end)] = deleteat!(copy(th_hat),i)

    cp, parv = simprobm(seed,bt,pv,qt,i,iqfunc,n,m,nth,mle_max_iter,bounded::Vector{Int64},positive::Vector{Int64},scalev,anmed,external=external,generatort=generatorp,likelihoodt=likelihoodp,format=format)

    if (cp<=0.999) && (cp>=0.001)
        FF = cp - qt
    else
        if cp < 0.001
            q1, q2, vc = qsortil(parv,i,0.001,0.005,0.05)
            xx = iqfunc(Array(vc),bt,Array(scalev[filter(x->x!=i,1:end)]'),Array(anmed[filter(x->x!=i,1:end)]'))[1]
            bb = (0.05-0.001)/(q2-q1)
            FF = 0.001 + (xx - q1)*bb - qt
        elseif cp > 0.999
            q1, q2 ,vc = qsortiu(parv,i,0.95,0.995,0.999)
            xx = iqfunc(Array(vc),bt,Array(scalev[filter(x->x!=i,1:end)]'),Array(anmed[filter(x->x!=i,1:end)]'))[1]
            bb = (0.999-0.95)/(q2-q1)
            FF = 0.999 + (xx - q2)*bb - qt
        end
    end

    return abs(FF)
end

# for test inversion
function phi_ti(x::Vector{Float64},i::Int64,qt::Float64,th_hat::Vector{Float64},n::Int64,m::Int64,nth::Int64,grid_model::Int64,mle_max_iter::Int64,seed::UInt128,bounded::Vector{Int64},positive::Vector{Int64},scalev::Vector{Float64},anmed::Vector{Float64};external::Array{Float64}=[0.0],generatorp::Function=nothing,likelihoodp::Function=nothing,format::Symbol=:Float64)
    bt = th_hat[i]
    pv = trns_one(x[1],i,Float64.(th_hat),scalev,bounded::Vector{Int64},positive::Vector{Int64})

    cp, parv = simprob_ti(seed,bt,pv,qt,i,n,m,nth,mle_max_iter,bounded,positive,scalev,anmed,external=external,generatort=generatorp,likelihoodt=likelihoodp,format=format)
    
    if (cp<=0.999) && (cp>=0.001)
        FF = cp - qt
    else
        if cp < 0.001
            q1, q2 = uqsortil(vec(parv),0.001,0.05)
            bb = (0.05-0.001)/(q2-q1)
            FF = 0.001 + (bt - q1)*bb - qt
        elseif cp > 0.999
            q1, q2 = uqsortiu(vec(parv),0.95,0.999)
            bb = (0.999-0.95)/(q2-q1)
            FF = 0.999 + (bt - q2)*bb - qt
        end
    end

    return abs(FF)
end

function phi_lin(iqfunc::Function,iqfunc_big::Function,iqfunc_proc::Function,x::Vector{Float64},sratio::Float64,fftol::Float64,iii::Int64,qt::Float64,th_hat::Vector{Float64},coord::Array{Float64},coord_med::Array{Float64},coordz::Vector{Float64},
    n::Int64,m::Int64,mg::Int64,nth::Int64,grid_model::Int64,grid_sub::Int64,mle_max_iter::Int64,seed::UInt128,bounded::Vector{Int64},positive::Vector{Int64},rvec::Vector{Int64},scalev::Vector{Float64},anmed::Vector{Float64};
    external::Array{Float64}=[0.0],generatorf::Function=nothing,likelihoodf::Function=nothing,format::Symbol=:Float64)

    bt = ilincrop(x,grid_model,grid_sub,nth)
    bt = trns_bt(iqfunc_big,bt,sratio,th_hat,iii,bounded,positive,grid_model,grid_sub,nth,coord,coord_med,coordz,scalev,anmed,fftol,qt)
    th_hat_c = deleteat!(copy(th_hat),iii)
    tti = iqfunc_big(BigFloat.(Array(th_hat_c')),BigFloat.(bt),BigFloat.(Array(scalev[filter(x->x!=iii,1:end)]')),BigFloat.(Array(anmed[filter(x->x!=iii,1:end)]')))[1]-th_hat[iii]

    if iii in positive
        tti = abs(tti)
    elseif iii in bounded
        tti = trianglewave(tti)
    end

    bts = copy(bt)
    bts[1] = bts[1] - tti
	
    gcord = copy(coord)

    dseed = Xoshiro.(rand(MersenneTwister(seed),0:UInt128(2)^128-1))

    FF = DArray((size(coord,1),1)) do inds
        arr = zeros(inds)

        for iip in inds[1]
            copy!(Random.default_rng(),dseed)
            pv = zeros(nth)
            pv[iii] = tti
            pv[filter(x->x!=iii,1:end)] = view(gcord,iip,:)
            cp, parv = simprob(bts,pv,qt,iii,iqfunc_proc,n,m,nth,mle_max_iter,bounded,positive,scalev,anmed,external=external,generatort=generatorf,likelihoodt=likelihoodf,format=format)
            if (cp<=0.999) && (cp>=0.001)
                arr[iip,1] = cp - qt
            else
                if cp<0.001
                    q1, q2, vc = qsortil(parv,iii,0.001,0.005,0.05)
                    xx = iqfunc_proc(Array(vc),bts,Array(scalev[filter(x->x!=iii,1:end)]'),Array(anmed[filter(x->x!=iii,1:end)]'))[1]
                    bb = (0.05-0.001)/(q2-q1)
                    arr[iip,1] = 0.001 + (xx - q1)*bb - qt
                elseif cp>0.999
                    q1, q2, vc = qsortiu(parv,iii,0.95,0.995,0.999)
                    xx = iqfunc_proc(Array(vc),bts,Array(scalev[filter(x->x!=iii,1:end)]'),Array(anmed[filter(x->x!=iii,1:end)]'))[1]
                    bb = (0.999-0.95)/(q2-q1)
                    arr[iip,1] = 0.999 + (xx - q2)*bb - qt
                end
            end
        end
        parent(arr)
    end
    FFF = convert(Array,FF[1:size(coord,1)])
    close(FF)

    return fixednorm(FFF)
end

function phi_fix(fratio::Float64,iqfunc::Function,iqfunc_big::Function,iqfunc_proc::Function,x::Vector{Float64},sratio::Float64,fftol::Float64,iii::Int64,qt::Float64,th_hat::Vector{Float64},coord::Array{Float64},coord_med::Array{Float64},coordz::Vector{Float64},
    n::Int64,m::Int64,mg::Int64,nth::Int64,grid_model::Int64,grid_sub::Int64,mle_max_iter::Int64,seed::UInt128,bounded::Vector{Int64},positive::Vector{Int64},rvec::Vector{Int64},scalev::Vector{Float64},anmed::Vector{Float64};
    external::Array{Float64}=[0.0],generatorf::Function=nothing,likelihoodf::Function=nothing,format::Symbol=:Float64)

    bt = trns_bt(iqfunc_big,x,sratio,th_hat,iii,bounded::Vector{Int64},positive::Vector{Int64},grid_model::Int64,grid_sub,nth::Int64,coord,coord_med,coordz,scalev,anmed,fftol,qt)
    th_hat_c = deleteat!(copy(th_hat),iii)
    tti = iqfunc_big(BigFloat.(Array(th_hat_c')),BigFloat.(bt),BigFloat.(Array(scalev[filter(x->x!=iii,1:end)]')),BigFloat.(Array(anmed[filter(x->x!=iii,1:end)]')))[1]-th_hat[iii]

    if iii in positive
        tti = abs(tti)
    elseif iii in bounded
        tti = trianglewave(tti)
    end

    bts = copy(bt)
    bts[1] = bts[1] - tti

    gcord = copy(coord)

    dseed = Xoshiro.(rand(MersenneTwister(seed::UInt128),0:UInt128(2)^128-1))

    FF = DArray((numbeta(grid_model+1,nth-1),1)) do inds
        arr = zeros(inds)

        for iip in inds[1]
            copy!(Random.default_rng(),dseed)
            pv = zeros(nth::Int64)
            pv[iii] = tti
            pv[filter(x->x!=iii,1:end)] = view(gcord,iip,:)
            cp, parv = simprob(bts,pv,qt,iii,iqfunc_proc,n,m,nth,mle_max_iter,bounded::Vector{Int64},positive::Vector{Int64},scalev,anmed,external=external,generatort=generatorf,likelihoodt=likelihoodf,format=format)
            if (cp<=0.999) && (cp>=0.001)
                arr[iip,1] = cp - qt
            else
                if cp<0.001
                    q1, q2, vc = qsortil(parv,iii,0.001,0.005,0.05)
                    xx = iqfunc_proc(Array(vc),bts,Array(scalev[filter(x->x!=iii,1:end)]'),Array(anmed[filter(x->x!=iii,1:end)]'))[1]
                    bb = (0.05-0.001)/(q2-q1)
                    arr[iip,1] = 0.001 + (xx - q1)*bb - qt
                elseif cp>0.999
                    q1, q2, vc = qsortiu(parv,iii,0.95,0.995,0.999)
                    xx = iqfunc_proc(Array(vc),bts,Array(scalev[filter(x->x!=iii,1:end)]'),Array(anmed[filter(x->x!=iii,1:end)]'))[1]
                    bb = (0.999-0.95)/(q2-q1)
                    arr[iip,1] = 0.999 + (xx - q2)*bb - qt
                end
            end
        end
        parent(arr)
    end
    FFF = convert(Array,FF[1:numbeta(grid_model+1,nth-1)])
    close(FF)

    ret = fratio.*FFF .+ x

    return ret
end

function phi_rob(iqfunc::Function,iqfunc_big::Function,iqfunc_proc::Function,x::Vector{Float64},sratio::Float64,fftol::Float64,iii::Int64,qt::Float64,th_hat::Vector{Float64},coord::Array{Float64},coord_med::Array{Float64},coordz::Vector{Float64},
    n::Int64,m::Int64,mg::Int64,nth::Int64,grid_model::Int64,grid_sub::Int64,mle_max_iter::Int64,seed::UInt128,bounded::Vector{Int64},positive::Vector{Int64},rvec::Vector{Int64},scalev::Vector{Float64},anmed::Vector{Float64};
    external::Array{Float64}=[0.0],generatorf::Function=nothing,likelihoodf::Function=nothing,format::Symbol=:Float64)

    bt = trns_bt(iqfunc_big,x,sratio,th_hat,iii,bounded,positive,grid_model,grid_sub,nth,coord,coord_med,coordz,scalev,anmed,fftol,qt)
    th_hat_c = deleteat!(copy(th_hat),iii)
    tti = iqfunc_big(BigFloat.(Array(th_hat_c')),BigFloat.(bt),BigFloat.(Array(scalev[filter(x->x!=iii,1:end)]')),BigFloat.(Array(anmed[filter(x->x!=iii,1:end)]')))[1]-th_hat[iii]

    if iii in positive
        tti = abs(tti)
    elseif iii in bounded
        tti = trianglewave(tti)
    end

    bts = copy(bt)
    bts[1] = bts[1] - tti
	
    gcord = copy(coord)

    # reorder coordinates to match the seeds
    nwork = nworkers()
    ncd = size(gcord,1)
    nbt = length(bts)
    ncolbt = Int64(ceil(nbt/nwork))
    ncolad = Int64(ceil((ncd-nbt+mod(nbt,nwork))/nwork))
    ons = [ones(Int64,nbt);zeros(Int64,nwork*ncolbt-nbt)]
    ons = reshape(ons,(nwork,ncolbt))
    ibt = zeros(Int64,size(ons,2),size(ons,1))
    ibtv = collect(1:nbt)
    ibtc = 1
    for i = 1:nwork
        nos = sum(ons[i,:])
        ibt[1:nos,i] = ibtv[ibtc:(ibtc+(nos-1))]
        ibtc = ibtc+nos
    end
    iad = [zeros(Int64,mod(nbt,nwork));collect(nbt+1:ncd);zeros(Int64,(nwork*ncolad-(ncd-nbt))-mod(nbt,nwork))]
    iad = reshape(iad,(nwork,ncolad))'
    icd = [ibt;iad]
    icd = reshape(icd,(nwork*(ncolbt+ncolad),1))
    icd = filter(x->x!=0,icd)
    gcord = gcord[icd,:]

    dseed = Xoshiro.(rand(MersenneTwister(seed),0:UInt128(2)^128-1))

    FF = DArray((size(coord,1),1)) do inds
        arr = zeros(inds)

        for iip in inds[1]
            copy!(Random.default_rng(),dseed)
            pv = zeros(nth)
            pv[iii] = tti
            pv[filter(x->x!=iii,1:end)] = view(gcord,iip,:)
            cp, parv = simprob(bts,pv,qt,iii,iqfunc_proc,n,m,nth,mle_max_iter,bounded,positive,scalev,anmed,external=external,generatort=generatorf,likelihoodt=likelihoodf,format=format)
            if (cp<=0.999) && (cp>=0.001)
                arr[iip,1] = cp - qt
            else
                if cp<0.001
                    q1, q2, vc = qsortil(parv,iii,0.001,0.005,0.05)
                    xx = iqfunc_proc(Array(vc),bts,Array(scalev[filter(x->x!=iii,1:end)]'),Array(anmed[filter(x->x!=iii,1:end)]'))[1]
                    bb = (0.05-0.001)/(q2-q1)
                    arr[iip,1] = 0.001 + (xx - q1)*bb - qt
                elseif cp>0.999
                    q1, q2, vc = qsortiu(parv,iii,0.95,0.995,0.999)
                    xx = iqfunc_proc(Array(vc),bts,Array(scalev[filter(x->x!=iii,1:end)]'),Array(anmed[filter(x->x!=iii,1:end)]'))[1]
                    bb = (0.999-0.95)/(q2-q1)
                    arr[iip,1] = 0.999 + (xx - q2)*bb - qt
                end
            end
        end
        parent(arr)
    end
    FFF = convert(Array,FF[1:size(coord,1)])
    close(FF)

    return fixednorm(FFF)
end

function empgen(seed::Xoshiro,cexcoor::Vector{Float64},mb::Int64,nth::Int64,n::Int64,mle_max_iter::Int64,bounded::Vector{Int64},positive::Vector{Int64},format::Symbol,external::Array{Float64},genera::Function,likeli::Function)

    if format==:BigFloat
        excoor = BigFloat.(copy(cexcoor))
        externalv = BigFloat.(copy(external))
    else
        excoor = Float64.(copy(cexcoor))
        externalv = Float64.(copy(external))
    end

    dseed = Xoshiro.(rand(seed,0:UInt128(2)^128-1,mb))

    exth_btv = zeros(mb,nth)
    workid = workers()
    if external==[0.0]
        mblock = DArray((mb,nth),workid,[Int(min(nworkers(),mb)),1]) do inds
            arr = zeros(inds[1],nth)
            for jjc in inds[1]
                copy!(Random.default_rng(),dseed[jjc])
                mcheck = zeros(Int64,1)
                arr[jjc,:] = mle_est(genera(excoor,n),excoor,mle_max_iter,bounded::Vector{Int64},positive::Vector{Int64},mcheck,likelihoodm=likeli)
            end
            parent(arr)
        end
        exth_btv = convert(Array,mblock)
        close(mblock)
    else
        mblock = DArray((mb,nth),workid,[Int(min(nworkers(),mb)),1]) do inds
            arr = zeros(inds[1],nth)
            for jjc in inds[1]
                copy!(Random.default_rng(),dseed[jjc])
                mcheck = zeros(Int64,1)
                arr[jjc,:] = mle_estx(genera(excoor,n,externalv),excoor,mle_max_iter,bounded::Vector{Int64},positive::Vector{Int64},mcheck,external=externalv,likelihoodm=likeli)
            end
            parent(arr)
        end
        exth_btv = convert(Array,mblock)
        close(mblock)
    end

    return exth_btv
end

function linsqrt(x::Vector{Float64},d::Float64)
    nm = sqrt(sum(x.^2))
    if nm >= 1
        return d.*x
    else
        ys = sqrt(d^2/(1-d))
        ss = d^2/(4 - 4*d)
        vv = d - sqrt(1+ss)*ys
        return (vv + sqrt(ss + nm)*ys)/nm.*x
    end
end

function rescheck(par0::Vector{Float64},jj::Int64,positive::Vector{Int64},bounded::Vector{Int64})
    par = copy(par0)
    insert!(par,jj,0.5)
    ret = []
    for i = positive
        if par[i] < 0.0
            ret = [ret;i]
        end
    end
    for i = bounded
        if par[i] > 1.0 || par[i] < -1.0
            ret = [ret;i]
        end
    end
    return ret
end

function widthp(qn::Float64,cth_btvv::Array{Float64},ithub::Vector{Float64},ithlb::Vector{Float64},nth::Int64,jj::Int64,gridq::Float64)
    ind_cut = collect(1:size(cth_btvv,1))
    for jjc = 1:(nth-1)
        cutlb = quantile(cth_btvv[:,jjc],(1-qn)/2)
        cutub = quantile(cth_btvv[:,jjc],1-(1-qn)/2)
        ind_cut = intersect(ind_cut,findall(x->x<=cutub&&x>=cutlb,cth_btvv[:,jjc]))
    end
    cth_cut = cth_btvv[ind_cut,:]

    ithlbq = zeros(nth-1)
    ithubq = zeros(nth-1)
    for jjc = 1:(nth-1)
        ithlbq[jjc] = quantile(cth_cut[:,jjc],(1-gridq)/2)
        ithubq[jjc] = quantile(cth_cut[:,jjc],1-(1-gridq)/2)
    end

    return abs(sum(((ithubq .- ithlbq)).^2) - sum((ithub[filter(x->x!=jj,1:end)] .- ithlb[filter(x->x!=jj,1:end)]).^2))
end

function censor_res(coord::Vector{Float64},positive::Vector{Int64},bounded::Vector{Int64})
    coordx = copy(coord)
    tol = 0.000001
    for i in positive
        coordx[i] = max(tol,coordx[i])
    end
    for i in bounded
        coordx[i] = min(1.0-tol,max(-1.0+tol,coordx[i]))
    end
    return coordx
end

function censorc_res(coordc::Array{Float64},jj::Int64,positive::Vector{Int64},bounded::Vector{Int64})
    coordx = copy(coordc)
    tol = 0.000001

    pos = copy(positive)
    pos[findall(x->x>jj,positive)] = positive[findall(x->x>jj,positive)] .- 1
    deleteat!(pos,findall(x->x==jj,positive))
    rng = copy(bounded)
    rng[findall(x->x>jj,bounded)] = bounded[findall(x->x>jj,bounded)] .- 1
    deleteat!(rng,findall(x->x==jj,bounded))

    for i in pos
        ind = findall(x->x<tol,coordx[:,i])
        coordx[ind,i] .= tol
    end
    for i in rng
        ind = findall(x->x<-1.0+tol,coordx[:,i])
        coordx[ind,i] .= -1.0+tol
        ind = findall(x->x>1.0-tol,coordx[:,i])
        coordx[ind,i] .= 1.0-tol
    end
    return coordx
end

function dampenc_res(coordc::Array{Float64},ratioc::Vector{Float64},centerc::Vector{Float64},jj::Int64,positive::Vector{Int64},bounded::Vector{Int64})
    tol = 0.5

    coordx = (coordc .- centerc').*ratioc' .+ centerc'

    pos = copy(positive)
    pos[findall(x->x>jj,positive)] = positive[findall(x->x>jj,positive)] .- 1
    deleteat!(pos,findall(x->x==jj,positive))
    rng = copy(bounded)
    rng[findall(x->x>jj,bounded)] = bounded[findall(x->x>jj,bounded)] .- 1
    deleteat!(rng,findall(x->x==jj,bounded))

    for i in pos
        inn = findall(x->x==1,coordc[:,i] .> coordx[:,i])
        ind = findall(x->x<centerc[i]*tol,coordx[:,i])
        ind = intersect(inn,ind)
        sta = min.(centerc[i]*tol,coordc[ind,i])
        dif = sta .- coordx[ind,i]
        ndif = dif./(centerc[i]*tol)
        ndam = -exp.(-ndif).*(1.0 .- exp.(ndif) .- (centerc[i]*tol .- sta)./(centerc[i]*tol)) .-
            (centerc[i]*tol .- sta)./(centerc[i]*tol)
        dam = ndam.*(centerc[i]*tol)
        coordx[ind,i] = sta .- dam
    end

    for i in rng
        # lower
        inn = findall(x->x==1,coordc[:,i] .> coordx[:,i])
        ind = findall(x->x<(-1.0 + (centerc[i] + 1.0)*tol),coordx[:,i])
        ind = intersect(inn,ind)
        sta = min.((-1.0 + (centerc[i] + 1.0)*tol),coordc[ind,i])
        dif = sta .- coordx[ind,i]
        ndif = dif./((centerc[i] + 1.0)*tol)
        ndam = -exp.(-ndif).*(1.0 .- exp.(ndif) .- ((-1.0 + (centerc[i] + 1.0)*tol) .- sta)./((centerc[i] + 1.0)*tol)) .-
            ((-1.0 + (centerc[i] + 1.0)*tol) .- sta)./((centerc[i] + 1.0)*tol)
        dam = ndam.*((centerc[i] + 1.0)*tol)
        coordx[ind,i] = sta .- dam
        # upper
        inn = findall(x->x==1,coordc[:,i] .< coordx[:,i])
        ind = findall(x->x>(1.0 - (1.0 - centerc[i])*tol),coordx[:,i])
        ind = intersect(inn,ind)
        sta = max.((1.0 - (1.0 - centerc[i])*tol),coordc[ind,i])
        dif = coordx[ind,i] .- sta
        ndif = dif./((1.0 - centerc[i])*tol)
        ndam = -exp.(-ndif).*(1.0 .- exp.(ndif) .- (sta .- (1.0 - (1.0 - centerc[i])*tol))./((1.0 - centerc[i])*tol)) .-
            (sta .- (1.0 - (1.0 - centerc[i])*tol))./((1.0 - centerc[i])*tol)
        dam = ndam.*((1.0 - centerc[i])*tol)
        coordx[ind,i] = sta .+ dam
    end
    return coordx
end

function dampf(x::Vector{Float64})
    return asin.(x)./(pi/2)
end

function dampenc_circ_res(coordc::Array{Float64},ratioc::Vector{Float64},centerc::Vector{Float64},jj::Int64,positive::Vector{Int64},bounded::Vector{Int64})
    tol = 0.99

    coordx = (coordc .- centerc').*ratioc' .+ centerc'

    pos = copy(positive)
    pos[findall(x->x>jj,positive)] = positive[findall(x->x>jj,positive)] .- 1
    deleteat!(pos,findall(x->x==jj,positive))
    rng = copy(bounded)
    rng[findall(x->x>jj,bounded)] = bounded[findall(x->x>jj,bounded)] .- 1
    deleteat!(rng,findall(x->x==jj,bounded))

    for i in pos
        inn = findall(x->x==1,coordc[:,i] .> coordx[:,i])
        ind = findall(x->x<centerc[i]*tol,coordx[:,i])
        ind = intersect(inn,ind)
        sta = min.(centerc[i]*tol,coordc[ind,i])
        dif = sta .- coordx[ind,i]
        if ind != []
            maxdif = maximum(centerc[i]*tol .- coordx[ind,i])
        else
            maxdif = 1.0
        end
        ndam = dampf(dif./maxdif)
        dam = ndam.*sta
        coordx[ind,i] = sta .- dam
    end

    for i in rng
        # lower
        inn = findall(x->x==1,coordc[:,i] .> coordx[:,i])
        ind = findall(x->x<(-1.0 + (centerc[i] + 1.0)*tol),coordx[:,i])
        ind = intersect(inn,ind)
        sta = min.((-1.0 + (centerc[i] + 1.0)*tol),coordc[ind,i])
        dif = sta .- coordx[ind,i]
        if ind != []
            maxdif = maximum((-1.0 + (centerc[i] + 1.0)*tol) .- coordx[ind,i])
        else
            maxdif = 1.0
        end
        ndam = dampf(dif./maxdif)
        dam = ndam.*(sta .+ 1.0)
        coordx[ind,i] = sta .- dam
        # upper
        inn = findall(x->x==1,coordc[:,i] .< coordx[:,i])
        ind = findall(x->x>(1.0 - (1.0 - centerc[i])*tol),coordx[:,i])
        ind = intersect(inn,ind)
        sta = max.((1.0 - (1.0 - centerc[i])*tol),coordc[ind,i])
        dif = coordx[ind,i] .- sta
        if ind != []
            maxdif = maximum(coordx[ind,i] .- (1.0 - (1.0 - centerc[i])*tol))
        else
            maxdif = 1.0
        end
        ndam = dampf(dif./maxdif)
        dam = ndam.*(1.0 .- sta)
        coordx[ind,i] = sta .+ dam
    end
    return censorc_res(coordx,jj,positive,bounded)
end

function hband(empd::Array{Float64},d::Int64,n::Int64)
    return ((4/(d+2))^(1/(d+4)).*n^(-1/(d+4)).*sqrt.(vec(var(empd,dims=1)))).^2
end

function denest(empd::Array{Float64},coord::Vector{Float64},h::Vector{Float64},d::Int64,n::Int64)
    return (2.0*pi)^(-d/2)/sqrt(prod(h))*sum(exp.(-sum((coord' .- empd).^2 ./h',dims=2)./2.0))/n
end

function flatd_r(seed::Xoshiro,empd::Array{Float64})
    ne = size(empd,1)
    k = size(empd,2)
    nf = Int(ceil(ne/2))
    if iseven(nf)
        nf = nf + 1
    end

    h = hband(empd,k,ne)

    workid = workers()
    nwork = nworkers()
    chsum = DArray((ne,1),workid,[Int(min(nwork,ne)),1]) do inds
        arr = zeros(inds)
        for j in inds[1]
            arr[j,1] = denest(empd,empd[j,:],h,k,ne)
        end
        parent(arr)
    end
    kdenw = convert(Array,chsum)
    close(chsum)
    kdenw = 1.0./(kdenw./minimum(kdenw))

    fdist = zeros(nf,k)
    nc = 1
    while nc == 1
        xi = rand(seed,1:ne)
        if rand(seed) <= kdenw[xi]
            fdist[nc,:] = empd[xi,:]
            nc += 1
            empd = empd[filter(x->x!=xi,1:end),:]
            ne -= 1
        end
    end
    while nc <= nf
        xi = rand(seed,1:ne)
        if rand(seed) <= kdenw[xi]
            xe = empd[xi,:]
            if findfirst(x->x<=10.0^(-14),abs.(fdist[1:(nc-1),:] .- xe')) != nothing && 
                rand(seed) > 0.1
                continue
            end
            rp = rand(seed)
            fdist[nc,:] = fdist[nc-1,:].*rp .+ xe.*(1.0 - rp)
            fdist[nc+1,:] = xe
            nc += 2
            empd = empd[filter(x->x!=xi,1:end),:]
            ne -= 1
        end
    end

    return fdist
end

function tip_match(seed::Xoshiro,dumth::Int64,initm::Vector{Float64},th_btvc::Array{Float64},th_hatc::Vector{Float64},qtleft1::Float64,jj::Int64,alpha::Float64,qt::Float64,nth::Int64,n::Int64,mb::Int64,med_gen_pre::Int64,format::Symbol,external::Array{Float64},mle_max_iter::Int64,bounded::Vector{Int64},positive::Vector{Int64},genera::Function,likeli::Function)
    tolm = 0.05
    cnt_ret_med = 1
    seed1 = copy(seed)
    @label ret_med
    if dumth == 1
        th_btvt = th_btvc .- median(th_btvc,dims=1) .+ th_hatc'
    elseif dumth == 0
        th_btvt = copy(th_btvc)
    end
    cmed = copy(initm)
    insert!(cmed,jj,qtleft1)
    seed2 = copy(seed1)
    hmed = mediantip(seed2,qt,cmed[filter(x->x!=jj,1:nth)],th_hatc,qtleft1,jj,n,med_gen_pre,nth,format,external,mle_max_iter,bounded,positive,genera,likeli)
    deleteat!(cmed,jj)
    cmed1 = copy(cmed)
    hmed1 = copy(hmed)
    dist1 = 10.0^10
    dist2 = 10.0^9
    p1 = 1.0
    q1 = 0.0
    cntleq = 1
    med_gen_act = copy(med_gen_pre)
    @label leqa1
    insert!(cmed,jj,qtleft1)
    seed2 = copy(seed1)
    hemp = empgen(seed2,cmed,mb,nth,n,mle_max_iter,bounded,positive,format,external,genera,likeli)
    seed2 = copy(seed1)
    hmed = mediantip(seed2,qt,cmed[filter(x->x!=jj,1:nth)],th_hatc,qtleft1,jj,n,med_gen_act,nth,format,external,mle_max_iter,bounded,positive,genera,likeli)
    cmed2 = cmed[filter(x->x!=jj,1:nth)]
    hmed2 = copy(hmed)
    hcratio = sqrt(sum((hmed2 .- hmed1).^2))/sqrt(sum((cmed2 .- cmed1).^2))
    if hcratio==Inf||isnan(hcratio)
        hcratio = 10.0
    end
    ddiff = (hmed2 .- hmed1)./(cmed2 .- cmed1)
    iind = findall(x->x==Inf||x==-Inf||isnan(x),ddiff)
    if iind != []
        ddiff[iind] .= 10.0
    end
    iind = findall(x->x==0.0,ddiff)
    if iind != []
        ddiff[iind] .= 10.0
    end
    cmed1 = cmed[filter(x->x!=jj,1:nth)]
    hmed1 = copy(hmed)

    if pseudo_incm(th_btvt,vec(hmed),tolm) == true
        hdiff = vec(median(th_btvt,dims=1)) .- hmed
        if sqrt(sum(hdiff.^2)) > max(10,10*sqrt(sum(vec(median(th_btvt,dims=1)).^2))) &&
            cntleq > 50+20*(nth-1)
            cmed = copy(initm)
            insert!(cmed,jj,qtleft1)
            seed1 = Xoshiro.(rand(seed,0:UInt128(2)^128-1))
            seed2 = copy(seed1)
            hmed = mediantip(seed2,qt,cmed[filter(x->x!=jj,1:nth)],th_hatc,qtleft1,jj,n,med_gen_pre,nth,format,external,mle_max_iter,bounded,positive,genera,likeli)
            deleteat!(cmed,jj)
            cmed1 = copy(cmed)
            hmed1 = copy(hmed)
            cntleq = 1
            @goto leqa1
        end
        dist3 = copy(sqrt(sum(hdiff.^2)))
        if dist2 >= dist1 && dist3 >= dist2
            tem = copy(p1)
            p1 = copy(q1)
            q1 = copy(tem)
        end
        dist1 = copy(dist2)
        dist2 = copy(dist3)
        ddiff = hdiff./ddiff
        hdiff = linsqrt(p1.*hdiff .+ q1.*ddiff,max(0.002,1/max(1.0,hcratio)/5.0))
        insert!(hdiff,jj,0.0)
        hcnt = 1
        @label leqa2
        cmed_temp = cmed .+ hdiff
        resch = rescheck(vec(cmed_temp[filter(x->x!=jj,1:nth)]),jj,positive,bounded)
        if resch != []
            hdiff[resch] = hdiff[resch]./2.0
            @goto leqa2
        end
        if pseudo_sepm(hemp[:,filter(x->x!=jj,1:nth)],vec(cmed_temp[filter(x->x!=jj,1:nth)]),0.95) == true
            hdiff = hdiff./2.0
            hcnt = hcnt + 1
            if hcnt > 100
                if cnt_ret_med > 10
                    print("\n[Warning] Coordinate points preparation keeps failing. If this message keeps repeating, it may be because the parameter estimation is numerically unstable or the necessary assumptions do not hold.")
                    tolm = min(0.99,tolm + 0.05)
                end
                seed1 = Xoshiro(rand(seed,0:UInt128(2)^128-1))
                cnt_ret_med = cnt_ret_med + 1
                @goto ret_med
            else
                @goto leqa2
            end
        else
            cmed = copy(cmed_temp[filter(x->x!=jj,1:nth)])
        end
        if cntleq > 500+200*(nth-1)
            if cnt_ret_med > 10
                print("\n[Warning] Coordinate points preparation keeps failing. If this message keeps repeating, it may be because the parameter estimation is numerically unstable or the necessary assumptions do not hold.")
                tolm = min(0.99,tolm + 0.05)
            end
            seed1 = Xoshiro(rand(seed,0:UInt128(2)^128-1))
            cnt_ret_med = cnt_ret_med + 1
            @goto ret_med
        else
            cntleq = cntleq + 1
            @goto leqa1
        end
    end

    return cmed, seed1
end

function med_match(seed::Xoshiro,seed_co::Xoshiro,cnt_ret_meda::Int64,tolmm::Float64,cmed::Vector{Float64},jj::Int64,grid_model::Int64,robust::Int64,mb::Int64,nth::Int64,n::Int64,mle_max_iter::Int64,bounded::Vector{Int64},positive::Vector{Int64},format::Symbol,external::Array{Float64},genera::Function,likeli::Function)
    seed1 = copy(seed)
    fcmed = copy(cmed)
    seed2 = copy(seed1)
    hemp = empgen(seed2,fcmed,mb,nth,n,mle_max_iter,bounded,positive,format,external,genera,likeli)
    hmed = vec(median(hemp[:,filter(x->x!=jj,1:nth)],dims=1))
    cmed1 = fcmed[filter(x->x!=jj,1:nth)]
    hmed1 = copy(hmed)
    dist1 = 10.0^10
    dist2 = 10.0^9
    p1 = 1.0
    q1 = 0.0
    cntff = 1
    mb_act = copy(mb)
    @label fa1
    if mod(cntff,100+100*(nth-1)) == 0
        mb_act = min(mb*10,Int(ceil(mb_act*1.5)))
    end
    seed2 = copy(seed1)
    hemp = empgen(seed2,fcmed,mb,nth,n,mle_max_iter,bounded,positive,format,external,genera,likeli)
    hmed = vec(median(hemp[:,filter(x->x!=jj,1:nth)],dims=1))
    cmed2 = fcmed[filter(x->x!=jj,1:nth)]
    hmed2 = copy(hmed)
    hcratio = sqrt(sum((hmed2 .- hmed1).^2))/sqrt(sum((cmed2 .- cmed1).^2))
    if hcratio==Inf||isnan(hcratio)
        hcratio = 5.0
    end
    ddiff = (hmed2 .- hmed1)./(cmed2 .- cmed1)
    iind = findall(x->x==Inf||x==-Inf||isnan(x),ddiff)
    if iind != []
        ddiff[iind] .= 5.0
    end
    iind = findall(x->x==0.0,ddiff)
    if iind != []
        ddiff[iind] .= 5.0
    end
    cmed1 = fcmed[filter(x->x!=jj,1:nth)]
    hmed1 = copy(hmed)
    hdiff = vec(median(hemp[:,filter(x->x!=jj,1:nth)],dims=1)) .- cmed[filter(x->x!=jj,1:nth)]
    smind = findall(x->abs(x)<10.0^(-5),hdiff)
    ccmed = cmed[filter(x->x!=jj,1:nth)]
    if smind != []
        ccmed[smind] = vec(median(hemp[:,filter(x->x!=jj,1:nth)],dims=1))[smind]
    end

    if pseudo_incm(hemp[:,filter(x->x!=jj,1:nth)],ccmed,tolmm) == true
        hdiff = vec(median(hemp[:,filter(x->x!=jj,1:nth)],dims=1)) .- cmed[filter(x->x!=jj,1:nth)]
        if sqrt(sum(hdiff.^2)) > max(10,10*sqrt(sum(cmed[filter(x->x!=jj,1:nth)].^2))) && cntff > 100
            fcmed = copy(cmed)
            seed1 = Xoshiro.(rand(seed1,0:UInt128(2)^128-1))
            seed2 = copy(seed1)
            hemp = empgen(seed2,fcmed,mb,nth,n,mle_max_iter,bounded,positive,format,external,genera,likeli)
            hmed = vec(median(hemp[:,filter(x->x!=jj,1:nth)],dims=1))
            cmed1 = fcmed[filter(x->x!=jj,1:nth)]
            hmed1 = copy(hmed)
            cntff = 1
            mb_act = min(mb*10,Int(ceil(mb_act*1.5)))
            @goto fa1
        end
        dist3 = copy(sqrt(sum(hdiff.^2)))
        if dist2 >= dist1 && dist3 >= dist2
            tem = copy(p1)
            p1 = copy(q1)
            q1 = copy(tem)
        end
        dist1 = copy(dist2)
        dist2 = copy(dist3)
        ddiff = hdiff./ddiff
        hdiff = linsqrt(p1.*hdiff .+ q1*ddiff,1/max(1.0,hcratio)/10.0)
        insert!(hdiff,jj,0.0)
        hcnt = 1
        @label fa2
        fcmed_temp = fcmed .- hdiff
        if pseudo_sepm(hemp[:,filter(x->x!=jj,1:nth)],vec(fcmed_temp[filter(x->x!=jj,1:nth)]),0.999) == true ||
            rescheck(vec(fcmed_temp[filter(x->x!=jj,1:nth)]),jj,positive,bounded) != []
            hdiff = hdiff./2.0
            hcnt = hcnt + 1
            if hcnt > 100
                if cnt_ret_meda > 10
                    print("\n[Warning] Coordinate points preparation keeps failing. If this message keeps repeating, it may be because the necessary assumptions do not hold.")
                    tolmm = min(0.99,tolmm + 0.05)
                end
                seed_co = Xoshiro(rand(seed,0:UInt128(2)^128-1))
                cnt_ret_meda = cnt_ret_meda + 1
                return nothing, seed_co, cnt_ret_meda, tolmm
            else
                @goto fa2
            end
        else
            fcmed = copy(fcmed_temp)
        end
        if cntff > 500+200*(nth-1)
            if cnt_ret_meda > 10
                print("\n[Warning] Coordinate points preparation keeps failing. If this message keeps repeating, it may be because the necessary assumptions do not hold.")
                tolmm = min(0.99,tolmm + 0.05)
            end
            seed_co = Xoshiro(rand(seed,0:UInt128(2)^128-1))
            cnt_ret_meda = cnt_ret_meda + 1
            return nothing, seed_co, cnt_ret_meda, tolmm
        else
            cntff = cntff + 1
            @goto fa1
        end
    end
    @label fa3
    nemp = max(200,(numbeta(grid_model+1,nth-1) + robust*(nth-1))*50)*2
    seed2 = copy(seed1)
    hemp = empgen(seed2,fcmed,nemp,nth,n,mle_max_iter,bounded,positive,format,external,genera,likeli)
    cth_btv = hemp[:,filter(x->x!=jj,1:nth)]
    findiff = vec(median(cth_btv,dims=1)) .- fcmed[filter(x->x!=jj,1:nth)]

    return cth_btv, seed_co, cnt_ret_meda, tolmm
end

function stretchd(seed::Xoshiro,demp::Array{Float64},demp_cen::Vector{Float64},cth_btv::Array{Float64},th_btv::Array{Float64},gridq::Float64,qt::Float64,th_hat::Vector{Float64},qtleft1::Float64,jj::Int64,n::Int64,med_gen_pre::Int64,nth::Int64,format::Symbol,external::Array{Float64},mle_max_iter::Int64,bounded::Vector{Int64},positive::Vector{Int64},genera::Function,likeli::Function)
    nstr = 100
    closq = 0.95
    seed1 = copy(seed)
    ndem = size(demp,1)
    @label stre1
    ic1 = demp[rand(seed,1:ndem),:]
    if pseudo_sepm(cth_btv,ic1,closq) == true
        @goto stre1
    end
    seed2 = copy(seed1)
    im1 = mediantip(seed2,qt,ic1,Float64.(th_hat[filter(x->x!=jj,1:nth)]),qtleft1,jj,n,med_gen_pre,nth,format,external,mle_max_iter,bounded,positive,genera,likeli)
    speedd = zeros(nstr,nth-1)
    tmedd = zeros(nstr,nth-1)
    stdsd = zeros(nstr,nth-1)
    for i = 1:nstr
        strcnt = 1
        @label stre2
        ic2 = demp[rand(seed,1:ndem),:]
        if pseudo_sepm(cth_btv,ic2,closq) == true
            @goto stre2
        end
        for j = 1:(nth-1)
            if ic2[j] == ic1[j]
                if strcnt < 50
                    strcnt = strcnt + 1
                    @goto stre2
                else
                    print("\n[Warning] Coordinate points preparation keeps failing. If this message keeps repeating, it may be because the necessary assumptions do not hold.")
                    speedd = speedd[1:(i-1),:]
                    tmedd = tmedd[1:(i-1),:]
                    stdsd = stdsd[1:(i-1),:]
                    @goto str_ex
                end
            end
        end
        seed2 = copy(seed1)
        im2 = mediantip(seed2,qt,ic2,Float64.(th_hat[filter(x->x!=jj,1:nth)]),qtleft1,jj,n,med_gen_pre,nth,format,external,mle_max_iter,bounded,positive,genera,likeli)
        tmedd[i,:] = im2
        stdsd[i,:] = ic2 .- ic1
        speedd[i,:] = (im2 .- im1)./(ic2 .- ic1)
        ic1 = copy(ic2)
        im1 = copy(im2)
    end
    @label str_ex
    spv = zeros(nth-1)
    for i = 1:(nth-1)
        lb1 = quantile(tmedd[:,i],0.02)
        ub1 = quantile(tmedd[:,i],0.98)
        ix1 = findall(x->x>lb1&&x<ub1,tmedd[:,i])
        lbs = quantile(stdsd[:,i],0.1)
        ixs = findall(x->x>lbs,stdsd[:,i])
        if intersect(ix1,ixs) == []
            spp = [median(speedd)]
        else
            spp = speedd[intersect(ix1,ixs),i]
        end
        lb2 = quantile(spp,0.1)
        ub2 = quantile(spp,0.9)
        ix2 = findall(x->x>=lb2&&x<=ub2,spp)
        spv[i] = abs(mean(spp[ix2]))
    end
    speedd = copy(spv)
    th_btvc = th_btv[:,filter(x->x!=jj,1:nth)]
    sbandlb = zeros(nth-1)
    sbandub = zeros(nth-1)
    for jjc = 1:(nth-1)
        sbandlb[jjc] = quantile(th_btvc[:,jjc],(1-gridq)/2)
        sbandub[jjc] = quantile(th_btvc[:,jjc],1-(1-gridq)/2)
    end
    sband = sbandub .- sbandlb
    tbandlb = zeros(nth-1)
    tbandub = zeros(nth-1)
    for jjc = 1:(nth-1)
        tbandlb[jjc] = quantile(cth_btv[:,jjc],(1-gridq)/2)
        tbandub[jjc] = quantile(cth_btv[:,jjc],1-(1-gridq)/2)
    end
    tband = tbandub .- tbandlb

    crtio = (sband./tband)./speedd
    crtiod = 2.0.*(sband./tband)./speedd
    crindd = findall(x->x<1.0,crtio)
    crtio[crindd] = min.(1.0,crtiod[crindd])
    demp = dampenc_circ_res(demp,crtio,demp_cen,jj,positive,bounded)

    return demp, crtio
end

# test inversion for single-parameter
function exactci_ti(datax::Array{Float64},external::Array{Float64}=[0.0];
    alpha::Float64=0.05,
    generator::Function=nothing,likelihood::Function=nothing,
    appx_order::Int64=2,appx_order_sub::Int64=2,num_grid::Int64=0,
    num_par::Int64=0,mle_max_iter::Int64=0,mle_init::Vector{Float64}=[0.0],
    prob_tol::Float64=0.005,
    conv_tol::Float64=0.04*prob_tol,
    opt_max_iter::Int64=0,num_sim::Int64=0,
	positive::Vector{Int64}=Vector{Int64}(undef,0),
    bounded::Vector{Int64}=Vector{Int64}(undef,0),
    seed::Xoshiro=Xoshiro(0),format::Symbol=:Float64,robust::Int64=3)

    if generator==nothing
        println("[Error] Specify the generator function correctly")
        error()
    end
    if likelihood==nothing
        println("[Error] Specify the likelihood function correctly")
        error()
    end

    genera(th::Vector{Float64},n::Int64) = Float64.(generator(th,n))
    likeli(th::Vector{Float64},y::Array{Float64}) = -likelihood(th,y)
    genera(th::Vector{BigFloat},n::Int64) = BigFloat.(generator(th,n))
    likeli(th::Vector{BigFloat},y::Array{BigFloat}) = -likelihood(th,y)

    genera(th::Vector{Float64},n::Int64,external::Array{Float64}) = Float64.(generator(th,n,external))
    likeli(th::Vector{Float64},y::Array{Float64},external::Array{Float64}) = -likelihood(th,y,external)
    genera(th::Vector{BigFloat},n::Int64,external::Array{BigFloat}) = BigFloat.(generator(th,n,external))
    likeli(th::Vector{BigFloat},y::Array{BigFloat},external::Array{BigFloat}) = -likelihood(th,y,external)

    # set up variables
    grid_model = copy(appx_order)
    grid_sub = copy(appx_order_sub)
    if num_par==0
        nth = count_th(likelihood,Float64.(datax);external=external)
    else
        nth = copy(num_par)
    end
    if mle_max_iter == 0
        mle_max_iter = 200 + 50*nth
    end
    if num_grid==0
        mg = Int64(ceil(numbeta(grid_model+1,nth-1)^(1/(nth-1))))
    else
        mg = copy(num_grid)
    end
    if mle_init==[0]
        if format==:BigFloat
            initv = BigFloat.(ones(nth).*0.5)
        else
            initv = Float64.(ones(nth).*0.5)
        end
    else
        if format==:BigFloat
            initv = BigFloat.(copy(mle_init))
        else
            initv = Float64.(copy(mle_init))
        end
    end
    nmtol = copy(conv_tol)
    fftol = copy(prob_tol)
    if opt_max_iter==0
        ni_max = 500
    else
        ni_max = copy(opt_max_iter)
    end
    if num_sim==0
        m = Int(ceil(1/conv_tol))*1
    else
        m = copy(num_sim)
    end
	positive = copy(positive)	
    bounded = copy(bounded)
    if setdiff(positive,bounded) != positive
        println("[Error] Sign and range constraints on the same parameter")
        error()
    end
    gridq = 0.9
    mb = max(1000,Int64(ceil(10/(min(1-gridq,gridq)/2))))

    # parameter estimation
    if format==:BigFloat
        datax = BigFloat.(copy(datax))
        externalv = BigFloat.(copy(external))
    else
        datax = Float64.(copy(datax))
        externalv = Float64.(copy(external))
    end
    if external==[0.0]
        mcheck = zeros(Int64,1)
        th_hat = mle_est(datax,initv,mle_max_iter,bounded::Vector{Int64},positive::Vector{Int64},mcheck,likelihoodm=likeli)
    else
        mcheck = zeros(Int64,1)
        th_hat = mle_estx(datax,initv,mle_max_iter,bounded::Vector{Int64},positive::Vector{Int64},mcheck,external=externalv,likelihoodm=likeli)
    end

    n = size(datax,1)

    # bootstrap
    th_btv = empgen(seed,Float64.(th_hat),mb,nth,n,mle_max_iter,bounded,positive,format,external,genera,likeli)

    th_btv = th_btv[.!vec(any(isnan,th_btv;dims=2)),:]

    ithlb = zeros(nth)
    ithub = zeros(nth)
    for jj = 1:nth
        ithlb[jj] = quantile(th_btv[:,jj],(1-gridq)/2)
        ithub[jj] = quantile(th_btv[:,jj],1-(1-gridq)/2)
    end

    scalev = (ithub .- ithlb)./2.0./5.0

    anmed = Float64.(copy(th_hat))

    rvec0 = zeros(Int64,nth)
    for jj = 1:nth
        if th_hat[jj] >= 0
            rvec0[jj] = 1
        else
            rvec0[jj] = -1
        end
    end

    # test inversion confidence interval
    qtlefti = zeros(nth)
    for jj = 1:nth
		print(" Parameter ",jj,"/",nth,"...")
        qt = 1-alpha

        count_ff = 1
        @label rep_pl

        seed1 = rand(seed,0:UInt128(2)^128-1)
        ppp1(x::Vector{Float64}) = phi_ti(x,jj,qt,Float64.(th_hat),n,m,nth,grid_model,mle_max_iter,seed1,bounded,positive,scalev,anmed,external=external,generatorp=genera,likelihoodp=likeli,format=format)
        solp = optimize(ppp1,[itrns_one(Float64(th_hat[jj]),jj,Float64.(th_hat),scalev,bounded::Vector{Int64},positive::Vector{Int64})],NelderMead(),
                        Optim.Options(iterations=ni_max,g_tol=nmtol))
        if Optim.iterations(solp)==ni_max
            print("\n[Warning] Increase opt_max_iter if repeated. (Current: ",ni_max,")")
            @goto rep_pl
        end
        para = Optim.minimizer(solp)
        qtleft1 = trns_one(para[1],jj,Float64.(th_hat),scalev,bounded::Vector{Int64},positive::Vector{Int64})

        count_f = 1
        initp = [itrns_one(Float64(qtleft1[1]),jj,Float64.(th_hat),scalev,bounded,positive)]
        rbeg = 1.0
        phi_tix(x::Vector{Float64}) = phi_ti(x,jj,qt,Float64.(th_hat),n,m,nth,grid_model,mle_max_iter,seed1,bounded,positive,scalev,anmed,external=external,generatorp=genera,likelihoodp=likeli,format=format)
        @label rep_f
        solf_x, solf_info = newuoa(phi_tix,initp,maxfun=ni_max,ftarget=fftol,rhobeg=rbeg,scale=1.0.*ones(length(initp)))
        if solf_info.fx > fftol
            if count_f < 2
                initp = copy(solf_x)
                rbeg = rbeg/4
                count_f = count_f + 1
                @goto rep_f
            else
                if count_ff <= 3
                    count_ff = count_ff + 1
                    @goto rep_pl
                else
                    print("\n[Warning] Taking too much time. If this message keeps repeating, it may be because the confidence interval has reached the boundary of the parameter space.")
                    count_ff = 1
                    @goto rep_pl
                end
            end
        else
            qtleft1 = copy(solf_x)
        end

        qtlefti = trns_one(qtleft1[1],jj,Float64.(th_hat),scalev,bounded,positive)
        if (alpha < 0.5 && qtlefti[jj] >= th_hat[jj]) || (alpha > 0.5 && qtlefti[jj] <= th_hat[jj]) || (abs(qtlefti[jj]) > 200*max(abs(ithub[jj]),abs(ithlb[jj])))
            print("\n[Warning] Numerical optimization unstable. If this message keeps repeating, it may be because of the inaccurate likelihood function.")
                @goto rep_pl
        end

        print("...")
    end

    return th_hat, [qtlefti]
end

struct init_splx <: Optim.Simplexer end
function Optim.simplexer(S::init_splx, initnm::Vector{Float64})
    ncoef = length(initnm)
    init_vert = fill(initnm,ncoef+1)
    initnmi = copy(initnm)
    initnmi[1] = initnm[1] + 2.0
    init_vert[2] = initnmi
    for kk = 3:(ncoef+1)
        initnmi = copy(initnm)
        initnmi[kk-1] = initnm[kk-1] + 0.025
        init_vert[kk] = initnmi
    end
    return init_vert
end    

function exactci_base(datax::Array{Float64},external::Array{Float64}=[0.0];
    alpha::Float64=0.05,
    generator::Function=nothing,likelihood::Function=nothing,
    appx_order::Int64=2,appx_order_sub::Int64=2,num_grid::Int64=0,
    num_par::Int64=0,mle_max_iter::Int64=0,mle_init::Vector{Float64}=[0.0],
    prob_tol::Float64=0.005,
    conv_tol::Float64=0.04*prob_tol,
    opt_max_iter::Int64=0,num_sim::Int64=0,
	positive::Vector{Int64}=Vector{Int64}(undef,0),
    bounded::Vector{Int64}=Vector{Int64}(undef,0),
    seed::Xoshiro=Xoshiro(0),format::Symbol=:Float64,robust::Int64=3)

    if generator==nothing
        println("[Error] Specify the generator function correctly")
        error()
    end
    if likelihood==nothing
        println("[Error] Specify the likelihood function correctly")
        error()
    end

    genera(th::Vector{Float64},n::Int64) = Float64.(generator(th,n))
    likeli(th::Vector{Float64},y::Array{Float64}) = -likelihood(th,y)
    genera(th::Vector{BigFloat},n::Int64) = BigFloat.(generator(th,n))
    likeli(th::Vector{BigFloat},y::Array{BigFloat}) = -likelihood(th,y)

    genera(th::Vector{Float64},n::Int64,external::Array{Float64}) = Float64.(generator(th,n,external))
    likeli(th::Vector{Float64},y::Array{Float64},external::Array{Float64}) = -likelihood(th,y,external)
    genera(th::Vector{BigFloat},n::Int64,external::Array{BigFloat}) = BigFloat.(generator(th,n,external))
    likeli(th::Vector{BigFloat},y::Array{BigFloat},external::Array{BigFloat}) = -likelihood(th,y,external)

    # set up variables
    grid_model = copy(appx_order)
    grid_sub = copy(appx_order_sub)
    if num_par==0
        nth = count_th(likelihood,Float64.(datax);external=external)
    else
        nth = copy(num_par)
    end
    if mle_max_iter == 0
        mle_max_iter = 200 + 50*nth
    end
    if num_grid==0
        mg = Int64(ceil(numbeta(grid_model+1,nth-1)^(1/(nth-1))))
    else
        mg = copy(num_grid)
    end
    if mle_init==[0]
        if format==:BigFloat
            initv = BigFloat.(ones(nth).*0.5)
        else
            initv = Float64.(ones(nth).*0.5)
        end
    else
        if format==:BigFloat
            initv = BigFloat.(copy(mle_init))
        else
            initv = Float64.(copy(mle_init))
        end
    end
    nmtol = copy(conv_tol)
    fftol = copy(prob_tol)
    if opt_max_iter==0
        ni_max = Int64(round(200 + 50*numbeta(grid_model+1,nth-1);sigdigits=2))
    else
        ni_max = copy(opt_max_iter)
    end
    if num_sim==0
        m = Int(ceil(1/conv_tol))*1
    else
        m = copy(num_sim)
    end
    med_gen = max(200,Int64(ceil(10/min(alpha,1-alpha))))
    med_gen_pre = max(200,Int64(ceil(10/min(alpha,1-alpha))))
    if setdiff(positive,bounded) != positive
        println("[Error] Sign and range constraints on the same parameter")
        error()
    end
    gridq = 0.9
    mb = max(1000,Int64(ceil(10/(min(1-gridq,gridq)/2))))

    # define h function everywhere
    exx = genfvs(nth,grid_model,grid_sub)
    iqfunc = genfun(exx,[:x,:bt,:scalevi,:anmed])
    iqfunc_proc = genfun(exx,[:x,:bt,:scalevi,:anmed])
    iqfunc_big = genfun_big(exx,[:x,:bt,:scalevi,:anmed])
    iqfunc_big_proc = genfun_big(exx,[:x,:bt,:scalevi,:anmed])
    nw = nworkers()
    ids = workers()
	for i in 1:nw
        iqfunc_proc = remotecall_fetch(genfun,ids[i],exx,[:x,:bt,:scalevi,:anmed])
        iqfunc_big_proc = remotecall_fetch(genfun_big,ids[i],exx,[:x,:bt,:scalevi,:anmed])
    end

    # parameter estimation
    if format==:BigFloat
        datax = BigFloat.(copy(datax))
        externalv = BigFloat.(copy(external))
    else
        datax = Float64.(copy(datax))
        externalv = Float64.(copy(external))
    end
    if external==[0.0]
        mcheck = zeros(Int64,1)
        th_hat = mle_est(datax,initv,mle_max_iter,bounded::Vector{Int64},positive::Vector{Int64},mcheck,likelihoodm=likeli)
    else
        mcheck = zeros(Int64,1)
        th_hat = mle_estx(datax,initv,mle_max_iter,bounded::Vector{Int64},positive::Vector{Int64},mcheck,external=externalv,likelihoodm=likeli)
    end

    n = size(datax,1)

    # bootstrap
    th_btv = empgen(seed,Float64.(th_hat),mb,nth,n,mle_max_iter,bounded,positive,format,external,genera,likeli)

    th_btv = th_btv[.!vec(any(isnan,th_btv;dims=2)),:]
    mth_btv = th_btv .- mean(th_btv,dims=1)
    vth_btv = sum(mth_btv.^2,dims=1)
    bt_cor = (mth_btv'*mth_btv)./sqrt.(vth_btv'*vth_btv)   # correlation matrix

    ithlb = zeros(nth)
    ithub = zeros(nth)
    for jj = 1:nth
        ithlb[jj] = quantile(th_btv[:,jj],(1-gridq)/2)
        ithub[jj] = quantile(th_btv[:,jj],1-(1-gridq)/2)
    end

    scalev = (ithub .- ithlb)./2.0./5.0

    anmed = Float64.(copy(th_hat))

    rvec0 = zeros(Int64,nth)
    for jj = 1:nth
        if th_hat[jj] >= 0
            rvec0[jj] = 1
        else
            rvec0[jj] = -1
        end
    end

    # invariant quantile confidence interval
    qtlefti = zeros(nth)
    for jj = 1:nth
        print(" Parameter ",jj,"/",nth,"...")
        qt = 1-alpha

        bt_corc = bt_cor[filter(x->x!=jj,1:end),filter(x->x!=jj,1:end)]

        # Optimization Step 0

        count_ff = 1
        kc = 0
        cnt_fpe = 1
        @label rep_pl

        seed1 = rand(seed,0:UInt128(2)^128-1)
        ppp1(x::Vector{Float64}) = phi_con(iqfunc,x,jj,qt,Float64.(th_hat),n,m,nth,grid_model,mle_max_iter,seed1,bounded,positive,scalev,anmed,external=external,generatorp=genera,likelihoodp=likeli,format=format)
        solp = optimize(ppp1,[itrns_one(Float64(th_hat[jj]),jj,Float64.(th_hat),scalev,bounded::Vector{Int64},positive::Vector{Int64})],NelderMead(),
                        Optim.Options(iterations=ni_max,g_tol=nmtol))
        if Optim.iterations(solp)==ni_max
            print("\n[Warning] Maximum iteration reached. Increasing opt_max_iter may help. (Current: ",ni_max,")")
            @goto rep_pl
        end
        para = Optim.minimizer(solp)
        qtleft1 = trns_one(para[1],jj,Float64.(th_hat),scalev,bounded::Vector{Int64},positive::Vector{Int64})

        # coordinate points preparation

        seed_co = copy(seed)
        cxtime = time()
        cnt_ret_meda = 1
        cnt_fixc = 1
        tolmm = 0.01
        @label ret_med
        cth_btv = th_btv[findall(x->(sign(0.5-alpha)+(0.5==alpha))*x<=(sign(0.5-alpha)+(0.5==alpha))*median(th_btv[:,jj]),th_btv[:,jj]),filter(x->x!=jj,1:nth)]
        initm = vec(median(cth_btv,dims=1))
        seed1 = copy(seed_co)
        cmed, seed_co = tip_match(seed1,0,initm,th_btv[:,filter(x->x!=jj,1:nth)],Float64.(th_hat[filter(x->x!=jj,1:nth)]),qtleft1,jj,alpha,qt,nth,n,mb,med_gen_pre,format,external,mle_max_iter,bounded,positive,genera,likeli)
        
        initm = cmed[filter(x->x!=jj,1:nth)]
        cth_btv,seed_co,cnt_ret_meda,tolmm = med_match(seed,seed_co,cnt_ret_meda,tolmm,cmed,jj,grid_model,robust,mb,nth,n,mle_max_iter,bounded,positive,format,external,genera,likeli)
        if cth_btv == nothing
            @goto ret_med
        end

        demp = flatd_r(seed,cth_btv)
        demp_cen = cmed[filter(x->x!=jj,1:nth)]

        demp,crtio = stretchd(seed,demp,demp_cen,cth_btv,th_btv,gridq,qt,Float64.(th_hat),qtleft1,jj,n,med_gen_pre,nth,format,external,mle_max_iter,bounded,positive,genera,likeli)

        cth_btvv = dampenc_circ_res(cth_btv,crtio,demp_cen,jj,positive,bounded)
		
        rvec = rvec0[filter(x->x!=jj,1:end)]

        @label rep_0l
        seed_fx = copy(seed_co)
        tcord, tcord_med, seed_co = fixcoord(cth_btv,initm,iqfunc_big,iqfunc_big_proc,med_gen,qt,th_btv[:,filter(x->x!=jj,1:nth)],demp,cth_btvv,gridq,seed,seed_fx,nth-1,numbeta(grid_model+1,nth-1),jj,Float64.(th_hat[filter(x->x!=jj,1:end)]),qtleft1,Array(scalev[filter(x->x!=jj,1:end)]'),Array(anmed[filter(x->x!=jj,1:end)]'),nth,grid_model,bt_corc,n,mb,format,external,mle_max_iter,bounded,positive,genera,likeli)
        if tcord == nothing
            seed_co = Xoshiro(rand(seed,0:UInt128(2)^128-1))
            if cnt_fixc > 10
                print("\n[Warning] Taking too much time in finding coordinates.")
            end
            cnt_fixc = cnt_fixc + 1
            @goto ret_med
        end

        # reorder
        evenind = evenp(grid_model,nth)
        reoind = reorderc(iqfunc_big,tcord_med,evenind,Array(scalev[filter(x->x!=jj,1:end)]'),Array(anmed[filter(x->x!=jj,1:end)]'))
        tcord = tcord[reoind,:]
        tcord_med = tcord_med[reoind,:]

        coordz = mgetcoordz(iqfunc_big,numbeta(grid_model+1,nth-1),tcord_med,Float64.(th_hat[filter(x->x!=jj,1:end)]),Array(scalev[filter(x->x!=jj,1:end)]'),Array(anmed[filter(x->x!=jj,1:end)]'))

        sratio = 1.0
        cnts = 1
        initp = zeros(numbeta(grid_model+1,nth-1))
        initp[1] = qtleft1+th_hat[jj]
        initp = itrns_bt(iqfunc_big,initp,sratio,Float64.(th_hat),jj,bounded::Vector{Int64},positive::Vector{Int64},grid_model::Int64,grid_sub,nth::Int64,tcord,tcord_med,coordz,scalev,anmed,fftol,qt)
        initp = lincrop(initp,grid_model,grid_sub,nth)

        # Optimization Step 1 - linearized

        @label rep_1l

        cnt = 1
        cnt3 = 1
        cntc = 1
        seed1 = rand(seed,0:UInt128(2)^128-1)
        phi_linx(x::Vector{Float64}) = phi_lin(iqfunc,iqfunc_big,iqfunc_proc,x,sratio,fftol,jj,qt,Float64.(th_hat),tcord,tcord_med,coordz,n,m,mg,nth,grid_model,grid_sub,mle_max_iter,seed1,bounded,positive,rvec,scalev,anmed,external=external,generatorf=genera,likelihoodf=likeli,format=format)
        soln = optimize(phi_linx,initp,NelderMead(initial_simplex=init_splx()),
                        Optim.Options(iterations=ni_max,g_tol=nmtol))
        if Optim.iterations(soln)==ni_max
            print("\n[Warning] Maximum iteration reached. Increasing opt_max_iter may help. (Current: ",ni_max,")")

            seed_fx = copy(seed_co)
            tcord, tcord_med, seed_co = fixcoord(cth_btv,initm,iqfunc_big,iqfunc_big_proc,med_gen,qt,th_btv[:,filter(x->x!=jj,1:nth)],demp,cth_btvv,gridq,seed,seed_fx,nth-1,numbeta(grid_model+1,nth-1),jj,Float64.(th_hat[filter(x->x!=jj,1:end)]),qtleft1,Array(scalev[filter(x->x!=jj,1:end)]'),Array(anmed[filter(x->x!=jj,1:end)]'),nth,grid_model,bt_corc,n,mb,format,external,mle_max_iter,bounded,positive,genera,likeli)
            if tcord == nothing
                seed_co = Xoshiro(rand(seed,0:UInt128(2)^128-1))
                if cnt_fixc > 10
                    print("\n[Warning] Taking too much time in finding coordinates.")
                end
                cnt_fixc = cnt_fixc + 1
                @goto ret_med
            end
            
            # reorder
            evenind = evenp(grid_model,nth)
            reoind = reorderc(iqfunc_big,tcord_med,evenind,Array(scalev[filter(x->x!=jj,1:end)]'),Array(anmed[filter(x->x!=jj,1:end)]'))
            tcord = tcord[reoind,:]
            tcord_med = tcord_med[reoind,:]

            coordz = mgetcoordz(iqfunc_big,numbeta(grid_model+1,nth-1),tcord_med,Float64.(th_hat[filter(x->x!=jj,1:end)]),Array(scalev[filter(x->x!=jj,1:end)]'),Array(anmed[filter(x->x!=jj,1:end)]'))

            @goto rep_1l
        end
        if soln.minimum > fftol

            # Optimization Step 2 - fixed-point iteration

            initp = copy(soln.minimizer)
            initp = ilincrop(initp,grid_model,grid_sub,nth)
            fratio = 30.0

            fmaxiter = max(20,2*numbeta(grid_model+1,nth-1))
            cnt = 1
            cnt2 = 1
            @label rep_fl
            phi_fixx(x::Vector{Float64}) = phi_fix(fratio,iqfunc,iqfunc_big,iqfunc_proc,x,sratio,fftol,jj,qt,Float64.(th_hat),tcord,tcord_med,coordz,n,m,mg,nth,grid_model,grid_sub,mle_max_iter,seed1,bounded,positive,rvec,scalev,anmed,external=external,generatorf=genera,likelihoodf=likeli,format=format)
            solnp = fixed_point(phi_fixx,initp;Algorithm=:Anderson,MaxIter=fmaxiter,
                ConvergenceMetric=norm(input,output)=fixednorm(output.-input),
                ConvergenceMetricThreshold=fratio*fftol,ReplaceInvalids=:ReplaceVector)

            if string(solnp.TerminationCondition_) == "InvalidInputOrOutputOfIteration"
                if cnts < 15
                    sratio = sratio*0.25
                    fratio = fratio/sratio
                    cnts = cnts + 1
                    @goto rep_fl
                else
                    if cnt_fpe < 3
                        cnt_fpe = cnt_fpe + 1
                        @goto rep_pl
                    else
                        print("\n[Warning] Fixed-point iteration error. If this message keeps repeating, it may be because the likelihood function is inaccurate or the confidence interval has reached the boundary.")
                        @goto rep_pl
                    end
                end
            end

            # distance increasing
            if (fixednorm(solnp.Outputs_[:,end] .- solnp.Inputs_[:,end]) > fixednorm(solnp.Outputs_[:,1] .- solnp.Inputs_[:,1])) && (solnp.Convergence_/fratio > fftol)
                if cnt2 < 105
                    fratio = fratio*0.25
                    fmaxiter = max(20,2*numbeta(grid_model+1,nth-1))
                    cnt2 = cnt2 + 1
                    cntc = 1
                    @goto rep_fl
                else
                    print("\n[Warning] Taking too much time. Changing appx_order may help.")
                    @goto rep_0l
                end
            elseif ((fixednorm(solnp.Outputs_[:,end] .- solnp.Inputs_[:,end]) == fixednorm(solnp.Outputs_[:,1] .- solnp.Inputs_[:,1])) && (cntc == 1)) && (solnp.Convergence_/fratio > fftol)
                cntc = 2
                initp = solnp.Inputs_[:,end]
                fmaxiter = max(60,6*numbeta(grid_model+1,nth-1)*Int(ceil(0.005/prob_tol)))
                @goto rep_fl
            elseif ((fixednorm(solnp.Outputs_[:,end] .- solnp.Inputs_[:,end]) == fixednorm(solnp.Outputs_[:,1] .- solnp.Inputs_[:,1])) && (cntc == 2)) && (solnp.Convergence_/fratio > fftol)
                kc = kc + 1
                if kc == 5
                    kc = 0
                    print("\n[Warning] Taking too much time. Changing appx_order may help. If this message keeps repeating, it may be because the confidence interval has reached the boundary of the parameter space.")
                end
                @goto rep_pl

            # distance decreasing but not enough
            elseif solnp.Convergence_/fratio > fftol
                cntc = 1
                initp = solnp.Inputs_[:,end]
                fmaxiter = max(60,6*numbeta(grid_model+1,nth-1)*Int(ceil(0.005/prob_tol)))
                @goto rep_fl
            end

            # max iteration reached
            if solnp.TerminationCondition_==:ReachedMaxIter
                if cnt3 < 10
                    cnt3 = cnt3 + 1
                    cntc = 1
                    initp = solnp.Inputs_[:,end]   # should change
                        @goto rep_fl
                else
                    @goto rep_0l
                end
            end

            # converged
            qtleft = solnp.FixedPoint_
            soliter = solnp.Iterations_
        else
            qtleft = Optim.minimizer(soln)
            qtleft = ilincrop(qtleft,grid_model,grid_sub,nth)
            soliter = Optim.iterations(soln)
        end

        # Optimization Step 3 - robustness check

        kk = robust*(nth-1)   # number of robustness check points

        seed_rb = copy(seed_co)
        tcorda, tcord_meda = robcoord(iqfunc_big,med_gen,qt,th_btv[:,filter(x->x!=jj,1:nth)],demp,cth_btvv,gridq,seed,seed_rb,tcord,tcord_med,nth-1,kk,jj,Float64.(th_hat[filter(x->x!=jj,1:end)]),qtleft1,Array(scalev[filter(x->x!=jj,1:end)]'),Array(anmed[filter(x->x!=jj,1:end)]'),nth,grid_model,bt_corc,n,mb,format,external,mle_max_iter,bounded,positive,genera,likeli)
        tcordc = vcat(tcord,tcorda)
        tcord_medc = vcat(tcord_med,tcord_meda)
        coordz_f = mgetcoordz_f(iqfunc_big,numbeta(grid_model+1,nth-1),tcord_medc,Float64.(th_hat[filter(x->x!=jj,1:end)]),Array(scalev[filter(x->x!=jj,1:end)]'),Array(anmed[filter(x->x!=jj,1:end)]'))

        count_f = 1
        aa = trns_bt(iqfunc_big,qtleft,sratio,Float64.(th_hat),jj,bounded,positive,grid_model,grid_sub,nth,tcord,tcord_med,coordz,scalev,anmed,fftol,qt)
        initp = itrns_bt(iqfunc_big,aa,sratio,Float64.(th_hat),jj,bounded,positive,grid_model,grid_sub,nth,tcordc,tcord_medc,coordz_f,scalev,anmed,fftol,qt)
        rbeg = 1.0
        @label rep_f
        phi_robx(x::Vector{Float64}) = phi_rob(iqfunc,iqfunc_big,iqfunc_proc,x,sratio,fftol,jj,qt,Float64.(th_hat),tcordc,tcord_medc,coordz_f,n,m,mg,nth,grid_model,grid_sub,mle_max_iter,seed1,bounded,positive,rvec,scalev,anmed,external=external,generatorf=genera,likelihoodf=likeli,format=format)
        solf_x, solf_info = newuoa(phi_robx,initp,maxfun=ni_max,ftarget=fftol,rhobeg=rbeg,scale=1.0.*ones(length(initp)))
        if solf_info.fx > fftol
            if count_f < 2
                initp = copy(solf_x)
                rbeg = rbeg/4
                count_f = count_f + 1
                @goto rep_f
            else
                if count_ff <= 3
                    count_ff = count_ff + 1
                    @goto rep_pl
                else
                    print("\n[Warning] Taking too much time. Changing appx_order may help. If this message keeps repeating, it may be because the confidence interval has reached the boundary of the parameter space.")
                    count_ff = 1
                    @goto rep_pl
                end
            end
        else
            qtleft = copy(solf_x)
            soliter = solf_info.nf
        end

        # stability check
        bt = trns_bt(iqfunc_big,qtleft,sratio,Float64.(th_hat),jj,bounded::Vector{Int64},positive::Vector{Int64},grid_model::Int64,grid_sub,nth::Int64,tcordc,tcord_medc,coordz_f,scalev,anmed,fftol,qt)
        qtlefti[jj] = iqfunc_big(BigFloat.(Array(deleteat!(Float64.(copy(th_hat)),jj)')),BigFloat.(bt),BigFloat.(Array(scalev[filter(x->x!=jj,1:end)]')),BigFloat.(Array(anmed[filter(x->x!=jj,1:end)]')))[1] - th_hat[jj]
        if (alpha < 0.5 && qtlefti[jj] >= th_hat[jj]) || (alpha > 0.5 && qtlefti[jj] <= th_hat[jj]) || (abs(qtlefti[jj]) > 200*max(abs(ithub[jj]),abs(ithlb[jj])))
            print("\n[Warning] Numerical optimization unstable. If this message keeps repeating, it may be because of an inaccurate likelihood function.")
                @goto rep_pl
        end

        print("...")
    end

    return th_hat, qtlefti
end

function exactci_left(datax::Array{Float64},external::Array{Float64}=[0.0];
    alpha::Float64=0.05,
    generator::Function=nothing,likelihood::Function=nothing,
    appx_order::Int64=2,appx_order_sub::Int64=2,num_grid::Int64=0,
    num_par::Int64=0,mle_max_iter::Int64=0,mle_init::Vector{Float64}=[0.0],
    prob_tol::Float64=0.005,
    conv_tol::Float64=0.04*prob_tol,
    opt_max_iter::Int64=0,num_sim::Int64=0,
	positive::Vector{Int64}=Vector{Int64}(undef,0),
    bounded::Vector{Int64}=Vector{Int64}(undef,0),
    seed::Xoshiro=Xoshiro(0),format::Symbol=:Float64,robust::Int64=3)

    if alpha < 0.001
        println("[Error] alpha too small")
        error()
    end

    if num_par==0
        nth = count_th(likelihood,Float64.(datax);external=external)
    else
        nth = copy(num_par)
    end
    if mle_max_iter == 0
        mle_max_iter = 200 + 50*nth
    end
    #print("Number of coefficients: ",numbeta(appx_order+1,nth-1),"\n")

    likeli(th::Vector{Float64},y::Array{Float64}) = -likelihood(th,y)
    likeli(th::Vector{BigFloat},y::Array{BigFloat}) = -likelihood(th,y)

    likeli(th::Vector{Float64},y::Array{Float64},external::Array{Float64}) = -likelihood(th,y,external)
    likeli(th::Vector{BigFloat},y::Array{BigFloat},external::Array{BigFloat}) = -likelihood(th,y,external)

    if mle_init==[0]
        if format==:BigFloat
            initv = BigFloat.(ones(nth).*0.5)
        else
            initv = Float64.(ones(nth).*0.5)
        end
    else
        if format==:BigFloat
            initv = BigFloat.(copy(mle_init))
        else
            initv = Float64.(copy(mle_init))
        end
    end

    # parameter estimation
    if format==:BigFloat
        dataxv = BigFloat.(copy(datax))
        externalv = BigFloat.(copy(external))
    else
        dataxv = Float64.(copy(datax))
        externalv = Float64.(copy(external))
    end
    if external==[0.0]
        mcheck = zeros(Int64,1)
        th_hat = mle_est(dataxv,initv,mle_max_iter,bounded,positive,mcheck,likelihoodm=likeli)
    else
        mcheck = zeros(Int64,1)
        th_hat = mle_estx(dataxv,initv,mle_max_iter,bounded,positive,mcheck,external=externalv,likelihoodm=likeli)
    end
    print("Parameter estimates: ",round.(Float64.(th_hat),sigdigits=4),"\n")

    cp = 100*(1.0 - alpha)
    cp = rstrip("$cp",'0')
    cp = rstrip("$cp",'.')
    print("Computing left-sided $cp% confidence intervals\n")

    print("Lower bounds:")
    if nth > 1
        thhat,thci = exactci_base(datax,external;
        alpha=alpha,
        generator=generator,likelihood=likelihood,
        appx_order=appx_order,appx_order_sub=appx_order_sub,num_grid=num_grid,
        num_par=num_par,mle_max_iter=mle_max_iter,mle_init=mle_init,
        conv_tol=conv_tol,prob_tol=prob_tol,
        opt_max_iter=opt_max_iter,num_sim=num_sim,
        positive=positive,
        bounded=bounded,
        seed=seed,format=format,robust=robust)
    else
        thhat,thci = exactci_ti(datax,external;
        alpha=alpha,
        generator=generator,likelihood=likelihood,
        appx_order=appx_order,appx_order_sub=appx_order_sub,num_grid=num_grid,
        num_par=num_par,mle_max_iter=mle_max_iter,mle_init=mle_init,
        conv_tol=conv_tol,prob_tol=prob_tol,
        opt_max_iter=opt_max_iter,num_sim=num_sim,
        positive=positive,
        bounded=bounded,
        seed=seed,format=format,robust=robust)
    end

    print("\nLower bounds: ",round.(Float64.(thci),sigdigits=4),"\n")
    return Float64.(thhat), thci
end

function exactci_right(datax::Array{Float64},external::Array{Float64}=[0.0];
    alpha::Float64=0.05,
    generator::Function=nothing,likelihood::Function=nothing,
    appx_order::Int64=2,appx_order_sub::Int64=2,num_grid::Int64=0,
    num_par::Int64=0,mle_max_iter::Int64=0,mle_init::Vector{Float64}=[0.0],
    prob_tol::Float64=0.005,
    conv_tol::Float64=0.04*prob_tol,
    opt_max_iter::Int64=0,num_sim::Int64=0,
	positive::Vector{Int64}=Vector{Int64}(undef,0),
    bounded::Vector{Int64}=Vector{Int64}(undef,0),
    seed::Xoshiro=Xoshiro(0),format::Symbol=:Float64,robust::Int64=3)

    if alpha < 0.001
        println("[Error] alpha too small")
        error()
    end

    if num_par==0
        nth = count_th(likelihood,Float64.(datax);external=external)
    else
        nth = copy(num_par)
    end
    if mle_max_iter == 0
        mle_max_iter = 200 + 50*nth
    end
    #print("Number of coefficients: ",numbeta(appx_order+1,nth-1),"\n")

    likeli(th::Vector{Float64},y::Array{Float64}) = -likelihood(th,y)
    likeli(th::Vector{BigFloat},y::Array{BigFloat}) = -likelihood(th,y)

    likeli(th::Vector{Float64},y::Array{Float64},external::Array{Float64}) = -likelihood(th,y,external)
    likeli(th::Vector{BigFloat},y::Array{BigFloat},external::Array{BigFloat}) = -likelihood(th,y,external)

    if mle_init==[0]
        if format==:BigFloat
            initv = BigFloat.(ones(nth).*0.5)
        else
            initv = Float64.(ones(nth).*0.5)
        end
    else
        if format==:BigFloat
            initv = BigFloat.(copy(mle_init))
        else
            initv = Float64.(copy(mle_init))
        end
    end

    # parameter estimation
    if format==:BigFloat
        dataxv = BigFloat.(copy(datax))
        externalv = BigFloat.(copy(external))
    else
        dataxv = Float64.(copy(datax))
        externalv = Float64.(copy(external))
    end
    if external==[0.0]
        mcheck = zeros(Int64,1)
        th_hat = mle_est(dataxv,initv,mle_max_iter,bounded,positive,mcheck,likelihoodm=likeli)
    else
        mcheck = zeros(Int64,1)
        th_hat = mle_estx(dataxv,initv,mle_max_iter,bounded,positive,mcheck,external=externalv,likelihoodm=likeli)
    end
    print("Parameter estimates: ",round.(Float64.(th_hat),sigdigits=4),"\n")

    cp = 100*(1.0 - alpha)
    cp = rstrip("$cp",'0')
    cp = rstrip("$cp",'.')
    print("Computing right-sided $cp% confidence intervals\n")

    print("Upper bounds:")
    if nth > 1
        thhat,thci = exactci_base(datax,external;
        alpha=1-alpha,
        generator=generator,likelihood=likelihood,
        appx_order=appx_order,appx_order_sub=appx_order_sub,num_grid=num_grid,
        num_par=num_par,mle_max_iter=mle_max_iter,mle_init=mle_init,
        conv_tol=conv_tol,prob_tol=prob_tol,
        opt_max_iter=opt_max_iter,num_sim=num_sim,
        positive=positive,
        bounded=bounded,
        seed=seed,format=format,robust=robust)
    else
        thhat,thci = exactci_ti(datax,external;
        alpha=1-alpha,
        generator=generator,likelihood=likelihood,
        appx_order=appx_order,appx_order_sub=appx_order_sub,num_grid=num_grid,
        num_par=num_par,mle_max_iter=mle_max_iter,mle_init=mle_init,
        conv_tol=conv_tol,prob_tol=prob_tol,
        opt_max_iter=opt_max_iter,num_sim=num_sim,
        positive=positive,
        bounded=bounded,
        seed=seed,format=format,robust=robust)
    end

    print("\nUpper bounds: ",round.(Float64.(thci),sigdigits=4),"\n")
    return Float64.(thhat), thci
end

function exactci(datax::Array{Float64},external::Array{Float64}=[0.0];
    alpha::Float64=0.05,
    generator::Function=nothing,likelihood::Function=nothing,
    appx_order::Int64=2,appx_order_sub::Int64=2,num_grid::Int64=0,
    num_par::Int64=0,mle_max_iter::Int64=0,mle_init::Vector{Float64}=[0.0],
    prob_tol::Float64=0.005,conv_tol::Float64=0.04*prob_tol,
    opt_max_iter::Int64=0,num_sim::Int64=0,
	positive::Vector{Int64}=Vector{Int64}(undef,0),
    bounded::Vector{Int64}=Vector{Int64}(undef,0),
    seed::Xoshiro=Xoshiro(0),format::Symbol=:Float64,robust::Int64=3)

    if alpha >= 0.5
        println("[Error] alpha should be smaller than 0.5.")
        error()
    end
    if alpha < 0.002
        println("[Error] alpha too small")
        error()
    end

    if num_par==0
        nth = count_th(likelihood,Float64.(datax);external=external)
    else
        nth = copy(num_par)
    end
    if mle_max_iter == 0
        mle_max_iter = 200 + 50*nth
    end
    #print("Number of coefficients: ",numbeta(appx_order+1,nth-1),"\n")

    likeli(th::Vector{Float64},y::Array{Float64}) = -likelihood(th,y)
    likeli(th::Vector{BigFloat},y::Array{BigFloat}) = -likelihood(th,y)

    likeli(th::Vector{Float64},y::Array{Float64},external::Array{Float64}) = -likelihood(th,y,external)
    likeli(th::Vector{BigFloat},y::Array{BigFloat},external::Array{BigFloat}) = -likelihood(th,y,external)

    if mle_init==[0]
        if format==:BigFloat
            initv = BigFloat.(ones(nth).*0.5)
        else
            initv = Float64.(ones(nth).*0.5)
        end
    else
        if format==:BigFloat
            initv = BigFloat.(copy(mle_init))
        else
            initv = Float64.(copy(mle_init))
        end
    end

    # parameter estimation
    if format==:BigFloat
        dataxv = BigFloat.(copy(datax))
        externalv = BigFloat.(copy(external))
    else
        dataxv = Float64.(copy(datax))
        externalv = Float64.(copy(external))
    end
    if external==[0.0]
        mcheck = zeros(Int64,1)
        th_hat = mle_est(dataxv,initv,mle_max_iter,bounded,positive,mcheck,likelihoodm=likeli)
    else
        mcheck = zeros(Int64,1)
        th_hat = mle_estx(dataxv,initv,mle_max_iter,bounded,positive,mcheck,external=externalv,likelihoodm=likeli)
    end
    print("Parameter estimates: ",round.(Float64.(th_hat),sigdigits=4))

    cp = 100*(1.0 - alpha)
    cp = rstrip("$cp",'0')
    cp = rstrip("$cp",'.')
    print("\nComputing two-sided $cp% confidence intervals")

    print("\nLower bounds:")
    if nth > 1
    th_hat, qtleft = exactci_base(datax,external;
    alpha=alpha/2,
    generator=generator,likelihood=likelihood,
    appx_order=appx_order,appx_order_sub=appx_order_sub,num_grid=num_grid,
    num_par=num_par,mle_max_iter=mle_max_iter,mle_init=mle_init,
    conv_tol=conv_tol,prob_tol=prob_tol,
    opt_max_iter=opt_max_iter,num_sim=num_sim,
	positive=positive,
    bounded=bounded,
    seed=seed,format=format,robust=robust)
    else
    th_hat, qtleft = exactci_ti(datax,external;
    alpha=alpha/2,
    generator=generator,likelihood=likelihood,
    appx_order=appx_order,appx_order_sub=appx_order_sub,num_grid=num_grid,
    num_par=num_par,mle_max_iter=mle_max_iter,mle_init=mle_init,
    conv_tol=conv_tol,prob_tol=prob_tol,
    opt_max_iter=opt_max_iter,num_sim=num_sim,
	positive=positive,
    bounded=bounded,
    seed=seed,format=format,robust=robust)
    end

    print("\nUpper bounds:")
    if nth > 1
    th_hat, qtright = exactci_base(datax,external;
    alpha=1-alpha/2,
    generator=generator,likelihood=likelihood,
    appx_order=appx_order,appx_order_sub=appx_order_sub,num_grid=num_grid,
    num_par=num_par,mle_max_iter=mle_max_iter,mle_init=mle_init,
    conv_tol=conv_tol,prob_tol=prob_tol,
    opt_max_iter=opt_max_iter,num_sim=num_sim,
	positive=positive,
    bounded=bounded,
    seed=seed,format=format,robust=robust)
    else
    th_hat, qtright = exactci_ti(datax,external;
    alpha=1-alpha/2,
    generator=generator,likelihood=likelihood,
    appx_order=appx_order,appx_order_sub=appx_order_sub,num_grid=num_grid,
    num_par=num_par,mle_max_iter=mle_max_iter,mle_init=mle_init,
    conv_tol=conv_tol,prob_tol=prob_tol,
    opt_max_iter=opt_max_iter,num_sim=num_sim,
	positive=positive,
    bounded=bounded,
    seed=seed,format=format,robust=robust)
    end

    print("\nLower bounds: ",round.(Float64.(qtleft),sigdigits=4))
    print("\nUpper bounds: ",round.(Float64.(qtright),sigdigits=4),"\n")
    return Float64.(th_hat), [qtleft qtright]
end

end


