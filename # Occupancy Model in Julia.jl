function loglikf_FPMS(params, mydata, fixpar)
    V = mydata[:V]
    L = mydata[:L]
    D = mydata[:D]
    K = mydata[:K]
    Du = mydata[:Du]
    caldata1 = mydata[:caldata1]
    caldata2 = mydata[:caldata2]
    
    npar = 1
    if isnan(fixpar[1])
        psi = 1 / (1 + exp(-params[npar]))
        npar += 1
    else
        psi = fixpar[1]
    end
    
    if isnan(fixpar[2])
        theta11 = 1 / (1 + exp(-params[npar]))
        npar += 1
    else
        theta11 = fixpar[2]
    end
    
    if isnan(fixpar[3])
        theta10 = 1 / (1 + exp(-params[npar]))
        npar += 1
    else
        theta10 = fixpar[3]
    end
    
    if isnan(fixpar[4])
        p11 = 1 / (1 + exp(-params[npar]))
        npar += 1
    else
        p11 = fixpar[4]
    end
    
    if isnan(fixpar[5])
        p10 = 1 / (1 + exp(-params[npar]))
        npar += 1
    else
        p10 = fixpar[5]
    end
    
    if isnan(fixpar[6])
        r11 = 1 / (1 + exp(-params[npar]))
    else
        r11 = fixpar[6]
    end
    
    # field data component (ambiguous and unambiguous method)
    ND = L - D # number of non detections (ambiguous method)
    tmpZ2_1 = p11^D * (1 - p11)^ND
    tmpZ2_0 = p10^D * (1 - p10)^ND
    tmp = theta11 * tmpZ2_1 + (1 - theta11) * tmpZ2_0
    tmpZ1_1 = prod(tmp, dims=1)
    tmp = theta10 * tmpZ2_1 + (1 - theta10) * tmpZ2_0
    tmpZ1_0 = prod(tmp, dims=1)
    NDu = K - Du # number of non detections (unambiguous method)
    tmpZ1_1u = r11^Du * (1 - r11)^NDu
    tmpZ1_0u = zeros(Du) .+ 1^NDu
    loglik = sum(log.(psi .* tmpZ1_1 .* tmpZ1_1u .+ (1 - psi) .* tmpZ1_0 .* tmpZ1_0u))
    
    # calibration data component (equipment level)
    caldata1_U = caldata1[:CalL1] .- caldata1[:CalD1]
    tmp = theta10 .* p11.^caldata1[:CalD1] .* (1 .- p11).^caldata1_U .+
          (1 - theta10) .* p10.^caldata1[:CalD1] .* (1 .- p10).^caldata1_U
    loglik += sum(log.(tmp))
    
    # calibration data component (PCR level)
    if caldata2[1] > 0
        loglik += caldata2[2] * log(p10) .+ (caldata2[1] - caldata2[2]) * log(1 - p10)
    end
    
    # change sign for minimization
    loglik = -loglik
end



function fit_model(mydata, fixpar, nstarts=10, meanstarts, sdstart=1, method="Nelder-Mead", dohess=false)
    parshat_m = fill(fixpar, (nstarts, 6)) # to hold estimates from each optimization
    theLLvals = fill(NaN, nstarts) # to hold the loglikelihood value from each optimization
    optimres = Vector{Any}(undef, nstarts) # to store complete model output from optim in each optimization
    
    for ii in 1:nstarts
        mystarts = randn(length(meanstarts)) .* sdstart .+ meanstarts # init to given values + noise
        tmp = optimize(p -> loglikf_FPMS(p, mydata, fixpar), mystarts, method, autodiff=:forward, hessian=dohess)
        theLLvals[ii] = tmp.minimum
        parshat_m[ii, .!isnan.(fixpar)] .= round.(plogis.(tmp.minimizer[.!isnan.(fixpar)]), digits=3)
        optimres[ii] = tmp
    end
    
    diff = theLLvals .- minimum(theLLvals)
    allres = hcat(parshat_m, theLLvals, round.(diff, digits=2))
    return Dict("myMLEs" => optimres[argmin(theLLvals)], "allres" => allres)
end

function get_profCIall(mydata, fixpar, nstarts=10, sdstart=2, method="Nelder-Mead", thestep=0.02)
    parnames = ["psi", "theta11", "theta10", "p11", "p10", "r11"]
    npar = count(isnan.(fixpar)) # total number of parameters in the model (those not fixed)
    meanstarts = zeros(npar - 1) # optim has one dimension less than the total number of parameters
    theseq = 0.01:thestep:0.99 # parameter values to sweep through
    profliks = fill(NaN, 6, length(theseq)) # to hold the profile loglik function values
    
    for ii in 1:6
        if isnan(fixpar[ii]) # this is a parameter for which to calculate the prof loglik
            for jj in 1:length(theseq)
                println(parnames[ii], "=", theseq[jj])
                fixpar2 = copy(fixpar)
                fixpar2[ii] = theseq[jj] # fix the parameter to the corresponding value
                m1 = fit_model(mydata, fixpar2, nstarts=nstarts, meanstarts, sdstart=sdstart, method=method)
                profliks[ii, jj] = m1["myMLEs"].minimum # keep value of ll function at the maximum
            end
        end
    end
    
    return Dict("profliks" => profliks, "theseq" => theseq)
end

function plot_profCIall(profliks, theseq, mypars, fixpar, ylimZoom=true, parnames=["psi", "theta11", "theta10", "p11", "p10", "r11"])
    for ii in 1:6
        if isnan(fixpar[ii])
            proflik = profliks[ii, :]
            themin = minimum(proflik)
            proflik .= proflik .- themin
            if ylimZoom
                myYlim = (-10, 0) # fix axes to this range
            else
                myYlim = (-max(max(proflik), 3.84), 0) # adjust for each case, but at least show 3.84 units
            end
            plot(theseq, -proflik, seriestype=:line, ylab="", xlab="", ylim=myYlim, xticks=[0, 0.5, 1], xtickfont=font(7))
            title!(parnames[ii], loc=:center, pad=2, font=font(7))
            scatter!(theseq, -proflik, markershape=:circle, markercolor=:black)
            IDin = findall(proflik .< 3.84/2)
            scatter!(theseq[IDin], -proflik[IDin], markershape=:circle, markercolor=:green)
            vline!([mypars[ii]], linecolor=:gray)
        end
    end
end

function psicond_FPMS(mypar, mydata)
    psi, theta11, theta10, p11, p10, r11 = mypar
    L, D, K, Du = mydata.L, mydata.D, mydata.K, mydata.Du
    
    # probability calculations (as in function computing the likelihood)
    ND = L - D # number of non detections (ambiguous method)
    tmpZ2_1 = p11.^D .* (1 .- p11).^ND
    tmpZ2_0 = p10.^D .* (1 .- p10).^ND
    tmp = theta11 .* tmpZ2_1 .+ (1 .- theta11) .* tmpZ2_0
    tmpZ1_1 = prod(tmp, dims=2)
    tmp = theta10 .* tmpZ2_1 .+ (1 .- theta10) .* tmpZ2_0
    tmpZ1_0 = prod(tmp, dims=2)
    NDu = K .- Du # number of non detections (unambiguous method)
    tmpZ1_1u = r11.^Du .* (1 .- r11).^NDu
    tmpZ1_0u = zeros(size(tmpZ1_1u))
    
    # numerator and denominator for the calculation of the conditional probability of presence
    mynum = psi .* tmpZ1_1 .* tmpZ1_1u # Pr(Z1=1 & survey data)
    myden = psi .* tmpZ1_1 .* tmpZ1_1u .+ (1 .- psi) .* tmpZ1_0 .* tmpZ1_0u # Pr(survey data)
    
    # conditional probability of presence
    psicond = mynum ./ myden # Pr(Z1=1|survey data)
    println(round.(psicond, digits=3)) # print the probability for each of the sampled sites
end