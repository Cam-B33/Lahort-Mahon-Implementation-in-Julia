# Occupancy Model in Julia
# Basic For Loop in Julia
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
    
    ND = L - D
    tmpZ2_1 = p11^D * (1 - p11)^ND
    tmpZ2_0 = p10^D * (1 - p10)^ND
    tmp = theta11 * tmpZ2_1 + (1 - theta11) * tmpZ2_0
    tmpZ1_1 = prod(tmp, dims=1)
    tmp = theta10 * tmpZ2_1 + (1 - theta10) * tmpZ2_0
    tmpZ1_0 = prod(tmp, dims=1)
    NDu = K - Du
    tmpZ1_1u = r11^Du * (1 - r11)^NDu
    tmpZ1_0u = 0^Du * 1^NDu
    loglik = sum(log.(psi * tmpZ1_1 .* tmpZ1_1u .+ (1 - psi) * tmpZ1_0 .* tmpZ1_0u))
    
    caldata1_U = caldata1[:CalL1] - caldata1[:CalD1]
    tmp = theta10 * p11^caldata1[:CalD1] .* (1 - p11).^caldata1_U .+
          (1 - theta10) * p10^caldata1[:CalD1] .* (1 - p10).^caldata1_U
    loglik += sum(log.(tmp))
    
    if caldata2[1] > 0
        loglik += caldata2[2] * log(p10) + (caldata2[1] - caldata2[2]) * log(1 - p10)
    end
    
    loglik = -loglik
    
  end

# Adding optim function
#Fisrt load optim package
using Pkg
Pkg.add("Optim")
using Optim
Pkg.add("Random")
using Random

#Creating fit model function
  function fit_model(mydata, fixpar, nstarts = 10, meanstarts, sdstart = 1, method = NelderMead, dohess = F){
 # object to hold estimates from each optimization
 parshat_m = fixpar
 theLLvals = repeat(NA, nstarts)
 #Create an empty list to store complte model output
    optimres = list()
    
    for ii in 1:nstarts{
        #initilaize given values + noise
        #randomly sample values from a normal distribution,  with length = meanstarts mean = meanstarts and sd = sdstart
       n = length(meanstarts)   
        mystarts = randn(n) .* sdstart .+ meanstarts
        tmp = Optim.optimize(x -> loglikf_FPMS(x, mydata, fixpar), mystarts, method, autodiff=:forward,
        theLLvals[ii] = tmp}})