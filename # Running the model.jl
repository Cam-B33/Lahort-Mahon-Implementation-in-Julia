# Running the model

include("# Occupancy Model in Julia.jl")

### 2016 eDNA DATA load

edna.data.Positive_PCR .= ifelse.(ismissing.(edna.data.Positive_PCR), 0, edna.data.Positive_PCR)
edna.data = df_all_samples


### 2016 NETTING DATA load
netting.data = eFish_df


# set up
S = 54 # number of sites
V = fill(9, 54) # number of eDNA samples
L = fill(8, 54, 9) # number of qPCRs
K = fill(0, S) # number of dip netting occasions
D = reshape(edna.data.Positive_PCR, 54, 9)' # number of positive qPCR detections for each site and eDNA sample
Du = zeros(S) # as.numeric(head(netting.data.Presence, 54) + tail(netting.data.Presence, 54))
tmpL = fill(8, 182) # for this species, we ran 8 equip blanks and 4 PCRs per sample
tmpU = fill(0.329, 182) # none were positive - set up empty vector of 0s
caldata1 = Dict("CalL1" => tmpL, "CalD1" => tmpU) # caldata format for c1
caldata2 = [256, 0] # c2 (no PCR blanks were positive - did 3 pcrs per plate * 4 plates)

mydata = Dict("V" => V, "L" => L, "K" => 0, "D" => D, "Du" => Du, "caldata1" => caldata1, "caldata2" => caldata2)
meanstarts = fill(0.5, 6)
fixpar = fill(NaN, 6)

D
# fit model using 100 random starts - model with lowest likelihood (i.e. theLLvals == 0.00) is final model
fit_model(mydata, fixpar, nstarts=100, meanstarts=[1, 2, 3], sdstart=1, method="Nelder-Mead", dohess=false)

# or fit model using steps of 0.001 and profile likelihood
profCI = get_profCIall(mydata, fixpar, nstarts=25, sdstart=2, thestep=0.001, method="Nelder-Mead")

plot_profCIall(profCI["profliks"], profCI["theseq"], mypars=fill(NaN, 6), fixpar, ylimZoom=false)

myests = fill(NaN, 6, 3)
for ii in 1:6
    temp = profCI["profliks"][ii, :] .- minimum(profCI["profliks"][ii, :])
    myests[ii, 1] = profCI["theseq"][temp .== minimum(temp)]
    myests[ii, 2:3] = extrema(profCI["theseq"][temp .< (3.84/2)])
end
println(myests)

mypar = [0.630, 0.444, 0.014, 0.854, 0.028, 0.97]
detect = psicond_FPMS(mypar, mydata)

detections_prob = DataFrame(detect)

detections_prob