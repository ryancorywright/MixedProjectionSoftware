using JuMP, Gurobi, LinearAlgebra, Suppressor, StatsBase, Compat, Mosek, DataFrames, CSV, Random
include("convex_penalties_edm.jl")
include("exact_edm.jl")

results_template = DataFrame(n=Int[], r=Real[], t_sdp=Real[], err_sdpdtr=Real[], nocorr_sdp=Int[], t_ex=Real[], err_exdtr=Real[], nocorr_ex=Int[], MIPGap=Real[], nodeCountEX=Int[], t_oa=Real[], err_oadtr=Real[], nocorr_oa=Int[], MIPGapOA=Real[], nodeCountOA=Int[], cutCountOA=Int[])

results_edmradio_ex=similar(results_template, 0)

# Gram matrix recovery test problem
r=2
d_radio=0.3 # You will need to vary this to reproduce all of the results in Table 6
n=30  # You will need to vary this to reproduce all of the results in Table 6
lambda=1.0*n^2
gamma=1.0/n

for ARG in ARGS #1-4
    d_radio=[0.1, 0.2, 0.3, 0.4][parse(Int, ARG)]

    xs=rand(n, r).-0.5 #generate n random points in r dimensional space uniformly distributed on [-0.5, 0.5]^r
    D=zeros(n,n) #the matrix we would like to recover.
    for i=1:n #in a loop is slightly slower, but unimportant
        for j=1:(i-1)
            D[i,j]=max(0.0, norm(xs[i,:].-xs[j,:])^2+0.01*(randn(1)[1])) #Corrupt the distance with some noise, in a symmetric fashion
            D[j,i]=D[i,j]
        end
    end



    # Get observed set of coordinates
    coords=[repeat(1:n, inner=size(1:n,1)) repeat(1:n, outer=size(1:n,1))];
    entries=Int[]
    for i=1:n^2
        if norm(D[coords[i,1], coords[i,2]])<=d_radio^2
            push!(entries, i) #Remark: D_ij=D_ji (including noise), so this ensures that if D_ij is in the radio range, so is D_ji; no need to account for symmetry directly
        end
    end

    numEntriesSampled=size(entries,1)


    # Solve the sdp relaxation
    t_sdp=@elapsed G_imputed, Y_relax=getMatrixTraceNorm_EDM_Gram(coords, entries, D, n, numEntriesSampled, d_radio, lambda, gamma, r);
    u,v,=svd(Y_relax)
    Y_ws=u[:,1:r]*u[:,1:r]'

    D_imputed=diag(G_imputed)*ones(n)'+ones(n)*diag(G_imputed)'.-2*G_imputed
    @show err_sdpDTR=norm(D_imputed.-D)/norm(D)
    @show sdp_corr=sum(abs.(D.-D_imputed).<1e-2)
    @show sdp_incorr =sum(abs.(D.-D_imputed).>=1e-2)

    # Commented out as the submitted version of the paper only includes multi tree
    t_oa=@elapsed G_imputed,nodecountOA, cutCountOA, miogapOA,=getMatrixFrobeniusNorm_cuttingplanesmultitree_exact_inout(coords, entries, D, n, numEntriesSampled, lambda, d_radio, r, gamma, Y_ws, Y_relax);
    D_imputed=diag(G_imputed)*ones(n)'+ones(n)*diag(G_imputed)'.-2*G_imputed
    @show err_oaDTR=norm(D_imputed.-D)/norm(D)
    @show oa_corr=sum(abs.(D.-D_imputed).<1e-2)
    @show oa_incorr =sum(abs.(D.-D_imputed).>=1e-2)

    # Commented out as the submitted version of the paper only includes multi tree
    t_oa=-1.0#@elapsed G_imputed,nodecountOA, cutCountOA, miogapOA,=getMatrixFrobeniusNorm_cuttingplanes_exact_inout(coords, entries, D, n, numEntriesSampled, lambda, d_radio, r, gamma, Y_ws, Y_relax);
    # D_imputed=diag(G_imputed)*ones(n)'+ones(n)*diag(G_imputed)'.-2*G_imputed
    @show err_oaDTR=-1.0#norm(D_imputed.-D)/norm(D)
    @show oa_corr=-1.0#sum(abs.(D.-D_imputed).<1e-2)
    @show oa_incorr =-1.0#sum(abs.(D.-D_imputed).>=1e-2)
    miogapOA=-1.0; nodeCountOA=-1; cutCountOA=-1

    t_ex=-1.0@elapsed G_imputed,miogap,nodeCountEX=getMatrixExact_EDM_Gram(coords, entries, D, n, r, numEntriesSampled, d_radio, lambda, gamma, D_imputed);
    # D_imputed=diag(G_imputed)*ones(n)'+ones(n)*diag(G_imputed)'.-2*G_imputed
    @show err_exDTR=-1.0#norm(D_imputed.-D)/norm(D)
    @show ex_corr=-1.0#sum(abs.(D.-D_imputed).<1e-2)
    @show ex_incorr =-1.0#sum(abs.(D.-D_imputed).>=1e-2)
    miogap=-1.0; nodeCountEX=-1


    push!(results_edmradio_ex, [n, r, t_sdp, err_sdpDTR, sdp_corr, t_ex, err_exDTR, ex_corr, miogap, nodeCountEX, t_oa, err_oaDTR, oa_corr, miogapOA, nodecountOA, cutCountOA])
end
CSV.write("oavsgurobi_edmgram.csv", results_edmradio_ex, append=true)
