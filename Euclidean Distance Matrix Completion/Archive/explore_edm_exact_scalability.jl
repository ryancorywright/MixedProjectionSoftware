using JuMP, Gurobi, LinearAlgebra, Suppressor, StatsBase, Compat, Mosek, DataFrames, CSV
include("convex_penalties_edm.jl")
include("exact_edm.jl")

results_template = DataFrame(n=Int[], xr=Real[], t_sdp=Real[], err_sdpdtr=Real[], err_sdpxtr=Real[], nocorr_sdp=Int[], t_ex=Real[], err_exdtr=Real[], err_exxtr=Real[], nocorr_ex=Int[], MIPGap=Real[])

results_edmradio=similar(results_template, 0)


# Radio test problem alla Biswas and Ye (2005)
r=0.2
n=30 #Number of points located on the grid
m=15 #Number of anchors located on the grid

xs=rand(n, 2).-0.5 #generate n random points in 2 dimensional space uniformly distributed on [-0.5, 0.5)
ys=rand(m, 2).-0.5 #generate m random anchor points in 2 dimensional space uniformly distributed on [-0.5, 0.5)
zs=[ys; xs]
D=zeros(n+m,n+m) #the matrix we would like to recover.
for i=1:(n+m) #in a loop is slightly slower, but unimportant
    for j=1:(i-1)
        D[i,j]=norm(zs[i,:].-zs[j,:])^2
        D[j,i]=norm(zs[i,:].-zs[j,:])^2
    end
end



# Get observed set of coordinates
coords=[repeat(1:(n+m), inner=size(1:(n+m),1)) repeat(1:(n+m), outer=size(1:(n+m),1))];
entries=Int[]
for i=1:(n+m)^2
    if norm(D[coords[i,1], coords[i,2]])<=r^2
        push!(entries, i)
    end
end

numEntriesSampled=size(entries,1)

for iter=1:1
    # Solve the sdp relaxation
    t_sdp=@elapsed X_imputed, G_imputed,X_full,=getMatrixTraceNorm_EDM(coords, entries, D, n, numEntriesSampled, ys, m);

    @show err_sdpXTR=norm(X_imputed.-xs)/norm(xs);
    D_imputed=diag(G_imputed)*ones(n)'+ones(n)*diag(G_imputed)'.-2*G_imputed
    @show err_sdpDTR=norm(D_imputed.-D[m+1:end, m+1:end])/norm(D[m+1:end, m+1:end])
    @show sdp_corr=sum(abs.(xs.-X_imputed)*ones(2).<1e-2)
    @show sdp_incorr =sum(abs.(xs.-X_imputed)*ones(2).>=1e-2)

    # Test out an exact approach (using e.g. Gurobi)
    t_ex=@elapsed X_imputed, G_imputed,miogap,=getMatrixExact_EDM(coords, entries, D, n, numEntriesSampled, ys, m, X_full);
    @show err_exXTR=norm(X_imputed.-xs)/norm(xs);
    D_imputed=diag(G_imputed)*ones(n)'+ones(n)*diag(G_imputed)'.-2*G_imputed
    @show err_exDTR=norm(D_imputed.-D[m+1:end, m+1:end])/norm(D[m+1:end, m+1:end])
    @show ex_corr=sum(abs.(xs.-X_imputed)*ones(2).<1e-2)
    @show ex_incorr =sum(abs.(xs.-X_imputed)*ones(2).>=1e-2)


    push!(results_edmradio, [n, m, r, t_sdp, err_sdpDTR, err_sdpXTR, sdp_corr, t_ex, err_exDTR, err_exXTR, ex_corr, miogap])
end
CSV.write("sdpvsexactgurobi_edm.csv", results_edmradio, append=true)
