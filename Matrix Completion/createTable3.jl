using JuMP, LinearAlgebra, Suppressor, StatsBase, CSV, DataFrames, Compat, Test, Random#, CPLEX
include("alt_min.jl")
include("convexpenalties.jl")


results_template = DataFrame(n=Int[], r=Int[], p=Real[], err_GD_msk=Real[], err_GD_msk_am=Real[], t_gd_msk=Real[], t_gd_msk_am=Real[])
results_exact=similar(results_template, 0)


for ARG in ARGS
    array_num = parse(Int, ARG)
    for n in [50*array_num] #1-12
        p=0.25
        r=5
        gamma=20/p
        M_L=randn(n, r)
        M_R=randn(n, r)
        M=M_L*M_R'

        numEntriesSampled=Int(round(p*n^2))
        coords=[repeat(1:n, inner=size(1:n,1)) repeat(1:n, outer=size(1:n,1))];
        entries=sample(1:n^2, Int(numEntriesSampled), replace=false)

        theIndices=zeros(n,n)
        for t=1:Int(numEntriesSampled)
            theIndices[coords[entries[t],1], coords[entries[t],2]]=1.0
        end

        k=r
        t_GD_msk=@elapsed X_GD,LB, UB, Y_relax=getMatrixFrobeniusNorm_greedy(coords, entries, M, n, numEntriesSampled, k, gamma)

        @show "SDP lower bound is: " LB
        u,sigma,v,=svd(X_GD)
        @show rnk_GD=sum(sigma.>1e-2);
        @show err_GD_msk = norm(X_GD.-M)/norm(M);

        t_GD_msk_am=@elapsed X_GD=alternatingminimization_matrixcompletion_solvesdprelax(coords, entries, M, n, numEntriesSampled, k, gamma)
        @show t_GD_msk_am
        u,sigma,v,=svd(X_GD)
        @show rnk_GD=sum(sigma.>1e-2);
        @show err_GD_am = norm(X_GD.-M)/norm(M);


        push!(results_exact, [n, r, p, err_GD_msk, err_GD_am,t_GD_msk, t_GD_msk_am])

    end
end

CSV.write("results_evaluatescalability.csv", results_exact, append=true)
