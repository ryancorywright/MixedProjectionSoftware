#ENV["MOSEKLM_LICENSE_FILE"]="/home/ryancw/mosek/mosek.lic" #needed on cluster
using JuMP, LinearAlgebra, Suppressor, StatsBase, CSV, DataFrames, Compat, Test, Random
include("exact_approaches.jl")
include("alt_min.jl")
include("convexpenalties.jl")


results_template = DataFrame(n=Int[], r=Int[], p=Real[], gamma=Real[], err_GD=Real[], rnk_GD=Int[], solve_GD=Real[], err_AM=Real[], rnk_AM=Int[], solve_AM=Real[], err_EX=Real[],
rnk_EX=Int[], solve_EX=Real[], nc_EX=Int[], gap_EX=Real[], err_OA=Real[], rnk_OA=Int[], solve_OA=Real[], nc_OA=Int[], gap_OA=Real[], cuts_OA=Int[],
err_OA_inout=Real[], rnk_OA_inout=Int[], solve_OA_inout=Real[], nc_OA_inout=Int[], gap_OA_inout=Real[], cuts_OA_inout=Int[])

results_exact=similar(results_template, 0)

# note: you will need to manually vary n, r, gamma to obtain all the output you want.
for ARG in ARGS #1-4
    for iter in 1:1
        #Random.seed!(1234)
        p=[0.05, 0.1, 0.15, 0.2][parse(Int, ARG)]
        n=10
        r=1
        gamma=100/p
        M_L=randn(n, r)
        M_L=round.(1000*M_L)./1000
        M_R=randn(n, r)
        M_R=round.(1000*M_R)./1000
        M=M_L*M_R'

        @show maximum(M)
        numEntriesSampled=Int(round(p*n^2))
        coords=[repeat(1:n, inner=size(1:n,1)) repeat(1:n, outer=size(1:n,1))];
        entries=sample(1:n^2, Int(numEntriesSampled), replace=false)

        theIndices=zeros(n,n)
        for t=1:Int(numEntriesSampled)
            theIndices[coords[entries[t],1], coords[entries[t],2]]=1.0
        end

        k=r


        t_GD=@elapsed X_GD,LB, UB, Y_relax=getMatrixFrobeniusNorm_greedy(coords, entries, M, n, numEntriesSampled, k, gamma, true)

        @show "SDP lower bound is: " LB
        u,sigma,v,=svd(X_GD)
        rnk_GD=sum(sigma.>1e-2);
        err_GD = norm(X_GD.-M)/norm(M);

        t_AM=@elapsed X_AM,=alternatingminimization_matrixcompletion(coords, entries, M, n, numEntriesSampled, k, gamma, 50, X_GD)

        # Test solution on overall data.
        u,sigma,v,=svd(X_AM)
        @show rnk_AM=sum(sigma.>1e-2);
        @show err_AM = norm(X_AM.-M)/norm(M);

        u_0=u[:,1:k]
        v_0=(Diagonal(sigma[1:k])*v[:,1:k]')'

        # # # Apply the cutting-plane method (with the in-out nethod)
        t_OA=@elapsed X_OA, nc_OA, cuts_OA, gap_OA=getMatrixFrobeniusNorm_cuttingplanes_exact_inout(coords, entries, M, n, numEntriesSampled, k, gamma, Y_relax, X_AM, true, true, false)
        # # Test solution on overall data.
        u,v,=svd(X_OA)
        @show rnk_OA=sum(v.>1e-2);
        @show err_OA=norm(X_OA.-M)/norm(M);

        # Apply the cutting-plane method (with the in-out nethod)
        t_OA_inout=@elapsed X_OA, nc_OA_inout, cuts_OA_inout, gap_OA_inout=getMatrixFrobeniusNorm_cuttingplanesmultitree_exact_inout(coords, entries, M, n, numEntriesSampled, k, gamma, Y_relax, X_AM, true)
        # Test solution on overall data.
        u,v,=svd(X_OA)
        @show rnk_OA_inout=sum(v.>1e-2);
        @show err_OA_inout = norm(X_OA.-M)/norm(M);


        t_EX=@elapsed X_EX, nc_EX, gap_EX=getMatrixFrobeniusNorm_exact(coords, entries, M, n, numEntriesSampled, k, gamma, false, u_0, v_0)

        # Test solution on overall data.
        u,v,=svd(X_EX)
        @show rnk_EX=sum(v.>1e-2);
        @show err_EX = norm(X_EX.-M)/norm(M);

        push!(results_exact, [n, r, p, gamma, err_GD, rnk_GD, t_GD, err_AM, rnk_AM, t_AM, err_EX, rnk_EX, t_EX, nc_EX, gap_EX, err_OA, rnk_OA, t_OA, nc_OA, gap_OA, cuts_OA, err_OA_inout, rnk_OA_inout, t_OA_inout, nc_OA_inout, gap_OA_inout, cuts_OA_inout])

        end
    end

CSV.write("benchmark_gurobi_vs_cuts_grbnolazy.csv", results_exact, append=true)
