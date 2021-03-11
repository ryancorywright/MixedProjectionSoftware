using JuMP, LinearAlgebra, Suppressor, StatsBase, CSV, DataFrames, Compat, Test, Random
include("exact_approaches.jl")
include("alt_min.jl")
include("convexpenalties.jl")


results_template = DataFrame(n=Int[], r=Int[], p=Real[], err_NN=Real[], rnk_NN=Int[],
solve_NN=Real[], err_GD=Real[], rnk_GD=Int[], solve_GD=Real[], err_AM=Real[], rnk_AM=Int[], solve_AM=Real[], err_OA=Real[], rnk_OA=Int[], solve_OA=Real[], nc_OA=Int[], gap_OA=Real[], cuts_OA=Int[])

results_exact=similar(results_template, 0)

for ARG in ARGS
    theArg=parse(Int, ARG)
    pvals=[0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4]
    p=pvals[theArg]
    for iter in 1:1

        n=50
        r=1
        gamma=100/p
        M_L=randn(n, r)
        M_R=randn(n, r)
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
        mu=1.0 #imposing constraints so mu doesn't matter
        t_NN=@elapsed X_NN=getMatrixNuclearNorm(coords, entries, M, n, numEntriesSampled, mu, true) #imposing constraints as noiseless so we want exact signal recovery asap
        u,sigma,v,=svd(X_NN)
        rnk_NN=sum(sigma.>1e-2);
        err_NN = norm(X_NN.-M)/norm(M);
        # Get SDP lower bound (for sake of comparison)
        t_GD=@elapsed X_GD,LB, UB, Y_relax=getMatrixFrobeniusNorm_greedy(coords, entries, M, n, numEntriesSampled, k, gamma, false)

        @show "SDP lower bound is: " LB
        u,sigma,v,=svd(X_GD)
        rnk_GD=sum(sigma.>1e-2);
        err_GD = norm(X_GD.-M)/norm(M);

        t_AM=@elapsed X_AM,=alternatingminimization_matrixcompletion(coords, entries, M, n, numEntriesSampled, k, gamma, 50, X_GD)#AM warm started with GD

        # Test solution on overall data.
        u,sigma,v,=svd(X_AM)
        @show rnk_AM=sum(sigma.>1e-2);
        @show err_AM = norm(X_AM.-M)/norm(M);

        u_0=u[:,1:k]
        v_0=(Diagonal(sigma[1:k])*v[:,1:k]')'

        # Apply the cutting-plane method (multi-tree version)
        t_OA=@elapsed X_OA, nc_OA, cuts_OA, gap_OA=getMatrixFrobeniusNorm_cuttingplanesmultitree_exact_inout(coords, entries, M, n, numEntriesSampled, k, gamma, Y_relax, X_AM, true)
        # Test solution on overall data.
        u,v,=svd(X_OA)
        @show rnk_OA=sum(v.>1e-2);
        @show err_OA = norm(X_OA.-M)/norm(M);

        push!(results_exact, [n, r, p, err_NN, rnk_NN, t_NN, err_GD, rnk_GD, t_GD, err_AM, rnk_AM, t_AM, err_OA, rnk_OA, t_OA, nc_OA, gap_OA, cuts_OA])

        end
    end

CSV.write("results_basispursuit_varyingp_n50_k1_gamma100onp.csv", results_exact, append=true)
