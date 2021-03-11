#ENV["MOSEKLM_LICENSE_FILE"]="/home/ryancw/mosek/mosek.lic" #needed on cluster
using JuMP, LinearAlgebra, Suppressor, StatsBase, CSV, DataFrames, Compat, Test, Random
include("exact_approaches.jl")
include("alt_min.jl")
include("convexpenalties.jl")


results_template = DataFrame(n=Int[], r=Int[], p=Real[], gamma=Real[], err=Real[], rnk=Int[], solvetime=Real[], nc=Int[], gap=Real[], cuts=Int[])

results_exact=similar(results_template, 0)

for ARG in ARGS
    for iter in 1:1
        p=0.2
        n=20
        r=1
        seeds=1:20
        gammaMults=[100.0, 500.0]
        array_num = parse(Int, ARG)


        theSeed=seeds[(array_num-1)÷2+1]
        gamma=gammaMults[(array_num-1)%2+1]
        Random.seed!(theSeed)
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


        t_EX=@elapsed X_EX, nc_EX, gap_EX=getMatrixFrobeniusNorm_exact(coords, entries, M, n, numEntriesSampled, k, gamma, false, u_0, v_0)
        u,v,=svd(X_EX)
        @show rnk_EX=sum(v.>1e-2);
        @show err_EX = norm(X_EX.-M)/norm(M);


        push!(results_exact, [n, r, p, gamma, err_EX, rnk_EX, t_EX, nc_EX, gap_EX, theSeed])
        CSV.write("sensitivity_analysis_grb_p"*string(p)*"_n_"*string(n)*".csv", results_exact, append=true)

        end
    end
