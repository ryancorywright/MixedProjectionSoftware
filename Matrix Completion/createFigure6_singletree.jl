#ENV["MOSEKLM_LICENSE_FILE"]="/home/ryancw/mosek/mosek.lic" #needed on cluster
using JuMP, LinearAlgebra, Suppressor, StatsBase, CSV, DataFrames, Compat, Test, Random
include("exact_approaches.jl")
include("alt_min.jl")
include("convexpenalties.jl")


results_template = DataFrame(n=Int[], r=Int[], p=Real[], gamma=Real[], err_OA_inout=Real[], rnk_OA_inout=Int[], solve_OA_inout=Real[], nc_OA_inout=Int[], gap_OA_inout=Real[], cuts_OA_inout=Int[], theSeed=Int[])

results_exact=similar(results_template, 0)

for ARG in ARGS
    for iter in 1:1
        p=0.2
        n=20
        r=1
        seeds=1:20
        gammaMults=[0.1, 	0.316227766, 	1, 	3.16227766, 	5, 	10, 	25, 	31.6227766, 	50, 	100, 	250, 	316.227766, 	500, 	1000, 	2500, 	3162.27766, 	10000, 	31622.7766, 	100000, 	316227.766] # You will need to alter the gamma values in here to generate the full range of points shown
        array_num = parse(Int, ARG)


        theSeed=seeds[(array_num-1)÷20+1]
        gamma=gammaMults[(array_num-1)%20+1]
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


        # # # Apply the cutting-plane method (with the in-out nethod)
        t_OA_inout=@elapsed X_OA, nc_OA_inout, cuts_OA_inout, gap_OA_inout=getMatrixFrobeniusNorm_cuttingplanes_exact_inout(coords, entries, M, n, numEntriesSampled, k, gamma, Y_relax, X_AM, true, true, false)
        # # Test solution on overall data.
        u,v,=svd(X_OA)
        @show rnk_OA_inout=sum(v.>1e-2);
        @show err_OA_inout = norm(X_OA.-M)/norm(M);



        push!(results_exact, [n, r, p, gamma, err_OA_inout, rnk_OA_inout, t_OA_inout, nc_OA_inout, gap_OA_inout, cuts_OA_inout, theSeed])

        end
    end

CSV.write("sensitivity_analysis_singletree_gamma.csv", results_exact, append=true)
