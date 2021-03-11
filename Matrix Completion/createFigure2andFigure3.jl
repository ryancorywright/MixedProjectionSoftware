
using JuMP, Mosek, MosekTools, LinearAlgebra, Suppressor, StatsBase, CSV, DataFrames, Ipopt, Compat
include("convexpenalties.jl")
include("alt_min.jl")



results_template = DataFrame(n=Int[], r=Int[], p=Real[], y=Real[],
    error_NN=Real[], error_GD=Real[], error_FN=Real[], error_BM=Real[], error_AM=Real[], rank_NN=Int[], rank_GD=Int[], rank_FN=Int[], rank_BM=Int[], rank_AM=Int[])

results_basispursuit=similar(results_template, 0)

pRange=collect(0.1:0.1:1.0) # You will also need to look at collect(0.05:0.1:0.95), we only did 1/4 here to avoid running ARGS=1-400 on sbatch which might have upset the cluster admin
pSize=size(pRange, 1)
yRange=collect(0.1:0.1:1.0) # You will also need to look at collect(0.05:0.1:0.95), we only did 1/4 here to avoid running ARGS=1-400 on sbatch which might have upset the cluster admin
ySize=size(yRange, 1)

for ARG in ARGS # Varies between 1 and 100
  n=100

  array_num = parse(Int, ARG)


  y=yRange[(array_num-1)÷ySize+1]
  p=pRange[(array_num-1)%pSize+1]
  r=Int(round(n*(1.0-sqrt(1.0-y*p))))
  if r==0
      r=1
  end
  k=r #assume we picked it via say cross-validation (can only show two dimensions on the plot).
  M_L=randn(n, r)
  M_R=randn(n, r)
  M=M_L*M_R'
  numEntriesSampled=Int(round(p*n^2))
  gamma=500/p #This value needs to be changed to reproduce the plots where gamma is a different value

  coords=[repeat(1:n, inner=size(1:n,1)) repeat(1:n, outer=size(1:n,1))];
  entries=sample(1:n^2, Int(numEntriesSampled), replace=false)

  ############## Nuclear norm  #######################################################
  X_NN=getMatrixNuclearNorm(coords, entries, M, n, numEntriesSampled, true)
  @show err_NN=norm(X_NN-M)/norm(M);
  u,v,w=svd(X_NN);
  @show rnk_NN=sum(v.>1e-2);



  ############## Greedy rounding  #####################################################
  @show k
  X_GD,=getMatrixFrobeniusNorm_greedy(coords, entries, M, n, numEntriesSampled, k, gamma, true)

  u,v,w=svd(X_GD);
  @show rnk_GD=sum(v.>1e-2);
  @show err_GD = norm(X_GD.-M)/norm(M);

  # ############# Frobenius random rounding ##############################################
  X_FN=getMatrixFrobeniusNorm_reg2ndstage(coords, entries, M, n, numEntriesSampled, k, gamma, true)
  u,v,w=svd(X_FN);
  rnk_FN=sum(v.>1e-2);
  err_FN = norm(X_FN.-M)/norm(M);

  # ############ Burer-Monterio ##########################################################
  X_BM=getMatrix_BM(coords, entries, M, n, numEntriesSampled, k, gamma)
  u,v,w=svd(X_BM);
  @show rnk_BM=sum(v.>1e-2);
  @show err_BM =norm(X_BM.-M)/norm(M);
  ########### Alternating Minimization ###########################################################
  X_AM=alternatingminimization_matrixcompletion(coords, entries, M, n, numEntriesSampled, k, gamma, 50, X_GD)
  u,v,=svd(X_AM)
  @show rnk_AM=sum(v.>1e-2);
  @show err_AM = norm(X_AM.-M)/norm(M);

  ########### Push results #############################################################

  push!(results_basispursuit, [n, r, p, y, err_NN, err_GD, err_FN, err_BM, err_AM, rnk_NN, rnk_GD, rnk_FN, rnk_BM, rnk_AM])
  CSV.write("signalrecoveryplot.csv", results_basispursuit, append=true)

end
