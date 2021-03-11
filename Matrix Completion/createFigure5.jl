
using JuMP, Mosek, MosekTools, LinearAlgebra, Suppressor, StatsBase, CSV, DataFrames, Ipopt, Compat
include("convexpenalties.jl")
include("alt_min.jl")

gammaMult=5.0 # Need to vary this to generate the four plots

results_template = DataFrame(n=Int[], r=Int[], p=Real[], y=Real[],
    error_GD=Real[], error_AM=Real[], rank_GD=Int[], rank_AM=Int[], LB_GD=Real[], UB_GD=Real[], UB_AM=Real[])

results_basispursuit=similar(results_template, 0)

pRange=collect(0.1:0.1:1.0)
pSize=size(pRange, 1)
yRange=collect(0.1:0.1:1.0)
ySize=size(yRange, 1)

for ARG in ARGS # 1-100
  for iter in 1:5
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
  gamma=gammaMult/p #setting this to be really high as a good proxy for exact constraints (no noise)

  coords=[repeat(1:n, inner=size(1:n,1)) repeat(1:n, outer=size(1:n,1))];
  entries=sample(1:n^2, Int(numEntriesSampled), replace=false)

  ############## Greedy rounding  #####################################################
  @show k
  X_GD,LB_GD, UB_GD,=getMatrixFrobeniusNorm_greedy(coords, entries, M, n, numEntriesSampled, k, gamma, false)

  u,v,w=svd(X_GD);
  @show rnk_GD=sum(v.>1e-2);
  @show err_GD = norm(X_GD.-M)/norm(M);

  ########### Alternating Minimization ###########################################################
  X_AM, UB_AM=alternatingminimization_matrixcompletion(coords, entries, M, n, numEntriesSampled, k, gamma, 50, X_GD)
  u,v,=svd(X_AM)
  @show rnk_AM=sum(v.>1e-2);
  @show err_AM = norm(X_AM.-M)/norm(M);

  ########### Push results #############################################################

  push!(results_basispursuit, [n, r, p, y, err_GD, err_AM, rnk_GD, rnk_AM, LB_GD, UB_GD, UB_AM])
end
  CSV.write("dualitygapplot_gamma5p0onp.csv", results_basispursuit, append=true)

end
