using JuMP, Gurobi, LinearAlgebra, Suppressor, StatsBase, Compat, Mosek
include("convex_penalties_edm.jl")
# Begin by constructing a simple 2d edm test matrix problem, based on Biswas and Ye (2005), Zuo et. al. (2010) among others

n=4 #Number of points located on the grid
m=0 #Number of anchors located on the grid

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

p=0.6
#k=2
numEntriesSampled=Int(round(p*(n+m)^2))

# Get observed set of coordinates
coords=[repeat(1:(n+m), inner=size(1:(n+m),1)) repeat(1:(n+m), outer=size(1:(n+m),1))];
entries=sample(1:(n+m)^2, Int(numEntriesSampled), replace=false)

# Solve the sdp relaxation and display the rank, relative MSE
X_imputed, G_imputed,=getMatrixTraceNorm_EDM(coords, entries, D, n, numEntriesSampled, ys, m)

@show err_XTR=norm(X_imputed.-xs)/norm(xs);
D_imputed=diag(G_imputed)*ones(n)'+ones(n)*diag(G_imputed)'.-2*G_imputed
@show err_DTR=norm(D_imputed.-D[m+1:end, m+1:end])/norm(D[m+1:end, m+1:end])
