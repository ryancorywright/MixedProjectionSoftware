using JuMP, Mosek, MosekTools, LinearAlgebra, Suppressor, StatsBase, Ipopt


function getMatrixTraceNorm_EDM_noisy(coords, entries, D, n, numEntriesSampled, ys, num_anchors, useRadiorange=false, radiorange=1.0)
    m=Model(Mosek.Optimizer)
    @variable(m, G[1:n+num_anchors+2, 1:n+num_anchors+2], PSD)
    @constraint(m, G[1:2,1:2].==[1.0 0.0; 0.0 1.0])
    # Note that G_full=[I X;X' G], so we can recover X from G_full quite simply when rank(G)=2 in the underlying relaxation.
    @variable(m, slack[1:numEntriesSampled])
    @variable(m, absslack[1:numEntriesSampled])
    @constraint(m, absslack.>=slack)
    @constraint(m, absslack.>=-slack)
    mu=100.0

    for t=1:Int(numEntriesSampled)
        if (coords[entries[t],1]<=num_anchors) && (coords[entries[t],2]<=num_anchors)
            #don't need to consider this case as doesn't yield any new information
        elseif (coords[entries[t],1]<=num_anchors) && (coords[entries[t],2]>num_anchors)
            e_k=zeros(n+num_anchors)
            e_k[coords[entries[t],2]]=1.0
            @constraint(m, Compat.dot([ys[coords[entries[t],1],:]; -e_k]*[ys[coords[entries[t],1],:]; -e_k]', G).==D[coords[entries[t],1], coords[entries[t],2]]+slack[t])
        elseif (coords[entries[t],1]>num_anchors) && (coords[entries[t],2]<=num_anchors)
            e_k=zeros(n+num_anchors)
            e_k[coords[entries[t],1]]=1.0
            @constraint(m,Compat.dot([ys[coords[entries[t],2],:]; -e_k]*[ys[coords[entries[t],2],:]; -e_k]', G).==D[coords[entries[t],1], coords[entries[t],2]]+slack[t])
        elseif (coords[entries[t],1]>num_anchors) && (coords[entries[t],2]>num_anchors)
            e_i=zeros(n+num_anchors)
            e_i[coords[entries[t],1]]=1.0
            e_k=zeros(n+num_anchors)
            e_k[coords[entries[t],2]]=1.0
            @constraint(m, Compat.dot([zeros(2); e_i.-e_k]*[zeros(2); e_i.-e_k]', G).==D[coords[entries[t],1], coords[entries[t],2]]+slack[t])
        end
    end
    if useRadiorange
        altentries=setdiff((1:(n+num_anchors)^2), entries)
        for t=1:((n+num_anchors)^2 -Int(numEntriesSampled))
            if (coords[altentries[t],1]<=num_anchors) && (coords[altentries[t],2]<=num_anchors)
                #don't need to consider this case as doesn't yield any new information
            elseif (coords[altentries[t],1]<=num_anchors) && (coords[altentries[t],2]>num_anchors)
                e_k=zeros(n+num_anchors)
                e_k[coords[altentries[t],2]]=1.0
                @constraint(m, Compat.dot([ys[coords[altentries[t],1],:]; -e_k]*[ys[coords[altentries[t],1],:]; -e_k]', G).>=radiorange^2)
            elseif (coords[altentries[t],1]>num_anchors) && (coords[altentries[t],2]<=num_anchors)
                e_k=zeros(n+num_anchors)
                e_k[coords[altentries[t],1]]=1.0
                @constraint(m,Compat.dot([ys[coords[altentries[t],2],:]; -e_k]*[ys[coords[altentries[t],2],:]; -e_k]', G).>=radiorange^2)
            elseif (coords[altentries[t],1]>num_anchors) && (coords[altentries[t],2]>num_anchors)
                e_i=zeros(n+num_anchors)
                e_i[coords[altentries[t],1]]=1.0
                e_k=zeros(n+num_anchors)
                e_k[coords[altentries[t],2]]=1.0
                @constraint(m, Compat.dot([zeros(2); e_i.-e_k]*[zeros(2); e_i.-e_k]', G).>=radiorange^2)
            end
        end
    end

    @objective(m, Min, sum(G[i,i] for i=1:n)+mu*sum(absslack))

    @suppress optimize!(m)
   return value.(G[3+num_anchors:end, 1:2]), value.(G[3+num_anchors:end, 3+num_anchors:end]), value.(G[3:end, 1:2]) #return X
end

function getMatrixTraceNorm_EDM(coords, entries, D, n, numEntriesSampled, ys, num_anchors, imposeTriangleInequalities=false)
    m=Model(Mosek.Optimizer)
    @variable(m, G[1:n+num_anchors+2, 1:n+num_anchors+2], PSD)
    @constraint(m, G[1:2,1:2].==[1.0 0.0; 0.0 1.0])
    # Note that G_full=[I X;X' G], so we can recover X from G_full quite simply when rank(G)=2 in the underlying relaxation.

    for t=1:Int(numEntriesSampled)
        if (coords[entries[t],1]<=num_anchors) && (coords[entries[t],2]<=num_anchors)
            #don't need to consider this case as doesn't yield any new information
        elseif (coords[entries[t],1]<=num_anchors) && (coords[entries[t],2]>num_anchors)
            e_k=zeros(n+num_anchors)
            e_k[coords[entries[t],2]]=1.0
            @constraint(m, Compat.dot([ys[coords[entries[t],1],:]; -e_k]*[ys[coords[entries[t],1],:]; -e_k]', G).==D[coords[entries[t],1], coords[entries[t],2]])
        elseif (coords[entries[t],1]>num_anchors) && (coords[entries[t],2]<=num_anchors)
            e_k=zeros(n+num_anchors)
            e_k[coords[entries[t],1]]=1.0
            @constraint(m,Compat.dot([ys[coords[entries[t],2],:]; -e_k]*[ys[coords[entries[t],2],:]; -e_k]', G).==D[coords[entries[t],1], coords[entries[t],2]])
        elseif (coords[entries[t],1]>num_anchors) && (coords[entries[t],2]>num_anchors)
            e_i=zeros(n+num_anchors)
            e_i[coords[entries[t],1]]=1.0
            e_k=zeros(n+num_anchors)
            e_k[coords[entries[t],2]]=1.0
            @constraint(m, Compat.dot([zeros(2); e_i.-e_k]*[zeros(2); e_i.-e_k]', G).==D[coords[entries[t],1], coords[entries[t],2]])
        end
    end
    @objective(m, Min, sum(G[i,i] for i=1:n))

    @suppress optimize!(m)
   return value.(G[3+num_anchors:end, 1:2]), value.(G[3+num_anchors:end, 3+num_anchors:end]), value.(G[3:end, 1:2]) #return X
end

function getMatrixTraceNorm_EDM_Gram(coords, entries, D, n, numEntriesSampled, d_radio, lambda, gamma, k)
    m=Model(Mosek.Optimizer)
    @variable(m, G[1:n, 1:n], PSD)
    @variable(m, G2[1:n, 1:n])
    @variable(m, theta[1:n, 1:n], Symmetric)
    @variable(m, Y[1:n, 1:n], Symmetric)
    @variable(m, xi[1:n, 1:n])
    @variable(m, absxi[1:n, 1:n])
    @constraint(m, absxi.>=xi)
    @constraint(m, absxi.>=-xi)
    @constraint(m, G.==G2) #Since Mosek can't have a matrix which is part of two seperate PSD constraints at once

    for t=1:Int(numEntriesSampled)
        it=coords[entries[t],1]
        jt=coords[entries[t],2]
        @constraint(m, G[it,it]+G[jt,jt]-2*G[it,jt]+xi[it,jt]==D[it,jt])
    end

    @constraint(m, Symmetric(Matrix(1.0I, n, n)-Y) in PSDCone());
    @constraint(m, Symmetric([Y G2; G2' theta]) in PSDCone());
    @constraint(m, sum(Y[i,i] for i=1:n)<=k)

    @objective(m, Min, sum(G[i,i] for i=1:n)+lambda*sum(absxi)+(1/(2*gamma))*sum(theta[i,i] for i=1:n))#sum(G[i,j]^2 for i=1:n for j=1:n))
    @show d_radio, lambda, gamma
    @suppress optimize!(m)
    @show objective_value(m)
    # @show value.(G)
    # @show value.(Y)
   return value.(G), value.(Y)
end
