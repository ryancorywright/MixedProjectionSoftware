using JuMP, Gurobi, MosekTools, LinearAlgebra, Suppressor, StatsBase, Ipopt, MathOptInterface

mutable struct cut
    intercept::Float64
    slope::Array{Float64,2}
end

mutable struct CutIterData #used for plotting performance of method
    time::Float64
    obj::Float64
    bound::Float64
end


function getMatrixExact_EDM(coords, entries, D, n, numEntriesSampled, ys, num_anchors, x_ws=zeros(n+num_anchors,2), useRadiorange=false, radiorange=1.0)
    radiorange=0.4
    m=Model(Gurobi.Optimizer)
    set_optimizer_attribute(m, "NonConvex", 2)
    set_optimizer_attribute(m, "MIPGap", 1e-2)
    set_optimizer_attribute(m, "MIPGapAbs", 1e-3)
    set_optimizer_attribute(m, "Heuristics", 0.5)
    set_optimizer_attribute(m, "TimeLimit", 60)
    @variable(m, G[1:n+num_anchors+2, 1:n+num_anchors+2], Symmetric)
    @variable(m, x[1:n+num_anchors, 1:2])
    @variable(m, y[1:n+num_anchors, 1:2])
    @variable(m, slack[1:n+num_anchors+2, 1:n+num_anchors+2])
    @variable(m, absslack[1:n+num_anchors+2, 1:n+num_anchors+2])
    @constraint(m, slack.<=absslack)
    @constraint(m, -slack.<=absslack)
    @variable(m, slack2[1:numEntriesSampled])
    @variable(m, absslack2[1:numEntriesSampled])
    @constraint(m, slack2.<=absslack2)
    @constraint(m, -slack2.<=absslack2)
    @constraint(m, x.==y)
    @constraint(m, x.<=0.5)
    @constraint(m, -x.<=0.5)
    @constraint(m, G[1:2,1:2].==[1.0 0.0; 0.0 1.0])
    # Note that G_full=[I X;X' G], so we can recover X from G_full quite simply when rank(G)=2 in the underlying relaxation.
    @constraint(m, G[3:n+num_anchors+2, 1:2].==x)
    # @constraint(m, x[1,1]>=x[1,2]+1e-2) #symmetry breaking
    # @constraint(m, x[:,1]'*x[:,2]>=-1e-1)
    # @constraint(m, x[:,1]'*x[:,2]<=1e-1) #orthogonality

    # @constraint(m, x[1:num_anchors,:].>=ys-2e-1)
    # @constraint(m, x[1:num_anchors,:].<=ys+2e-1)


    @show size(x*y')
    @show size(G[3:n+num_anchors+2, 3:n+num_anchors+2])
    @constraint(m, G[3:n+num_anchors+2, 3:n+num_anchors+2].==x*y'.+slack[3:n+num_anchors+2, 3:n+num_anchors+2])
    # @constraint(m, G[3:n+num_anchors+2, 3:n+num_anchors+2].<=x*y'.+1e-4*ones(n+num_anchors, n+num_anchors))

    for t=1:Int(numEntriesSampled)
        if (coords[entries[t],1]<=num_anchors) && (coords[entries[t],2]<=num_anchors)
            #don't need to consider this case as doesn't yield any new information
        elseif (coords[entries[t],1]<=num_anchors) && (coords[entries[t],2]>num_anchors)
            e_k=zeros(n+num_anchors)
            e_k[coords[entries[t],2]]=1.0
            @constraint(m, Compat.dot([ys[coords[entries[t],1],:]; -e_k]*[ys[coords[entries[t],1],:]; -e_k]', G).==D[coords[entries[t],1], coords[entries[t],2]]+slack2[t])
        elseif (coords[entries[t],1]>num_anchors) && (coords[entries[t],2]<=num_anchors)
            e_k=zeros(n+num_anchors)
            e_k[coords[entries[t],1]]=1.0
            @constraint(m,Compat.dot([ys[coords[entries[t],2],:]; -e_k]*[ys[coords[entries[t],2],:]; -e_k]', G).==D[coords[entries[t],1], coords[entries[t],2]]+slack2[t])
        elseif (coords[entries[t],1]>num_anchors) && (coords[entries[t],2]>num_anchors)
            e_i=zeros(n+num_anchors)
            e_i[coords[entries[t],1]]=1.0
            e_k=zeros(n+num_anchors)
            e_k[coords[entries[t],2]]=1.0
            @constraint(m, Compat.dot([zeros(2); e_i.-e_k]*[zeros(2); e_i.-e_k]', G).==D[coords[entries[t],1], coords[entries[t],2]]+slack2[t])
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


    @objective(m, Min,  sum(absslack)+sum(absslack2))
    @show size(x_ws)
    set_start_value.(x, x_ws)
    optimize!(m)
    @show objective_value(m), objective_bound(m)
    @show MOI.get(m, MOI.NodeCount())
   return value.(G[3+num_anchors:end, 1:2]), value.(G[3+num_anchors:end, 3+num_anchors:end]), MathOptInterface.get(m, MathOptInterface.RelativeGap()) #return X
end

function getMatrixExact_EDM_Gram(coords, entries, D, n, r, numEntriesSampled, d_radio, lambda, gamma, G_ws=zeros(n,n))
    m=Model(Gurobi.Optimizer)
    set_optimizer_attribute(m, "NonConvex", 2)
    set_optimizer_attribute(m, "MIPGap", 1e-2)
    set_optimizer_attribute(m, "MIPGapAbs", 1e-3)
    set_optimizer_attribute(m, "FuncPieceError", 1e-6)
    set_optimizer_attribute(m, "FuncPieceLength", 1e-5)
    set_optimizer_attribute(m, "TimeLimit", 120)

    @variable(m, G[1:n, 1:n], Symmetric)
    @variable(m, x[1:n, 1:r])
    @variable(m, slack[1:n, 1:n])
    @variable(m, absslack[1:n, 1:n])
    @constraint(m, slack.<=absslack)
    @constraint(m, -slack.<=absslack)

    @constraint(m, G.>=x*x'.-1e-8*ones(n,n))
    @constraint(m, G.<=x*x'.+1e-8*ones(n,n))

    for t=1:Int(numEntriesSampled)
        it=coords[entries[t],1]
        jt=coords[entries[t],2]
        @constraint(m, G[it,it]+G[jt,jt]-2*G[it,jt]==D[it,jt]+slack[it,jt])
    end


    @objective(m, Min,  sum(G[i,i] for i=1:n)+lambda*sum(absslack)+(1/(2*gamma))*sum(G[i,j]^2 for i=1:n for j=1:n))
    set_start_value.(G, G_ws)
    optimize!(m)
    @show objective_value(m), objective_bound(m)
    @show MOI.get(m, MOI.NodeCount())
   return value.(G), MathOptInterface.get(m, MathOptInterface.RelativeGap()), MOI.get(m, MOI.NodeCount()) #return G
end

function getMatrixFrobeniusNorm_cuttingplanes_exact_inout(coords, entries, M, n, numEntriesSampled, lambda, d_radio, k=sqrt(n), gamma=200*sqrt(n)/(numEntriesSampled), Y_ws=zeros(n,n), stabilizationPoint=k/n*Diagonal(ones(n)), useInOut=true, useSOCs=true, useKelleys=false)

    m=Model(Mosek.Optimizer) #First solve the continuous relaxation
    set_optimizer_attribute(m, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-10)
    set_optimizer_attribute(m, "MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-10)
    @variable(m, Y[1:n, 1:n], PSD)
    @constraint(m, Symmetric(Matrix(1.0I, n, n)-Y) in PSDCone());
    @variable(m, theta>=-1e5)

    @objective(m, Min, theta)
    @constraint(m, sum(Y[i,i] for i=1:n)<=k)
    CurrentTime=time()

    ######### Start defining variables for the exact method ##############
    m_ex=Model(Gurobi.Optimizer)
    set_optimizer_attribute(m_ex, "NonConvex", 2)
    set_optimizer_attribute(m_ex, "TimeLimit", 1800.0)
    set_optimizer_attribute(m_ex, "MIPGap", 1e-2)
    set_optimizer_attribute(m_ex, "FuncPieceError", 1e-3)
    set_optimizer_attribute(m_ex, "FuncPieceLength", 1e-3)
    set_optimizer_attribute(m_ex, "Threads", 12)
    set_optimizer_attribute(m_ex, "Presolve", 0)

    @variable(m_ex, Y_ex[1:n, 1:n], Symmetric)
    @constraint(m_ex, sum(Y_ex[i,i] for i=1:n)<=k)
    @variable(m_ex, U_ex[1:n, 1:k])

    @variable(m_ex, theta_ex>=-1e4)
    @objective(m_ex, Min, theta_ex) #can penalize the absolute slack as appropriate

    @constraint(m_ex, defineY[i=1:n, j=i:n], Y_ex[i,j].<=U_ex[i,:]'*U_ex[j,:]+1e-4) #Note that we need only impose this in upper triangle, as Y is symmetric
    @constraint(m_ex, defineY2[i=1:n, j=i:n], Y_ex[i,j].>=U_ex[i,:]'*U_ex[j,:]-1e-4)

    bcdata = CutIterData[]

    for i=1:k
        for j=i:k
            ind=1.0*(i==j)
            @constraint(m_ex, U_ex[:,i]'*U_ex[:,j]<=1.0*ind+1e-4)
            @constraint(m_ex, U_ex[:,i]'*U_ex[:,j]>=1.0*ind-1e-4)
        end
    end



    if useSOCs
        @constraint(m_ex, imposeSOC[i=1:n, j=1:n], [Y_ex[i,i]+Y_ex[j,j]; Y_ex[i,i]-Y_ex[j,j]; 2.0*Y_ex[i,j]] in SecondOrderCone())
    end

    @constraint(m_ex, imposeDiag[i=1:n], 0.0<=Y_ex[i,i]<=1.0)

    theIndices=zeros(n,n)
    for t=1:Int(numEntriesSampled)
        theIndices[coords[entries[t],1], coords[entries[t],2]]=1.0
        theIndices[coords[entries[t],2], coords[entries[t],1]]=1.0 #Recall that the matrix is known to be symmetric
    end

    ############ Start the cut loop #######################
    theCutPool=cut[]
    UB=1e12
    LB=-1e12
    rootStabilizationTrick = :inOut
     #Use the solution to the SDP relaxation instead, where appropriate
    rootCutsSense=1
    ε= 1e-10
    λ = (rootStabilizationTrick == :inOut) ? .1 : 1.
    δ = (rootStabilizationTrick == :inOut || rootStabilizationTrick == :twoEps) ? 2*ε : 0.
    rootCutsLim=200
    rootCutCount = 0
    oaRootCutCount=0
    consecutiveNonImprov_1 = 0
    consecutiveNonImprov_2 = 0
    # stabilizationPoint=Y_ws
    if useInOut
        Y_best=zeros(n,n)
        for epoch in 1:200
          @suppress optimize!(m)
          Ystar=value.(Y)
          stabilizationPoint += Ystar; stabilizationPoint /= 2

          if LB >= objective_value(m) - eps()
            if consecutiveNonImprov_1 == 5
              consecutiveNonImprov_2 += 1
            else
              consecutiveNonImprov_1 += 1
            end
          else
            if consecutiveNonImprov_1 < 5
              consecutiveNonImprov_1 = 0
            end
            consecutiveNonImprov_2 = 0
          end
          LB = max(LB, objective_value(m))

          if consecutiveNonImprov_1 == 5
            λ = 1
          elseif consecutiveNonImprov_2 == 5
            δ = 0.
          end

          Y0 = λ*Ystar + (1-λ)*stabilizationPoint .+ δ*Diagonal(ones(n))
          if useKelleys #override in-out bit of method
              Y0=Ystar
          end

          # Perform an SVD step here to ensure still in convex hull
          u,v,=svd(Y0)
          v=v*k/sum(v)
          v.=min.(ones(n), max.(v, zeros(n)))

          Y0=u*Diagonal(v)*u'


          # Get a cut by solving the dual subproblem
          # Solve subproblem for fixed Y, in alpha
          f_Y, alpha_t=getEDMcut(Y0, theIndices, M, n, gamma, lambda)
          if f_Y<UB
              Y_best=Y0
              UB=f_Y
          end
          LB=max(LB, objective_value(m))
          # Impose constraint
          @constraint(m, theta>=f_Y-gamma/2*Compat.dot(alpha_t*alpha_t', Y-Y0))

          @constraint(m_ex, theta_ex>=f_Y-gamma/2*Compat.dot(alpha_t*alpha_t', U_ex*U_ex'-Y0))
          # Add cut to pool (add in later if it works)
          @show LB, UB, epoch, time()-CurrentTime
          if abs(UB-LB) <= 1e-4 || consecutiveNonImprov_2 >= 10
              break
          end
        end
    end

    # After the cutting-plane method on the continuous relaxation has terminated, apply an outer-approximation method on the exact problem
    # Using lazy callbacks (faster than a multi-tree method)
    cutCount=0
    maxCutCount=1e6
    best_UB=1e6
    function add_lazy_callback(cb_data)
        if cutCount<=maxCutCount
            # Y0=zeros(n,n)
            # for i=1:n
            #     for j=1:n
            #         Y0[i,j]=callback_value(cb_data, Y_ex[i,j])
            #     end
            # end
            # if eigvals(Y0)[1]<=-0.1
            #     u_t=eigvecs(Y0)[:,1]
            #     con3 = @build_constraint(Compat.dot(Y_ex,u_t*u_t') >=0.0)
            #     MOI.submit(m_ex, MOI.LazyConstraint(cb_data), con3)
            #     noPSDCuts+=1
            # end
            U0=zeros(n,k)
            for i=1:n
                for j=1:k
                    U0[i,j]=callback_value(cb_data, U_ex[i,j])
                end
            end
            for j=1:k
                U0[:,j]=U0[:,j]/norm(U0[:,j])
            end
            Y0=U0*U0'

            f_Y, alpha_t=getEDMcut(Y0, theIndices, M, n, gamma, lambda)
            if f_Y<best_UB
                @show best_UB=f_Y
            end
            b_t=f_Y+gamma/2*Compat.dot(alpha_t*alpha_t', Y0)

            A_t=(gamma/2)*alpha_t*alpha_t'

            con = @build_constraint(theta_ex>=b_t-Compat.dot(A_t, Y_ex))
            MOI.submit(m_ex, MOI.LazyConstraint(cb_data), con)
            cutCount+=1
        end
    end
    MOI.set(m_ex, MOI.LazyConstraintCallback(), add_lazy_callback)

    noPSDCuts=0
    maxPSDCuts=0
    function addPSDCuts(cb_data)
        if noPSDCuts<=maxPSDCuts
            Y0=zeros(n,n)
            for i=1:n
                for j=1:n
                    Y0[i,j]=callback_value(cb_data, Y_ex[i,j])
                end
            end
            if eigvals(Y0)[1]<=-0.1
                u_t=eigvecs(Y0)[:,1]
                con = @build_constraint(Compat.dot(Y_ex,u_t*u_t') >=0.0)
                MOI.submit(m_ex, MOI.UserCut(cb_data), con)
                noPSDCuts+=1
            end
        end
    end
    #MOI.set(m_ex, MOI.UserCutCallback(), addPSDCuts)

    f_Y, alpha_t=getEDMcut(Y_ws, theIndices, M, n, gamma, lambda)

    @constraint(m_ex, theta_ex>=f_Y-gamma/2*Compat.dot(alpha_t*alpha_t', U_ex*U_ex'-Y_ws))

    f_Y_ws=f_Y


    set_start_value.(Y_ex, Y_ws)

    optimize!(m_ex)

    @show cutCount, best_UB
    # Recover G_ex at the very end (alla rounding)
    Y_ex0=value.(Y_ex)
    u,v,=svd(Y_ex0)
    # @show v
    f_Y, alpha_t=getEDMcut(Y_ex0, theIndices, M, n, gamma, lambda)

    # Apply greedy rounding mechanism to purge any accumulated numerical errors
    u,v,w=LinearAlgebra.svd(Y_ex0)

    diag_v=zeros(n)
    diag_v[sortperm(v, rev=true)[1:k]].=1.0
    Y_rounded=u*Diagonal(diag_v)*u'

    # solve for G_ex, given the rounded Y

    m2=Model(Mosek.Optimizer)
    @variable(m2, G2[1:n, 1:n], PSD)
    @variable(m2, slack2[1:n, 1:n])
    @variable(m2, absslack2[1:n, 1:n])
    @constraint(m2, slack2.<=absslack2)
    @constraint(m2, -slack2.<=absslack2)
    for t=1:numEntriesSampled
        i=coords[entries[t],1]
        j=coords[entries[t],2]
       @constraint(m2, G2[i,i]+G2[j,j]-2*G2[i,j]+slack2[i,j]==M[i,j])
    end

    @objective(m2, Min, 1.0/(2.0*gamma)*sum(G2[i,j]^2 for i=1:n for j=1:n)+lambda*sum(absslack2)+sum(G2[i,i] for i=1:n))
    @constraint(m2, G2.==Y_rounded*G2)
    @suppress optimize!(m2)
    @show objective_value(m2)
    G_best=value.(G2)

   return G_best, MOI.get(m_ex, MOI.NodeCount()), cutCount, objective_value(m_ex)-objective_bound(m_ex)
   # Note: LB from the in-out method, which may not always converge to the sdo bound; UB from the greedy rounding which may sometimes differ from Gurobi's bound
end

function getMatrixFrobeniusNorm_cuttingplanesmultitree_exact_inout(coords, entries, M, n, numEntriesSampled, lambda, d_radio, k=sqrt(n), gamma=200*sqrt(n)/(numEntriesSampled), Y_ws=zeros(n,n), stabilizationPoint=k/n*Diagonal(ones(n)), useInOut=true)
    m=Model(Mosek.Optimizer) #First solve the continuous relaxation
    set_optimizer_attribute(m, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-10)
    set_optimizer_attribute(m, "MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-10)
    @variable(m, Y[1:n, 1:n], PSD)
    @constraint(m, Symmetric(Matrix(1.0I, n, n)-Y) in PSDCone());
    @variable(m, theta>=-1e4)
    @objective(m, Min, theta)
    @constraint(m, sum(Y[i,i] for i=1:n)<=k)

    m_ex=Model(Gurobi.Optimizer) #Second solve the exact problem
    set_optimizer_attribute(m_ex, "NonConvex", 2)
    set_optimizer_attribute(m_ex, "TimeLimit", 300.0)
    set_optimizer_attribute(m_ex, "MIPGap", 1e-2)
    set_optimizer_attribute(m_ex, "FuncPieceError", 1e-4)
    set_optimizer_attribute(m_ex, "FuncPieceLength", 1e-4)
    set_optimizer_attribute(m_ex, "Threads", 12)
    set_optimizer_attribute(m_ex, "Presolve", 0)
    set_optimizer_attribute(m_ex, "FuncMaxVal", 1e4)
    #set_optimizer_attribute(m_ex, "ResultFile", "initialMPSFile.mps")

    @variable(m_ex, Y_ex[1:n, 1:n], Symmetric)
    @constraint(m_ex, sum(Y_ex[i,i] for i=1:n)<=k)
    @variable(m_ex, U_ex[1:n, 1:k])

    @constraint(m_ex, strengthenY[i=1:n], Y_ex[i,i]>=sum(U_ex[i,t]^2 for t=1:k))
    @constraint(m_ex, defineY[i=1:n, j=i:n], Y_ex[i,j].<=U_ex[i,:]'*U_ex[j,:]+1e-4) #Note that we need only impose this in upper triangle, as Y is symmetric
    @constraint(m_ex, defineY2[i=1:n, j=i:n], Y_ex[i,j].>=U_ex[i,:]'*U_ex[j,:]-1e-4) #First 1e-3 is free

    for i=1:k
        for j=i:k
            ind=1.0*(i==j)
            @constraint(m_ex, U_ex[:,i]'*U_ex[:,j]>=1.0*ind-1e-4)
            @constraint(m_ex, U_ex[:,i]'*U_ex[:,j]<=1.0*ind+1e-4)
        end
    end
    # # Strengthen the formulation using second-order cone inequalities on Y
    @constraint(m_ex, imposeSOC[i=1:n, j=1:n], [Y_ex[i,i]+Y_ex[j,j]; Y_ex[i,i]-Y_ex[j,j]; 2.0*Y_ex[i,j]] in SecondOrderCone())
    @constraint(m_ex, imposeDiag[i=1:n], 0.0<=Y_ex[i,i]<=1.0)
    @variable(m_ex, theta_ex>=-1e4)
    @objective(m_ex, Min, theta_ex)

    theIndices=zeros(n,n)
    for t=1:Int(numEntriesSampled)
        theIndices[coords[entries[t],1], coords[entries[t],2]]=1.0
        theIndices[coords[entries[t],2], coords[entries[t],1]]=1.0 #Recall that everything is symmetric
    end

    UB=1e12
    LB=-1e12
    rootStabilizationTrick = :inOut
     #Use the solution to the SDP relaxation instead, where appropriate
    rootCutsSense=1
    ε= 1e-10
    λ = (rootStabilizationTrick == :inOut) ? .1 : 1.
    δ = (rootStabilizationTrick == :inOut || rootStabilizationTrick == :twoEps) ? 2*ε : 0.
    rootCutsLim=200
    rootCutCount = 0
    oaRootCutCount=0
    consecutiveNonImprov_1 = 0
    consecutiveNonImprov_2 = 0
    if useInOut
        Y_best=zeros(n,n)
        for epoch in 1:200
          @suppress optimize!(m)
          Ystar=value.(Y)
          stabilizationPoint += Ystar; stabilizationPoint /= 2

          if LB >= objective_value(m) - eps()
            if consecutiveNonImprov_1 == 5
              consecutiveNonImprov_2 += 1
            else
              consecutiveNonImprov_1 += 1
            end
          else
            if consecutiveNonImprov_1 < 5
              consecutiveNonImprov_1 = 0
            end
            consecutiveNonImprov_2 = 0
          end
          LB = max(LB, objective_value(m))

          if consecutiveNonImprov_1 == 5
            λ = 1
          elseif consecutiveNonImprov_2 == 5
            δ = 0.
          end

          Y0 = λ*Ystar + (1-λ)*stabilizationPoint .+ δ*Diagonal(ones(n))

          # Perform an SVD step here to ensure still in convex hull
          u,v,=svd(Y0)
          v=v*k/sum(v)
          v.=min.(ones(n), max.(v, zeros(n)))

          Y0=u*Diagonal(v)*u'

          f_Y, alpha_t=getEDMcut(Y0, theIndices, M, n, gamma, lambda)
          if f_Y<UB
              Y_best=Y0
              UB=f_Y
          end
          LB=max(LB, objective_value(m))
          # Impose constraint
          @constraint(m, theta>=f_Y-gamma/2*Compat.dot(alpha_t*alpha_t', Y-Y0))
          # Also impose constraint in exact master problem (this is the whole point of the cut loop)
          if abs(f_Y-LB)>1e-3
              @show "Applied cut"
              @constraint(m_ex, theta_ex>=f_Y-gamma/2*Compat.dot(alpha_t*alpha_t', Y_ex-Y0))
          end
          #@constraint(m_ex, theta_ex>=f_Y-gamma/2*Compat.dot(alpha_t*alpha_t', U_ex*U_ex'-Y0))
          # @show maximum(abs.(alpha_t*alpha_t'))
          # @show minimum(abs.(alpha_t*alpha_t'))
          # @show f_Y
          # Add cut to pool (add in later if it works)
          @show LB, UB, epoch
          if abs(UB-LB) <= 1e-4 || consecutiveNonImprov_2 >= 10
              break
          end
        end
    end



    U_ws,=svd(Y_ws) #Compute Y_ws given X_ws, then use SVD to make sure it is indeed of rank at most k
    U_ws=U_ws[:, 1:k] # Julia automatically orders the columns of U in terms of the size of their singular values

    #set_start_value.(Y_ex, Y_ws)
    set_start_value.(U_ex, U_ws)


    f_Y, alpha_t=getEDMcut(Y_ws, theIndices, M, n, gamma, lambda)

    @constraint(m_ex, theta_ex>=f_Y-gamma/2*Compat.dot(alpha_t*alpha_t', Y_ex-Y_ws))

    noPSDCuts=0
    maxPSDCuts=100
    function addPSDCuts(cb_data)
        if noPSDCuts<=maxPSDCuts
            Y0=zeros(n,n)
            for i=1:n
                for j=1:n
                    Y0[i,j]=callback_value(cb_data, Y_ex[i,j])
                end
            end
            if eigvals(Y0)[1]<=-0.1
                u_t=eigvecs(Y0)[:,1]
                con = @build_constraint(Compat.dot(Y_ex,u_t*u_t') >=0.0)
                MOI.submit(m_ex, MOI.UserCut(cb_data), con)
                noPSDCuts+=1
            end
        end
    end
    MOI.set(m_ex, MOI.UserCutCallback(), addPSDCuts)

    UB_master=1e12
    LB_master=-1e12
    cutCount=0
    Y_best=zeros(n,n)
    maxOAIters=50
    consecNonImprove=0 #Use to decide if we need to perform a more expensive iteration, since the lower bound isn't changing
    for t=1:maxOAIters
        if t>1
            set_optimizer_attribute(m_ex, "MIPGap", 1e-1/t)
            set_optimizer_attribute(m_ex, "Threads", 12)
            set_optimizer_attribute(m_ex, "Presolve", 0)
            noPSDCuts=0
            set_optimizer_attribute(m_ex, "TimeLimit", min(300.0*t, 3600.0))
            #set_optimizer_attribute(m_ex, "StartNodeLimit", 1000) #Limits the number of nodes used to compute a warm-start, in order to avoid excess time spent on the WS
            set_optimizer_attribute(m_ex, "PoolSolutions", 2) #No point in keeping a large number of solutions; just slows down the next iteration
            if t<=2
                set_start_value.(Y_ex, value.(Y_ex))
                set_start_value.(U_ex, value.(U_ex))
            end
            if consecNonImprove>=2
                set_optimizer_attribute(m_ex, "TimeLimit", 3600.0)  #Expensive iteration, to move the bound
                consecNonImprove=0
            end

        end

        @time optimize!(m_ex)
        Y_iter=value.(Y_ex)
        U_iter=value.(U_ex)
        # Impose soc callback cut
        if eigvals(Y_iter-U_iter*U_iter')[1]<=-0.01
            u_t=eigvecs(Y_iter-U_iter*U_iter')[:,1]
            @constraint(m_ex, [u_t'*Y_ex*u_t+1.0, u_t'*Y_ex*u_t-1.0, 2.0*U_iter'*u_t] in SecondOrderCone())
        end
        # Apply cut at fractional point first
        f_Y, alpha_t=getEDMcut(Y_iter, theIndices, M, n, gamma, lambda)
        @show f_Y
        @constraint(m_ex, theta_ex>=f_Y-gamma/2*Compat.dot(alpha_t*alpha_t', U_ex*U_ex'-Y_iter))
        # Now also apply cut at rounded version of point
        u,v,=svd(Y_iter)
        Y_iter=u[:,1:k]*u[:,1:k]'
        LB_diff=objective_bound(m_ex)-LB_master
        LB_master=max(LB_master, objective_bound(m_ex))
        f_Y, alpha_t=getEDMcut(Y_iter, theIndices, M, n, gamma, lambda)
        @show f_Y, v
        @constraint(m_ex, theta_ex>=f_Y-gamma/2*Compat.dot(alpha_t*alpha_t', U_ex*U_ex'-Y_iter))
        if f_Y<UB_master
            UB_master=f_Y
            Y_best=Y_iter
        elseif LB_diff<1e-3 #Neither bound moved, so need a more expensive iteration
            consecNonImprove+=1
        end
        cutCount+=1
        @show LB_master, UB_master, t
        if abs(UB_master-LB_master) <= 1e-2
            break
        end
    end
    Y_rounded=Y_best

    m2=Model(Mosek.Optimizer)
    @variable(m2, G2[1:n, 1:n], PSD)
    @variable(m2, slack2[1:n, 1:n])
    @variable(m2, absslack2[1:n, 1:n])
    @constraint(m2, slack2.<=absslack2)
    @constraint(m2, -slack2.<=absslack2)
    for t=1:numEntriesSampled
        i=coords[entries[t],1]
        j=coords[entries[t],2]
       @constraint(m2, G2[i,i]+G2[j,j]-2*G2[i,j]+slack2[i,j]==M[i,j])
    end

    @objective(m2, Min, 1.0/(2.0*gamma)*sum(G2[i,j]^2 for i=1:n for j=1:n)+lambda*sum(absslack2)+sum(G2[i,i] for i=1:n))
    @constraint(m2, G2.==Y_rounded*G2)
    @suppress optimize!(m2)
    @show objective_value(m2)
    G_best=value.(G2)

   return G_best, MOI.get(m_ex, MOI.NodeCount()), cutCount, objective_value(m_ex)-objective_bound(m_ex)
end


# Version where the objective is to minimize the trace of G (i.e., many solutions are feasible, since the problem is underdetermined)
function getEDMcut(Y0::Array{Float64, 2}, theIndices::Array{Float64, 2}, D::Array{Float64, 2}, n::Int64, gamma::Float64, lambda::Float64)
    u,v=svd(Y0)
    u0=u*Diagonal(sqrt.(v))

    m2=Model(Mosek.Optimizer)
    set_optimizer_attribute(m2, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-8)
    set_optimizer_attribute(m2, "MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-8)
    @variable(m2, alpha[1:n, 1:n], Symmetric)
    @variable(m2, r[1:n, 1:n])
    @variable(m2, Pi[1:n, 1:n], Symmetric)
    @variable(m2, rho[1:n])
    @constraint(m2, (2.0*Pi.+diagm(ones(n)).+alpha.-2.0*diagm(rho)) in PSDCone()) #Note from Ryan: the diagm(ones(n)) term implies there is trace(G) term in the primal objective
    @constraint(m2, Pi.*(ones(n,n).-theIndices).==0.0)
    @constraint(m2, r.==alpha'*u0)
    @constraint(m2, Pi.<=lambda) #Corresponds to a slack term in the constraint, which is penalized with an l_1 penalty term, so ofv=lambda||eps||_1+tr(G)-can add a lambda to tr(G) if desired
    @constraint(m2, -Pi.<=lambda)
    @constraint(m2, Pi*ones(n).==rho)
    @objective(m2, Max, sum((Pi[i,j]*D[i,j])*theIndices[i,j] for i=1:n for j=1:n)-(gamma/2.0)*Compat.dot(r,r))
    @suppress optimize!(m2)
    # @show objective_value(m2)
    alpha_t=value.(alpha)
    # Get UB and cut from second-stage optimization problem
    f_Y=objective_value(m2)
    return f_Y, alpha_t
end
