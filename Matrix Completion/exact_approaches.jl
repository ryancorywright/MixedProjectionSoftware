using JuMP, Gurobi, LinearAlgebra, Suppressor, StatsBase, Compat, MathOptInterface

mutable struct cut
    intercept::Float64
    slope::Array{Float64,2}
end

mutable struct CutIterData #used for plotting performance of method
    time::Float64
    obj::Float64
    bound::Float64
end

# This is a QCQP formulation of the problem
function getMatrixFrobeniusNorm_exact(coords, entries, M, n, numEntriesSampled, k=sqrt(n), gamma=200*sqrt(n)/(numEntriesSampled), imposeConstraintsLazily=true, u_init=rand(n,k), v_init=rand(n,k))

    m=Model(Gurobi.Optimizer)
    set_optimizer_attribute(m, "NonConvex", 2)
    set_optimizer_attribute(m, "TimeLimit", 3600.0)
    set_optimizer_attribute(m, "MIPGap", 1e-2)
    @variable(m, u[1:n, 1:k])
    @variable(m, v[1:n, 1:k])
    @variable(m, X[1:n, 1:n])
    for l=1:(k-1)
        for l2=2:k
            @constraint(m, u[:,l]'*u[:,l2]<=1e-4) #orthogonality constraint, since columns are orthogonal in an svd decomposition.
            @constraint(m, u[:,l]'*u[:,l2]>=-1e-4)
            @constraint(m, v[:,l]'*v[:,l2]<=1e-4)
            @constraint(m, v[:,l]'*v[:,l2]>=-1e-4)
        end
    end
    @constraint(m, X.>=u*v'.-1e-6*ones(n,n))
    @constraint(m, X.<=u*v'.+1e-6*ones(n,n))
    @constraint(m, 0 .<= -X.+n*ones(n,n))
    @constraint(m, 0 .<= X.+n*ones(n,n))
    for i=1:k
        @constraint(m, [1.0; u[:,i]] in SecondOrderCone())
    end
    @constraint(m, 0 .<=u.+1.0*ones(n,k))
    @constraint(m, 0 .<=v.+n*ones(n,k))
    @constraint(m, 0 .<=-u.+1.0*ones(n,k))
    @constraint(m, 0 .<=-v.+n*ones(n,k))


    @objective(m, Min, 1.0/(2.0*gamma)*sum(X[i,j]^2 for i=1:n for j=1:n)+0.5*sum((X[coords[entries[t],1], coords[entries[t],2]]-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)))
    optimize!(m)

    return value.(u)*value.(v)', MOI.get(m, MOI.NodeCount()), objective_value(m)-objective_bound(m)
end

function getMatrixFrobeniusNorm_cuttingplanes_exact_inout(coords, entries, M, n, numEntriesSampled, k=sqrt(n), gamma=200*sqrt(n)/(numEntriesSampled), stabilizationPoint=k/n*Diagonal(ones(n)), X_ws=zeros(n,n), useInOut=true, useSOCs=true, useKelleys=false)
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
    set_optimizer_attribute(m_ex, "TimeLimit", 30000.0)
    set_optimizer_attribute(m_ex, "MIPGap", 1e-4)
    set_optimizer_attribute(m_ex, "FuncPieceError", 1e-6)
    set_optimizer_attribute(m_ex, "FuncPieceLength", 1e-5)
    set_optimizer_attribute(m_ex, "Threads", 12)
    set_optimizer_attribute(m_ex, "Presolve", 0)

    @variable(m_ex, Y_ex[1:n, 1:n], Symmetric)
    @constraint(m_ex, sum(Y_ex[i,i] for i=1:n)<=k)
    @variable(m_ex, U_ex[1:n, 1:k])

    @variable(m_ex, theta_ex>=-1e4)
    @objective(m_ex, Min, theta_ex)

    @constraint(m_ex, defineY[i=1:n, j=i:n], Y_ex[i,j].<=U_ex[i,:]'*U_ex[j,:]+1e-4) #Note that we need only impose this in upper triangle, as Y is symmetric
    @constraint(m_ex, defineY2[i=1:n, j=i:n], Y_ex[i,j].>=U_ex[i,:]'*U_ex[j,:]-1e-4)

    bcdata = CutIterData[]

    for i=1:k
        for j=i:k
            ind=1.0*(i==j)
            @constraint(m_ex, U_ex[:,i]'*U_ex[:,j]<=1.0*ind+1e-6)
            @constraint(m_ex, U_ex[:,i]'*U_ex[:,j]>=1.0*ind-1e-6)
        end
    end
    if useSOCs
    # Strengthen the formulation using second-order cone inequalities on Y
        @constraint(m_ex, imposeSOC[i=1:n, j=1:n], [Y_ex[i,i]+Y_ex[j,j]; Y_ex[i,i]-Y_ex[j,j]; 2.0*Y_ex[i,j]] in SecondOrderCone())
        #@constraint(m_ex, revSOC[i=1:n, j=1:n], Y_ex[i,i]*Y_ex[j,j]<=Y_ex[i,j]^2+1e-4)
    # else # omitted, but could potentially use a diagonally dominant matrix approximation per Ahmadi/Hall
    #     @constraint(m_ex, imposeDD1[i=1:n, j=1:n], Y_ex[i,i]+Y_ex[j,j]+2.0*Y_ex[i,j]>=0.0)
    #     @constraint(m_ex, imposeDD2[i=1:n, j=1:n], Y_ex[i,i]+Y_ex[j,j]-2.0*Y_ex[i,j]>=0.0)
    end

    @constraint(m_ex, imposeDiag[i=1:n], 0.0<=Y_ex[i,i]<=1.0)


    theIndices=zeros(n,n)
    for t=1:Int(numEntriesSampled)
        theIndices[coords[entries[t],1], coords[entries[t],2]]=1.0
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
          f_Y, alpha_t=getmatrixcompcut(Y0, theIndices, M, n, gamma)
          if f_Y<UB
              Y_best=Y0
              UB=f_Y
          end
          LB=max(LB, objective_value(m))
          # Impose constraint
          @constraint(m, theta>=f_Y-gamma/2*Compat.dot(alpha_t*alpha_t', Y-Y0))

          @constraint(m_ex, theta_ex>=f_Y-gamma/2*Compat.dot(alpha_t*alpha_t', Y_ex-Y0))
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
            Y0=zeros(n,n)
            for i=1:n
                for j=1:n
                    Y0[i,j]=callback_value(cb_data, Y_ex[i,j])
                end
            end
            if eigvals(Y0)[1]<=-0.1
                u_t=eigvecs(Y0)[:,1]
                con3 = @build_constraint(Compat.dot(Y_ex,u_t*u_t') >=0.0)
                MOI.submit(m_ex, MOI.LazyConstraint(cb_data), con3)
                noPSDCuts+=1
            end
            U0=zeros(n,k)
            for i=1:n
                for j=1:k
                    U0[i,j]=callback_value(cb_data, U_ex[i,j])
                end
            end

            for j=1:k
                if norm(U0[:,j])>1e-4
                    U0[:,j]=U0[:,j]/norm(U0[:,j])
                else
                    U0[1,j]=1.0
                    U0[:,j]=U0[:,j]/norm(U0[:,j])
                end
            end
            #@show U0'*U0
            Y0=U0*U0'

            f_Y, alpha_t=getmatrixcompcut(Y0, theIndices, M, n, gamma)
            if f_Y<best_UB
                @show best_UB=f_Y
            end
            b_t=f_Y+gamma/2*Compat.dot(alpha_t*alpha_t', Y0)
            # f_0=b_t-floor(b_t)

            A_t=(gamma/2)*alpha_t*alpha_t'#+1e-3*Diagonal(ones(n))

            con = @build_constraint(theta_ex>=b_t-Compat.dot(A_t, Y_ex))
            MOI.submit(m_ex, MOI.LazyConstraint(cb_data), con)
            #MOI.submit(m_ex, MOI.LazyConstraint(cb_data), con2)
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

    U_ws,=svd(X_ws) #Compute Y_ws given X_ws, then use SVD to make sure it is indeed of rank at most k
    U_ws=U_ws[:, 1:k] # Julia automatically orders the columns of U in terms of the size of their singular values

    Y_ws=U_ws*U_ws'


    f_Y, alpha_t=getmatrixcompcut(Y_ws, theIndices, M, n, gamma)

    @constraint(m_ex, theta_ex>=f_Y-gamma/2*Compat.dot(alpha_t*alpha_t', Y_ex-Y_ws))

    f_Y_ws=f_Y


    # set_start_value(theta_ex, f_Y_ws)
    set_start_value.(U_ex, U_ws)
    set_start_value.(Y_ex, Y_ws)

    optimize!(m_ex)

    @show cutCount, best_UB

    Y_ex0=zeros(n,n)
    status = MOI.get(m_ex, MOI.TerminationStatus())
    primalstatus=MOI.get(m_ex, MOI.PrimalStatus())
    nodecount= MOI.get(m_ex, MOI.NodeCount())
    objval= -1.0
    objbound= 0.0

    if status == MOI.OPTIMAL || primalstatus==MOI.FEASIBLE_POINT
        # Set Y_ex0 equal to incumbent
        Y_ex0.=value.(Y_ex)
        objval=objective_value(m_ex)
        objbound=objective_bound(m_ex)
    else
        # No incumbent, set Y_ex0 equal to warm start (Gurobi hasn't found a solution due to numerical resolution issues), keep obj val and obj bound such that gap is -1.0
        Y_ex0.=Y_ws
    end

    # Recover X_ex at the very end (alla rounding)

    # Apply greedy rounding mechanism to purge any accumulated numerical errors
    u,v,w=LinearAlgebra.svd(Y_ex0)

    diag_v=zeros(n)
    diag_v[sortperm(v, rev=true)[1:k]].=1.0
    Y_rounded=u*Diagonal(diag_v)*u'

    # solve for X_ex

    m2=Model(Mosek.Optimizer)
    @variable(m2, X2[1:n, 1:n])
    @objective(m2, Min, 1.0/(2.0*gamma)*sum(X2[i,j]^2 for i=1:n for j=1:n)+sum((X2[coords[entries[t],1], coords[entries[t],2]]-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)))
    @constraint(m2, X2.==Y_rounded*X2)
    @suppress optimize!(m2)
    X_best=value.(X2)

   return X_best, nodecount, cutCount, objval-objbound
   # Note: LB from the in-out method, which may not always converge to the sdo bound; UB from the greedy rounding which may sometimes differ from Gurobi's bound
end

function getMatrixFrobeniusNorm_cuttingplanesmultitree_exact_inout(coords, entries, M, n, numEntriesSampled, k=sqrt(n), gamma=200*sqrt(n)/(numEntriesSampled), stabilizationPoint=k/n*Diagonal(ones(n)), X_ws=zeros(n,n), useInOut=true, fileName="")
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
    set_optimizer_attribute(m_ex, "TimeLimit", 30.0)
    set_optimizer_attribute(m_ex, "MIPGap", 1e-2)
    set_optimizer_attribute(m_ex, "FuncPieceError", 1e-6)
    set_optimizer_attribute(m_ex, "FuncPieceLength", 1e-5)
    set_optimizer_attribute(m_ex, "Threads", 12)
    set_optimizer_attribute(m_ex, "Presolve", 0)
    #set_optimizer_attribute(m_ex, "Heuristics", 0)
    set_optimizer_attribute(m_ex, "FuncMaxVal", 1e4)
    # set_optimizer_attribute(m_ex, "ResultFile", "theResultFile.sol")

    @variable(m_ex, Y_ex[1:n, 1:n], Symmetric)
    @constraint(m_ex, sum(Y_ex[i,i] for i=1:n)==k)
    @variable(m_ex, U_ex[1:n, 1:k])

    @constraint(m_ex, strengthenY0[i=1:n], Y_ex[i,i]>=sum(U_ex[i,t]^2 for t=1:k)) #Strengthen the relaxation
    @constraint(m_ex, strengthenY1[i=1:n, j=1:n], [Y_ex[i,i]+Y_ex[j,j]-2.0*Y_ex[i,j]+1.0;Y_ex[i,i]+Y_ex[j,j]-2.0*Y_ex[i,j]-1.0; U_ex[i,:]-U_ex[j,:]] in SecondOrderCone())
    @constraint(m_ex, strengthenY2[i=1:n, j=1:n], [Y_ex[i,i]+Y_ex[j,j]+2.0*Y_ex[i,j]+1.0;Y_ex[i,i]+Y_ex[j,j]+2.0*Y_ex[i,j]-1.0; U_ex[i,:]+U_ex[j,:]] in SecondOrderCone())


    @constraint(m_ex, defineY[i=1:n, j=i:n], Y_ex[i,j].<=U_ex[i,:]'*U_ex[j,:]+1e-4)
    @constraint(m_ex, defineY2[i=1:n, j=i:n], Y_ex[i,j].>=U_ex[i,:]'*U_ex[j,:]-1e-4)

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
    # theCutPool=Cut[]
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

          f_Y, alpha_t=getmatrixcompcut(Y0, theIndices, M, n, gamma)
          if f_Y<UB
              Y_best=Y0
              UB=f_Y
          end
          LB=max(LB, objective_value(m))
          # Impose constraint
          @constraint(m, theta>=f_Y-gamma/2*Compat.dot(alpha_t*alpha_t', Y-Y0))
          # Also impose constraint in exact master problem (this is the whole point of the cut loop)
          @constraint(m_ex, theta_ex>=f_Y-gamma/2*Compat.dot(alpha_t*alpha_t', Y_ex-Y0))
          # Add cut to pool (add in later if it works)
          @show LB, UB, epoch
          if abs(UB-LB) <= 1e-4 || consecutiveNonImprov_2 >= 10
              break
          end
        end
    end



    U_ws,=svd(X_ws) #Compute Y_ws given X_ws, then use SVD to make sure it is indeed of rank at most k
    U_ws=U_ws[:, 1:k] # Julia automatically orders the columns of U in terms of the size of their singular values
    Y_ws=U_ws*U_ws'
    # @show Y_ws[1,1]
    # @show typeof(Y_ws[1,1])

    set_start_value.(Y_ex, Y_ws)
    set_start_value.(U_ex, U_ws)



    f_Y, alpha_t=getmatrixcompcut(Y_ws, theIndices, M, n, gamma)
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
    nc_agg=0
    Y_best=zeros(n,n)
    maxOAIters=20
    epochData = DataFrame(n=Int64[], gamma=Float64[], iter=Int64[], nodes=Float64[])

    theNodeData = similar(epochData, 0)

    consecNonImprove=0 #Use to decide if we need to perform a more expensive iteration, since the lower bound isn't changing
    for t=1:maxOAIters
        if t>1
            set_optimizer_attribute(m_ex, "MIPGap", 1e-1/t)
            set_optimizer_attribute(m_ex, "Threads", 12)
            set_optimizer_attribute(m_ex, "Presolve", 0)
            noPSDCuts=0
            set_optimizer_attribute(m_ex, "TimeLimit", min(10.0*t, 300.0))
            set_optimizer_attribute(m_ex, "PoolSolutions", 2) #No point in keeping a large number of solutions; just slows down the next iteration
            if t<=2
                set_start_value.(Y_ex, value.(Y_ex))
                set_start_value.(U_ex, value.(U_ex))
            end
            if consecNonImprove>=2
                set_optimizer_attribute(m_ex, "TimeLimit", 300.0)  #Expensive iteration, to move the bound
                consecNonImprove=0
            end

        end

        @time optimize!(m_ex)
        U_iter=value.(U_ex)
        Y_iter=value.(Y_ex)
        # Impose soc callback cut
        if eigvals(Y_iter-U_iter*U_iter')[1]<=-0.01
            u_t=eigvecs(Y_iter-U_iter*U_iter')[:,1]
            @constraint(m_ex, [u_t'*Y_ex*u_t+1.0, u_t'*Y_ex*u_t-1.0, 2.0*U_iter'*u_t] in SecondOrderCone())
        end
        # Impose OA cut at the exact value identified by Gurobi
        f_Y, alpha_t=getmatrixcompcut(Y_iter, theIndices, M, n, gamma)
        @constraint(m_ex, theta_ex>=f_Y-gamma/2*Compat.dot(alpha_t*alpha_t', Y_ex-Y_iter))

        # Now round the matrix to ensure that it is infact an orthogonal projection matrix, and impose a cut there too.
        u,v,=svd(Y_iter)
        Y_iter=u[:,1:k]*u[:,1:k]'
        LB_diff=objective_bound(m_ex)-LB_master
        LB_master=max(LB_master, objective_bound(m_ex))
        f_Y, alpha_t=getmatrixcompcut(Y_iter, theIndices, M, n, gamma)
        @constraint(m_ex, theta_ex>=f_Y-gamma/2*Compat.dot(alpha_t*alpha_t', Y_ex-Y_iter))
        if f_Y<UB_master
            UB_master=f_Y
            Y_best=Y_iter
        elseif LB_diff<1e-3 #Neither bound moved, so need a more expensive iteration
            consecNonImprove+=1
        end
        @show typeof(n)
        @show typeof(gamma)
        @show typeof(cutCount)
        @show typeof(MOI.get(m_ex, MOI.NodeCount()))
        @show size(MOI.get(m_ex, MOI.NodeCount()))
        nc_agg+=MOI.get(m_ex, MOI.NodeCount())
        push!(theNodeData, [n, gamma, cutCount, MOI.get(m_ex, MOI.NodeCount())])
        cutCount+=1
        @show LB_master, UB_master, t
        @show MOI.get(m_ex, MOI.NodeCount())
        if abs(UB_master-LB_master) <= 1e-2
            break
        end
    end



    CSV.write("nodesvsiters"*fileName*".csv", theNodeData)


    @show cutCount
    @show MOI.get(m_ex, MOI.NodeCount())
    # Recover X_ex at the very end (alla rounding)
    Y_ex0=Y_best#value.(Y_ex)
    # Apply greedy rounding mechanism to purge any accumulated numerical errors
    u,v,w=LinearAlgebra.svd(Y_ex0)

    diag_v=zeros(n)
    diag_v[sortperm(v, rev=true)[1:k]].=1.0
    Y_rounded=u*Diagonal(diag_v)*u'

    # solve for X_ex

    m2=Model(Mosek.Optimizer)
    @variable(m2, X2[1:n, 1:n])
    @objective(m2, Min, 1.0/(2.0*gamma)*sum(X2[i,j]^2 for i=1:n for j=1:n)+sum((X2[coords[entries[t],1], coords[entries[t],2]]-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)))
    @constraint(m2, X2.==Y_rounded*X2)
    @suppress optimize!(m2)
    X_best=value.(X2)


   return X_best, MOI.get(m_ex, MOI.NodeCount()), cutCount, objective_value(m_ex)-max(objective_bound(m_ex), objective_bound(m)), nc_agg
   # Note: LB from the in-out method, which may not always converge to the sdo bound; UB from the greedy rounding which may sometimes differ from Gurobi's bound
end


function getmatrixcompcut(Y0::Array{Float64, 2}, theIndices::Array{Float64, 2}, M::Array{Float64, 2}, n::Int64, gamma::Float64)
    u,v=svd(0.5*Y0+0.5*Y0')

    u0=u*Diagonal(sqrt.(v))
    m2=Model(Mosek.Optimizer)
    set_optimizer_attribute(m2, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-8)
    set_optimizer_attribute(m2, "MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-8)
    @variable(m2, alpha[1:n, 1:n])
    @variable(m2, r[1:n, 1:n])
    @constraint(m2, r.==alpha'*u0)
    @constraint(m2, alpha.*(ones(n,n)-theIndices).==0.0) # If not in set of indices then eliminate variable
    @objective(m2, Max, -0.5*sum((alpha[i,j]+M[i,j])^2*theIndices[i,j] for i=1:n for j=1:n)-(gamma/2.0)*Compat.dot(r,r))
    @suppress optimize!(m2)
    alpha_t=value.(alpha).*theIndices
    # Get UB and cut from second-stage optimization problem
    f_Y=objective_value(m2)+0.5*sum((M.*theIndices).^2)
    return f_Y, alpha_t
end
# Pareto-optimal version, assuming we have access to k
function getmatrixcompcut_pareto(Y0::Array{Float64, 2}, theIndices::Array{Float64, 2}, M::Array{Float64, 2}, n::Int64, gamma::Float64, k::Int64)
    u,v=svd(Y0)
    u_inner=(sqrt(k)/sqrt(n))*Diagonal(ones(n)) #For the Pareto-optimal cut

    u0=u*Diagonal(sqrt.(v))
    m2=Model(Mosek.Optimizer)
    set_optimizer_attribute(m2, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-8)
    set_optimizer_attribute(m2, "MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-8)
    @variable(m2, alpha[1:n, 1:n])
    @variable(m2, r[1:n, 1:n])
    @constraint(m2, r.==alpha'*u0)
    @constraint(m2, alpha.*(ones(n,n)-theIndices).==0.0) # If not in set of indices then eliminate variable
    @objective(m2, Max, -0.5*sum((alpha[i,j]+M[i,j])^2*theIndices[i,j] for i=1:n for j=1:n)-(gamma/2.0)*Compat.dot(r,r))
    @suppress optimize!(m2)
    # Get UB and cut from second-stage optimization problem
    f_Y=objective_value(m2)+0.5*sum((M.*theIndices).^2)

    m3=Model(Mosek.Optimizer)
    set_optimizer_attribute(m3, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-8)
    set_optimizer_attribute(m3, "MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-8)
    @variable(m3, alpha2[1:n, 1:n])
    @variable(m3, r2[1:n, 1:n])
    @constraint(m3, r2.==alpha2'*u0)
    @variable(m3, r3[1:n, 1:n])
    @constraint(m3, r3.==alpha2'*u_inner)
    @constraint(m3, alpha2.*(ones(n,n)-theIndices).==0.0) # If not in set of indices then eliminate variable
    @constraint(m3, -0.5*sum((alpha2[i,j]+M[i,j])^2*theIndices[i,j] for i=1:n for j=1:n)-(gamma/2.0)*Compat.dot(r2,r2)>=objective_value(m2)) #Ensures that the cut supplies the right OFV
    @objective(m3, Max, -0.5*sum((alpha2[i,j]+M[i,j])^2*theIndices[i,j] for i=1:n for j=1:n)-(gamma/2.0)*Compat.dot(r3,r3))
    @suppress optimize!(m3)
    alpha_t=value.(alpha2).*theIndices
    return f_Y, alpha_t
end
