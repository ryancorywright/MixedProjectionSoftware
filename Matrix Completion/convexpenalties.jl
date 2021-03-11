using JuMP, Mosek, MosekTools, LinearAlgebra, Suppressor, StatsBase#, SCS#, Ipopt


function getMatrixNuclearNorm(coords, entries, M_noisy, n, numEntriesSampled, mu=sqrt(2*numEntriesSampled/n), imposeConstraints=false) #mu picked in line with Candes and Plan
    m=Model(Mosek.Optimizer)
    @variable(m, X[1:n, 1:n])
    @variable(m, U[1:n, 1:n])
    @variable(m, V[1:n, 1:n])
    if imposeConstraints
        for t=1:Int(numEntriesSampled)
            @constraint(m, X[coords[entries[t],1], coords[entries[t],2]]==M_noisy[coords[entries[t],1], coords[entries[t],2]])
        end
        @objective(m, Min, sum(U[i,i] for i=1:n)+sum(V[i,i] for i=1:n))
    else
        @objective(m, Min, mu*sum(U[i,i] for i=1:n)+mu*sum(V[i,i] for i=1:n)+sum((X[coords[entries[t],1], coords[entries[t],2]]-M_noisy[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)));
    end
    @constraint(m, Symmetric([U X; X' V]) in PSDCone());
    @suppress optimize!(m)
   return value.(X)
end

function getMatrixFrobeniusNorm_reg2ndstage(coords, entries, M, n, numEntriesSampled, k=sqrt(n), gamma=200*sqrt(n)/(numEntriesSampled), imposeConstraints=false)
    m=Model(Mosek.Optimizer)

    @variable(m, X[1:n, 1:n])
    @variable(m, Y[1:n, 1:n], Symmetric)
    @variable(m, theta[1:n, 1:n])


    if imposeConstraints
        for t=1:Int(numEntriesSampled)
            @constraint(m, X[coords[entries[t],1], coords[entries[t],2]]==M[coords[entries[t],1], coords[entries[t],2]])
        end
        @objective(m, Min, 1.0/(2.0*gamma)*sum(theta[i,i] for i=1:n))
    else
        @objective(m, Min, 1.0/(2.0*gamma)*sum(theta[i,i] for i=1:n)+sum((X[coords[entries[t],1], coords[entries[t],2]]-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)))
    end

    @constraint(m, Symmetric(Matrix(1.0I, n, n)-Y) in PSDCone());

    @constraint(m, Symmetric([Y X; X' theta]) in PSDCone());
    @constraint(m, sum(Y[i,i] for i=1:n)<=k)


    @suppress optimize!(m)
    u,v,w=LinearAlgebra.svd(value.(Y))

    # do 100 iterations of randomized rounding and keep best solution in terms of in-sample objective.
    X_best=zeros(n,n)
    ofv_best=1e10
    numIters=100
    for i=1:numIters
        v_rounded=(rand(n).<v)
        if sum(v_rounded)<=k
            Y_rounded=u*Diagonal(v_rounded)*u'
            m2=Model(with_optimizer(Mosek.Optimizer))
            @variable(m2, X2[1:n, 1:n])
            @objective(m2, Min, 1.0/(2.0*gamma)*sum(X2[i,j]^2 for i=1:n for j=1:n)+sum((X2[coords[entries[t],1], coords[entries[t],2]]-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)))
            @constraint(m2, X2.==Y_rounded*X2)
            @suppress optimize!(m2)
            if objective_value(m2) < ofv_best
               ofv_best = objective_value(m2)
               X_best=value.(X2)
            end
        end
    end
   return X_best
end

function getMatrixFrobeniusNorm_greedy(coords, entries, M, n, numEntriesSampled, k=sqrt(n), gamma=200*sqrt(n)/(numEntriesSampled), imposeConstraints=false, useRotation=false)
    m=Model(Mosek.Optimizer)

    @variable(m, X[1:n, 1:n])
    @variable(m, Y[1:n, 1:n], Symmetric)
    @variable(m, theta[1:n, 1:n])
    @variable(m, sigma>=0.0)


    @constraint(m, sigma>=1.0/(2.0*gamma)*sum(theta[i,i] for i=1:n))
    @SDconstraint(m, Matrix(1.0I, n, n)>=Y);

    if imposeConstraints
        for t=1:Int(numEntriesSampled)
            @constraint(m, X[coords[entries[t],1], coords[entries[t],2]]==M[coords[entries[t],1], coords[entries[t],2]])
        end

        @objective(m, Min, sigma)
    else
        @objective(m, Min, sigma+0.5*sum((X[coords[entries[t],1], coords[entries[t],2]]-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)))
    end
    @constraint(m, Symmetric([Y X; X' theta]) in PSDCone());
    @constraint(m, sum(Y[i,i] for i=1:n)<=k)

    @suppress optimize!(m)
    @show objective_value(m)
    u,v,w=LinearAlgebra.svd(value.(Y))

    diag_v=zeros(n)
    if useRotation
        # Compute alpha, use to compute weights (leaving out -1/gamma term as scales out from squaring)
        @show "Using rotation"
        alpha_star=pinv(value.(Y))*value.(X) #strictly speaking only proportional to this
        w_full=alpha_star*alpha_star'
        diag_v[sortperm(diag(w_full), rev=true)[1:k]].=1.0
    else
        diag_v[sortperm(v, rev=true)[1:k]].=1.0
        # diag_v2[sortperm(v2, rev=true)[1:k]].=1.0
    end
    @show sum(diag_v)
    Y_rounded=u*Diagonal(diag_v)*u'
    # Y_rounded2=u2*Diagonal(diag_v2)*u2'
    @show sum(abs.(Y_rounded))

    m2=Model(with_optimizer(Mosek.Optimizer))
    @variable(m2, X2[1:n, 1:n])
    @objective(m2, Min, 1.0/(2.0*gamma)*sum(X2[i,j]^2 for i=1:n for j=1:n)+0.5*sum((X2[coords[entries[t],1], coords[entries[t],2]]-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)))
    @constraint(m2, X2.==Y_rounded*X2)
    # @constraint(m2, X2.==X2*Y_rounded2)
    @suppress optimize!(m2)
    X_final=value.(X2)
    @show sum(abs.(value.(X2)))
    #@show 1.0/(2.0*gamma)*sum(value(X2[i,j])^2 for i=1:n for j=1:n)+0.5*sum((value(X2[coords[entries[t],1], coords[entries[t],2]])-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled))

   return X_final, objective_value(m), objective_value(m2), value.(Y)
end

function getMatrix_BM(coords, entries, M, n, numEntriesSampled, k=sqrt(n), gamma=200*sqrt(n)/(numEntriesSampled)) #can't impose constraints here because then we need to warm-start with a low-rank feasible solution, and if we have a method for doing that then we don't need to solve this problem.

    X_best=zeros(n,n)
    ofv_best=1e10
    numIters=1 #keeping it at 1 for now, since IPopt is rather slow with even a moderate number of constraints.

    m=Model(with_optimizer(Ipopt.Optimizer))

    @variable(m, X[1:n, 1:n])
    @variable(m, U[1:n, 1:k])
    @variable(m, V[1:k, 1:n])

    @objective(m, Min, 1.0/(2.0*gamma)*sum(X[i,j]^2 for i=1:n for j=1:n)+sum((X[coords[entries[t],1], coords[entries[t],2]]-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)))
    @constraint(m, X.>=U*V.-1e-6*ones(n,n))
    @constraint(m, X.<=U*V.+1e-6*ones(n,n))

    #@constraint(m, X.==U*V) #non-convex constraint encoding low-rank relation

    for iter=1:numIters
      U_init=randn(n,k) #same order of magnitude as actual solution
      V_init=randn(k,n)
      JuMP.set_start_value.(U, U_init)
      JuMP.set_start_value.(V, V_init)

      @suppress optimize!(m)
      if objective_value(m) < ofv_best
         @show ofv_best = objective_value(m)
         @show iter
         X_best=value.(X)
      end
    end


   return X_best
end

function getMatrixFrobeniusNorm_rademacher(coords, entries, M, n, numEntriesSampled, k=sqrt(n), gamma=200*sqrt(n)/(numEntriesSampled), imposeConstraints=false)
    m=Model(with_optimizer(Mosek.Optimizer))

    @variable(m, X[1:n, 1:n])
    @variable(m, Y[1:n, 1:n], Symmetric) #Need symmetry on Y because of the SVD step.
    @variable(m, theta[1:n, 1:n])

    @constraint(m, Symmetric(Matrix(1.0I, n, n)-Y) in PSDCone());

    if imposeConstraints
        for t=1:Int(numEntriesSampled)
            @constraint(m, X[coords[entries[t],1], coords[entries[t],2]]==M[coords[entries[t],1], coords[entries[t],2]])
        end
        @objective(m, Min, 1.0/(2.0*gamma)*sum(theta[i,i] for i=1:n))
    else
        @objective(m, Min, 1.0/(2.0*gamma)*sum(theta[i,i] for i=1:n)+sum((X[coords[entries[t],1], coords[entries[t],2]]-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)))
    end
    @constraint(m, Symmetric([Y X; X' theta]) in PSDCone());
    @constraint(m, sum(Y[i,i] for i=1:n)<=k)


    @suppress optimize!(m)
    u,v,w=LinearAlgebra.svd(value.(Y))


    # do 100 iterations of randomized rounding and keep best solution in terms of in-sample objective.
    X_best=zeros(n,n)
    ofv_best=1e10
    numIters=100
    for i=1:numIters
            Ξ=(2.0)*(rand(n,k).<=0.5).-1.0 # using distributions package would be more efficient, but introduces weird dependency issues
            Y_intermediate=(1.0/n)*u*Diagonal(sqrt.(v))*Ξ*Ξ'*Diagonal(sqrt.(v))*u'
            u2,v2,w2=LinearAlgebra.svd(Y_intermediate)
            v_rounded=zeros(n)
            v_rounded[sortperm(v2, rev=true)[1:k]].=1.0

            Y_rounded=u2*Diagonal(v_rounded)*u2'
            m2=Model(with_optimizer(Mosek.Optimizer))
            @variable(m2, X2[1:n, 1:n])
            @objective(m2, Min, 1.0/(2.0*gamma)*sum(X2[i,j]^2 for i=1:n for j=1:n)+sum((X2[coords[entries[t],1], coords[entries[t],2]]-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)))
            @constraint(m2, X2.==Y_rounded*X2)
            @suppress optimize!(m2)
            if objective_value(m2) < ofv_best
               ofv_best = objective_value(m2)
               X_best=value.(X2)
            end
    end
    return X_best
end

function getMatrixFrobeniusNorm_restrictedcoorddescent(coords, entries, M, n, numEntriesSampled, k=sqrt(n), gamma=200*sqrt(n)/(numEntriesSampled), imposeConstraints=false)
    m=Model(with_optimizer(Mosek.Optimizer))

    @variable(m, X[1:n, 1:n])
    @variable(m, Y[1:n, 1:n], Symmetric) #Need symmetry on Y because of the SVD step.
    @variable(m, theta[1:n, 1:n])

    @constraint(m, Symmetric(Matrix(1.0I, n, n)-Y) in PSDCone());

    if imposeConstraints
        for t=1:Int(numEntriesSampled)
            @constraint(m, X[coords[entries[t],1], coords[entries[t],2]]==M[coords[entries[t],1], coords[entries[t],2]])
        end
        @objective(m, Min, 1.0/(2.0*gamma)*sum(theta[i,i] for i=1:n))
    else
        @objective(m, Min, 1.0/(2.0*gamma)*sum(theta[i,i] for i=1:n)+sum((X[coords[entries[t],1], coords[entries[t],2]]-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)))
    end
    @constraint(m, Symmetric([Y X; X' theta]) in PSDCone());
    @constraint(m, sum(Y[i,i] for i=1:n)<=k)


    @suppress optimize!(m)
    u,v,w=LinearAlgebra.svd(value.(Y))

    # This gives us our u, v
    S=sortperm(v, rev=true)[1:min(2*k, n)]
    Y_search=zeros(n,n)
    for t=1:k
            best_ofv=1e10
            j=0
        for i in S
            # Evaluate f(Y+u_i u_i')
            m2=Model(with_optimizer(Mosek.Optimizer))
            @variable(m2, X2[1:n, 1:n])
            @objective(m2, Min, 1.0/(2.0*gamma)*sum(X2[i,j]^2 for i=1:n for j=1:n)+sum((X2[coords[entries[t],1], coords[entries[t],2]]-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)))
            @constraint(m2, X2.==(Y_search.+u[:,i]*u[:,i]')*X2)
            @suppress optimize!(m2)

            if objective_value(m2)<best_ofv
                best_ofv=objective_value(m2)
                j=i
            end
        end
        filter!(x->x≠j, S)
        Y_search=Y_search+u[:,j]*u[:,j]'
    end

# Finally, evaluate X and return it (could keep track of X as we go, but the idea of the paper is to work with Y wherever possible so we should really be doing that)
m2=Model(with_optimizer(Mosek.Optimizer))
@variable(m2, X2[1:n, 1:n])
@objective(m2, Min, 1.0/(2.0*gamma)*sum(X2[i,j]^2 for i=1:n for j=1:n)+sum((X2[coords[entries[t],1], coords[entries[t],2]]-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)))
@constraint(m2, X2.==Y_search*X2)
@suppress optimize!(m2)

return value.(X2)
end

# Note: this function is currently not used in the paper
function getMatrixFrobeniusNorm_greedystrengthened(coords, entries, M, n, numEntriesSampled, k=sqrt(n), gamma=200*sqrt(n)/(numEntriesSampled), imposeConstraints=false, useRotation=false)
    m=Model(with_optimizer(Mosek.Optimizer))
    U_bound=0.3

    @variable(m, X[1:n, 1:n])

    @variable(m, Y[1:n, 1:n], Symmetric)
    @variable(m, Y_sub[1:n, 1:n, 1:k])
    @variable(m, Y_sub2[1:n, 1:n, 1:k, 1:k])
    @variable(m, W[1:n, 1:k, 1:k])

    @variable(m, theta[1:n, 1:n])
    @variable(m, sigma>=0.0)
    @variable(m, U[1:n, 1:k])

    @constraint(m, Y.==sum(Y_sub[:,:,t] for t=1:k))

    @constraint(m, traceoneconstraint[t=1:k], sum(Y_sub[i,i,t] for i=1:n).==1.0)

    @constraint(m, imposeSDPSlice[t=1:k], ([1.0 U[:,t]' ;U[:,t] Y_sub[:,:,t] ]) in PSDCone());

    @constraint(m, imposeSDPSlice2[t=1:k, t2=(t+1):k], ([1.0 U[:,t]' U[:,t2]' ;U[:,t] Y_sub2[:,:,t, t] Y_sub2[:,:,t, t2]; U[:,t2] Y_sub2[:,:,t,t2]' Y_sub2[:,:,t2,t2]]) in PSDCone());

    @constraint(m, tracezeroconstraint[t=1:k, t2=(t+1):k], sum(Y_sub2[i,i,t, t2] for i=1:n).==0.0)

    @constraint(m, definediag[t=1:k], Y_sub[:,:,t].==Y_sub2[:,:,t,t])

    @constraint(m, strengthen4[i=1:n, t=1:k, t2=1:k], W[i,t,t2]==Y_sub2[i,i,t,t2])

    @constraint(m, McCormick5[i=1:n, t1=1:k, t2=1:k], W[i,t1,t2]>=-U_bound*(U[i,t1]+U[i,t2])-U_bound^2)
    @constraint(m, McCormick6[i=1:n, t1=1:k, t2=1:k], W[i,t1,t2]>=U_bound*(U[i,t1]+U[i,t2])-U_bound^2)
    @constraint(m, McCormick7[i=1:n, t1=1:k, t2=1:k], W[i,t1,t2]<=U_bound*(U[i,t1]-U[i,t2])+U_bound^2)
    @constraint(m, McCormick8[i=1:n, t1=1:k, t2=1:k], W[i,t1,t2]<=U_bound*(-U[i,t1]+U[i,t2])+U_bound^2)
    # @constraint(m, McCormick9[i=1:n, j=(i+1):n, t=1:k], Y_sub[i,j,t]<=U_bound^2)
    # @constraint(m, McCormick10[i=1:n, j=(i+1):n, t=1:k], Y_sub[i,j,t]>=-U_bound^2) #Product of U_it, U_jt but ||U||_2 \leq 1

    @constraint(m, orthogonality[t1=1:k, t2=1:k], sum(W[i,t1,t2] for i=1:n).==1.0*(t1==t2))
    @constraint(m, strengthen3[i=1:n, t=1:k], Y_sub[i,i,t].==W[i,t,t])

    @constraint(m, Symmetric(Matrix(1.0I, n, n)-Y) in PSDCone());
    @constraint(m, sigma>=1.0/(2.0*gamma)*sum(theta[i,i] for i=1:n))

    if imposeConstraints
        for t=1:Int(numEntriesSampled)
            @constraint(m, X[coords[entries[t],1], coords[entries[t],2]]==M[coords[entries[t],1], coords[entries[t],2]])
        end

        @objective(m, Min, sigma)
    else
        @objective(m, Min, sigma+0.5*sum((X[coords[entries[t],1], coords[entries[t],2]]-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)))
    end
    @constraint(m, Symmetric([Y X; X' theta]) in PSDCone());

    @suppress optimize!(m)
    @show objective_value(m)

    u,v,w=LinearAlgebra.svd(value.(Y))

    diag_v=zeros(n)
    diag_v[sortperm(v, rev=true)[1:k]].=1.0
    Y_rounded=u*Diagonal(diag_v)*u'
    @show sum(abs.(Y_rounded))

    m2=Model(with_optimizer(Mosek.Optimizer))
    @variable(m2, X2[1:n, 1:n])
    @objective(m2, Min, 1.0/(2.0*gamma)*sum(X2[i,j]^2 for i=1:n for j=1:n)+0.5*sum((X2[coords[entries[t],1], coords[entries[t],2]]-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)))
    @constraint(m2, X2.==Y_rounded*X2)
    @suppress optimize!(m2)
    X_best=value.(X2)
    @show sum(abs.(value.(X2)))

   return X_best, objective_value(m), objective_value(m2), value.(Y)
end

function getMatrixFrobeniusNorm_cuttingplanes_kelley(coords, entries, M, n, numEntriesSampled, k=sqrt(n), gamma=200*sqrt(n)/(numEntriesSampled), imposeConstraints=false)
    m=Model(Mosek.Optimizer)

    theIndices=zeros(n,n)
    for t=1:Int(numEntriesSampled)
        theIndices[coords[entries[t],1], coords[entries[t],2]]=1.0
    end

    @variable(m, Y[1:n, 1:n], PSD)
    @constraint(m, Symmetric(Matrix(1.0I, n, n)-Y) in PSDCone());
    @variable(m, theta>=-1e5)
    @objective(m, Min, theta)
    @constraint(m, sum(Y[i,i] for i=1:n)<=k)

    UB=1e12
    LB=-1e12
    max_iter=200
    Y_best=zeros(n,n)
    for iter=1:max_iter

        @suppress optimize!(m)

        LB=objective_value(m)

        # Solve subproblem for fixed Y, in alpha
        m2=Model(Ipopt.Optimizer)
        #set_start_value.(all_variables(m2), value.(all_variables(m2)))

        @variable(m2, alpha[1:n, 1:n])
        @constraint(m2, alpha.*(ones(n,n)-theIndices).==0.0) #if not in set of indices then eliminate variable
        @objective(m2, Max, -0.5*sum((alpha[i,j]+M[i,j])^2*theIndices[i,j] for i=1:n for j=1:n)-(gamma/2.0)*Compat.dot(alpha*alpha', value.(Y)))
        set_start_value.(alpha, rand(n,n))
        @suppress optimize!(m2)
        alpha_t=value.(alpha)
        # Get UB and cut from second-stage optimization problem
        f_Y=objective_value(m2)+0.5*sum((M.*theIndices).^2)
        # Impose constraint
        @constraint(m, theta>=f_Y-gamma/2*Compat.dot(alpha_t*alpha_t', Y-value.(Y)))
        #set_start_value.(all_variables(m), value.(all_variables(m)))
        if UB>f_Y
            Y_best=value.(Y)
            UB=f_Y
        end
        @show UB, LB, iter
        if abs.(UB-LB)<1e-3
            iter=max_iter
            break
        end
    end

    u,v,w=LinearAlgebra.svd(Y_best)

    diag_v=zeros(n)
    diag_v[sortperm(v, rev=true)[1:k]].=1.0
    Y_rounded=u*Diagonal(diag_v)*u'

    # greedy rounding

    m2=Model(with_optimizer(Mosek.Optimizer))
    @variable(m2, X2[1:n, 1:n])
    @objective(m2, Min, 1.0/(2.0*gamma)*sum(X2[i,j]^2 for i=1:n for j=1:n)+sum((X2[coords[entries[t],1], coords[entries[t],2]]-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)))
    @constraint(m2, X2.==Y_rounded*X2)
    @suppress optimize!(m2)
    X_best=value.(X2)

   return X_best, LB, objective_value(m2)
end

function getMatrixFrobeniusNorm_cuttingplanes_inout(coords, entries, M, n, numEntriesSampled, k=sqrt(n), gamma=200*sqrt(n)/(numEntriesSampled), stabilizationPoint=k/n*Diagonal(ones(n)))
    m=Model(Mosek.Optimizer)
    set_optimizer_attribute(m, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-10)
    set_optimizer_attribute(m, "MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-10)

    theIndices=zeros(n,n)
    for t=1:Int(numEntriesSampled)
        theIndices[coords[entries[t],1], coords[entries[t],2]]=1.0
    end

    @variable(m, Y[1:n, 1:n], PSD)
    @constraint(m, Symmetric(Matrix(1.0I, n, n)-Y) in PSDCone());
    @variable(m, theta>=-1e4)
    @objective(m, Min, theta)
    @constraint(m, sum(Y[i,i] for i=1:n)<=k)

    UB=1e12
    LB=-1e12
    rootStabilizationTrick = :inOut
     #Use the solution to the SDP relaxation instead, where appropriate
    rootCutsSense=1
    ε= 1e-10
    λ = (rootStabilizationTrick == :inOut) ? .1 : 1.
    δ = (rootStabilizationTrick == :inOut || rootStabilizationTrick == :twoEps) ? 2*ε : 0.
    rootCutsLim=1000
    rootCutCount = 0
    oaRootCutCount=0
    consecutiveNonImprov_1 = 0
    consecutiveNonImprov_2 = 0
    # theCutPool=Cut[]

    Y_best=zeros(n,n)
    for epoch in 1:1000
      #@show epoch, LB, UB, oaRootCutCount
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


      # Get a cut by solving the dual subproblem
      # Solve subproblem for fixed Y, in alpha
      m2=Model(Mosek.Optimizer)
      @variable(m2, alpha[1:n, 1:n])
      @constraint(m2, alpha.*(ones(n,n)-theIndices).==0.0) #if not in set of indices then eliminate variable
      @objective(m2, Max, -0.5*sum((alpha[i,j]+M[i,j])^2*theIndices[i,j] for i=1:n for j=1:n)-(gamma/2.0)*Compat.dot(alpha*alpha', Y0))
      set_start_value.(alpha, rand(n,n))
      @suppress optimize!(m2)
      alpha_t=value.(alpha)
      # Get UB and cut from second-stage optimization problem
      f_Y=objective_value(m2)+0.5*sum((M.*theIndices).^2)
      @show f_Y
      @show sum(diag(Y0))
      if f_Y<UB
          Y_best=Y0
          UB=f_Y
      end
      LB=max(LB, objective_value(m))
      # Impose constraint
      @constraint(m, theta>=f_Y-gamma/2*Compat.dot(alpha_t*alpha_t', Y-Y0))
      # Add cut to pool (add in later if it works)
      @show LB, UB, epoch
      if abs(UB-LB) <= 1e-2 || consecutiveNonImprov_2 >= 10
          break
      end
    end


    u,v,w=LinearAlgebra.svd(Y_best)

    diag_v=zeros(n)
    diag_v[sortperm(v, rev=true)[1:k]].=1.0
    Y_rounded=u*Diagonal(diag_v)*u'

    # greedy rounding

    m2=Model(with_optimizer(Mosek.Optimizer))
    @variable(m2, X2[1:n, 1:n])
    @objective(m2, Min, 1.0/(2.0*gamma)*sum(X2[i,j]^2 for i=1:n for j=1:n)+sum((X2[coords[entries[t],1], coords[entries[t],2]]-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)))
    @constraint(m2, X2.==Y_rounded*X2)
    @suppress optimize!(m2)
    X_best=value.(X2)

   return X_best, LB, objective_value(m2)
end
