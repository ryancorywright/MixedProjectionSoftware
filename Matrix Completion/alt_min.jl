using Ipopt, OSQP, DataFrames

results_template_alg = DataFrame(n=Int[], lb=Real[], ub=Real[])



function alternatingminimization_matrixcompletion(coords, entries, M, n, numEntriesSampled, k=sqrt(n), gamma=200*sqrt(n)/(numEntriesSampled),
                                            innerEpochs::Int=20, X_ws::Array{Float64, 2}=zeros(n,n))

  X_init=zeros(n,n)

  theIndices=zeros(n,n)
  for t=1:Int(numEntriesSampled)
      theIndices[coords[entries[t],1], coords[entries[t],2]]=1.0
  end
  #X_init=M.*theIndices
  u_t,sigma,=svd(X_ws)
  u_t=u_t[:, 1:k]
  v_t=zeros(n,k)
  ofv_current=1e10
  ofv_prev=1e12

 # Begin alternating min loop

  start_time = time()

      for epoch in 1:innerEpochs #we already have a value of alpha, from warm-start.

        # Minimization over v for a given u
        m=Model(with_optimizer(Mosek.Optimizer))
        @variable(m, v[1:n, 1:k])
        @objective(m, Min, 1.0/(2.0*gamma)*Compat.dot(u_t*v', u_t*v')+0.5*sum((Compat.dot(u_t[coords[entries[t],1],:], v[coords[entries[t],2],:])-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)))
        @suppress optimize!(m)
        v_t.=value.(v)

        # Minimization over u for a given v
        m2=Model(with_optimizer(Mosek.Optimizer))
        @variable(m2, u[1:n, 1:k])
        @objective(m2, Min, 1.0/(2.0*gamma)*Compat.dot(u*v_t', u*v_t')+0.5*sum((Compat.dot(u[coords[entries[t],1],:], v_t[coords[entries[t],2],:])-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)))
        @suppress optimize!(m2)
        u_t.=value.(u)

        X_t=u_t*v_t'
        ofv_current=1.0/(2.0*gamma)*sum(X_t[i,j]^2 for i=1:n for j=1:n)+sum((X_t[coords[entries[t],1], coords[entries[t],2]]-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled))

        if abs.(ofv_current-ofv_prev)<1e-4
            break
        end
        ofv_prev=ofv_current

      end
      @show ofv_current
  return u_t*v_t', ofv_current
end

function alternatingminimization_matrixcompletion_solvesdprelax(coords, entries, M, n, numEntriesSampled, k=sqrt(n), gamma=200*sqrt(n)/(numEntriesSampled),
                                            innerEpochs::Int=500)


  theIndices=zeros(n,n)
  for t=1:Int(numEntriesSampled)
      theIndices[coords[entries[t],1], coords[entries[t],2]]=1.0
  end
  X_t=(1.0+1.0/gamma)^(-1.0)*M.*theIndices #Note: no benefit in projecting onto space of rank-k matrices, will obtain samve svd.

  ofv_current=1e10
  ofv_prev=1e12
  ofv_dual=-1e10
  Y_t=zeros(n,n)
  Y_int=zeros(n,n)
  Y_int_prev=zeros(n,n)
  X_avg=zeros(n,n)
  t_k=(1.0+sqrt(5.0))/2.0
  t_k_prev=1.0
  X_int=X_t
  omega=2.0
  X_int_prev=X_t

  eps_tol=1e-3

  results_convergence=similar(results_template_alg, 0)


 # Begin alternating min loop

  start_time = time()


      for epoch in 1:innerEpochs

        # Minimization over Y for a given X_t
        U,sigma,=svd(X_t)
        sum_sigma=sum(sigma)
        if epoch<=innerEpochs-10
            sigma=max.(sigma, (1e-2/epoch)*ones(n))
        else
            sigma=max.(sigma, 1e-8*ones(n))
        end
        sigma.=(sum_sigma/sum(sigma))*sigma

        size(U)

        m=Model(Mosek.Optimizer)
        set_optimizer_attribute(m, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-8) #increase tolerance when nearer the optimal solution (i.e. have run more epochs).
        set_optimizer_attribute(m, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-8)
        set_optimizer_attribute(m, "MSK_DPAR_INTPNT_QO_TOL_DFEAS", 1e-8)
        set_optimizer_attribute(m, "MSK_DPAR_INTPNT_CO_TOL_INFEAS", 1e-8)

        if epoch<=innerEpochs-10
            @variable(m, theta[1:n]>=max((1e-2/epoch), 1e-5))
        else #obtain some lift by relaxing requirement that Y remains "far" away from center
            @variable(m, theta[1:n]>=1e-5)
        end
        @constraint(m, sum(theta)<=k)
        @constraint(m, theta.<=1.0)
        @variable(m, w[1:n])
        # @variable(m, Y[1:n, 1:n])
        # @constraint(m, Y.==sum(theta[i]*U[:,i]*U[:,i]' for i=1:n))
        @constraint(m, imposeSOCP[i=1:n], [w[i]+theta[i]; w[i]-theta[i]; 2.0*sigma[i]] in SecondOrderCone())
        @objective(m, Min, 1.0/(2.0*gamma)*sum(w))#+(n/50)/(2.0*epoch)*Compat.dot(Y-Y_t, Y-Y_t))
         @suppress optimize!(m)
        Y_int=U*Diagonal(value.(theta))*U'
        Y_t=Y_int+(t_k_prev-1.0)/(t_k)*(Y_int-Y_int_prev)
        Y_int_prev=Y_int
        invtheta=1.0./max.(value.(theta), 1e-5)


        X_int, ofv_current, omega=getX_t_new(M, theIndices, U*Diagonal(invtheta)*U', gamma, X_t, n, epoch, omega)
        X_t=X_int+(t_k_prev-1.0)/(t_k)*(X_int-X_int_prev)
        X_int_prev=X_int
        t_k_prev=t_k
        t_k=(1.0+sqrt(4.0*t_k^2+1.0))/2.0


        if epoch%50==0
            # @show ofv_current, epoch
            # Compute dual objective value and use this in the termination criteria?
            theDualBound=getDualBound(M, theIndices, Y_t, gamma, n, k)
            ofv_dual=max(ofv_dual, theDualBound)
            @show ofv_current, epoch, "duality gap is:", (ofv_current-ofv_dual), time()-start_time
        end
        # Remark: major bottleneck is actually the time to build the model; takes 0.4s to build and 0.04s to solve at n=100.
        # Update objective value, this should be non-increasing compared to the previous iterate.
        # Remark: each iterate gives a feasible projection matrix which can be randomly rounded as needed.
        if (ofv_current-ofv_dual)/(abs(ofv_dual)+1e-4)<1e-4 || ofv_prev-ofv_current<1e-3
            break
        end
        # @show ofv_current, ofv_prev
        #push!(results_convergence, [n, ofv_dual,ofv_current])

        ofv_prev=ofv_current

      end
    #CSV.write("plotconvergence_seqaltmin.csv", results_convergence, append=true)

      # Finally, perform greedy rounding on Y_t, alla the sdp relaxation.
      u,v,w=LinearAlgebra.svd(Y_t)
      diag_v=zeros(n)
      diag_v[sortperm(v, rev=true)[1:k]].=1.0
      Y_rounded=u*Diagonal(diag_v)*u'

      m3=Model(with_optimizer(Mosek.Optimizer))
      @variable(m3, X2[1:n, 1:n])
      @objective(m3, Min, 1.0/(2.0*gamma)*sum(X2[i,j]^2 for i=1:n for j=1:n)+0.5*sum((X2[coords[entries[t],1], coords[entries[t],2]]-M[coords[entries[t],1], coords[entries[t],2]])^2 for t in 1:Int(numEntriesSampled)))
      @constraint(m3, X2.==Y_rounded*X2)
      @suppress optimize!(m3)
      X_final=value.(X2)

  return X_final
end


function getX_t_new(M, theIndices, Y_pinv_t, gamma, X_t, n, epoch, omega)

    X_iter=X_t
    X_opt=X_t
    X_prev=X_t
    residual=1e2
    residual_prev=1e3
    gs_tol=max(1e-3/epoch, 1e-5)
    iters=1
    omega=2.0 #Omega>2 is not recommended in any textbooks, but seems to work well here. Remark: omega is effectively global
    while residual>gs_tol && iters<1000 && residual<1e3 #Use Gauss-Seidel approach
        X_prev=X_iter
        L=(1.0/gamma)*(Y_pinv_t)+(1.0/epoch)*Diagonal(ones(n))
        X_iter=(omega*L)\(theIndices.*(M-X_iter)+(1.0/epoch)*X_t-(1.0-omega)*L*X_iter)
        residual_prev=residual
        residual=norm(((1.0/gamma)*(Y_pinv_t*X_iter)+(1.0/epoch)*(X_iter-X_t)+theIndices.*(X_iter-M)))
        if residual_prev<residual-1e-6 #Increasing omega allows convergence to occur with a worse conditioned system, but more slowly.
            omega*=1.5
        end
        iters+=1
    end
    if residual<gs_tol
        X_opt=X_iter
    else #Not going to converge, so use IPopt
        @show "Warning: Using Solver"
        @show residual, omega, iters

        m2=Model(OSQP.Optimizer)
        set_optimizer_attribute(m2, "max_iter", 1e4)
        set_optimizer_attribute(m2, "eps_prim_inf", 1e-3)
        set_optimizer_attribute(m2, "eps_abs", 1e-4)
        @variable(m2, X[1:n, 1:n])
        @constraint(m2, ((1.0/gamma)*(Y_pinv_t*X)+(1.0/epoch)*(X-X_t)+theIndices.*(X-M)).==0.0) #precondition with X_t^\dag for numerical stability.
        @time JuMP.optimize!(m2)
        X_opt=value.(X)
    end

    Residual = theIndices.*(X_opt.-M)
    ofv_current=0.5*Compat.dot(Residual, Residual)+0.5*Compat.dot(X_opt, -Residual)-(0.5/epoch)*Compat.dot(X_opt-X_t, X_t)
    return X_opt, ofv_current, omega
end

function getDualBound(M, theIndices, Y_current, gamma, n, k)
    # Search for optimal alpha, then compute bound
    #m3=Model(Ipopt.Optimizer)
    # set_optimizer_attribute(m3, "linear_solver", "mumps")
    # set_optimizer_attribute(m3, "tol", 1e-4)
    u,v=svd(Y_current)
    u0=u*Diagonal(sqrt.(v))
    m3=Model(Mosek.Optimizer)
        #set_optimizer_attribute(m3, "tol", 1e-4)
    set_optimizer_attribute(m3, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-4)
    set_optimizer_attribute(m3, "MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1e-4)
    set_optimizer_attribute(m3, "MSK_DPAR_INTPNT_CO_TOL_INFEAS", 1e-4)
    set_optimizer_attribute(m3, "MSK_DPAR_INTPNT_CO_TOL_MU_RED", 1e-4)
    set_optimizer_attribute(m3, "MSK_DPAR_INTPNT_CO_TOL_NEAR_REL", 1e5)
    set_optimizer_attribute(m3, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1e-4)
    set_optimizer_attribute(m3, "MSK_DPAR_INTPNT_QO_TOL_DFEAS", 1e-4)
    set_optimizer_attribute(m3, "MSK_DPAR_INTPNT_QO_TOL_INFEAS", 1e-6)
    set_optimizer_attribute(m3, "MSK_DPAR_INTPNT_QO_TOL_MU_RED", 1e-4)
    set_optimizer_attribute(m3, "MSK_DPAR_INTPNT_QO_TOL_PFEAS", 1e-4)
    set_optimizer_attribute(m3, "MSK_DPAR_INTPNT_QO_TOL_REL_GAP", 1e-4)
    set_optimizer_attribute(m3, "MSK_DPAR_INTPNT_TOL_DFEAS", 1e-4)
    # m3=Model(OSQP.Optimizer)
    # set_optimizer_attribute(m3, "max_iter", 1e4)

    @variable(m3, alpha[1:n, 1:n])
    @variable(m3, r[1:n, 1:n])
    @constraint(m3, r.==alpha'*u0)
    @constraint(m3, alpha.*(ones(n,n)-theIndices).==0.0) #if not in set of indices then eliminate variable
    @objective(m3, Max, -0.5*sum((alpha[i,j]+M[i,j])^2*theIndices[i,j] for i=1:n for j=1:n)-(gamma/2.0)*Compat.dot(r,r))#-(gamma/2.0)*Compat.dot(alpha*alpha', Y_current))
    #-0.5*-(gamma/2.0)*Compat.dot(alpha_t*alpha_t', Y_t)+0.5*sum((M.*theIndices).^2)

    set_start_value.(alpha, rand(n,n))
    @time optimize!(m3)
    alpha_t=value.(alpha)
    u_dual,sigma_dual,=svd(alpha_t*alpha_t')
    dualBound=-0.5*sum((alpha_t[i,j]+M[i,j])^2*theIndices[i,j] for i=1:n for j=1:n)-(gamma/2.0)*sum(sigma_dual[i] for i=1:k)+0.5*sum((M.*theIndices).^2)

    return dualBound
end
