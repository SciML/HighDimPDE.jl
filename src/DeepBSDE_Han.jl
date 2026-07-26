# called whenever sdealg is not specified.
"""
$(TYPEDSIGNATURES)

Returns a `PIDESolution` object. 

# Arguments: 
- `maxiters`: The number of training epochs. Defaults to `300`
- `trajectories`: The number of trajectories simulated for training. Defaults to `100`

To use [SDE Algorithms](https://diffeq.sciml.ai/stable/solvers/sde_solve/) use [`DeepBSDE`](@ref)
"""
function solve(
        prob::ParabolicPDEProblem,
        alg::DeepBSDE;
        dt,
        abstol = 1.0f-6,
        verbose = false,
        maxiters = 300,
        save_everystep = false,
        trajectories,
        ensemblealg = EnsembleThreads(),
        limits = false,
        trajectories_upper = 1000,
        trajectories_lower = 1000,
        maxiters_limits = 10
    )
    X0 = prob.x
    ts = prob.tspan[1]:dt:prob.tspan[2]
    d = length(X0)
    g, f, μ, σ, p = prob.g, prob.f, prob.μ, prob.σ, prob.p

    #hidden layer
    opt = alg.opt
    u0 = alg.u0
    σᵀ∇u = alg.σᵀ∇u
    model = (u0, σᵀ∇u)

    function sol(model)
        u0, σᵀ∇u = model
        return map(1:trajectories) do j
            u = u0(X0)[1]
            X = X0
            for i in 1:(length(ts) - 1)
                t = ts[i]
                _σᵀ∇u = σᵀ∇u[i](X)
                dW = sqrt(dt) * randn(d)
                u = u - f(X, u, _σᵀ∇u, p, t) * dt + _σᵀ∇u' * dW
                X = X .+ μ(X, p, t) * dt .+ σ(X, p, t) * dW
            end
            X, u
        end
    end

    function loss(model)
        return mean(sum(abs2, g(X) - u) for (X, u) in sol(model))
    end

    iters = eltype(X0)[]
    losses = eltype(X0)[]

    opt_state = Flux.setup(opt, model)
    for _ in 1:maxiters
        gs = Flux.gradient(model) do model_
            loss(model_)
        end
        Flux.update!(opt_state, model, gs[1])
        save_everystep && push!(iters, u0(X0)[1])
        l = loss(model)
        push!(losses, l)
        verbose && println("Current loss is: $l")
        l < abstol && break
    end

    if limits == false
        if save_everystep
            sol = PIDESolution(X0, ts, losses, iters, u0)
        else
            sol = PIDESolution(X0, ts, losses, u0(X0)[1], u0)
        end
        return sol
    else
        A = haskey(prob.kwargs, :A) ? prob.kwargs.A : nothing
        u_domain = prob.x0_sample

        verbose && println("Upper limit")
        sdeProb = SDEProblem(μ, σ, X0, prob.tspan)
        ensembleprob = EnsembleProblem(sdeProb)
        sim = solve(
            ensembleprob,
            EM(),
            ensemblealg,
            dt = dt,
            trajectories = trajectories_upper,
            prob.kwargs...
        )
        function sol_high(model)
            u0, σᵀ∇u = model
            return map(sim.u) do u
                xsde = u.u
                U = g(xsde[end])
                u = u0(X0)[1]
                for i in length(ts):-1:3
                    t = ts[i]
                    _σᵀ∇u = σᵀ∇u[i - 1](xsde[i - 1])
                    dW = sqrt(dt) * randn(d)
                    U = U .+ f(xsde[i - 1], U, _σᵀ∇u, p, t) * dt .- _σᵀ∇u' * dW
                end
                U
            end
        end

        loss_(model) = sum(sol_high(model)) / trajectories_upper

        opt_state_limits = Flux.setup(Flux.Adam(0.01), model)
        for _ in 1:maxiters_limits
            gs = Flux.gradient(model) do model_
                loss_(model_)
            end
            Flux.update!(opt_state_limits, model, gs[1])
            l = loss_(model)
            verbose && println("Current loss is: $l")
            l < abstol && break
        end
        u_high = loss_(model)

        verbose && println("Lower limit")
        # Function to precalculate the f values over the domain
        function give_f_matrix(X, urange, σᵀ∇u, p, t)
            return map(urange) do u
                f(X, u, σᵀ∇u, p, t)
            end
        end

        #The Legendre transform that uses the precalculated f values.
        function legendre_transform(f_matrix, a, urange)
            le = a .* (collect(urange)) .- f_matrix
            return maximum(le)
        end

        function sol_low()
            return map(1:trajectories_lower) do j
                u = u0(X0)[1]
                X = X0
                I = zero(eltype(u))
                Q = zero(eltype(u))
                for i in 1:(length(ts) - 1)
                    t = ts[i]
                    _σᵀ∇u = σᵀ∇u[i](X)
                    dW = sqrt(dt) * randn(d)
                    u = u - f(X, u, _σᵀ∇u, p, t) * dt + _σᵀ∇u' * dW
                    X = X .+ μ(X, p, t) * dt .+ σ(X, p, t) * dW
                    f_matrix = give_f_matrix(X, u_domain, _σᵀ∇u, p, ts[i])
                    a_ = A[
                        findmax(
                            collect(A) .* u .-
                                collect(
                                legendre_transform(f_matrix, a, u_domain)
                                    for a in A
                            )
                        )[2],
                    ]
                    I = I + a_ * dt
                    Q = Q + exp(I) * legendre_transform(f_matrix, a_, u_domain)
                end
                I, Q, X
            end
        end
        u_low = sum(exp(I) * g(X) - Q for (I, Q, X) in sol_low()) / (trajectories_lower)
        if save_everystep
            sol = PIDESolution(X0, ts, losses, iters, u0(X0)[1], (u_low, u_high))
        else
            sol = PIDESolution(X0, ts, losses, u0(X0)[1], u0, (u_low, u_high))
        end
        return sol
    end
end #pde_solve
