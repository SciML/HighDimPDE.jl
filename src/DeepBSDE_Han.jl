# called whenever sdealg is not specified.
function DiffEqBase.solve(prob::ParabolicPDEProblem,
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
        maxiters_limits = 10,)
    X0 = prob.x
    ts = prob.tspan[1]:dt:prob.tspan[2]
    d = length(X0)
    g, f, μ, σ, p = prob.g, prob.f, prob.μ, prob.σ, prob.p

    data = Iterators.repeated((), maxiters)

    #hidden layer
    opt = alg.opt
    u0 = alg.u0
    σᵀ∇u = alg.σᵀ∇u
    ps = Flux.params(u0, σᵀ∇u...)

    function sol()
        map(1:trajectories) do j
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

    function loss()
        mean(sum(abs2, g(X) - u) for (X, u) in sol())
    end

    iters = eltype(X0)[]
    losses = eltype(X0)[]

    callback = function ()
        save_everystep && push!(iters, u0(X0)[1])
        l = loss()
        push!(losses, l)
        verbose && println("Current loss is: $l")
        l < abstol && Flux.stop()
    end

    Flux.train!(loss, ps, data, opt; cb = callback)

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
        sim = solve(ensembleprob,
            EM(),
            ensemblealg,
            dt = dt,
            trajectories = trajectories_upper,
            prob.kwargs...)
        function sol_high()
            map(sim.u) do u
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

        loss_() = sum(sol_high()) / trajectories_upper

        ps = Flux.params(u0, σᵀ∇u...)
        callback = function ()
            l = loss_()
            verbose && println("Current loss is: $l")
            l < abstol && Flux.stop()
        end
        dataS = Iterators.repeated((), maxiters_limits)
        Flux.train!(loss_, ps, dataS, ADAM(0.01); cb = callback)
        u_high = loss_()

        verbose && println("Lower limit")
        # Function to precalculate the f values over the domain
        function give_f_matrix(X, urange, σᵀ∇u, p, t)
            map(urange) do u
                f(X, u, σᵀ∇u, p, t)
            end
        end

        #The Legendre transform that uses the precalculated f values.
        function legendre_transform(f_matrix, a, urange)
            le = a .* (collect(urange)) .- f_matrix
            return maximum(le)
        end

        function sol_low()
            map(1:trajectories_lower) do j
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
                    a_ = A[findmax(collect(A) .* u .-
                                   collect(legendre_transform(f_matrix, a, u_domain)
                                           for a in A))[2]]
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
