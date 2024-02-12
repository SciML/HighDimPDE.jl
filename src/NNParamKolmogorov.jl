"""
Algorithm for solving Backward Kolmogorov Equations.

```julia
HighDimPDE.NNKolmogorov(chain, opt)
```
Arguments:
- `chain`: A Chain neural network with a d-dimensional output.
- `opt`: The optimizer to train the neural network. Defaults to `ADAM(0.1)`.
[1] Berner Julius et al. "Numerically solving parametric families of high-dimensional Kolmogorov partial differential equations via deep learning."
"""
struct NNParamKolmogorov{C, O} <: HighDimPDEAlgorithm
    chain::C
    opt::O
end

NNParamKolmogorov(chain; opt = Flux.ADAM(0.1)) = NNParamKolmogorov(chain, opt)

"""
$(TYPEDSIGNATURES)

Returns a `PIDESolution` object.

# Arguments

- `sdealg`: a SDE solver from [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/solvers/sde_solve/). 
    If not provided, the plain vanilla [DeepBSDE](https://arxiv.org/abs/1707.02568) method will be applied.
    If provided, the SDE associated with the PDE problem will be solved relying on 
    methods from DifferentialEquations.jl, using [Ensemble solves](https://diffeq.sciml.ai/stable/features/ensemble/) 
    via `sdealg`. Check the available `sdealg` on the 
    [DifferentialEquations.jl doc](https://diffeq.sciml.ai/stable/solvers/sde_solve/).
- `maxiters`: The number of training epochs. Defaults to `300`
- `trajectories`: The number of trajectories simulated for training. Defaults to `100`
- `dps::NamedTuple`: The sampling interval for ranges of parameters. Should have keys : `p_sigma`, 'p_mu` and `p_phi`
- Extra keyword arguments passed to `solve` will be further passed to the SDE solver.
"""
function DiffEqBase.solve(prob::ParabolicPDEProblem,
        pdealg::NNParamKolmogorov,
        sdealg = EM();
        ensemblealg = EnsembleThreads(),
        abstol = 1.0f-6,
        verbose = false,
        maxiters = 300,
        trajectories = 1000,
        save_everystep = false,
        use_gpu = false,
        dps = (0.01,),
        dt,
        dx,
        kwargs...)
    tspan = prob.tspan
    sigma = prob.σ
    mu = prob.μ
    noise_rate_prototype = get(prob.kwargs, :noise_rate_prototype, nothing)

    p_domain = prob.kwargs.p_domain
    xspan = prob.kwargs.xspan

    xspans = isa(xspan, Tuple) ? [xspan] : xspan
    d = length(xspans)

    phi = prob.g

    ts = tspan[1]:dt:tspan[2]
    xs = map(xspans) do xspan
        xspan[1]:dx:xspan[2]
    end

    p_defaults = (p_sigma = nothing, p_mu = nothing, p_phi = nothing)

    p_domain = merge(p_defaults, prob.kwargs.p_domain)
    p_prototype = merge(p_defaults, prob.kwargs.p_prototype)
    dps = merge(p_defaults, dps)

    chain = pdealg.chain
    ps = Flux.params(chain)
    opt = pdealg.opt

    xi = mapreduce(x -> rand(x, 1, trajectories), vcat, xs)
    ti = rand(ts, 1, trajectories)

    ps_sigma, ps_mu, ps_phi = map(zip(p_domain,
        p_prototype,
        dps)) do (domain, prototype, dp)
        # domain , prototype, dp = p_domain[key], p_prototype[key], dps[key]
        isnothing(domain) && return
        return rand(domain[1]:dp:domain[2], size(prototype)..., trajectories)
    end

    total_dims = mapreduce(*, (ti, xi, ps_mu, ps_sigma, ps_phi)) do y
        isnothing(y) && return 1
        *(size(y)[1:(end - 1)]...)
    end

    train_data = mapreduce(vcat, (ti, xi, ps_mu, ps_sigma, ps_phi)) do y
        isnothing(y) && return rand(0, trajectories) # empty matrix
        reshape(y, :, trajectories)
    end

    ps_sigma_iterator = !isnothing(ps_sigma) ?
                        eachslice(ps_sigma, dims = length(size(ps_sigma))) :
                        collect(Iterators.repeated(nothing, trajectories))
    ps_mu_iterator = !isnothing(ps_mu) ? eachslice(ps_mu, dims = length(size(ps_mu))) :
                     collect(Iterators.repeated(nothing, trajectories))
    ps_phi_iterator = !isnothing(ps_phi) ? eachslice(ps_phi, dims = length(size(ps_phi))) :
                      collect(Iterators.repeated(nothing, trajectories))
    # return xi, ti, ps_sigma_iterator[1]
    prob_func = (prob, i, repeat) -> begin
        sigma_(dx, x, p, t) = sigma(dx, x, ps_sigma_iterator[i], t)
        mu_(dx, x, p, t) = mu(dx, x, ps_mu_iterator[i], t)
        SDEProblem(mu_,
            sigma_,
            xi[:, i],
            (tspan[1], ti[:, 1][1]),
            noise_rate_prototype = noise_rate_prototype)
    end

    output_func = (sol, i) -> (sol.u[end], false)

    sdeprob = SDEProblem(mu,
        sigma,
        xi[:, 1],
        tspan;
        noise_rate_prototype = noise_rate_prototype)

    ensembleprob = EnsembleProblem(sdeprob,
        prob_func = prob_func,
        output_func = output_func)
    # return train_data , ps_sigma_iterator, ps_mu_iterator
    sol = solve(ensembleprob, sdealg, ensemblealg; trajectories = trajectories, dt = dt)
    # return train_data, sol
    # Y = reduce(hcat, phi.(eachcol(x_sde)))
    Y = reduce(hcat, phi.(eachcol(Array(sol)), ps_phi_iterator))

    Y = use_gpu ? Y |> gpu : Y
    train_data = use_gpu ? train_data |> gpu : train_data
    #MSE Loss Function
    loss(m, x, y) = Flux.mse(m(x), y)

    losses = AbstractFloat[]

    opt_state = Flux.setup(opt, chain)
    for epoch in 1:maxiters
        gs = Flux.gradient(chain) do model
            loss(model, train_data, Y)
        end
        Flux.update!(opt_state, chain, gs[1])
        l = loss(chain, train_data, Y)
        @info "Current Epoch: $epoch Current Loss: $l"
        push!(losses, l)
    end

    sol_func = (x0, t, _p_sigma, _p_mu, _p_phi) -> begin
        ps = map(zip(p_prototype, (_p_sigma, _p_mu, _p_phi))) do (prototype, p)
            @assert typeof(prototype) == typeof(p)
            !isnothing(prototype) && return reshape(p, :, 1)
            return nothing
        end
        ps = filter(x -> !isnothing(x), ps)
        chain(vcat(reshape(t, :, 1), reshape(x0, :, 1), ps...))
    end

    train_out = chain(train_data)
    PIDESolution(xi, ts, losses, train_out, sol_func, nothing)
end #solve
