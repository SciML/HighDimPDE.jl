"""
Algorithm for solving Kolmogorov Equations.

```julia
HighDimPDE.NNKolmogorov(chain, opt)
```
Arguments:
- `chain`: A Chain neural network with a d-dimensional output.
- `opt`: The optimizer to train the neural network. Defaults to `ADAM(0.1)`.
[1]Beck, Christian, et al. "Solving stochastic differential equations and Kolmogorov equations by means of deep learning." arXiv preprint arXiv:1806.00421 (2018).
"""
struct NNKolmogorov{C, O} <: HighDimPDEAlgorithm
    chain::C
    opt::O
end
NNKolmogorov(chain; opt = Flux.ADAM(0.1)) = NNKolmogorov(chain, opt)

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
- Extra keyword arguments passed to `solve` will be further passed to the SDE solver.
"""
function DiffEqBase.solve(
        prob::ParabolicPDEProblem,
        pdealg::HighDimPDE.NNKolmogorov,
        sdealg;
        ensemblealg = EnsembleThreads(),
        abstol = 1.0f-6,
        verbose = false,
        maxiters = 300,
        trajectories = 1000,
        save_everystep = false,
        use_gpu = false,
        dt,
        dx,
        kwargs...
    )
    tspan = prob.tspan
    sigma = prob.σ
    μ = prob.μ

    noise_rate_prototype = get(prob.kwargs, :noise_rate_prototype, nothing)
    phi = prob.g

    xspan = prob.kwargs.xspan

    xspans = isa(xspan, Tuple) ? [xspan] : xspan

    d = length(xspans)
    ts = tspan[1]:dt:tspan[2]
    xs = map(xspans) do xspan
        xspan[1]:dx:xspan[2]
    end
    N = size(ts)
    T = tspan[2]

    #hidden layer
    chain = pdealg.chain
    opt = pdealg.opt
    ps = Flux.params(chain)
    xi = mapreduce(x -> rand(x, 1, trajectories), vcat, xs)
    #Finding Solution to the SDE having initial condition xi. Y = Phi(S(X , T))
    sdeproblem = SDEProblem(
        μ,
        sigma,
        xi[:, 1],
        tspan,
        noise_rate_prototype = noise_rate_prototype
    )

    function prob_func(prob, i, repeat)
        return SDEProblem(
            prob.f,
            xi[:, i],
            prob.tspan,
            noise_rate_prototype = prob.noise_rate_prototype
        )
    end
    output_func(sol, i) = (sol.u[end], false)
    ensembleprob = EnsembleProblem(
        sdeproblem,
        prob_func = prob_func,
        output_func = output_func
    )
    sim = solve(
        ensembleprob,
        sdealg,
        ensemblealg,
        dt = dt,
        trajectories = trajectories,
        adaptive = false
    )

    x_sde = Array(sim)

    y = reduce(hcat, phi.(eachcol(x_sde)))

    y = use_gpu ? y |> gpu : y
    xi = use_gpu ? xi |> gpu : xi

    #MSE Loss Function
    loss(m, x, y) = Flux.mse(m(x), y)

    losses = AbstractFloat[]

    opt_state = Flux.setup(opt, chain)
    for epoch in 1:maxiters
        gs = Flux.gradient(chain) do model
            loss(model, xi, y)
        end
        Flux.update!(opt_state, chain, gs[1])
        l = loss(chain, xi, y)
        @info "Current Epoch: $epoch Current Loss: $l"
        push!(losses, l)
    end
    # Flux.train!(loss, chain, data, opt; cb = callback)
    chainout = chain(xi)
    xi, chainout
    return PIDESolution(xi, ts, losses, chainout, chain, nothing)
end
