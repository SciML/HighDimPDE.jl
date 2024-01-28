"""
Algorithm for solving Backward Kolmogorov Equations.

```julia
NeuralPDE.NNKolmogorov(chain, opt, sdealg, ensemblealg )
```
Arguments:
- `chain`: A Chain neural network with a d-dimensional output.
- `opt`: The optimizer to train the neural network. Defaults to `ADAM(0.1)`.
- `sdealg`: The algorithm used to solve the discretized SDE according to the process that X follows. Defaults to `EM()`.
- `ensemblealg`: The algorithm used to solve the Ensemble Problem that performs Ensemble simulations for the SDE. Defaults to `EnsembleThreads()`. See
  the [Ensemble Algorithms](https://diffeq.sciml.ai/stable/features/ensemble/#EnsembleAlgorithms-1)
  documentation for more details.
- - `kwargs`: Additional arguments splatted to the SDE solver. See the
  [Common Solver Arguments](https://diffeq.sciml.ai/dev/basics/common_solver_opts/)
  documentation for more details.
[1]Beck, Christian, et al. "Solving stochastic differential equations and Kolmogorov equations by means of deep learning." arXiv preprint arXiv:1806.00421 (2018).
"""
struct NNKolmogorov{C, O} <: HighDimPDEAlgorithm
    chain::C
    opt::O
end
NNKolmogorov(chain; opt = Flux.ADAM(0.1)) = NNKolmogorov(chain, opt)

function DiffEqBase.solve(prob::Union{PIDEProblem, SDEProblem},
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
        kwargs...)
    tspan = prob.tspan
    sigma = prob.σ
    μ = prob.μ
    noise_rate_prototype = prob.kwargs.noise_rate_prototype
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
    sdeproblem = SDEProblem(μ,
        sigma,
        xi,
        tspan,
        noise_rate_prototype = noise_rate_prototype)
    function prob_func(prob, i, repeat)
        SDEProblem(prob.f,
            xi[:, i],
            prob.tspan,
            noise_rate_prototype = prob.noise_rate_prototype)
    end
    output_func(sol, i) = (sol.u[end], false)
    ensembleprob = EnsembleProblem(sdeproblem,
        prob_func = prob_func,
        output_func = output_func)
    sim = solve(ensembleprob,
        sdealg,
        ensemblealg,
        dt = dt,
        trajectories = trajectories,
        adaptive = false)

    x_sde = Array(sim)

    y = reduce(hcat, phi.(eachcol(x_sde)))

    if use_gpu == true
        y = y |> gpu
        xi = xi |> gpu
    end
    data = Iterators.repeated((xi, y), maxiters)
    if use_gpu == true
        data = data |> gpu
    end

    #MSE Loss Function
    loss(x, y) = Flux.mse(chain(x), y)

    losses = AbstractFloat[]
    callback = function ()
        l = loss(xi, y)
        verbose && println("Current loss is: $l")
        push!(losses, l)
        l < abstol && Flux.stop()
    end

    Flux.train!(loss, ps, data, opt; cb = callback)
    chainout = chain(xi)
    xi, chainout
    return PIDESolution(xi, ts, losses, chainout, chain, nothing)
end
