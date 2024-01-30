"""
```julia
NNStopping(models, opt)
```

[Deep Optimal Stopping](https://arxiv.org/pdf/1908.01602.pdf), S Becker, P Cheridito, A Jentzen3, and T Welti.

## Arguments
- `models::Vector{Flux.Chain}`: A vector of Flux.Chain where each model corresponds to a specific timestep from the timespan (tspan). The overall length of the vector should be `length(timesteps) - 1`.
- `opt`: the optimization algorithm to be used to optimize the neural networks. Defaults to `ADAM(0.1)`.

## Example
d = 3 # Number of assets in the stock
r = 0.05 # interest rate
beta = 0.2 # volatility
T = 3 # maturity
u0 = fill(90.0, d) # initial stock value
delta = 0.1 # delta
f(du, u, p, t) = du .= (r - delta) * u # drift
sigma(du, u, p, t) = du .= beta * u # diffusion
tspan = (0.0, T)
N = 9 # discretization parameter
dt = T / (N)
K = 100.00 # strike price

# payoff function
function g(x, t)
    return exp(-r * t) * (max(maximum(x) - K, 0))
end

prob = PIDEProblem(f, sigma, u0, tspan; payoff = g)
models = [Chain(Dense(d + 1, 32, tanh), BatchNorm(32, tanh), Dense(32, 1, sigmoid))
          for i in 1:N]

opt = Flux.Optimisers.Adam(0.01)
alg = NNStopping(models, opt)

sol = solve(prob, alg, SRIW1(); dt = dt, trajectories = 1000, maxiters = 1000, verbose = true)
```
"""
struct NNStopping{M, O}
    m::M
    opt::O
end

struct NNStoppingModelArray{M}
    ms::M
end

Flux.@functor NNStoppingModelArray

function (model::NNStoppingModelArray)(X, G)
    XG = cat(X, reshape(G, 1, size(G)...), dims = 1)
    broadcast((x, m) -> m(x), eachslice(XG, dims = 2)[2:end], model.ms)
end

function DiffEqBase.solve(prob::PIDEProblem,
        pdealg::NNStopping,
        sdealg;
        verbose = false,
        maxiters = 300,
        trajectories = 100,
        dt = eltype(prob.tspan)(0),
        ensemblealg = EnsembleThreads(),
        kwargs...)
    g = prob.kwargs[:payoff]

    sde_prob = SDEProblem(prob.μ, prob.σ, prob.x, prob.tspan)
    ensemble_prob = EnsembleProblem(sde_prob)
    sim = solve(ensemble_prob,
        sdealg,
        trajectories = trajectories,
        dt = dt,
        adaptive = false)

    m = NNStoppingModelArray(pdealg.m)
    opt = pdealg.opt
    tspan = prob.tspan
    ts = tspan[1]:dt:tspan[2]
    N = length(ts) - 1

    G = reduce(hcat, map(u -> map(i -> g(u.u[i], u.t[i]), 1:(N + 1)), sim))

    function nn_loss(m)
        preds = m(Array(sim), G)
        preds = reduce(vcat, preds)
        un_gs = map(eachcol(preds), eachcol(G)) do pred, g_
            local u_sum = pred[1]
            uns = map(1:N) do i
                i == 1 && return pred[i]
                # i == N && return (1 - u_sum)
                res = pred[i] * (1 - u_sum)
                u_sum += res
                return res
            end

            return sum(uns .* g_[2:end])
        end
        return -1 * mean(un_gs)
    end

    opt_state = Flux.setup(opt, m)
    for epoch in 1:maxiters
        # sim = solve(ensemble_prob, EM(), trajectories = M, dt = dt)
        gs = Flux.gradient(m) do model
            nn_loss(model)
        end
        Flux.update!(opt_state, m, gs[1])
        l = nn_loss(m)
        verbose && @info "Current Epoch : $epoch  Current Loss :$l"
    end

    final_preds = m(Array(sim), G)
    final_preds = reduce(vcat, final_preds)
    un_gs = map(eachcol(final_preds), eachcol(G)) do pred, g_
        local u_sum = pred[1]
        uns = map(1:N) do i
            i == 1 && return pred[i]
            res = pred[i] * (1 - u_sum)
            u_sum += res
            return res
        end
        uns
    end

    ns = map(un_gs) do un
        argmax(cumsum(un) + un .>= 1)
    end

    tss = getindex.(Ref(ts), ns .+ 1)
    payoff = mean(map((t, u) -> g(u(t), t), tss, sim))
    return (payoff = payoff, stopping_time = tss)
end
