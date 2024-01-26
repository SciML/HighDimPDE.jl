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

function DiffEqBase.solve(prob::SDEProblem,
        pdealg::NNStopping,
        sdealg;
        verbose = false,
        maxiters = 300,
        trajectories = 100,
        dt = eltype(prob.tspan)(0),
        ensemblealg = EnsembleThreads(),
        kwargs...)
    g = prob.kwargs[:payoff]

    sde_prob = SDEProblem(prob.f, prob.u0, prob.tspan)
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

    G = reduce(hcat, map(u -> map(i -> g(u.t[i], u.u[i]), 1:(N + 1)), sim))

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
    payoff = mean(map((t, u) -> g(t, u(t)), tss, sim))
    return (payoff = payoff, stopping_time = tss)
end
