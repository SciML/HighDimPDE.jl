_copy(t::Tuple) = t
_copy(t) = t
function _copy(opt::O) where  O<:Flux.Optimise.AbstractOptimiser
    return O([_copy(getfield(opt,f)) for f in fieldnames(typeof(opt))]...)
end

"""
    DeepSplitting(nn, K=1, opt = Flux.Optimise.Adam(0.01), λs = nothing, mc_sample =  NoSampling())

Deep splitting algorithm.

# Arguments
* `nn`: a [Flux.Chain](https://fluxml.ai/Flux.jl/stable/reference/models/layers/#Flux.Chain), or more generally a [functor](https://github.com/FluxML/Functors.jl).
* `K`: the number of Monte Carlo integrations.
* `opt`: optimizer to be used. By default, `Flux.Optimise.Adam(0.01)`.
* `λs`: the learning rates, used sequentially. Defaults to a single value taken from `opt`.
* `mc_sample::MCSampling` : sampling method for Monte Carlo integrations of the non-local term. Can be `UniformSampling(a,b)`, `NormalSampling(σ_sampling, shifted)`, or `NoSampling` (by default).

# Example
```julia
hls = d + 50 # hidden layer size
d = 10 # size of the sample

# Neural network used by the scheme
nn = Flux.Chain(Dense(d, hls, tanh),
                Dense(hls,hls,tanh),
                Dense(hls, 1, x->x^2))

alg = DeepSplitting(nn, K=10, opt = Flux.Optimise.Adam(), λs = [5e-3,1e-3],
                    mc_sample = UniformSampling(zeros(d), ones(d)) )
```
"""
struct DeepSplitting{NN, F, O, L, MCS} <: HighDimPDEAlgorithm
    nn::NN
    K::F
    opt::O
    λs::L
    mc_sample!::MCS # Monte Carlo sample
end

function DeepSplitting(nn;
        K = 1,
        opt::O = Flux.Optimise.Adam(0.01),
        λs::L = nothing,
        mc_sample = NoSampling()) where {
        O <: Flux.Optimise.AbstractOptimiser,
        L <: Union{Nothing, Vector{N}} where {N <: Number},
}
    isnothing(λs) ? λs = [opt.eta] : nothing
    DeepSplitting(nn, K, opt, λs, mc_sample)
end

"""
$(TYPEDSIGNATURES)

Returns a `PIDESolution` object.

# Arguments
- `maxiters`: number of iterations per time step. Can be a tuple, where `maxiters[1]` is used for the training of the neural network used in the first time step (which can be long) and `maxiters[2]` is used for the rest of the time steps.
- `batch_size` : the batch size.
- `abstol` : threshold for the objective function under which the training is stopped.
- `verbose` : print training information.
- `verbose_rate` : rate for printing training information (every `verbose_rate` iterations).
- `use_cuda` : set to `true` to use CUDA.
- `cuda_device` : integer, to set the CUDA device used in the training, if `use_cuda == true`.
"""
function DiffEqBase.solve(prob::Union{PIDEProblem, ParabolicPDEProblem},
        alg::DeepSplitting,
        dt;
        batch_size = 1,
        abstol = 1.0f-6,
        verbose = false,
        maxiters = 300,
        use_cuda = false,
        cuda_device = nothing,
        verbose_rate = 100)
    if use_cuda
        if CUDA.functional()
            @info "Training on CUDA GPU"
            CUDA.allowscalar(false)
            !isnothing(cuda_device) ? CUDA.device!(cuda_device) : nothing
            _device = Flux.gpu
        else
            error("CUDA not functional, deactivate `use_cuda` and retry")
        end
    else
        @info "Training on CPU"
        _device = Flux.cpu
    end

    ## unbin stuff
    neumann_bc = prob.neumann_bc |> _device
    x0 = prob.x |> _device
    mc_sample! = alg.mc_sample! |> _device
    x0_sample! = prob.x0_sample |> _device

    d = size(x0, 1)
    K = alg.K
    opt = alg.opt
    λs = alg.λs
    g, μ, σ, p = prob.g, prob.μ, prob.σ, prob.p

    f = if isa(prob, ParabolicPDEProblem)
        (y, z, v_y, v_z, ∇v_y, ∇v_z, p, t) -> prob.f(y, v_y, ∇v_y, p, t )
    else
        prob.f
    end
    T = eltype(x0)

    # neural network model
    nn = alg.nn |> _device
    vi = g
    # fix for deepcopy
    vj = Flux.fmap(nn) do x
        x isa AbstractArray && return copy(x)
        x
    end
    ps = Flux.params(vj)

    dt = convert(T, dt)
    ts = prob.tspan[1]:(dt - eps(T)):prob.tspan[2]
    N = length(ts) - 1

    usol = [g(x0 |> cpu)[]]
    nns = Any[g]
    losses = [Vector{eltype(prob.x)}() for net in 1:(N + 1)]

    # allocating
    x0_batch = repeat(x0, 1, batch_size)
    y1 = similar(x0_batch)
    y0 = similar(y1)
    z = similar(x0, d, batch_size, K) # for MC non local integration

    # checking element types
    eltype(mc_sample!) == T || !_integrate(mc_sample!) ? nothing :
    error("Element type of `mc_sample` not the same as element type of `x`")

    function splitting_model(y0, y1, z, t)
        # TODO: for now hardcoded because of a bug in Zygote differentiation rules for adjoints
        # vi_y1, ∇vi = Zygote.pullback(vi, y1)
        # _int = reshape(sum(f(y1, z, vi_y1, vi(z), ∇vi(y1)[1], ∇vi(z)[1], p, t), dims = 3), 1, :)
        ∇vi(x) = [0.0f0]
        _int = reshape(sum(f(y1, z, vi(y1), vi(z), ∇vi(y1), ∇vi(z), p, t), dims = 3), 1, :)
        return vj(y0) - (vi(y1) + dt * _int / K)
    end

    function loss(y0, y1, z, t)
        u = splitting_model(y0, y1, z, t)
        return sum(u .^ 2) / batch_size
    end

    # calculating SDE trajectories
    function sde_loop!(y0, y1, dWall)
        randn!(dWall) # points normally distributed for brownian motion
        x0_sample!(y1) # points for initial conditions
        for i in 1:size(dWall, 3)
            t = ts[N + 1 - i]
            dW = @view dWall[:, :, i]
            y0 .= y1
            y1 .= y0 .+ μ(y0, p, t) .* dt .+ σ(y0, p, t) .* sqrt(dt) .* dW
            if !isnothing(neumann_bc)
                y1 .= _reflect(y0, y1, neumann_bc[1], neumann_bc[2])
            end
        end
    end

    for net in 1:N
        # preallocate dWall
        dWall = similar(x0, d, batch_size, N + 1 - net) # for SDE

        verbose && println("Step $(net) / $(N) ")
        t = ts[net]
        # first of maxiters used for first nn, second used for the other nn
        _maxiters = length(maxiters) > 1 ? maxiters[min(net, 2)] : maxiters[]

        for λ in λs
            opt_net = _copy(opt) # starting with a new optimiser state at each time step
            opt_net.eta = λ
            verbose &&
                println("Training started with ", typeof(opt_net), " and λ :", opt_net.eta)
            for epoch in 1:_maxiters
                y1 .= x0_batch
                # generating sdes
                sde_loop!(y0, y1, dWall)

                if _integrate(mc_sample!)
                    # generating z for MC non local integration
                    mc_sample!(z, y1)
                end

                # training
                gs = Flux.gradient(ps) do
                    loss(y0, y1, z, t)
                end
                Flux.Optimise.update!(opt_net, ps, gs) # update parameters

                # report on training
                if epoch % verbose_rate == 1
                    l = loss(y0, y1, z, t) # explicitly computing loss every verbose_rate
                    verbose && println("Current loss is: $l")
                    push!(losses[net], l)
                    if l < abstol
                        break
                    end
                end
                if epoch == maxiters
                    l = loss(y0, y1, z, t)
                    push!(losses[net + 1], l)
                    verbose && println("Final loss for step $(net) / $(N) is: $l")
                end
            end
        end
        # saving
        # fix for deepcopy
        vi = Flux.fmap(vj) do x
            x isa AbstractArray && return copy(x)
            x
        end
        # vj = deepcopy(nn)
        # ps = Flux.params(vj)
        push!(usol, cpu(vi(reshape(x0, d, 1)))[])
        push!(nns, vi |> cpu)
    end

    # return
    sol = PIDESolution(x0, ts, losses, usol, nns)
    return sol
end
