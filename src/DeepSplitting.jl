"""
DeepSplitting(f, K, opt, mc_sample)
Deep splitting algorithm for solving non local non linear PDES.

Arguments:
* `f`: a [Flux.Chain](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Chain),
or more generally a [functor](https://github.com/FluxML/Functors.jl) function
* `K`: the number of Monte Carlo integrations
* `opt`: optimiser to be use. By default, `Flux.ADAM(0.1)`.
* `mc_sample::MCSampling` : sampling method for Monte Carlo integrations of the non local term.
Can be `UniformSampling(a,b)`, `NormalSampling(σ_sampling)`, or `NoSampling` (by default).

"""
struct DeepSplitting{NN,F,O,MCS} <: HighDimPDEAlgorithm
    nn::NN
    K::F
    opt::O
    mc_sample!::MCS # Monte Carlo sample
end

function DeepSplitting(nn; K=1, opt=Flux.ADAM(0.1), mc_sample::MCSampling = NoSampling()) 
    DeepSplitting(nn, K, opt, mc_sample)
end

function solve(
    prob::PIDEProblem,
    alg::DeepSplitting,
    dt;
    batch_size = 1,
    abstol = 1f-6,
    verbose = false,
    maxiters = 300,
    use_cuda = false,
    )
    if use_cuda
        if CUDA.functional()
            @info "Training on CUDA GPU"
            CUDA.allowscalar(false)
            _device = Flux.gpu
        else
            error("CUDA not functional, deactivate `use_cuda` and retry")
        end
    else
        @info "Training on CPU"
        _device = Flux.cpu
    end

    # unbin stuff
    u_domain = prob.u_domain |> _device # domain on which we want to approximate u, nothing if only one point wanted
    neumann_bc = prob.neumann_bc |> _device
    x0 = prob.x |> _device
    mc_sample! =  alg.mc_sample! |> _device

    d  = size(x0,1)
    K = alg.K
    opt = alg.opt
    g,f,μ,σ,p = prob.g,prob.f,prob.μ,prob.σ,prob.p

    # neural network model
    nn = alg.nn |> _device
    vi = g
    vj = deepcopy(nn)
    ps = Flux.params(vj)

    # output solution
    if isnothing(u_domain)
        sample_initial_points! = NoSampling()
        usol = [g(x0 |>cpu)[]]
        T = eltype(x0)
    else
        usol = Any[g]
        T = eltype(u_domain)
        sample_initial_points! = UniformSampling(u_domain[1], u_domain[2])
    end

    dt = convert(T,dt)
    ts = prob.tspan[1]:dt-eps(T):prob.tspan[2]
    N = length(ts) - 1

    # allocating
    x0_batch = repeat(x0, 1, batch_size)
    y1 = similar(x0_batch)
    y0 = similar(y1)
    z = similar(x0, d, batch_size, K) # for MC non local integration

    # checking element types
    eltype(mc_sample!) == T || !_integrate(mc_sample!) ? nothing : error("Type of mc_sample! not the same as x0")

    function splitting_model(y0, y1, z, t)
        # TODO: for now hardcoded because of a bug in Zygote differentiation rules for adjoints
        # ∇vi(x) = vcat(first.(Flux.jacobian.(vi, eachcol(x)))...)'
        ∇vi(x) = [0f0]
        # Monte Carlo integration
        _int = reshape(sum(f(y1, z, vi(y1), vi(z), ∇vi(y1), ∇vi(z), p, t), dims = 3), 1, :)
        vj(y0) - (vi(y1) + dt * _int / K)
    end

    function loss(y0, y1, z, t)
        u = splitting_model(y0, y1, z, t)
        return sum(u.^2) / batch_size
    end

    # calculating SDE trajectories
    function sde_loop!(y0, y1, dWall)
        randn!(dWall) # points normally distributed for brownian motion
        sample_initial_points!(y1) # points uniformly distributed for initial conditions
        for i in 1:size(dWall,3)
            # @show i
            # not sure about this one
            t = ts[N + 1 - i]
            dW = @view dWall[:,:,i]
            y0 .= y1
            y1 .= y0 .+ μ(y0,p,t) .* dt .+ σ(y0,p,t) .* sqrt(dt) .* dW
            if !isnothing(neumann_bc)
                y1 .= _reflect_GPU(y0, y1, neumann_bc[1], neumann_bc[2])
            end
        end
    end

    for net in 1:N
        # preallocate dWall
        dWall = similar(x0, d, batch_size, N + 1 - net) # for SDE

        verbose && println("Step $(net) / $(N) ")
        t = ts[net]

        # @showprogress
        for epoch in 1:maxiters
            y1 .= x0_batch
            # generating sdes
            sde_loop!(y0, y1, dWall)

            if _integrate(mc_sample!)
                # generating z for MC non local integration
                mc_sample!(z, y1)
                # need to reflect the sampling
                if !isnothing(neumann_bc) && typeof(mc_sample!) == NormalSampling
                    _y1K = similar(z)
                    _y1K .= y1
                    z .= _reflect_GPU(_y1K, z, neumann_bc[1], neumann_bc[2])
                end
            end

            # training
            gs = Flux.gradient(ps) do
                loss(y0, y1, z, t)
            end
            Flux.Optimise.update!(opt, ps, gs) # update parameters
            
            # report on training
            if epoch % 100 == 1
                l = loss(y0, y1, z, t)
                verbose && println("Current loss is: $l")
                l < abstol && break
            end
            if epoch == maxiters
                l = loss(y0, y1, z, t)
                verbose && println("Current loss is: $l")
            end
        end

        # saving
        vi = deepcopy(vj)
        # vj = deepcopy(nn)
        # ps = Flux.params(vj)
        if isnothing(u_domain)
            # reshape used in the case where there are some normalisation in layers
            push!(usol, cpu(vi(reshape(x0, d, 1)))[] )
        else
            push!(usol, vi |> cpu)
        end
    end

    # return
    if isnothing(u_domain)
        # sol = DiffEqBase.build_solution(prob, alg, ts, usol)
        x0 = x0 |> cpu
        sol = x0, ts, usol
    else
        sample_initial_points!(y1)
        xgrid = [reshape(y1[:,i], d, 1) for i in 1:size(y1,2)] .|> cpu #reshape needed for batch size
        sol = xgrid, ts, usol
    end
    return sol
end

