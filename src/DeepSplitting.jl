"""
Deep splitting algorithm for solving non local non linear PDES.

Arguments:
* `nn`: a Flux.jl chain with a d-dimensional input and a 1-dimensional output,
* `K`: the number of Monte Carlo integrations
* `opt`: optimiser to be use. By default, `Flux.ADAM(0.1)`.
* `mc_sample::MCSampling` : sampling method for Monte Carlo integrations of the non local term.
Can be `UniformSampling(a,b)`, `NormalSampling(σ_sampling)`, or `NoSampling` (by default).
"""
struct DeepSplitting{C1,O} <: HighDimPDEAlgorithm
    nn::C1
    K::Int
    opt::O
    mc_sample::MCSampling # Monte Carlo sample
end

function DeepSplitting(nn; K=1, opt=Flux.ADAM(0.1), mc_sample::MCSampling = NoSampling()) 
    DeepSplitting(nn, K, opt, mc_sample)
end

function DiffEqBase.__solve(
    prob::PIDEProblem,
    alg::DeepSplitting;
    dt,
    batch_size = 1,
    abstol = 1f-6,
    verbose = false,
    maxiters = 300,
    use_cuda = false,
    neumann = nothing
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
    u_domain = prob.u_domain
    X0 = prob.X0 |> _device
    ts = prob.tspan[1]:dt:prob.tspan[2]
    dt = convert(eltype(X0),dt)
    N = length(ts) - 1
    d  = length(X0)
    K = alg.K
    opt = alg.opt
    g,f,μ,σ,p = prob.g,prob.f,prob.μ,prob.σ,prob.p
    mc_sample =  alg.mc_sample

    #hidden layer
    nn = alg.nn |> _device
    vi = g
    vj = deepcopy(nn)
    ps = Flux.params(vj)

    y0 = similar(X0,d,batch_size)
    y1 = similar(X0,d,batch_size)
    # output solution is a cpu array
    usol = Any[g]

    # checking element types
    eltype(mc_sample) == eltype(X0) || !_integrate(mc_sample) ? nothing : error("Type of mc_sample not the same as X0")
    eltype(g(X0)) == eltype(X0) ? nothing : error("Type of `g(X0)` not matching type of X0")
    eltype(f(X0, X0, vi(X0), vi(X0), 0f0, 0f0, dt)) == eltype(X0) ? nothing : error("Type of non linear function `f(X0)` not matching type of X0")

    function splitting_model(y0, y1, z, t)
        ∇vi(x) = 0f0#gradient(vi,x)[1]
        zi = z[:,:,1]
        _int = f(y1, zi, vi(y1), vi(zi), ∇vi(y1), ∇vi(y1), t)
        # Monte Carlo integration
        # z is the variable that gets integrated
        for i in 2:K
             zi = z[:,:,i]
            _int += f(y1, zi, vi(y1), vi(zi), ∇vi(y1), ∇vi(y1), t)
        end
        vj(y0) - (vi(y1) + dt * _int / K)
    end

    function loss(y0, y1, z, t)
        u = splitting_model(y0, y1, z, t)
        return mean(u.^2)
    end

    # calculating SDE trajectories
    function sde_loop!(y0, y1, dWall, u_domain, neumann)
        randn!(dWall) # points normally distributed for brownian motion
        sample_initial_points!(y1, u_domain) # points uniformly distributed for initial conditions
        for i in 1:size(dWall,3)
            # not sure about this one
            t = ts[N + 1 - i]
            dW = @view dWall[:,:,i]
            y0 .= y1
            y1 .= y0 .+ μ(y0,p,t) .* dt .+ σ(y0,p,t) .* sqrt(dt) .* dW
            if !isnothing(neumann)
                y1 .= _reflect_GPU(y0, y1, neumann[1], neumann[2])
            end
        end
        return y0, y1
    end

    for net in 1:N
        # preallocate dWall
        # verbose && println("preallocating dWall")
        dWall = similar(X0, d, batch_size, N + 1 - net) # for SDE
        z = similar(X0, d, batch_size, K) # for MC non local integration

        verbose && println("Step $(net) / $(N) ")
        t = ts[net]

        # @showprogress
        for epoch in 1:maxiters
            # verbose && println("epoch $epoch")

            # verbose && println("sde loop")
            sde_loop!(y0, y1, dWall, u_domain, neumann)
            # verbose && println("mc samples")
            if _integrate(mc_sample)
                for i in 1:K
                    zi = @view z[:,:,i]
                    zi .= mc_sample(y0)
                end
            end

            # verbose && println("training gradient")
            gs = Flux.gradient(ps) do
                loss(y0, y1, z, t)
            end
            Flux.Optimise.update!(opt, ps, gs) # update parameters
            # report on train
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
        vi = deepcopy(vj)
        vj = deepcopy(nn)
        ps = Flux.params(vj)
        push!(usol, vi)
    end
    sample_initial_points!(y1, u_domain)
    xgrid = [reshape(y1[:,i],d,1) for i in 1:size(y1,2)] #reshape needed for batch size
    return xgrid,usol
end

function sample_initial_points!(y, u_domain)
    T = eltype(y)
    rand!(y)
    m = sum(u_domain) / convert(T,2)
    y .= (y .- convert(T,0.5)) * (u_domain[2] - u_domain[1]) .+ m
    return y 
end