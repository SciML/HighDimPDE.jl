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
    mc_sample!::MCSampling # Monte Carlo sample
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
    u_domain = prob.u_domain |> _device
    x0 = prob.x |> _device
    d  = size(x0,1)
    K = alg.K
    opt = alg.opt
    g,f,μ,σ,p = prob.g,prob.f,prob.μ,prob.σ,prob.p
    mc_sample! =  alg.mc_sample!
    #hidden layer
    nn = alg.nn |> _device
    vi = g
    vj = deepcopy(nn)
    ps = Flux.params(vj)

    # output solution
    if isnothing(u_domain)
        sample_initial_points! = NoSampling()
        usol = [g(x0)[]]
        T = eltype(x0)
    else
        usol = Any[g]
        T = eltype(u_domain)
        sample_initial_points! = UniformSampling(u_domain[:,1], u_domain[:,2])
    end

    dt = convert(T,dt)
    ts = prob.tspan[1]:dt-eps(T):prob.tspan[2]
    N = length(ts) - 1

    # allocating
    y0 = similar(x0,d,batch_size)
    y1 = similar(x0,d,batch_size)
    z = similar(x0, d, batch_size, K) # for MC non local integration

    # checking element types
    eltype(mc_sample!) == T || !_integrate(mc_sample!) ? nothing : error("Type of mc_sample! not the same as x0")

    function splitting_model(y0, y1, z, t)
        ∇vi(x) = 0f0#gradient(vi,x)[1]
        # Monte Carlo integration
        # z is the variable that gets integrated
        _int = sum(f(y1, z, vi(y1), vi(z), ∇vi(y1), ∇vi(z), t), dims = 3) 
        vj(y0) - (vi(y1) + dt * _int / K)
    end

    function loss(y0, y1, z, t)
        u = splitting_model(y0, y1, z, t)
        return mean(u.^2)
    end

    # calculating SDE trajectories
    function sde_loop!(y0, y1, dWall)
        randn!(dWall) # points normally distributed for brownian motion
        sample_initial_points!(y1) # points uniformly distributed for initial conditions
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
        dWall = similar(x0, d, batch_size, N + 1 - net) # for SDE

        verbose && println("Step $(net) / $(N) ")
        t = ts[net]

        # @showprogress
        for epoch in 1:maxiters
            # verbose && println("epoch $epoch")

            # verbose && println("sde loop")
            sde_loop!(y0, y1, dWall)
            # verbose && println("mc samples")
            if _integrate(mc_sample!)
                # TODO : remove loop?
                for i in 1:K
                    zi = @view z[:,:,i]
                    mc_sample!(zi, y0)
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
        if isnothing(u_domain)
            push!(usol, mean(vi(x0)) |> cpu)
        else
            push!(usol, vi |> cpu)
        end
    end
    if isnothing(u_domain)
        sol = DiffEqBase.build_solution(prob,alg,ts,usol)
    else
        sample_initial_points!(y1)
        xgrid = [reshape(y1[:,i],d,1) for i in 1:size(y1,2)] .|> cpu #reshape needed for batch size
        sol = xgrid,usol
    end
    return sol
end

