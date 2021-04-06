using DiffEqBase
abstract type NeuralPDEAlgorithm <: DiffEqBase.AbstractODEAlgorithm end
using Flux, Zygote, LinearAlgebra
using ProgressMeter: @showprogress
using CUDA

## this one is in fact not used!
"""
    PIDEProblem(g,f, μ, σ, x0, tspan)
A non local non linear PDE problem.
Consider `du/dt = l(u) + \\int f(u,x) dx`; where l is the nonlinear Lipschitz function
# Arguments
* `g` : The terminal condition for the equation.
* `f` : The function f(u(x),u(y),du(x),du(y),x,y)
* `μ` : The drift function of X from Ito's Lemma
* `μ` : The noise function of X from Ito's Lemma
* `x0`: The initial X for the problem.
* `tspan`: The timespan of the problem.
"""
struct PIDEProblem{G,F,Mu,Sigma,X,T,P,A,UD,K} <: DiffEqBase.DEProblem
    g::G
    f::F
    μ::Mu
    σ::Sigma
    X0::X
    tspan::Tuple{T,T}
    p::P
    A::A
    u_domain::UD
    kwargs::K
    PIDEProblem(g,f,μ,σ,X0,tspan,p=nothing;A=nothing,u_domain=nothing,kwargs...) = new{typeof(g),typeof(f),
                                                         typeof(μ),typeof(σ),
                                                         typeof(X0),eltype(tspan),
                                                         typeof(p),typeof(A),typeof(u_domain),typeof(kwargs)}(
                                                         g,f,μ,σ,X0,tspan,p,A,u_domain,kwargs)
end

Base.summary(prob::PIDEProblem) = string(nameof(typeof(prob)))

function Base.show(io::IO, A::PIDEProblem)
  println(io,summary(A))
  print(io,"timespan: ")
  show(io,A.tspan)
end

"""
Deep splitting algorithm for solving non local non linear PDES.

Arguments:
* `chain`: a Flux.jl chain with a d-dimensional input and a 1-dimensional output,
* `strategy`: determines which training strategy will be used,
* `init_params`: the initial parameter of the neural network,
* `phi`: a trial solution,
* `derivative`: method that calculates the derivative.

"""

struct NNPDEDS{C1,O} <: NeuralPDEAlgorithm
    nn::C1
    K::Int
    opt::O
end
NNPDEDS(nn;K=1,opt=Flux.ADAM(0.1)) = NNPDEDS(nn,K,opt)

function DiffEqBase.solve(
    # prob::PIDEProblem,
    prob::PIDEProblem,
    alg::NNPDEDS,
    mc_sample;
    dt,
    batch_size = 1,
    abstol = 1f-6,
    verbose = false,
    maxiters = 300,
    save_everystep = false,
    give_limit = false,
    ensemblealg = EnsembleThreads(),
    maxiters_upper = 10,
    use_cuda = false
    )

    function splitting_model(y0,y1,t)
        # Monte Carlo integration
        # z is the variable that gets integreated
        _int = zeros(Float32,1,batch_size) |> _device
        for _ in 1:K
             z = mc_sample(y0)
             ∇vi(x) = 0f0#gradient(vi,x)[1]
            _int = _int + f(y1, z, vi(y1), vi(z), ∇vi(y1), ∇vi(y1), p, t)
        end
        vj(y0) - (vi(y1) + dt * _int / K)
    end

    function loss(y0,y1,t)
        u = splitting_model(y0,y1,t)
        return mean(u.^2)
    end

    # calculating SDE trajectories
    function sde_loop!(y0,y1,dWall,net)
        rgen!(dWall)
        for i in 1:size(dWall,3)
            # not sure about this one
            t = ts[N + 1 - i]
            dW = @view dWall[:,:,i]
            y0 .= y1
            y1 .= y0 .+ μ(y0,p,t) .* dt .+ σ(y0,p,t) .* sqrt(dt) .* dW
        end
        return y0, y1
    end

    if use_cuda && CUDA.functional()
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        _device = Flux.gpu
        rgen! = CUDA.randn!
    else
        @info "Training on CPU"
        _device = Flux.cpu
        rgen! = randn!
    end

    X0 = prob.X0 |> _device
    ts = prob.tspan[1]:dt:prob.tspan[2]
    N = length(ts)
    d  = length(X0)
    g,f,μ,σ,p = prob.g,prob.f,prob.μ,prob.σ,prob.p

    #hidden layer
    nn = alg.nn |> _device
    vi = g |> _device
    vj = deepcopy(nn)

    ps = Flux.params(vj)

    K = alg.K
    opt = alg.opt
    N = length(ts) - 1

    # preallocate y0, y1
    y0 = repeat(X0[:],1,batch_size)
    y1 = repeat(X0[:],1,batch_size)

    for net in 1:N
        # preallocate dWall
        dWall = zeros(Float32, d, batch_size, N + 1 - net) |> _device

        verbose && println("Step $(net) / $(N) ")
        t = ts[net]

        # @showprogress
        for epoch in 1:maxiters
            y0 .= repeat(X0[:],1,batch_size)
            y1 .= repeat(X0[:],1,batch_size)
            sde_loop!(y0, y1, dWall, net)
            gs = Flux.gradient(ps) do
                loss(y0,y1,t)
            end
            Flux.Optimise.update!(opt, ps, gs) # update parameters
            # report on train
            if epoch % 100 == 1
                l = loss(y0,y1,t)
                verbose && println("Current loss is: $l")
                l < abstol && break
            end
            if epoch == maxiters
                l = loss(y0,y1,t)
                verbose && println("Current loss is: $l")
                # we change abstol as we can not get more precise over time
                abstol = 1.05 * l
            end
        end
        vi = deepcopy(vj)
    end

    mean(vj(X0))
    # save_everystep ? iters : u0(X0)[1]
end
