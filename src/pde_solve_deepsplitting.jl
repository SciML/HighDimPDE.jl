using DiffEqBase
using Flux, Zygote, LinearAlgebra, StochasticDiffEq
import NeuralPDE
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

struct NNPDEDS{C1,O} <: NeuralPDE.NeuralPDEAlgorithm
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
    sdealg = EM(),
    ensemblealg = EnsembleThreads(),
    maxiters_upper = 10,
    )

    println("leeeeeets goooo")

    X0 = prob.X0
    ts = prob.tspan[1]:dt:prob.tspan[2]
    N = length(ts)
    d  = length(X0)
    g,f,μ,σ,p = prob.g,prob.f,prob.μ,prob.σ,prob.p

    # this is where you should put the sde loop
    data = Iterators.repeated((), maxiters)


    #hidden layer
    nn = alg.nn
    K = alg.K
    opt = alg.opt
    N = length(ts) - 1

    vi(x) = g(x)
    vj = deepcopy(nn)

    for net in 1:N
        verbose && println("Step $(net) / $(N) ")
        ps = Flux.params(vj)
        # this is the splitting model
        function splitting_model()
            # calculating SDE trajectories
            # Todo : calculate this outside of the splitting model
            # in the data
            # because this code is differentiated which is slooow
            # for this look here
            # https://fluxml.ai/Flux.jl/stable/training/training/#Custom-Training-loops-1
            y0 = y1 = repeat(X0[:],1,batch_size)
            for i in 1:(N - net)
                # not sure about this one
                t = ts[N- i]
                dW = sqrt(dt)*randn(d,batch_size)
                y0 = y1
                y1  = y0 .+ μ(y0,p,t)*dt .+ σ(y0,p,t)*dW
            end
        # Monte Carlo integration
        # z is the variable that gets integreated
            _int = zeros(1,batch_size)
            t = ts[net]
            for _ in 1:K
                 z = mc_sample(y0)
                 ∇vi(x) = 0.#gradient(vi,x)[1]
                _int += f(y1, z, vi(y1), vi(z),∇vi(y1) ,∇vi(y1) , p, t)
            end
            u = vj(y0) - (vi(y1) + dt * _int / K)
        end

        function loss()
            mean(sum(abs2,u) for u in splitting_model())
        end

        iters = eltype(X0)[]

        function cb()
            # save_everystep && push!(iters, vi(X0)[1])
            l = loss()
            verbose && println("Current loss is: $l")
            l < abstol && Flux.stop()
        end

        Flux.train!(loss, ps, data, opt; cb = cb)
        vi = deepcopy(vj)
    end
    vj(X0)[1]
    # save_everystep ? iters : u0(X0)[1]
end
