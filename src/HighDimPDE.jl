module HighDimPDE
    using Reexport
    @reexport using DiffEqBase
    using Statistics
    using Flux, Zygote, LinearAlgebra
    # using ProgressMeter: @showprogress
    using CUDA
    using Random
    using SparseArrays

    abstract type HighDimPDEAlgorithm <: DiffEqBase.AbstractODEAlgorithm end


    struct NLFunction{F} <: DiffEqBase.AbstractODEFunction{false}
        f::F
    end
    (f::NLFunction)(args...) = f.f(args...)

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
    # Options
    * `u_domain` : the domain, correspoding to the hypercube
    ∏_i [u_domain[1][i], u_domain[2][i]]. In this case the problem has reflecting boundary conditions
    """
    struct PIDEProblem{X0Type,uType,tType,G,F,Mu,Sigma,P,A,UD,K} <: DiffEqBase.AbstractODEProblem{uType,tType,false}
        u0::uType
        g::G # initial condition
        f::F # nonlinear part
        μ::Mu
        σ::Sigma
        X0::X0Type
        tspan::tType
        p::P
        A::A
        u_domain::UD
        kwargs::K
        PIDEProblem(g, f, μ, σ, X0, tspan, p=nothing;A=nothing,
                                                    u_domain=nothing,
                                                    kwargs...) = new{typeof(X0),
                                                                    typeof(g(X0)),
                                                                    typeof(tspan),
                                                                    NLFunction,
                                                                    NLFunction,
                                                                    typeof(μ),
                                                                    typeof(σ),
                                                                    typeof(p),typeof(A),typeof(u_domain),typeof(kwargs)}(
                                                                    g(X0),NLFunction(g),NLFunction(f),μ,σ,X0,tspan,p,A,u_domain,kwargs)
    end

    Base.summary(prob::PIDEProblem) = string(nameof(typeof(prob)))

    function Base.show(io::IO, A::PIDEProblem)
    println(io,summary(A))
    print(io,"timespan: ")
    show(io,A.tspan)
    end

    function _initializer(use_cuda)
        if use_cuda && CUDA.functional()
            @info "Training on CUDA GPU"
            CUDA.allowscalar(false)
            global _device = Flux.gpu
            global rgen! = CUDA.randn!
            global rgen_uni! = CUDA.rand!
        else
            @info "Training on CPU"
            global _device = Flux.cpu
            global rgen! = randn!
            global rgen_uni! = rand!
        end
    end

    _initializer(use_cuda)
    include("MCSample.jl")
    include("reflect.jl")
    include("DeepSplitting.jl")
    include("MLP.jl")

    export PIDEProblem, DeepSplitting, MLP

    export NormalSampling, UniformSampling, NoSampling
end