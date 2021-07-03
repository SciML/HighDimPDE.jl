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
    `[u_domain[1], u_domain[2]]^size(x0,1)`. 
    In this case the problem has Neumann boundary conditions.
    """
    struct PIDEProblem{uType,G,F,Mu,Sigma,xType,tType,P,UD,K} <: DiffEqBase.AbstractODEProblem{uType,tType,false}
        u0::uType
        g::G # initial condition
        f::F # nonlinear part
        μ::Mu
        σ::Sigma
        x::xType
        tspan::tType
        p::P
        u_domain::UD
        kwargs::K
    end

    function PIDEProblem(g, f, μ, σ, tspan, p=nothing;
                        x0=nothing,
                        u_domain=nothing,
                        kwargs...)
    if isnothing(x0) && !isnothing(u_domain)
        x = u_domain[:,1]
        println(g(Y))
        println(typeof(g(Y)))
    elseif !isnothing(x0) && isnothing(u_domain)
        x = x0
        println("hey")
    else
        error("Need to provide whether `x0` or `u_domain`")
    end
    println(g(Y))
    PIDEProblem{typeof(g(Y)),
                NLFunction,
                NLFunction,
                typeof(μ),
                typeof(σ),
                typeof(x),
                typeof(tspan),
                typeof(p),typeof(u_domain),typeof(kwargs)}(g(Y),NLFunction(g),NLFunction(f),μ,σ,x,tspan,p,u_domain,kwargs)
    end

    Base.summary(prob::PIDEProblem) = string(nameof(typeof(prob)))

    function Base.show(io::IO, A::PIDEProblem)
    println(io,summary(A))
    print(io,"timespan: ")
    show(io,A.tspan)
    end

    include("MCSample.jl")
    include("reflect.jl")
    include("DeepSplitting.jl")
    include("MLP.jl")

    export PIDEProblem, DeepSplitting, MLP

    export NormalSampling, UniformSampling, NoSampling, solve
end