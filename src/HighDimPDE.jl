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
        PIDEProblem(g,f, μ, σ, x, tspan)
    A non local non linear PDE problem.
    Consider `du/dt = 1/2 Tr(\\sigma \\sigma^T) Δu(t,x) + μ ∇u(t,x) + \\int f(u,x) dx`; where f is the nonlinear Lipschitz function
    # Arguments
    * `g` : The terminal condition for the equation.
    * `f` : The function f(u(x),u(y),du(x),du(y),x,y)
    * `μ` : The drift function of X from Ito's Lemma
    * `σ` : The noise function of X from Ito's Lemma
    * `tspan`: The timespan of the problem.
    # Options
    * `u_domain` : the domain of the solution required, correspoding to the hypercube
    `u_domain[:,1] × u_domain[:,2]`. 
    * `x`: the point of the solution required
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

    function PIDEProblem(g, f, μ, σ, tspan;
                                    p=nothing,
                                    x=nothing,
                                    u_domain=nothing,
                                    kwargs...)
    if isnothing(x) && !isnothing(u_domain)
        size(u_domain,2) == 2 ? nothing : error("`u_domain` needs to be of dimension `(d,2)`")
        x = u_domain[:,1]
    elseif !isnothing(x) && !isnothing(u_domain)
        error("Need to provide whether `x` or `u_domain`")
    end

    eltype(g(x)) == eltype(x) ? nothing : error("Type of `g(x)` not matching type of x")
    eltype(f(x, x, g(x), g(x), 0f0, 0f0, tspan[1])) == eltype(x) ? nothing : error("Type of non linear function `f(x)` not matching type of x")

    PIDEProblem{typeof(g(x)),
                NLFunction,
                NLFunction,
                typeof(μ),
                typeof(σ),
                typeof(x),
                typeof(tspan),
                typeof(p),typeof(u_domain),typeof(kwargs)}(g(x),NLFunction(g),NLFunction(f),μ,σ,x,tspan,p,u_domain,kwargs)
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