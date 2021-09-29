module HighDimPDE
    using Reexport
    @reexport using DiffEqBase
    using Statistics
    using Flux, Zygote, LinearAlgebra
    using Functors
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
        PIDEProblem(g, f, μ, σ, x, tspan, p = nothing, x=nothing, u_domain=nothing, neumann_bc=nothing)

    Defines a Partial Integro Differential Problem, of the form `du/dt = 1/2 Tr(\\sigma \\sigma^T) Δu(t,x) + μ ∇u(t,x) + \\int f(u,x) dx`; where f is a nonlinear Lipschitz function
    
    # Arguments
    * `g` : The initial condition g(x, p, t).
    * `f` : The function f(x, y, u(x, t), u(y, t), ∇u(x, t), ∇u(y, t), p, t)
    * `μ` : The drift function of X from Ito's Lemma μ(x, p, t)
    * `σ` : The noise function of X from Ito's Lemma σ(x, p, t)
    * `tspan`: The timespan of the problem.
    * `p`: the parameter 
    * `x`: the point of the solution required
    * `u_domain` : if provided, approximating the solution on the hypercube `u_domain[1] × u_domain[2]`. 
    * `neumann_bc`: if provided, neumann boundary conditions on the hypercube `neumann_bc[1] × neumann_bc[2]`. 
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
        u_domain::Tuple{X,X} # for DeepSplitting only
        neumann_bc::Union{Nothing,Tuple{X,X}} # neumann boundary conditions
        kwargs::K
    end

    function PIDEProblem(g, f, μ, σ, tspan;
                                    p=nothing,
                                    x=nothing,
                                    u_domain=nothing,
                                    neumann_bc=nothing,
                                    kwargs...)

    @assert eltype(x) <: AbstractFloat 
    if isnothing(x) && !isnothing(u_domain)
        x = first(u_domain)
    elseif isnothing(x) && isnothing(u_domain)
        error("Need to provide whether `x` or `u_domain`")
    end

    eltype(g(x)) == eltype(x) ? nothing : error("Type of `g(x)` must match type of x")
    eltype(f(x, x, g(x), g(x), 0f0, 0f0, prob.p, tspan[1])) == eltype(x) ? nothing : error("Type of non linear function `f(x)` must type of x")

    PIDEProblem{typeof(g(x)),
                NLFunction,
                NLFunction,
                typeof(μ),
                typeof(σ),
                typeof(x),
                typeof(tspan),
                typeof(p),typeof(u_domain),typeof(neumann_bc),typeof(kwargs)}(
                g(x), NLFunction(g), NLFunction(f), μ, σ, x, tspan, p, u_domain, neumann_bc, kwargs)
    end

    Base.summary(prob::PIDEProblem) = string(nameof(typeof(prob)))

    function Base.show(io::IO, A::PIDEProblem)
    println(io, summary(A))
    print(io, "timespan: ")
    show(io, A.tspan)
    end

    include("MCSample.jl")
    include("reflect.jl")
    include("DeepSplitting.jl")
    include("MLP.jl")

    export PIDEProblem, DeepSplitting, MLP

    export NormalSampling, UniformSampling, NoSampling, solve
end