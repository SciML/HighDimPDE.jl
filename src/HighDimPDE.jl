"""
$(DocStringExtensions.README)
"""
module HighDimPDE
    using Reexport
    using DocStringExtensions
    @reexport using DiffEqBase
    using Statistics
    using Flux, Zygote, LinearAlgebra
    using Functors
    # using ProgressMeter: @showprogress
    using CUDA
    using Random
    using SparseArrays

    abstract type HighDimPDEAlgorithm <: DiffEqBase.AbstractODEAlgorithm end

    """
        PIDEProblem(g, f, μ, σ, x, tspan, p = nothing, x0_sample=nothing, neumann_bc=nothing)

    Defines a Partial Integro Differential Problem, of the form
    ```math
    \\begin{aligned}
        \\frac{du}{dt} &= \\tfrac{1}{2} \\text{Tr}(\\sigma \\sigma^T) \\Delta u(x, t) + \\mu \\nabla u(x, t) \\\\
        &\\quad + \\int f(x, y, u(x, t), u(y, t), ( \\nabla_x u )(x, t), ( \\nabla_x u )(y, t), p, t) dy,
    \\end{aligned}
    ```
    with `` u(x,0) = g(x)``.

    ## Arguments

    * `g` : initial condition, of the form `g(x, p, t)`.
    * `f` : nonlinear function, of the form `f(x, y, u(x, t), u(y, t), ∇u(x, t), ∇u(x, t), p, t)`.
    * `μ` : drift function, of the form `μ(x, p, t)`.
    * `σ` : diffusion function `σ(x, p, t)`.
    * `x`: point where `u(x,t)` is approximated. Is required even in the case where `x0_sample` is provided.
    * `tspan`: timespan of the problem.
    * `p`: the parameter vector.
    * `x0_sample` : sampling method for `x0`. Can be `UniformSampling(a,b)`, `NormalSampling(σ_sampling, shifted)`, or `NoSampling` (by default). If `NoSampling`, only solution at the single point `x` is evaluated.
    * `neumann_bc`: if provided, Neumann boundary conditions on the hypercube `neumann_bc[1] × neumann_bc[2]`.
    """
    struct PIDEProblem{uType,G,F,Mu,Sigma,xType,tType,P,UD,NBC,K} <: DiffEqBase.AbstractODEProblem{uType,tType,false}
        u0::uType
        g::G # initial condition
        f::F # nonlinear part
        μ::Mu
        σ::Sigma
        x::xType
        tspan::Tuple{tType,tType}
        p::P
        x0_sample::UD # for DeepSplitting only
        neumann_bc::NBC # neumann boundary conditions
        kwargs::K
    end

    function PIDEProblem(g, f, μ, σ, x::Vector{X}, tspan;
                                    p=nothing,
                                    x0_sample=NoSampling(),
                                    neumann_bc::NBC=nothing,
                                    kwargs...) where {X <: AbstractFloat, NBC <: Union{Nothing, AbstractVector}}

    @assert eltype(tspan) <: AbstractFloat "`tspan` should be a tuple of Float"

    isnothing(neumann_bc) ? nothing : @assert eltype(eltype(neumann_bc)) <: eltype(x)
    @assert eltype(g(x)) == eltype(x) "Type of `g(x)` must match type of x"
    @assert(eltype(f(x, x, g(x), g(x), x, x, p, tspan[1])) == eltype(x),
        "Type of non linear function `f(x)` must type of x")

    PIDEProblem{typeof(g(x)),
                typeof(g),
                typeof(f),
                typeof(μ),
                typeof(σ),
                typeof(x),
                eltype(tspan),
                typeof(p),
                typeof(x0_sample),
                typeof(neumann_bc),
                typeof(kwargs)}(
                g(x), g, f, μ, σ, x, tspan, p, x0_sample, neumann_bc, kwargs)
    end

    Base.summary(prob::PIDEProblem) = string(nameof(typeof(prob)))

    function Base.show(io::IO, A::PIDEProblem)
        println(io, summary(A))
        print(io, "timespan: ")
        show(io, A.tspan)
    end

    struct PIDESolution{X0,Ts,L,Us,NNs}
        x0::X0
        ts::Ts
        losses::L
        us::Us # array of solution evaluated at x0, ts[i]
        ufuns::NNs # array of parametric functions
    end

    Base.summary(prob::PIDESolution) = string(nameof(typeof(prob)))

    function Base.show(io::IO, A::PIDESolution)
        println(io, summary(A))
        print(io, "timespan: ")
        show(io, A.ts)
        print(io, "\nu(x,t): ")
        show(io, A.us)
    end

    include("MCSample.jl")
    include("reflect.jl")
    include("DeepSplitting.jl")
    include("MLP.jl")

    export PIDEProblem, PIDESolution, DeepSplitting, MLP

    export NormalSampling, UniformSampling, NoSampling, solve
end
