"""
$(DocStringExtensions.README)
"""
module HighDimPDE
using DocStringExtensions # for $(SIGNATURES)
using Reexport
using DocStringExtensions
@reexport using DiffEqBase
using SciMLSensitivity
using StochasticDiffEq
using Statistics
using Flux, Zygote, LinearAlgebra
using Functors
# using ProgressMeter: @showprogress
using Tracker
using CUDA, cuDNN
using Random
using SparseArrays

abstract type HighDimPDEAlgorithm <: DiffEqBase.AbstractODEAlgorithm end
abstract type AbstractPDEProblem <: SciMLBase.AbstractSciMLProblem end

Base.summary(prob::AbstractPDEProblem) = string(nameof(typeof(prob)))

function Base.show(io::IO, A::AbstractPDEProblem)
    println(io, summary(A))
    print(io, "timespan: ")
    show(io, A.tspan)
end

include("MCSample.jl")

struct PIDEProblem{uType, G, F, Mu, Sigma, xType, tType, P, UD, NBC, K} <:
       DiffEqBase.AbstractODEProblem{uType, tType, false}
    u0::uType
    g::G # initial condition
    f::F # nonlinear part
    μ::Mu
    σ::Sigma
    x::xType
    tspan::Tuple{tType, tType}
    p::P
    x0_sample::UD # the domain of u to be solved
    neumann_bc::NBC # neumann boundary conditions
    kwargs::K
end

"""
$(SIGNATURES)

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
* `f` : nonlinear function, of the form `f(x, y, u(x, t), u(y, t), ∇u(x, t), ∇u(y, t), p, t)`.
* `μ` : drift function, of the form `μ(x, p, t)`.
* `σ` : diffusion function `σ(x, p, t)`.
* `x`: point where `u(x,t)` is approximated. Is required even in the case where `x0_sample` is provided. Determines the dimensionality of the PDE.
* `tspan`: timespan of the problem.
* `p`: the parameter vector.
* `x0_sample` : sampling method for `x0`. Can be `UniformSampling(a,b)`, `NormalSampling(σ_sampling, shifted)`, or `NoSampling` (by default). If `NoSampling`, only solution at the single point `x` is evaluated.
* `neumann_bc`: if provided, Neumann boundary conditions on the hypercube `neumann_bc[1] × neumann_bc[2]`.
"""
function PIDEProblem(μ,
    σ,
    x0::Union{Nothing, AbstractArray},
    tspan::TF,
    g,
    f;
    p::Union{Nothing, AbstractVector} = nothing,
    x0_sample::Union{Nothing, AbstractSampling} = NoSampling(),
    neumann_bc::Union{Nothing, AbstractVector} = nothing,
    kw...) where {TF <: Tuple{AbstractFloat, AbstractFloat}}


    isnothing(neumann_bc) ? nothing : @assert eltype(eltype(neumann_bc)) <: eltype(x0)

    @assert(eltype(f(x0, x0, g(x0), g(x0), x0, x0, p, tspan[1]))==eltype(x0),
                "Type returned by non linear function `f` must match the type of `x0`")

    @assert eltype(g(x0))==eltype(x0) "Type of `g(x)` must match the Type of x"

    PIDEProblem{typeof(g(x0)),
    typeof(g),
    typeof(f),
    typeof(μ),
    typeof(σ),
    typeof(x0),
    eltype(tspan),
    typeof(p),
    typeof(x0_sample),
    typeof(neumann_bc),
    typeof(kw)}(g(x0),
    g,
    f,
    μ,
    σ,
    x0,
    tspan,
    p,
    x0_sample,
    neumann_bc,
    kw)

end

struct ParabolicPDEProblem{uType, G, F, Mu, Sigma, xType, tType, P, UD, NBC, K} <:
    DiffEqBase.AbstractODEProblem{uType, tType, false}
 u0::uType
 g::G # initial condition
 f::F # nonlinear part
 μ::Mu
 σ::Sigma
 x::xType
 tspan::Tuple{tType, tType}
 p::P
 x0_sample::UD # the domain of u to be solved
 neumann_bc::NBC # neumann boundary conditions
 kwargs::K
end

"""
$(SIGNATURES)

Defines a Parabolic Partial Differential Equation of the form:
- Semilinear Parabolic Partial Differential Equation 
    * f -> f(X, u, σᵀ∇u, p, t)
- Kolmogorov Differential Equation
    * f -> `nothing`
    * x0 -> nothing, xspan must be provided.
- Obstacle Partial Differential Equation 
    * f -> `nothing`
    * g -> `nothing`
    * discounted payoff function provided.

## Arguments

* `μ` : drift function, of the form `μ(x, p, t)`.
* `σ` : diffusion function `σ(x, p, t)`.
* `x`: point where `u(x,t)` is approximated. Is required even in the case where `x0_sample` is provided. Determines the dimensionality of the PDE.
* `tspan`: timespan of the problem.
* `g` : initial condition, of the form `g(x, p, t)`.
* `f` : nonlinear function, of the form  `f(X, u, σᵀ∇u, p, t)`

## Optional Arguments 
* `p`: the parameter vector.
* `x0_sample` : sampling method for `x0`. Can be `UniformSampling(a,b)`, `NormalSampling(σ_sampling, shifted)`, or `NoSampling` (by default). If `NoSampling`, only solution at the single point `x` is evaluated.
* `neumann_bc`: if provided, Neumann boundary conditions on the hypercube `neumann_bc[1] × neumann_bc[2]`.
* `xspan`: The domain of the independent variable `x`
* `payoff`: The discounted payoff function. Required when solving for optimal stopping problem (Obstacle PDEs).
"""
function ParabolicPDEProblem(μ,
        σ,
        x0::Union{Nothing, AbstractArray},
        tspan::TF;
        g = nothing,
        f = nothing,
        p::Union{Nothing, AbstractVector} = nothing,
        xspan::Union{Nothing, TF, AbstractVector{<:TF}} = nothing,
        x0_sample::Union{Nothing, AbstractSampling} = NoSampling(),
        neumann_bc::Union{Nothing, AbstractVector} = nothing,
        payoff = nothing,
        kw...) where {TF <: Tuple{AbstractFloat, AbstractFloat}}

    # Check the Initial Condition Function returns correct types.
    isnothing(g) && @assert !isnothing(payoff) "Either of `g` or `payoff` must be provided."

    isnothing(neumann_bc) ? nothing : @assert eltype(eltype(neumann_bc)) <: eltype(x0)

    @assert !isnothing(x0)||!isnothing(xspan) "Either of `x0` or `xspan` must be provided."

    !isnothing(f) && @assert(eltype(f(x0, eltype(x0)(0.0), x0, p, tspan[1]))==eltype(x0),
                    "Type of non linear function `f(x)` must type of x")

    # Wrap kwargs : 
    kw = NamedTuple(kw)
    prob_kw = (xspan = xspan, payoff = payoff)
    kwargs = merge(prob_kw, kw)

    # If xspan isa Tuple, then convert it as a Vector{Tuple} with single element
    xspan = isa(xspan, Tuple) ? [xspan] : xspan

    # if `x0` is not provided, pick up the lower-bound of `xspan`.
    x0 = isnothing(x0) ? first.(xspan) : x0

    # Initial Condition 
    u0 = if haskey(kw, :p_prototype)
        u0 = g(x0, kw.p_prototype.p_phi)
    else
        !isnothing(g) ? g(x0) : payoff(x0, 0.0)
    end
    @assert eltype(u0)==eltype(x0) "Type of `g(x)` must match the Type of x"

    ParabolicPDEProblem{typeof(u0),
        typeof(g),
        typeof(f),
        typeof(μ),
        typeof(σ),
        typeof(x0),
        eltype(tspan),
        typeof(p),
        typeof(x0_sample),
        typeof(neumann_bc),
        typeof(kwargs)}(u0,
        g,
        f,
        μ,
        σ,
        x0,
        tspan,
        p,
        x0_sample,
        neumann_bc,
        kwargs)
end

struct PIDESolution{X0, Ts, L, Us, NNs, Ls}
    x0::X0
    ts::Ts
    losses::L
    us::Us # array of solution evaluated at x0, ts[i]
    ufuns::NNs # array of parametric functions
    limits::Ls
end
function PIDESolution(x0, ts, losses, usols, ufuns, limits = nothing)
    PIDESolution{typeof(x0),
        typeof(ts),
        typeof(losses),
        typeof(usols),
        typeof(ufuns),
        typeof(limits)}(x0,
        ts,
        losses,
        usols,
        ufuns,
        limits)
end

Base.summary(prob::PIDESolution) = string(nameof(typeof(prob)))

function Base.show(io::IO, A::PIDESolution)
    println(io, summary(A))
    print(io, "timespan: ")
    show(io, A.ts)
    print(io, "\nu(x,t): ")
    show(io, A.us)
end

include("reflect.jl")
include("DeepSplitting.jl")
include("DeepBSDE.jl")
include("DeepBSDE_Han.jl")
include("MLP.jl")
include("NNStopping.jl")

export PIDEProblem, ParabolicPDEProblem, PIDESolution, DeepSplitting, DeepBSDE, MLP, NNStopping

export NormalSampling, UniformSampling, NoSampling, solve
end
