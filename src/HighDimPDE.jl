"""
$(DocStringExtensions.README)
"""
module HighDimPDE
import DocStringExtensions
using DocStringExtensions: SIGNATURES, TYPEDSIGNATURES
using SciMLBase: AbstractODEAlgorithm, AbstractODEProblem, AbstractSciMLProblem,
    EnsembleProblem, EnsembleSerial, EnsembleThreads, SDEProblem
import SciMLBase: remake, solve
import SciMLSensitivity
using StochasticDiffEq: EM
using Statistics: mean
import Flux
using Flux: cpu, gpu
using LinearAlgebra: dot
using Functors: @functor
import Tracker
import CUDA
using Random: rand!, randn!

abstract type HighDimPDEAlgorithm <: AbstractODEAlgorithm end
abstract type AbstractPDEProblem <: AbstractSciMLProblem end

Base.summary(prob::AbstractPDEProblem) = string(nameof(typeof(prob)))

function Base.show(io::IO, A::AbstractPDEProblem)
    println(io, summary(A))
    print(io, "timespan: ")
    return show(io, A.tspan)
end

include("MCSample.jl")

"""
    PIDEProblem

Problem definition for a partial integro-differential equation solved by HighDimPDE.

# Fields
- `u0`: initial PDE state evaluated at `x`.
- `g`: terminal or initial condition function.
- `f`: nonlinear local or nonlocal PDE contribution.
- `μ`: drift function.
- `σ`: diffusion function.
- `x`: PDE evaluation point.
- `tspan`: start and end times.
- `p`: user-supplied model parameters.
- `x0_sample`: strategy used to sample initial states.
- `neumann_bc`: optional lower and upper Neumann boundary vectors.
- `kwargs`: additional solver-specific problem metadata.
"""
struct PIDEProblem{uType, G, F, Mu, Sigma, xType, tType, P, UD, NBC, K} <:
    AbstractODEProblem{uType, tType, false}
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
    PIDEProblem(μ, σ, x0, tspan, g, f; p = nothing, x0_sample = NoSampling(), neumann_bc = nothing, kwargs...)

Define a partial integro-differential equation problem of the form
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
* `neumann_bc`: Neumann boundary conditions on the hypercube `neumann_bc[1] × neumann_bc[2]`.

# Examples
```julia
using LinearAlgebra: I

μ(x, p, t) = zero(x)
σ(x, p, t) = I
g(x) = sum(abs2, x)
f(x, y, ux, uy, dux, duy, p, t) = zero(x)
prob = PIDEProblem(μ, σ, zeros(2), (0.0, 1.0), g, f)
```
"""
function PIDEProblem(
        μ,
        σ,
        x0::Union{Nothing, AbstractArray},
        tspan::TF,
        g,
        f;
        p::Union{Nothing, AbstractVector} = nothing,
        x0_sample::Union{Nothing, AbstractSampling} = NoSampling(),
        neumann_bc::Union{Nothing, AbstractVector} = nothing,
        kw...
    ) where {TF <: Tuple{AbstractFloat, AbstractFloat}}
    isnothing(neumann_bc) ? nothing : @assert eltype(eltype(neumann_bc)) <: eltype(x0)

    @assert(
        eltype(f(x0, x0, g(x0), g(x0), x0, x0, p, tspan[1])) == eltype(x0),
        "Type returned by non linear function `f` must match the type of `x0`"
    )

    @assert eltype(g(x0)) == eltype(x0) "Type of `g(x)` must match the Type of x"

    return PIDEProblem{
        typeof(g(x0)),
        typeof(g),
        typeof(f),
        typeof(μ),
        typeof(σ),
        typeof(x0),
        eltype(tspan),
        typeof(p),
        typeof(x0_sample),
        typeof(neumann_bc),
        typeof(kw),
    }(
        g(x0),
        g,
        f,
        μ,
        σ,
        x0,
        tspan,
        p,
        x0_sample,
        neumann_bc,
        kw
    )
end

"""
    ParabolicPDEProblem

Problem definition for a semilinear parabolic PDE solved by HighDimPDE.

# Fields
- `u0`: initial PDE state evaluated at `x`.
- `g`: terminal or initial condition function, or `nothing` for payoff problems.
- `f`: nonlinear PDE contribution, or `nothing` for linear equations.
- `μ`: drift function.
- `σ`: diffusion function.
- `x`: PDE evaluation point.
- `tspan`: start and end times.
- `p`: user-supplied model parameters.
- `x0_sample`: strategy used to sample initial states.
- `neumann_bc`: optional lower and upper Neumann boundary vectors.
- `kwargs`: additional problem metadata such as `xspan`, `payoff`, and parameter domains.
"""
struct ParabolicPDEProblem{uType, G, F, Mu, Sigma, xType, tType, P, UD, NBC, K} <:
    AbstractODEProblem{uType, tType, false}
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
    ParabolicPDEProblem(μ, σ, x0, tspan; g = nothing, f = nothing, p = nothing, kwargs...)

Define a semilinear parabolic PDE problem of the form
```math
\\begin{aligned}
    \\frac{du}{dt} &= \\tfrac{1}{2} \\text{Tr}(\\sigma \\sigma^T) \\Delta u(x, t) + \\mu \\nabla u(x, t) \\\\
    &\\quad +  f(x, u(x, t), ( \\nabla_x u )(x, t), p, t)
\\end{aligned}
```

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

# Examples
```julia
using LinearAlgebra: I

μ(x, p, t) = zero(x)
σ(x, p, t) = I
g(x) = sum(abs2, x)
f(x, u, du, p, t) = zero(u)
prob = ParabolicPDEProblem(μ, σ, zeros(2), (0.0, 1.0); g, f)
```
"""
function ParabolicPDEProblem(
        μ,
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
        kw...
    ) where {TF <: Tuple{AbstractFloat, AbstractFloat}}

    # Check the Initial Condition Function returns correct types.
    isnothing(g) && @assert !isnothing(payoff) "Either of `g` or `payoff` must be provided."

    isnothing(neumann_bc) ? nothing : @assert eltype(eltype(neumann_bc)) <: eltype(x0)

    @assert !isnothing(x0)||!isnothing(xspan) "Either of `x0` or `xspan` must be provided."

    !isnothing(f) && @assert(
        eltype(f(x0, eltype(x0)(0.0), x0, p, tspan[1])) == eltype(x0),
        "Type of non linear function `f(x)` must type of x"
    )

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
    @assert eltype(u0) == eltype(x0) "Type of `g(x)` must match the Type of x"

    return ParabolicPDEProblem{
        typeof(u0),
        typeof(g),
        typeof(f),
        typeof(μ),
        typeof(σ),
        typeof(x0),
        eltype(tspan),
        typeof(p),
        typeof(x0_sample),
        typeof(neumann_bc),
        typeof(kwargs),
    }(
        u0,
        g,
        f,
        μ,
        σ,
        x0,
        tspan,
        p,
        x0_sample,
        neumann_bc,
        kwargs
    )
end

function remake(
        prob::PIDEProblem;
        u0 = missing,
        g = missing,
        f = missing,
        μ = missing,
        σ = missing,
        x = missing,
        tspan = missing,
        p = missing,
        x0_sample = missing,
        neumann_bc = missing,
        kwargs = missing,
        interpret_symbolicmap = true,
        use_defaults = false
    )
    return PIDEProblem(
        ismissing(u0) ? prob.u0 : u0,
        ismissing(g) ? prob.g : g,
        ismissing(f) ? prob.f : f,
        ismissing(μ) ? prob.μ : μ,
        ismissing(σ) ? prob.σ : σ,
        ismissing(x) ? prob.x : x,
        ismissing(tspan) ? prob.tspan : tspan,
        ismissing(p) ? prob.p : p,
        ismissing(x0_sample) ? prob.x0_sample : x0_sample,
        ismissing(neumann_bc) ? prob.neumann_bc : neumann_bc,
        ismissing(kwargs) ? prob.kwargs : kwargs,
    )
end

function remake(
        prob::ParabolicPDEProblem;
        u0 = missing,
        g = missing,
        f = missing,
        μ = missing,
        σ = missing,
        x = missing,
        tspan = missing,
        p = missing,
        x0_sample = missing,
        neumann_bc = missing,
        kwargs = missing,
        interpret_symbolicmap = true,
        use_defaults = false
    )
    return ParabolicPDEProblem(
        ismissing(u0) ? prob.u0 : u0,
        ismissing(g) ? prob.g : g,
        ismissing(f) ? prob.f : f,
        ismissing(μ) ? prob.μ : μ,
        ismissing(σ) ? prob.σ : σ,
        ismissing(x) ? prob.x : x,
        ismissing(tspan) ? prob.tspan : tspan,
        ismissing(p) ? prob.p : p,
        ismissing(x0_sample) ? prob.x0_sample : x0_sample,
        ismissing(neumann_bc) ? prob.neumann_bc : neumann_bc,
        ismissing(kwargs) ? prob.kwargs : kwargs,
    )
end

"""
    PIDESolution(x0, ts, losses, usols, ufuns[, limits])

Solution container storing the evaluated states, learned solution functions, training
losses, and optional domain limits produced by HighDimPDE solvers.

# Fields
- `x0`: points at which the solution is evaluated.
- `ts`: saved time points.
- `losses`: training-loss history.
- `us`: solution values evaluated at `x0`.
- `ufuns`: learned solution functions or neural networks.
- `limits`: optional lower and upper bounds produced by limit computations.

# Examples
```julia
sol = PIDESolution(zeros(2), [0.0, 1.0], [1.0], [0.0, 1.0], nothing)
```
"""
struct PIDESolution{X0, Ts, L, Us, NNs, Ls}
    x0::X0
    ts::Ts
    losses::L
    us::Us # array of solution evaluated at x0, ts[i]
    ufuns::NNs # array of parametric functions
    limits::Ls
    function PIDESolution(x0, ts, losses, usols, ufuns, limits = nothing)
        return new{
            typeof(x0),
            typeof(ts),
            typeof(losses),
            typeof(usols),
            typeof(ufuns),
            typeof(limits),
        }(
            x0,
            ts,
            losses,
            usols,
            ufuns,
            limits
        )
    end
end

Base.summary(prob::PIDESolution) = string(nameof(typeof(prob)))

function Base.show(io::IO, A::PIDESolution)
    println(io, summary(A))
    print(io, "timespan: ")
    show(io, A.ts)
    print(io, "\nu(x,t): ")
    return show(io, A.us)
end

include("reflect.jl")
include("DeepSplitting.jl")
include("DeepBSDE.jl")
include("DeepBSDE_Han.jl")
include("MLP.jl")
include("NNStopping.jl")
include("NNKolmogorov.jl")
include("NNParamKolmogorov.jl")

export PIDEProblem, ParabolicPDEProblem, PIDESolution, DeepSplitting, DeepBSDE, MLP,
    NNStopping
export NNKolmogorov, NNParamKolmogorov
export NormalSampling, UniformSampling, NoSampling
end
