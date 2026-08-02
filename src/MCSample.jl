abstract type AbstractSampling{T} end
Base.eltype(::AbstractSampling{T}) where {T} = eltype(T)

# Monte Carlo AbstractSampling
abstract type MCSampling{T} <: AbstractSampling{T} end

"""
    UniformSampling(a, b)

Uniform sampling for the Monte Carlo integration, in the hypercube `[a, b]^2`.

# Fields
- `a`: lower bounds of the sampling hypercube.
- `b`: upper bounds of the sampling hypercube.

# Examples
```julia
sampler = UniformSampling(fill(-1.0, 2), fill(1.0, 2))
samples = sampler(zeros(2))
```
"""
struct UniformSampling{A} <: MCSampling{A}
    a::A
    b::A
end
@functor UniformSampling

function (mc_sample::UniformSampling{T})(x_mc, kwargs...) where {T}
    Tel = eltype(T)
    rand!(x_mc)
    m = (mc_sample.b + mc_sample.a) ./ convert(Tel, 2)
    return x_mc .= (x_mc .- convert(Tel, 0.5)) .* (mc_sample.b - mc_sample.a) .+ m
end

"""
    NormalSampling(σ)
    NormalSampling(σ, shifted)

Normal sampling method for the Monte Carlo integration.

# Fields
- `σ`: standard deviation applied to each normally distributed sample.
- `shifted::Bool`: whether sampling adds the evaluation point after drawing a sample.

# Arguments
* `σ`: the standard deviation of the sampling
* `shifted` : if true, the integration is shifted by `x`. Defaults to false.

# Examples
```julia
sampler = NormalSampling(0.1)
samples = sampler(zeros(2))
```
"""
struct NormalSampling{T} <: MCSampling{T}
    σ::T
    shifted::Bool # if true, we shift integration by x when invoking mc_sample::MCSampling(x)
end
@functor NormalSampling

NormalSampling(σ) = NormalSampling(σ, false)

function (mc_sample::NormalSampling)(x_mc)
    randn!(x_mc)
    return x_mc .*= mc_sample.σ
end

function (mc_sample::NormalSampling)(x_mc, x)
    mc_sample(x_mc)
    return mc_sample.shifted ? x_mc .+= x : nothing
end

"""
    NoSampling()

Sampling strategy that disables Monte Carlo integration and evaluates the PDE at the
provided state only.

# Examples
```julia
NoSampling()(zeros(2)) === nothing
```
"""
struct NoSampling <: AbstractSampling{Nothing} end

(mc_sample::NoSampling)(x...) = nothing

function _integrate(::MCS) where {MCS <: AbstractSampling}
    if MCS <: NoSampling
        return false
    else
        return true
    end
end
