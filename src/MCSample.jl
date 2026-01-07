abstract type AbstractSampling{T} end
Base.eltype(::AbstractSampling{T}) where {T} = eltype(T)

# Monte Carlo AbstractSampling
abstract type MCSampling{T} <: AbstractSampling{T} end

"""
    UniformSampling(a, b)

Uniform sampling for the Monte Carlo integration, in the hypercube `[a, b]^2`.
"""
struct UniformSampling{A} <: MCSampling{A}
    a::A
    b::A
end
@functor UniformSampling

function (mc_sample::UniformSampling{T})(x_mc, kwargs...) where {T}
    Tel = eltype(T)
    a = mc_sample.a
    b = mc_sample.b
    rand!(x_mc)
    # Fuse all operations to avoid temporaries: x_mc = a + (b - a) * x_mc
    # This is equivalent to uniform sampling in [a, b]
    @inbounds for i in eachindex(x_mc)
        x_mc[i] = a[i] + (b[i] - a[i]) * x_mc[i]
    end
    return x_mc
end

"""
    NormalSampling(σ)
    NormalSampling(σ, shifted)

Normal sampling method for the Monte Carlo integration.

# Arguments
* `σ`: the standard deviation of the sampling
* `shifted` : if true, the integration is shifted by `x`. Defaults to false.
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

struct NoSampling <: AbstractSampling{Nothing} end

(mc_sample::NoSampling)(x...) = nothing

function _integrate(::MCS) where {MCS <: AbstractSampling}
    if MCS <: NoSampling
        return false
    else
        return true
    end
end
