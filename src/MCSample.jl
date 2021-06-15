"""
Sampling method for the Monte Carlo integration
"""
abstract type MCSampling{T} end
Base.eltype(mc_sample::MCSampling{T}) where T = T

"""
Uniform sampling method for the Monte Carlo integration, in the hypercube `[a, b]^2`.
"""
struct UniformSampling{T <: Real} <: MCSampling{T}
    a::T
    b::T
end

function (mc_sample::UniformSampling)(x)
    T = eltype(x)
    x_mc = similar(x)
    rand!(x_mc)
    m = (mc_sample.b + mc_sample.a)/2
    x_mc .= (x_mc .- convert(T,0.5)) * (mc_sample.b - mc_sample.a) .+ m
    return x_mc 
end

"""
Normal sampling method for the Monte Carlo integration.

Arguments:
* `σ`: the standard devation of the sampling
* `shifted` : if true, the integration is shifted by `x`
"""
struct NormalSampling{T<:Real} <: MCSampling{T}
    σ::T
    shifted::Bool # if true, we shift integration by x when invoking mc_sample::MCSampling(x)
end

NormalSampling(σ::Real) = NormalSampling(σ,false)

function (mc_sample::NormalSampling)(x)
    x_mc = similar(x)
    randn!(x_mc)
    x_mc .*=  mc_sample.σ  
    mc_sample.shifted ? x_mc .+= x : nothing
    return x_mc
end                                    

struct NoSampling <: MCSampling{Nothing} end

(mc_sample::NoSampling)(x) = nothing

function _integrate(::MCS) where {MCS <: MCSampling}
    if MCS <: NoSampling
        return false
    else
        return true
    end
end