"""
Sampling method for the Monte Carlo integration
"""
abstract type MCSampling{T} end
Base.eltype(::MCSampling{T}) where T = eltype(T)

"""
Uniform sampling method for the Monte Carlo integration, in the hypercube `[a, b]^2`.
"""
struct UniformSampling{T} <: MCSampling{T} 
    a::T
    b::T
end
@functor UniformSampling


function (mc_sample::UniformSampling{T})(x_mc, kwargs...) where T
    Tel = eltype(T)
    rand!(x_mc)
    m = (mc_sample.b + mc_sample.a) ./ convert(Tel,2)
    x_mc .= (x_mc .- convert(Tel,0.5)) .* (mc_sample.b - mc_sample.a) .+ m
end


"""
Normal sampling method for the Monte Carlo integration.

Arguments:
* `σ`: the standard devation of the sampling
* `shifted` : if true, the integration is shifted by `x`
"""
struct NormalSampling{T} <: MCSampling{T}
    σ::T
    shifted::Bool # if true, we shift integration by x when invoking mc_sample::MCSampling(x)
end
@functor NormalSampling

NormalSampling(σ) = NormalSampling(σ,false)

function (mc_sample::NormalSampling)(x_mc)
    randn!(x_mc)
    x_mc .*=  mc_sample.σ  
end

function (mc_sample::NormalSampling)(x_mc, x)
    mc_sample(x_mc)
    mc_sample.shifted ? x_mc .+= x : nothing
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