# reflection of b on the segment [0,1]
function R_0_1(b::T) where T <: AbstractFloat
    fb = floor(b)
    b = mod(fb, T(2)) == zero(T) ? b - fb : one(T) - (b - fb)
    return b
end

# reflection of b on the segment [s,e]
function __reflect(b::T, s::T, e::T) where {T <: AbstractFloat}
    return (e-s) * R_0_1((b-s)/(e-s)) + s
end

"""
    _reflect(a,b,s,e)

reflection of the Brownian motion `B` where `B_{t-1} = a` and  `B_{t} = b` 
on the hypercube `[s,e]^d` where `d = size(a,1)`.
"""
function _reflect(a, b, s, e)
    @assert all((a .>= s) .& (a .<= e)) "a = $a not in hypercube"
    @assert size(a) == size(b) "a not same dim as b"
    __reflect.(b, s, e)
end