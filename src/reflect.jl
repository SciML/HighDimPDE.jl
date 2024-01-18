"""
    _reflect(a,b,s,e)

reflection of the Brownian motion `B` where `B_{t-1} = a` and  `B_{t} = b` 
on the hypercube `[s,e]^d` where `d = size(a,1)`.
"""
# Used by `MLP` algorithm.
function _reflect(a::T, b::T, s::T, e::T) where {T <: Vector}
    r = 2
    n = zeros(size(a))
    # first checking if b is in the hypercube
    all((a .>= s) .& (a .<= e)) ? nothing : error("a = $a not in hypercube")
    size(a) == size(b) ? nothing : error("a not same dim as b")
    for i in 1:length(a)
        if b[i] < s[i]
            rtemp = (a[i] - s[i]) / (a[i] - b[i])
            if rtemp < r
                r = rtemp
                n .= 0
                n[i] = -1
            end
        elseif b[i] > e[i]
            rtemp = (e[i] - a[i]) / (b[i] - a[i])
            if rtemp < r
                r = rtemp
                n .= 0
                n[i] = 1
            end
        end
    end
    while r < 1
        c = a + r * (b - a)
        a = c
        b = b - 2 * n * (dot(b - c, n))
        r = 2
        for i in 1:length(a)
            if b[i] < s[i]
                rtemp = (a[i] - s[i]) / (a[i] - b[i])
                if rtemp < r
                    r = rtemp
                    n .= 0
                    n[i] = -1
                end
            elseif b[i] > e[i]
                rtemp = (e[i] - a[i]) / (b[i] - a[i])
                if rtemp < r
                    r = rtemp
                    n .= 0
                    n[i] = 1
                end
            end
        end
    end
    return b
end

# Used by `DeepSplitting` algorithm.
function _reflect(a::T, b::T, s, e) where {T <: AbstractArray}
    @assert all((a .>= s) .& (a .<= e)) "a = $a not in hypercube"
    @assert size(a)==size(b) "a not same dim as b"
    out1 = b .< s
    out2 = b .> e
    out = out1 .| out2
    n = similar(a)
    n .= 0
    # Allocating
    while any(out)
        rtemp1 = @. (s - a) #left
        rtemp2 = @. (e - a) #right
        div = @. (out * (b - a) + !out)
        rtemp = (rtemp1 .* out1 .+ rtemp2 .* out2) ./ div .+ (.!(out1 .| out2))
        rmin = minimum(rtemp, dims = 1)
        n .= rtemp .== minimum(rtemp; dims = 1)
        c = @. (a + (b - a) * rmin)
        b = @. (b - 2 * n * (b - c))
        a = c
        @. out1 = b < s
        @. out2 = b > e
        @. out = out1 | out2
    end
    return b
end
