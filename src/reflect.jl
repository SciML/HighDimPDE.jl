"""
    _reflect(a,b,s,e)

reflection of the Brownian motion `B` where `B_{t-1} = a` and  `B_{t} = b`
on the hypercube `[s,e]^d` where `d = size(a,1)`.
"""
# Used by `MLP` algorithm.
function _reflect(a::T, b::T, s::T, e::T) where {T <: Vector}
    # Use copies to avoid mutating inputs
    a_curr = copy(a)
    b_curr = copy(b)
    _reflect!(a_curr, b_curr, s, e)
    return b_curr
end

"""
    _reflect!(a, b, s, e)

In-place version of `_reflect` that mutates both `a` and `b`.
`a` is used as working storage and `b` contains the result.
"""
function _reflect!(a::T, b::T, s::T, e::T) where {T <: Vector}
    elT = eltype(T)
    r = elT(2)
    n_idx = 0  # Track which index has the normal, 0 means no reflection needed
    n_val = zero(elT)  # The normal value at that index (+1 or -1)

    # first checking if a is in the hypercube
    @inbounds for i in eachindex(a)
        if a[i] < s[i] || a[i] > e[i]
            error("a = $a not in hypercube")
        end
    end
    length(a) == length(b) || error("a not same dim as b")

    @inbounds for i in eachindex(a)
        if b[i] < s[i]
            rtemp = (a[i] - s[i]) / (a[i] - b[i])
            if rtemp < r
                r = rtemp
                n_idx = i
                n_val = -one(elT)
            end
        elseif b[i] > e[i]
            rtemp = (e[i] - a[i]) / (b[i] - a[i])
            if rtemp < r
                r = rtemp
                n_idx = i
                n_val = one(elT)
            end
        end
    end

    while r < 1
        # c = a + r * (b - a), but we'll compute in-place
        # a = c
        @inbounds for i in eachindex(a)
            a[i] = a[i] + r * (b[i] - a[i])
        end

        # b = b - 2 * n * dot(b - c, n)
        # Since n is a unit vector with only one non-zero element at n_idx,
        # dot(b - c, n) = (b[n_idx] - c[n_idx]) * n_val = (b[n_idx] - a[n_idx]) * n_val
        # (after c is stored in a)
        dot_val = (b[n_idx] - a[n_idx]) * n_val
        b[n_idx] = b[n_idx] - 2 * n_val * dot_val

        r = elT(2)
        n_idx = 0
        n_val = zero(elT)

        @inbounds for i in eachindex(a)
            if b[i] < s[i]
                rtemp = (a[i] - s[i]) / (a[i] - b[i])
                if rtemp < r
                    r = rtemp
                    n_idx = i
                    n_val = -one(elT)
                end
            elseif b[i] > e[i]
                rtemp = (e[i] - a[i]) / (b[i] - a[i])
                if rtemp < r
                    r = rtemp
                    n_idx = i
                    n_val = one(elT)
                end
            end
        end
    end
    return b
end

# Used by `DeepSplitting` algorithm.
function _reflect(a::T, b::T, s, e) where {T <: AbstractArray}
    @assert all((a .>= s) .& (a .<= e)) "a = $a not in hypercube"
    @assert size(a) == size(b) "a not same dim as b"
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
