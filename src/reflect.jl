"""
    _reflect(a,b,s,e)
reflection of the vector (b-a) from a on the cube [s,e]^2
"""
function _reflect(a,b,s::Real,e::Real)
    r = 2; n = zeros(size(a))
    # first checking if b is in the hypercube
    all((a .>= s) .& (a .<= e)) ? nothing : error("a = $a not in hypercube")
    size(a) == size(b) ? nothing : error("a not same dim as b")
    #TODO: change "for i in 1:length(a)" to "for i in 1:size(a,2)"
    # right now the scheme is not efficient, as it proceeds one reflection for one batch at a time
    for i in 1:length(a)
        if b[i] < s
            rtemp = (a[i] - s) / (a[i] - b[i])
            if rtemp < r
                r = rtemp
                n .= 0
                n[i] = -1
            end
        elseif  b[i] > e
            rtemp =  (e - a[i]) / (b[i]- a[i])
            if rtemp < r
                r = rtemp
                n .= 0
                n[i] = 1
            end
        end
    end
    while r < 1
        c = a + r * ( b - a )
        # dat = hcat(a,c)
        # Plots.plot3d!(dat[1,:],dat[2,:],dat[3,:],label = "",color="blue")
        a = c
        b = b - 2 * n * ( dot(b-c,n))
        r = 2;
        for i in 1:length(a)
            if b[i] < s
                rtemp = (a[i] - s) / (a[i] - b[i])
                if rtemp < r
                    r = rtemp
                    n .= 0
                    n[i] = -1
                end
            elseif  b[i] > e
                rtemp =  (e - a[i]) / (b[i]- a[i])
                if rtemp < r
                    r = rtemp
                    n .= 0
                    n[i] = 1
                end
            end
        end
    end
    # dat = hcat(a,b)
    # Plots.plot3d!(dat[1,:],dat[2,:],dat[3,:],label = "",color="blue")
    return b
end


"""
    _reflect_GPU(a,b,s,e)
reflection of the Brownian motion `B` where `B_{t-1} = a` and  `B_{t} = b` 
on the hypercube `[s,e]^d` where `d = size(a,1)`

"""
function _reflect_GPU(a, b, s::Real, e::Real)
    all((a .>= s) .& (a .<= e)) ? nothing : error("a = $a not in hypercube")
    size(a) == size(b) ? nothing : error("a not same dim as b")
    out1 = b .< s
    out2 = b .> e
    out = out1 .| out2
    n = similar(a)
    n .= 0
    # Allocating
    while any(out)
        rtemp1 = @. (s - a) #left
        rtemp2 = @. (e - a) #right
        div = @. (out * (b-a) + !out)
        rtemp = (rtemp1 .* out1 .+ rtemp2 .* out2) ./ div .+ (.!(out1 .| out2))
        rmin = minimum(rtemp,dims=1)
        n .= rtemp .== minimum(rtemp;dims=1)
        c = @. (a + (b-a) * rmin)
        b = @. ( b - 2 * n * (b-c) )
        a = c
        @. out1 = b < s
        @. out2 = b > e
        @. out = out1 | out2
    end
    return b
end

function _out_indices(b, s::R, e::R) where {R<:Real}
    out1 = findall(b .< s)
    out2 = findall(b .> e)
    out = vcat(out1, out2)
    return out1, out2, out
end


"""
    _rtemp_lower!(rtemp, a, b, s, out_lower)
Update slice of `rtemp` corresponding to the lower boundary `s`.
"""
function _rtemp_lower!(rtemp, a, b, s, out_lower)
    for i in 1:length(out_lower)
        idx = out_lower[i]
        rtemp[i] = @. (s - a[idx]) / (b[idx] - a[idx])
    end
end

"""
    _rtemp_lower!(rtemp, a, b, e, out_upper, offset)
Update slice of `rtemp` corresponding to the upper boundary `e`.
NOTE: An offset needs to be provided, corresponding to the number of dimensions where `b` is below the lower boundary.
(SEE `rtemp_lower!`).
"""
function _rtemp_upper!(rtemp, a, b, e, out_upper, offset)
    for i in 1:length(out_upper)
        idx = out_upper[i]
        rtemp[i + offset] = @. (e - a[idx]) / (b[idx] - a[idx])
    end
end


"""
    _discard_reflected_out_index!(out, out1, out2, rmin_idx)
Checks if `b[n_idx]` (which corresponds to `out[rmin_idx]`) is within the specified boundary `[s, e]` after reflection along dimension `n_idx`.
The corresponding `out`-indices are removed before iterating the reflection.
"""
function _discard_reflected_out_index!(out, out1, out2, rmin_idx)
    if rmin_idx <= offset
        deleteat!(out1, rmin_idx)
        deleteat!(out, rmin_idx)
    else
        deleteat!(out2, rmin_idx - length(out1))
        deleteat!(out, rmin_idx)
    end
end


function _swap_boundary_outs!(out1, out2, n_idx, rmin_idx)
    offset = length(out1)
    # Should occur after reflecting OUT-OF-BOUNDS, where
    # dimension `n_idx` is assigned to other `out`-vector
    # for next reflection.
    if rmin_idx <= offset
        deleteat!(out1, rmin_idx)
        push!(out2, n_idx)
    else
        deleteat!(out2, rmin_idx - offset)
        push!(out1, n_idx)
    end
end

function _reflect_outs(a, b, s::Real, e::Real)
    all((a .>= s) .& (a .<= e)) ? nothing : error("a = $a not in hypercube")
    size(a) == size(b) ? nothing : error("a not same dim as b")

    # Initialize vectors of indices corresponding to dim of trajectories out of bounds
    out1, out2, out = _out_indices(b, s, e)
    rtemp = Vector{Float32}(undef, length(out))
    while length(out) > 0
        # TODO: Preallocate rtemp once, then remove elements with deleteat!(rtemp, rmin_idx)
        _rtemp_lower!(rtemp, a, b, s, out1)
        _rtemp_upper!(rtemp, a, b, e, out2, length(out1))
        
        rmin, rmin_idx = findmin(rtemp)
        n_idx = out[rmin_idx]
        @. a += (b - a) * rmin
        b[n_idx] = -b[n_idx] + 2 * a[n_idx]
        if s < b[n_idx] && b[n_idx] < e
            # Within boundary after reflection, can reduce number of reflecting dimensions.
            _discard_reflected_out_index!(out, out1, out2, rmin_idx)
            deleteat!(rtemp, rmin_idx)
        else
            # Reflected outside boundary, need to move element between out1 and out2
            # TODO: Reassign out-indices manually via _swap_boundary_outs!, for now, just recompute out-indices.
            # _swap_boundary_outs!(out1, out2, n_idx, rmin_idx)
            out1, out2, out = _out_indices(b, s, e)
        end
    end
    return b
end