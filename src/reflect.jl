using SparseArrays

"""
    _reflect(a,b,s,e)
reflection of the vector (b-a) from a on the cube [s,e]^2
"""
function _reflect(a,b,s,e)
    r = 2; n = zeros(size(a))
    # first checking if b is in the hypercube
    # if it is not, then r becomes less than one

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
    _reflect(a,b,s,e)
reflection of the vector (b-a) from a on the cube [s,e]^d
"""
function _reflect_GPU2(a, #first point
                        b, # second point
                        s, # [s,e]^d
                        e, # [s,e]^d
                        d, # [s,e]^d
                        batch_size,
                        _device
                        )
    T = eltype(a)
    prod((a .>= s) .* (a .<= e)) ? nothing : error("a not in hypercube")
    out1 = b .< s |> _device
    out2 = b .> e |> _device
    # Allocating
    rtemp_ones = ones(T,size(a)) |> _device
    rtemp1 = similar(a)
    rtemp2 = similar(a)
    rtemp = similar(a)
    rmin = minimum(rtemp,dims=1)
    imin = argmin(rtemp,dims=1)
    n = similar(a)
    c = similar(a)
    while sum(out1 .+ out2) > 0
        @. rtemp1 = (a - s) / (a - b) #left
        @. rtemp2 = (e - a) / (b - a) #right
        rtemp .= rtemp1 .* out1 .+ rtemp2 .* out2 .+ rtemp_ones .*(.!(out1 .| out2))
        imin .= argmin(rtemp,dims=1)
        rmin .= minimum(rtemp,dims=1)
        n .= 0.
        n[imin] = rtemp_ones
        @. c =  (a + (b-a) * rmin)
        @. b = ( b - 2 * n * (b-c) )
        @. a = c
        @. out1 = b < s
        @. out2 = b > e
    end
    return b
end
