"""
    _reflect(a,b,s,e)
reflection of the vector (b-a) from a on the cube [s,e]^2
"""
function _reflect(a,b,s::Real,e::Real)
    r = 2; n = zeros(size(a))
    all((a .>= s) .& (a .<= e)) ? nothing : error("a = $a not in hypercube")
    size(a) == size(b) ? nothing : error("a not same dim as b")
    # if it is not, then r becomes less than one

    # first checking if b is in the hypercube
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
reflection of the Brownian motion B where B_{t-1} = a and  B_{t} = b 
on the hypercube [s,e]^d where d = size(a,1)

#Arguments
* `_device` : Flux.gpu or Flux.cpu
"""
function _reflect_GPU(a, b, s::Real, e::Real, _device)
    T = eltype(a)
    prod((a .>= s) .* (a .<= e)) ? nothing : error("a = $a not in hypercube")
    prod(size(a) .== size(b)) ? nothing : error("a not same dim as b")
    out1 = b .< s |> _device
    out2 = b .> e |> _device
    out = out1 .| out2
    n = zeros(size(a)) |> _device
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
