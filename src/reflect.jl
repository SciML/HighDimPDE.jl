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

# """
#     _reflect(a,b,s,e)
# reflection of the vector (b-a) from a on the cube [s,e]^2
# """
# function _reflect_GPU(a,b,s,e,batch_size,_device)
#     n = zeros(size(a)) |> _device 
#     # s = fill(s,batch_size) |> _device
#     # e = fill(e,batch_size) |> _device 
#     # first checking if b is in the hypercube
#     # if it is not, then r becomes less than one
#     r = fill(2.,batch_size) |> _device
#     rtemp = similar(r)
#     rtemparg = zeros(batch_size) |> _device 
#     #TODO: change "for i in 1:length(a)" to "for i in 1:size(a,1)"
#     # right now the scheme is not efficient, as it proceeds one reflection for one batch at a time
#     for i in 1:size(a,1)
#         bi = @view b[i,:]
#         ai = @view a[i,:]
#         ni = @view n[i,:]
#         if prod(bi .< s)
#             @. rtemp = (ai - s) / (ai - bi)
#             if prod(rtemp .< r)
#                 r = rtemp
#                 n .= 0
#                 n[i] = -1
#             end
#         elseif  prod(bi .> e)
#             rtemp =  (e - a[i]) / (b[i]- a[i])
#             if rtemp < r
#                 r = rtemp
#                 n .= 0
#                 n[i] = 1
#             end
#         end
#     end
#     while r < 1
#         c = a + r * ( b - a )
#         # dat = hcat(a,c)
#         # Plots.plot3d!(dat[1,:],dat[2,:],dat[3,:],label = "",color="blue")
#         a = c
#         b = b - 2 * n * ( dot(b-c,n))
#         r = 2;
#         for i in 1:length(a)
#             if b[i] < s
#                 rtemp = (a[i] - s) / (a[i] - b[i])
#                 if rtemp < r
#                     r = rtemp
#                     n .= 0
#                     n[i] = -1
#                 end
#             elseif  b[i] > e
#                 rtemp =  (e - a[i]) / (b[i]- a[i])
#                 if rtemp < r
#                     r = rtemp
#                     n .= 0
#                     n[i] = 1
#                 end
#             end
#         end
#     end
#     # dat = hcat(a,b)
#     # Plots.plot3d!(dat[1,:],dat[2,:],dat[3,:],label = "",color="blue")
#     return b
# end

using SparseArrays
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
                        n # sparse matrix that is used to store reflection side
                        )
    T = eltype(a)
    prod((a .>= s) .* (a .<= e)) ? nothing : error("a not in hypercube")
    out1 = b .< s 
    out2 = b .> e 
    while sum(out1 .+ out2) > 0
        rtemp1 = @. (a - s) / (a - b) #left 
        rtemp2 = @. (e - a) / (b - a) #right
        rtemp = ones(T,size(a))
        rtemp[out1] .= rtemp1[out1]
        rtemp[out2] .= rtemp2[out2]
        imin = argmin.(eachcol(rtemp))
        rmin = minimum(rtemp,dims=1)
        n .= sparse(imin,1:batch_size,one(T),d,batch_size)
        c = @. (a + (b-a) * rmin)
        b = @.( b - 2 * n * (b-c) )
        a = c
        out1 = b .< s 
        out2 = b .> e 
    end
    return b
end

