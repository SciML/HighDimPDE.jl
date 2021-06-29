using CUDA
using Flux
using BenchmarkTools
using Random

using HighDimPDE
if CUDA.functional()
    CUDA.allowscalar(false)
    _device = Flux.gpu
    rgen! = CUDA.randn!
end
# testing reflection on batchsize
d = 100
batch_size = 10000
#
for d in [2^i for i in 1:17]
    Random.seed!(1)
    X0 = fill(0.0f0,d)
    X1 = X0 + randn(d)
    println("\n\nd = $d\n")
    if d <= 16384
        args = copy.((X0, X1, -1, 1))
        @time HighDimPDE._reflect(args...)
        args = copy.((X0, X1, -1, 1))
        @time HighDimPDE._reflect_GPU(args...)
    end
    args = copy.((X0, X1, -1, 1))
    @time HighDimPDE._reflect_outs(args...)
end
y0 = CUDA.zeros(d,batch_size)
y1 = CUDA.randn(size(y0)...)
@btime _reflect_GPU2($y0,$y1,-1f0,1f0,_device)

@btime CUDA.similar(y0)


function imin2array()
    y1 = CUDA.randn(1000,1000)
    imin = argmin(y1,dims=1) |> Array
    n = zeros(size(y1))
    n[imin] .= 1
    n = b |> gpu
end
@btime imin2array()
@btime CUDA.zeros(size(y1))


n = CUDA.zeros(size(y1))
CUDA.allowscalar(true)
function imin_scalar()
    y1 = CUDA.randn(1000,1000)
    imin = argmin(y1,dims=1)
    n .= 0.
    n[imin] .= 1
end
@btime imin_scalar()
