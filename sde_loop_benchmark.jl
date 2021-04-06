using Flux, CUDA

d = 10
N = 10
batch_size = 1000
use_cuda = true

if use_cuda && CUDA.functional()
    @info "Training on CUDA GPU"
    CUDA.allowscalar(false)
    device = Flux.gpu
else
    @info "Training on CPU"
    device = Flux.cpu
end

function sde_loop(d,batch_size)
    X0 = zeros(Float32,d)
    y0 = repeat(X0[:],1,batch_size)
    y1 = repeat(X0[:],1,batch_size)
    dt = 0.1
    dWall = sqrt(dt) * randn(d,batch_size,N)
    for i in 1:N
        dW = @view dWall[:,:,i]
        y0 .= y1
        y1 .= y0 .+ 0. * dt .+ 1. .* dW
    end
    return y0, y1
end


function sde_loop_CUDA(d,batch_size)
    X0 = CUDA.zeros(Float32,d)
    y0 = CUDA.repeat(X0[:],1,batch_size)
    y1 = CUDA.repeat(X0[:],1,batch_size)
    dt = 0.1
    dWall = sqrt(dt) * CUDA.randn(d,batch_size,N)
    for i in 1:N
        dW = @view dWall[:,:,i]
        y0 .= y1
        y1 .= y0 .+ 0. * dt .+ 1. .* dW
    end
    return y0, y1
end

using BenchmarkTools

@btime CUDA.randn(1000,1000)
@btime randn(1000,1000)

@btime sde_loop(d,batch_size)
@btime sde_loop_CUDA(d,batch_size)
