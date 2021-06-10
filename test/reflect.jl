### testing reflect method
using Test
using CUDA
using Random
using Revise
using Flux
using HighDimPDE

@testset "CPU reflect methods" begin
    # testing reflection on batchsize 0
    d = 10
    X0 = fill(0.0f0,d)
    X1 = X0 + randn(d)
    X11 = HighDimPDE._reflect(X0,X1,-1,1)
    @test prod(X11 .< 1) && prod(X11 .> -1)

    @testset "testing equivalence _reflect methods" begin
        batch_size = 1000
        a = repeat(X0[:],1,batch_size)
        b = a + 2 .* randn(size(a))
        a = hcat(X0,a)
        b = hcat(X1,b)
        s = -1f0; e = 1f0;
        b_gpu = HighDimPDE._reflect_GPU(a,b,s,e,cpu)
        @test prod(b_gpu .< 1) && prod(b_gpu .> -1)
        @test prod((b_gpu[:,1] .≈ X11[:,1]))
    end

    @testset "testing NaN reflect" begin
        # testing reflection on batchsize
        batch_size = 100
        y0 = repeat(X0[:],1,batch_size) 
        y1 = repeat(X0[:],1,batch_size)
        dWall = zeros(Float32, d, batch_size, 1000 )
        randn!(dWall) .* 10
        for i in 1:1000
            dW = @view dWall[:,:,i]
            y0 .= y1
            y1 .= y0 .+  dW
            y1 .= HighDimPDE._reflect_GPU(y0,y1,-1f0,1f0,cpu)
        end
        @test count(isnan.(y1)) == 0
    end
end

# testing reflection on batchsize
if CUDA.functional()
    @testset "GPU reflect methods" begin 

        CUDA.allowscalar(false)
        _device = Flux.gpu
        @testset "testing equivalence cpu gpu" begin
            batch_size = 1000
            a = repeat(X0[:],1,batch_size)
            b = a + 2 .* randn(size(a))
            a = hcat(X0,a) |> _device
            b = hcat(X1,b) |> _device
            s = -1f0; e = 1f0;
            b_gpu = HighDimPDE._reflect_GPU(a,b,s,e,_device)
            b_cpu = b_gpu |> cpu
            @test prod(b_cpu .< 1) && prod(b_cpu .> -1)
            @test prod((b_cpu[:,1] .≈ X11[:,1]))
        end

        @testset "testing NaN reflect" begin
            # testing reflection on batchsize
            batch_size = 10000
            y0 = repeat(X0[:],1,batch_size) |> _device
            y1 = repeat(X0[:],1,batch_size) |> _device
            dWall = zeros(Float32, d, batch_size, 100 ) |> _device
            randn!(dWall)
            for i in 1:100
                dW = @view dWall[:,:,i]
                y0 .= y1
                y1 .= y0 .+  dW
                y1 .= HighDimPDE._reflect_GPU(y0,y1,-1f0,1f0,_device)
            end
            @test count(isnan.(y1)) == 0
        end
    end
end
