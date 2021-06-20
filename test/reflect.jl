### testing reflect method
using Test
using CUDA
using Random
# using Revise
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
        b_gpu = HighDimPDE._reflect_GPU(a,b,s,e)
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
            y1 .= HighDimPDE._reflect_GPU(y0,y1,-1f0,1f0)
        end
        @test count(isnan.(y1)) == 0
    end
end

# testing reflection on batchsize
if CUDA.functional()
    CUDA.allowscalar(false)

    @testset "GPU reflect methods" begin 
        d = 10
        X0 = fill(0.0f0,d)
        X1 = X0 + randn(d)
        X11 = HighDimPDE._reflect(X0,X1,-1,1)
        @testset "testing equivalence cpu gpu" begin
            batch_size = 1000
            a = repeat(X0[:],1,batch_size)
            b = a + 2 .* randn(size(a))
            a = hcat(X0,a) |> gpu
            b = hcat(X1,b) |> gpu
            s = -1f0; e = 1f0;
            b_gpu = HighDimPDE._reflect_GPU(a,b,s,e)
            b_cpu = b_gpu |> cpu
            @test prod(b_cpu .< 1) && prod(b_cpu .> -1)
            @test prod((b_cpu[:,1] .≈ X11[:,1]))
        end

        @testset "testing NaN reflect" begin
            # testing reflection on batchsize
            batch_size = 10000
            y0 = repeat(X0[:],1,batch_size) |> gpu
            y1 = repeat(X0[:],1,batch_size) |> gpu
            dWall = zeros(Float32, d, batch_size, 100 ) |> gpu
            randn!(dWall)
            for i in 1:100
                dW = @view dWall[:,:,i]
                y0 .= y1
                y1 .= y0 .+  dW
                y1 .= HighDimPDE._reflect_GPU(y0,y1,-1f0,1f0)
            end
            @test count(isnan.(y1)) == 0
        end
    end
end

@testset "CPU index reflect methods" begin
    d = 1000
    X0 = fill(0.0f0,d)
    X1 = X0 + randn(d)
    @testset "test equivalence of index with cpu/gpu" begin
        args = (X0, X1, -1, 1)
        @test HighDimPDE._reflect(copy.(args)...) ≈ HighDimPDE._reflect_GPU(copy.(args)...)
        @test HighDimPDE._reflect(copy.(args)...) ≈ HighDimPDE._reflect_outs(copy.(args)...)
        @test HighDimPDE._reflect_GPU(copy.(args)...) ≈ HighDimPDE._reflect_outs(copy.(args)...)
    end
    
end