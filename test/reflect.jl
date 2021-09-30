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
    X0 = fill(0.0,d)
    X1 = X0 + randn(d)
    X11 = HighDimPDE._reflect(X0, X1, fill(-1.,d), fill(1.,d))
    @test all((X11 .< 1) .& (X11 .> -1))

    @testset "testing NaN reflect" begin
        # testing reflection on batchsize
        y0 = X0[:]
        y1 = X0[:]
        dWall = zeros(Float32, d, 1000 )
        randn!(dWall) .* 10
        for i in 1:1000
            dW = @view dWall[:,i]
            y0 .= y1
            y1 .= y0 .+  dW
            y1 .= HighDimPDE._reflect(y0, y1, fill(-1.,d), fill(1.,d))
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
        X1 = X0 + randn(Float32,d)
        X11 = HighDimPDE._reflect(X0, X1, fill(-1f0,d), fill(1f0,d))
        @testset "testing equivalence cpu gpu" begin
            batch_size = 1000
            a = repeat(X0[:],1,batch_size)
            b = a + 2 .* randn(size(a))
            a = hcat(X0,a) |> gpu
            b = hcat(X1,b) |> gpu
            s = -1f0; e = 1f0;
            b_gpu = HighDimPDE._reflect(a,b,s,e)
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
                y1 .= HighDimPDE._reflect(y0,y1,-1f0,1f0)
            end
            @test count(isnan.(y1)) == 0
        end
    end
end
