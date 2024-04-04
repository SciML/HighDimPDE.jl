using HighDimPDE, Random, Test, CUDA

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "GPU MCSampling" begin
        # uniform sampling
        mc_sample = UniformSampling(-1.0f0, 1.0f0)
        X = CUDA.zeros(Float32, 10, 8000)
        @test size(mc_sample(X)) == size(X)
        @test typeof(mc_sample(X)) == typeof(X)
        @test all(-1.0f0 .< mc_sample(X) .< 1.0f0)

        # normal sampling
        mc_sample = NormalSampling(1.0f0, false)
        @test size(mc_sample(X)) == size(X)
        @test typeof(mc_sample(X)) == typeof(X)

        # normal sampling - true
        mc_sample = NormalSampling(1.0, true)
        @test size(mc_sample(X)) == size(X)
        @test typeof(mc_sample(X)) == typeof(X)
    end
end

@testset "CPU MCSampling" begin
    # uniform sampling
    mc_sample = UniformSampling(0.0, 1.0)
    X = zeros(10, 8000)
    @test size(mc_sample(X)) == size(X)
    @test typeof(mc_sample(X)) == typeof(X)
    @test all(0.0f0 .< mc_sample(X) .< 1.0f0)

    # normal sampling
    mc_sample = NormalSampling(1.0, false)
    @test size(mc_sample(X)) == size(X)
    @test typeof(mc_sample(X)) == typeof(X)

    # normal sampling - true
    mc_sample = NormalSampling(1.0, true)
    @test size(mc_sample(X)) == size(X)
    @test typeof(mc_sample(X)) == typeof(X)
end
