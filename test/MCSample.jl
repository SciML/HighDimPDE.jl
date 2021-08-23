using HighDimPDE, Random, Test, CUDA

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "GPU MCSampling" begin 
        # uniform sampling
        mc_sample = UniformSampling(-1f0,1f0)
        X = CUDA.zeros(Float32,10,8000)
        @test size(mc_sample(X)) == size(X)
        @test typeof(mc_sample(X)) == typeof(X)
        @test all( -1f0 .< mc_sample(X) .< 1f0 )

        # normal sampling
        mc_sample = NormalSampling(1f0,false)
        @test size(mc_sample(X)) == size(X)
        @test typeof(mc_sample(X)) == typeof(X)

        # normal sampling - true
        mc_sample = NormalSampling(1.,true)
        @test size(mc_sample(X)) == size(X)
        @test typeof(mc_sample(X)) == typeof(X)
    end
end

@testset "CPU MCSampling" begin 
    # uniform sampling
    mc_sample = UniformSampling(0.,1.)
    X = zeros(10,8000)
    @test size(mc_sample(X)) == size(X)
    @test typeof(mc_sample(X)) == typeof(X)
    @test all( 0f0 .< mc_sample(X) .< 1f0 )

    # normal sampling
    mc_sample = NormalSampling(1.,false)
    @test size(mc_sample(X)) == size(X)
    @test typeof(mc_sample(X)) == typeof(X)

    # normal sampling - true
    mc_sample = NormalSampling(1.,true)
    @test size(mc_sample(X)) == size(X)
    @test typeof(mc_sample(X)) == typeof(X)
end
