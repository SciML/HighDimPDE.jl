using HighDimPDE
using Random
using Test
using Flux

# using the DeepSplitting alg
batch_size = 1000
train_steps = 1000
K = 5

tspan = (0.0,0.5)
dt = 0.1f0  # time step
μ(X,p,t) = 0.0 # advection coefficients
σ(X,p,t) = 0.1 # diffusion coefficients


anal_res = [1.398, 1.9567, 5.3554]
ds = [1,2,5]
atols = [5e-2,1e-1,5e-1]

@testset "DeepSplitting algorithm - Hamel example - CPU" begin
        σ_sampling = 0.1
        for i in 1:length(ds)
            d = ds[i]

            hls = d + 50 #hidden layer size

            nn = Flux.Chain(Dense(d,hls,tanh),
                    Dense(hls,hls,tanh),
                    Dense(hls,1)) # Neural network used by the scheme

            opt = Flux.Optimiser(ExpDecay(0.1,
                            0.1,
                            2000,
                            1e-4),
                            ADAM() )#optimiser
            alg = DeepSplitting(nn, K=K, opt = opt )


            X0 = fill(0.,d)  # initial point
            g(X) = 2.0^(d/2)* exp.(-2. * π  * sum( X.^2, dims=1))   # initial condition
            m(x) = - 0.5 * sum(x.^2, dims=1)
            f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = max.(0.0, v_y) .* (m(y) - max.(0.0, v_z) .* m(z) .* (2.0 * π)^(d/2) .* σ_sampling^d .* exp.(0.5 * sum(z.^2, dims = 1) / σ_sampling^2)) # nonlocal nonlinear part of the
            mc_sample(x) = randn(d,batch_size) * σ_sampling #montecarlo samples

            # defining the problem
            prob = PIDEProblem(g, f, μ, σ, X0, tspan, 
                                # u_domain=[-1f0,1f0]
                                )
            # solving
            @time sol = solve(prob, 
                            alg, 
                            mc_sample, 
                            dt=dt, 
                            verbose = true, 
                            abstol=2e-3,
                            maxiters = train_steps,
                            batch_size=batch_size,
                            use_cuda = false)
            @test isapprox(sol.u[end], anal_res[i],atol = atols[i])
            println("Deep splitting CPU, d = $d, u1 = $(sol.u[end])")
        end
end


@testset "DeepSplitting algorithm - allen cahn reflected example - CPU" begin
    u_domain = [-5f-1,5f-1]
    for i in 1:length(ds)
        d = ds[i]

        hls = d + 50 #hidden layer size

        nn = Flux.Chain(Dense(d,hls,tanh),
                Dense(hls,hls,tanh),
                Dense(hls,1)) # Neural network used by the scheme

        opt = Flux.Optimiser(ExpDecay(0.1,
                        0.1,
                        2000,
                        1e-4),
                        ADAM() )#optimiser
        alg = DeepSplitting(nn, K=K, opt = opt )


        X0 = fill(0f0,d)  # initial point
        g(X) = exp.(-0.25f0 * sum(X.^2,dims=1))   # initial condition
        a(u) = u - u^3
        f(y,z,v_y,v_z,∇v_y,∇v_z, t) = a.(v_y) .- a.(v_z) #.* Float32(π^(d/2)) * σ_sampling^d .* exp.(sum(z.^2, dims = 1) / σ_sampling^2) # nonlocal nonlinear part of the
        mc_sample(x) = (rand(Float64,d,batch_size) .- 0.5) * (u_domain[2]-u_domain[1]) .+ mean(u_domain) # uniform distrib in u_domain

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, X0, tspan, 
                            u_domain = u_domain
                            )
        # solving
        @time sol = solve(prob, 
                        alg, 
                        mc_sample, 
                        dt=dt, 
                        verbose = true, 
                        abstol=1e-5,
                        maxiters = train_steps,
                        batch_size=batch_size,
                        use_cuda = false)
        @test !isnan(sol.u[end])
        println("Deep splitting CPU, d = $d, u1 = $(sol.u[end])")
    end
end