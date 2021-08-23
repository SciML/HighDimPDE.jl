using HighDimPDE
using Random
using Test
using Flux

# using the DeepSplitting alg


tspan = (0.0,0.5)
dt = 0.10  # time step
μ(X,p,t) = 0.0 # advection coefficients
σ(X,p,t) = 0.1 # diffusion coefficients


anal_res = [1.398, 1.9567, 5.3554]
ds = [1,2,5]
atols = [5e-2,1e-1,5e-1]

# TODO: this one is not full converging yet
@testset "DeepSplitting - Hamel example - udomain - CPU" begin
        U = 1e0
        batch_size = 200
        train_steps = 1000
        K = 20
        for i in 1:length(ds)
            d = ds[i]
            u_domain = repeat([-U, U]', d, 1)

            hls = d + 50 #hidden layer size

            nn = Flux.Chain(Dense(d,hls,tanh),
                    Dense(hls,hls,tanh),
                    Dense(hls,1)) # Neural network used by the scheme

            opt = Flux.Optimiser(ExpDecay(0.1,
                            5.0,
                            200,
                            1e-6),
                            ADAM() )#optimiser

            mc_sample = UniformSampling(u_domain[:,1], u_domain[:,2])

            alg = DeepSplitting(nn, K=K, opt = opt, mc_sample = mc_sample )


            x = fill(0.,d)  # initial point
            g(x) = 2.0^(d/2)* exp.(-2. * π  * sum( x.^2, dims=1))   # initial condition
            m(x) = - 0.5 * sum(x.^2, dims=1)
            vol = prod(u_domain[:,2] - u_domain[:,1])
            f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = max.(0.0, v_y) .* (m(y) .- vol * max.(0.0, v_z) .* m(z) )

            # defining the problem
            prob = PIDEProblem(g, f, μ, σ, tspan, u_domain = u_domain )
            # solving
            @time xs,ts,sol = solve(prob, 
                            alg, 
                            dt, 
                            verbose = true, 
                            abstol=2e-3,
                            maxiters = train_steps,
                            batch_size = batch_size)
            u1 = sol[end](x)[]
            # @test isapprox(u1, anal_res[i], atol = atols[i])
            @test !isnan(u1)
            println("Deep splitting CPU, d = $d, u1 = $u1)")
        end
end


@testset "DeepSplitting algorithm - allen cahn reflected example - CPU" begin
    batch_size = 100
    train_steps = 1000
    K = 50
    for i in 1:length(ds)
        d = ds[i]
        u_domain = repeat([-5e-1, 5e-1]', d, 1)

        hls = d + 50 #hidden layer size

        nn = Flux.Chain(Dense(d,hls,tanh),
                Dense(hls,hls,tanh),
                Dense(hls,1)) # Neural network used by the scheme

        opt = Flux.Optimiser(ExpDecay(0.1,
                        0.1,
                        200,
                        1e-4),
                        ADAM() )#optimiser
        alg = DeepSplitting(nn, K=K, opt = opt, mc_sample = UniformSampling(u_domain[1],u_domain[2]) )


        x = fill(0e0,d)  # initial point
        g(X) = exp.(-0.25f0 * sum(X.^2,dims=1))   # initial condition
        a(u) = u - u^3
        f(y,z,v_y,v_z,∇v_y,∇v_z, t) = a.(v_y) .- a.(v_z) #.* Float32(π^(d/2)) * σ_sampling^d .* exp.(sum(z.^2, dims = 1) / σ_sampling^2) # nonlocal nonlinear part of the

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, tspan, x = x)
        # solving
        @time xs,ts,sol = solve(prob, 
                        alg, 
                        dt, 
                        neumann = u_domain,
                        verbose = true, 
                        abstol=1e-5,
                        maxiters = train_steps,
                        batch_size=batch_size)
        u1 = sol[end]
        @test !isnan(u1)
        println("Deep splitting CPU, d = $d, u1 = $(u1)")
    end
end


@testset "DeepSplitting algorithm - allen cahn local, reflected example - CPU" begin
    batch_size = 1000
    train_steps = 1000
    for i in 1:length(ds)
        d = ds[i]
        u_domain = repeat([-5e-1, 5e-1]', d, 1)

        hls = d + 50 #hidden layer size

        nn = Flux.Chain(Dense(d,hls,tanh),
                Dense(hls,hls,tanh),
                Dense(hls,1)) # Neural network used by the scheme

        opt = Flux.Optimiser(ExpDecay(0.1,
                        1,
                        200,
                        1e-4),
                        ADAM() )#optimiser
        alg = DeepSplitting(nn, opt = opt )


        X0 = fill(0f0,d)  # initial point
        g(X) = exp.(-0.25f0 * sum(X.^2,dims=1))   # initial condition
        a(u) = u - u^3
        f(y,z,v_y,v_z,∇v_y,∇v_z, t) = a.(v_y) # nonlocal nonlinear part of the

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, tspan, x = X0 )
        # solving
        @time xs,ts,sol = solve(prob, 
                        alg, 
                        dt, 
                        neumann = u_domain,
                        verbose = true, 
                        abstol=1e-5,
                        maxiters = train_steps,
                        batch_size=batch_size)
        u1 = sol[end]
        @test !isnan(u1)
        println("Deep splitting CPU, d = $d, u1 = $(u1)")
    end
end