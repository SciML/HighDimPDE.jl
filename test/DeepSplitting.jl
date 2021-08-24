using HighDimPDE
using Random
using Test
using Flux
using Statistics
# import DiffEqFlux: FastChain, FastDense
#TODO : to be modified
error_l2(u, uanal) = abs(uanal) >= 10 * eps(eltype(uanal)) ? sqrt((u - uanal)^2 / u^2) : zeros(eltype(uanal))

# https://en.wikipedia.org/wiki/Heat_equation#Fundamental_solutions
@testset "DeepSplitting - heat equation - single point" begin
    for d in [1, 10, 50]
        println("test for d = ", d)
        tspan = (0f0, 5f-1)
        dt = 5f-1  # time step
        μ(X,p,t) = 0f0 # advection coefficients
        σ(X,p,t) = 1f-1 # diffusion coefficients

        u_anal(x, t) = sum(x.^2) + d * σ(0., 0., 0.)^2 * t
        g(x) = sum(x.^2, dims=1)

        # d = 10
        x0 = fill(0f0,d)
        hls = d + 10 #hidden layer size

        nn = Flux.Chain(Dense(d,hls,relu),
                        Dense(hls,hls,relu),
                        Dense(hls,1)) # Neural network used by the scheme

        opt = ADAM(0.01) #optimiser
        alg = DeepSplitting(nn, opt = opt)

        f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = 0f0 .* v_y #TODO: this fix is not nice

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, tspan, 
                            x = x0
                            )
        # solving
        xs,ts,sol = solve(prob, alg, dt, 
                        verbose = false, 
                        use_cuda = true,
                        maxiters = 1000,
                        batch_size = 10000)
        u1 = sol[end]
        u1_anal = u_anal(x0, tspan[end])
        e_l2 = error_l2(u1, u1_anal)
        println("error_l2 = ", e_l2, "\n")
        @test e_l2 < 0.1
    end
end

@testset "DeepSplitting - heat equation - interval" begin
    for d in [1, 10, 50]
        println("test for d = ", d)
        tspan = (0f0, 5f-1)
        dt = 5f-1  # time step
        μ(X,p,t) = 0f0 # advection coefficients
        σ(X,p,t) = 1f-1 # diffusion coefficients

        u_anal(x, t) = sum(x.^2) + d * σ(0., 0., 0.)^2 * t
        g(x) = sum(x.^2, dims=1)

        # d = 10
        u_domain = repeat([-5f-1, 5f-1]', d, 1)
        hls = d + 10 #hidden layer size

        nn = Flux.Chain(Dense(d,hls,relu),
                        Dense(hls,hls,relu),
                        Dense(hls,1)) # Neural network used by the scheme

        opt = ADAM(0.01) #optimiser
        alg = DeepSplitting(nn, opt = opt)

        f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = 0f0 .* v_y #TODO: this fix is not nice

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, tspan, 
                            u_domain = u_domain,
                            )
        # solving
        xs,ts,sol = solve(prob, alg, dt, 
                        verbose = false, 
                        use_cuda = true,
                        maxiters = 1000,
                        batch_size = 20000)
        u1 = [sol[end](x)[] for x in xs]
        u1_anal = u_anal.(xs, tspan[end])
        e_l2 = mean(error_l2.(u1, u1_anal))
        println("error_l2 = ", e_l2, "\n")
        @test e_l2 < 0.1
    end
end

@testset "DeepSplitting - heat equation - interval - Neumann BC" begin
    # no known analytical solutions, so we compare 2 runs
    sols = []
    xs = []
    d = 5
    for _ in 1:2
        tspan = (0f0, 5f-1)
        dt = 5f-1  # time step
        μ(X,p,t) = 0f0 # advection coefficients
        σ(X,p,t) = 1f-1 # diffusion coefficients

        u_anal(x, t) = sum(x.^2) + d * σ(0., 0., 0.)^2 * t
        g(x) = sum(x.^2, dims=1)

        # d = 10
        u_domain = repeat([-5f-1, 5f-1]', d, 1)
        hls = d + 10 #hidden layer size

        nn = Flux.Chain(Dense(d,hls,relu),
                        Dense(hls,hls,relu),
                        Dense(hls,1)) # Neural network used by the scheme

        opt = ADAM(0.01) #optimiser
        alg = DeepSplitting(nn, opt = opt)

        f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = 0f0 .* v_y #TODO: this fix is not nice

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, tspan, 
                            u_domain = u_domain,
                            )
        # solving
        xs,ts,sol = solve(prob, alg, dt, 
                        verbose = false, 
                        use_cuda = true,
                        maxiters = 1000,
                        batch_size = 20000,
                        neumann = u_domain)
        push!(sols, sol[end])
    end
    u1 = [sols[1](x)[] for x in xs]
    u1_anal = [sols[2](x)[] for x in xs]
    e_l2 = mean(error_l2.(u1, u1_anal))
    println("error_l2 = ", e_l2, "\n")
    @test e_l2 < 0.1

    # for fun: plotting
    # close("all")    
    # fig, ax = plt.subplots()
    # xgrid = hcat([x[1:2] for x in xs]...)
    # #Deepsplitting sol
    # ax.scatter(xgrid[1,:], xgrid[2,:], c = reduce(vcat, sols[end].(xs))[:], s=.2)
    # ax.set_title("DeepSplitting")
    # gcf()
end

@testset "DeepSplitting - exponential" begin
    for d in [1, 3, 10]
        println("test for d = ", d)
        tspan = (0f0, 5f-1)
        dt = 5f-2  # time step
        μ(X,p,t) = 0f0 # advection coefficients
        σ(X,p,t) = 0f0 # diffusion coefficients
        r = 1f-1

        u_anal(x, t) = (g(x) * exp(r * t))[]
        g(x) = sum(x.^2, dims=1) .+ 2f0

        # d = 10
        u_domain = repeat([-5f-1, 5f-1]', d, 1)
        hls = d + 50 #hidden layer size

        nn = Flux.Chain(Dense(d,hls,relu),
                        Dense(hls,hls,relu),
                        Dense(hls,1)) # Neural network used by the scheme

        opt = ADAM(0.01) #optimiser
        alg = DeepSplitting(nn, opt = opt)

        f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = r * v_y #TODO: this fix is not nice

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, tspan, 
                            u_domain = u_domain,
                            )
        # solving
        xs,ts,sol = solve(prob, alg, dt, 
                        verbose = false, 
                        use_cuda = true,
                        maxiters = 1000,
                        batch_size = 100)
        u1 = [sol[end](x)[] for x in xs]
        u1_anal = u_anal.(xs, tspan[end])
        e_l2 = mean(error_l2.(u1, u1_anal))
        println("error_l2 = ", e_l2, "\n")
        @test e_l2 < 0.1
    end
end

# TODO: Victor, you stopped here (24/08/2021)
# think of a linear non local term instead of something crazy as below first (look onyx)
# probably that you want to try some of the tests of neuralPDE (e.g. allen cahn equations)

@testset "DeepSplitting - Hamel example - udomain - CPU" begin
    tspan = (0f0,2f-1)
    dt = 1f-2 # time step
    μ(X,p,t) = 0f0 # advection coefficients
    σ(X,p,t) = 1f-1 #1f-1 # diffusion coefficients

    # Analytic sol
    function _SS(x, t, p)
        d = length(x)
        MM = σ(x, p, t) * ones(d)
        SSt = MM .* ((MM .* sinh.(MM *t) .+ ss0 .* cosh.( MM * t)) ./ (MM .* cosh.(MM * t ) .+ ss0 .* sinh.(MM * t)))
        return SSt
    end

    function uanal(x, t, p)
            d = length(x)
            return (2*π)^(-d/2) * prod(_SS(x, t, p) .^(-1/2)) * exp(-0.5 *sum(x .^2 ./ _SS(x, t, p)) )
    end

    sols = []
    xs = []
    for d in [1,2,5]
        ss0 = 1f-2#std g0
        U = 5f-1
        u_domain = repeat([-U,U]', d, 1)

        ##############################
        ####### Neural Network #######
        ##############################
        batch_size = 1000
        train_steps = 500
        K = 1

        hls = d + 50 #hidden layer size

        nn_batch = Flux.Chain(
                # BatchNorm(d, affine = true, dim = 1),
                Dense(d, hls, tanh),
                # BatchNorm(hls, affine = true, dim = 1),
                Dense(hls, hls, tanh),
                Dense(hls, hls, tanh),
                # BatchNorm(hls, affine = true, dim = 1),
                Dense(hls, 1, relu)) # Neural network used by the scheme, with batch normalisation

        opt = ADAM(0.001)#optimiser
        alg = DeepSplitting(nn_batch, K=K, opt = opt, mc_sample = UniformSampling(u_domain[:,1], u_domain[:,2]) )

        ##########################
        ###### PDE Problem #######
        ##########################
        g(x) = Float32((2*π)^(-d/2)) * ss0^(- Float32(d) * 5f-1) * exp.(-5f-1 *sum(x .^2f0 / ss0, dims = 1)) # initial condition
        m(x) = - 5f-1 * sum(x.^2, dims=1)
        vol = prod(u_domain[:,2] - u_domain[:,1])
        f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) =  v_y .* (m(y) .- vol * v_z .* m(z)) # nonlocal nonlinear part of the
        # TODO: Victor pay attention, there is no max while this could cause the non convergence!
        # this is how it used to be
        # f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = max.(0.0, v_y) .* (m(y) .- vol * max.(0.0, v_z) .* m(z) )

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, tspan, 
                            u_domain = u_domain
                            )
        # solving
        xgrid,ts,sol = solve(prob, 
                        alg, 
                        dt, 
                        verbose = true, 
                        # abstol = 2f-1,
                        maxiters = train_steps,
                        batch_size = batch_size,
                        use_cuda = true
                        )
        u1 = [sol[end](x)[] for x in xgrid]
        u1_anal = uanal.(xgrid, tspan[end], Ref(Dict()))
        e_l2 = mean(error_l2.(u1, u1_anal))
        println("error_l2 = ", e_l2, "\n") # TODO: Victor, this is throwing an Inf because of small values - that should be fixed
        @test e_l2 < 0.1
        push!(sols, sol[end])
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