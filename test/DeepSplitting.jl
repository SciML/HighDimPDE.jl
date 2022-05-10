using HighDimPDE
using Random
using Test
using Flux
using Statistics
using CUDA
if CUDA.functional() 
    use_cuda = true 
    cuda_device = 7
else
    use_cuda = false
    cuda_device = nothing
end
# import DiffEqFlux: FastChain, FastDense

#relative error l2
function rel_error_l2(u, uanal) 
    if abs(uanal) >= 10 * eps(eltype(uanal))
        sqrt((u - uanal)^2 / u^2) 
    else # overflow
        abs(u-uanal)
    end
end

# https://en.wikipedia.org/wiki/Heat_equation#Fundamental_solutions
@testset "DeepSplitting - heat equation - single point" begin # solving at one unique point
    batch_size = 10000
    for d in [1, 10, 50]
        println("test for d = ", d)
        tspan = (0f0, 5f-1)
        dt = 5f-1  # time step
        μ(x, p, t) = 0f0 # advection coefficients
        σ(x, p, t) = 1f-1 # diffusion coefficients

        u_anal(x, t) = sum(x.^2) + d * σ(0., 0., 0.)^2 * t
        g(x) = sum(x.^2, dims=1)

        # d = 10
        x0 = fill(2f0,d)

        hls = d + 10 #hidden layer size

        nn = Flux.Chain(Dense(d,hls,relu),
                        Dense(hls,hls,relu),
                        Dense(hls,1)) # Neural network used by the scheme

        opt = ADAM(0.01) #optimiser
        alg = DeepSplitting(nn, opt = opt)

        f(y, z, v_y, v_z, p, t) = 0f0 .* v_y

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, x0, tspan)
        # solving
        sol = solve(prob, alg, dt, 
                    verbose = true, 
                    use_cuda = use_cuda,
                    maxiters = 1000,
                    batch_size = batch_size,
                    cuda_device = cuda_device)
        u1 = sol.us[end]
        u1_anal = u_anal(x0, tspan[end])
        e_l2 = rel_error_l2(u1, u1_anal)
        println("rel_error_l2 = ", e_l2, "\n")
        @test e_l2 < 0.1
    end
end

@testset "DeepSplitting - heat equation - interval" begin # solving on interval [-5f-1, 5f-1]^d
    batch_size = 20000
    for d in [1, 10, 50]
        println("test for d = ", d)
        tspan = (0f0, 5f-1)
        dt = 5f-1  # time step
        μ(x, p, t) = 0f0 # advection coefficients
        σ(x, p, t) = 1f-1 # diffusion coefficients

        u_anal(x, t) = sum(x.^2) + d * σ(0., 0., 0.)^2 * t
        g(x) = sum(x.^2, dims=1)

        # d = 10
        x0_sample = UniformSampling(fill(-5f-1, d), fill(5f-1, d))
        x0 = fill(2f0,d)

        hls = d + 10 #hidden layer size

        nn = Flux.Chain(Dense(d,hls,relu),
                        Dense(hls,hls,relu),
                        Dense(hls,1)) # Neural network used by the scheme

        opt = ADAM(0.01) #optimiser
        alg = DeepSplitting(nn, opt = opt)

        f(y, z, v_y, v_z, p, t) = 0f0 .* v_y #TODO: this fix is not nice

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, x0, tspan, x0_sample = x0_sample)
        # solving
        sol = solve(prob, alg, dt, 
                    verbose = false, 
                    use_cuda = use_cuda,
                    maxiters = 1000,
                    batch_size = batch_size,
                    cuda_device = cuda_device)
        xs = x0_sample(repeat(x0,1,batch_size))
        u1 = [sol.ufuns[end](x)[] for x in eachcol(xs)]
        u1_anal = [u_anal(x, tspan[end]) for x in eachcol(xs) ]
        e_l2 = mean(rel_error_l2.(u1, u1_anal))
        println("rel_error_l2 = ", e_l2, "\n")
        @test e_l2 < 0.1
    end
end

@testset "DeepSplitting - heat equation - interval - Neumann BC" begin
    # no known analytical solutions, so we compare 2 runs
    sols = []
    d = 5
    ∂ = fill(5f-1, d)
    x0_sample = UniformSampling(-∂, ∂)
    x0 = fill(25f-2,d)
    batch_size = 1000
    for _ in 1:2
        tspan = (0f0, 5f-1)
        dt = 5f-1  # time step
        μ(x, p, t) = 0f0 # advection coefficients
        σ(x, p, t) = 1f-1 # diffusion coefficients

        u_anal(x, t) = sum(x.^2) + d * σ(0., 0., 0.)^2 * t
        g(x) = sum(x.^2, dims=1)

        hls = d + 10 #hidden layer size

        nn = Flux.Chain(Dense(d,hls,relu),
                        Dense(hls,hls,relu),
                        Dense(hls,1)) # Neural network used by the scheme

        opt = ADAM(0.01) #optimiser
        alg = DeepSplitting(nn, opt = opt)

        f(y, z, v_y, v_z, p, t) = 0f0 .* v_y #TODO: this fix is not nice

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, x0, tspan, 
                            x0_sample = x0_sample,
                            neumann_bc = [-∂, ∂]
                            )
        # solving
        sol = solve(prob, alg, dt, 
                    verbose = false, 
                    use_cuda = use_cuda,
                    maxiters = 1000,
                    batch_size = batch_size,
                    cuda_device = cuda_device
                    )
        push!(sols, sol)
    end
    xs = x0_sample(repeat(x0,1,batch_size))
    u1 = [sols[1].ufuns[end](x)[] for x in eachcol(xs)]
    u2 = [sols[2].ufuns[end](x)[] for x in eachcol(xs)]
    e_l2 = mean(rel_error_l2.(u1, u2))
    println("rel_error_l2 = ", e_l2, "\n")
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

@testset "DeepSplitting - exponential - interval" begin
    batch_size = 1000
    for d in [1, 3, 10]
        println("test for d = ", d)
        tspan = (0f0, 5f-1)
        dt = 5f-2  # time step
        μ(x, p, t) = 0f0 # advection coefficients
        σ(x, p, t) = 0f0 # diffusion coefficients
        r = 1f-1

        u_anal(x, t) = (g(x) * exp(r * t))[]
        g(x) = sum(x.^2, dims=1) .+ 2f0

        # d = 10
        x0_sample = UniformSampling(fill(-5f-1, d), fill(5f-1, d))
        x0 = fill(2f0,d)

        hls = d + 50 #hidden layer size

        nn = Flux.Chain(Dense(d,hls,relu),
                        Dense(hls,hls,relu),
                        Dense(hls,1)) # Neural network used by the scheme

        opt = ADAM(0.01) #optimiser
        alg = DeepSplitting(nn, opt = opt)

        f(y, z, v_y, v_z, p, t) = r * v_y #TODO: this fix is not nice

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, x0, tspan, 
                            x0_sample = x0_sample,
                            )
        # solving
        sol = solve(prob, alg, dt, 
                    verbose = false, 
                    use_cuda = use_cuda,
                    maxiters = 1000,
                    batch_size = batch_size,
                    cuda_device = cuda_device)
        
        xs = x0_sample(repeat(x0,1,batch_size))
        u1 = [sol.ufuns[end](x)[] for x in eachcol(xs)]
        u1_anal = [u_anal(x, tspan[end]) for x in eachcol(xs) ]
        e_l2 = mean(rel_error_l2.(u1, u1_anal))
        println("rel_error_l2 = ", e_l2, "\n")
        @test e_l2 < 0.1
    end
end


@testset "DeepSplitting algorithm - allen cahn" begin
    batch_size = 2000
    train_steps = 500

    tspan = (3f-1, 6f-1)
    dt = 2f-2  # time step

    μ(x, p, t) = 0f0 # advection coefficients
    σ(x, p, t) = 1f0 # diffusion coefficients

    for d in [10]

        hls = d + 50 #hidden layer size

        nn = Flux.Chain(Dense(d,hls,tanh),
                        Dense(hls,hls,tanh),
                        Dense(hls,1)) # Neural network used by the scheme

        opt = ADAM(1e-3) #optimiser
        alg = DeepSplitting(nn, opt = opt )


        X0 = fill(0f0,d)  # initial point
        g(X) =  1f0 ./ (2f0 .+ 4f-1 * sum(X.^2, dims=1))   # initial condition
        a(u) = u - u^3
        f(y, z, v_y, v_z, p, t) = - a.(v_y) # nonlocal nonlinear part of the

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, X0, tspan,)
        # solving
        @time sol = solve(prob, 
                        alg, 
                        dt, 
                        verbose = false, 
                        # abstol = 1e-5,
                        use_cuda = use_cuda,
                        maxiters = train_steps,
                        batch_size=batch_size,
                        cuda_device = cuda_device)
        u1 = sol.us[end]
        # value coming from \cite{Beck2017a}
        e_l2 = rel_error_l2(u1, 0.30879)
        @test e_l2 < 0.5 # this is quite high as a relative error. 
        println("d = $d, rel_error_l2 = $e_l2")
    end
end

@testset "DeepSplitting - allen cahn local-  Neumann BC" begin
    batch_size = 1000
    train_steps = 1000

    tspan = (0f0, 5f-1)
    dt = 5f-2  # time step

    μ(x, p, t) = 0f0 # advection coefficients
    σ(x, p, t) = 1f0 # diffusion coefficients

    for d in [1,2,5]
        ∂ = fill(5f-1, d)

        hls = d + 50 #hidden layer size

        nn = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1)) # Neural network used by the scheme

        opt = ADAM(1e-2) #optimiser
        alg = DeepSplitting(nn, opt = opt )


        X0 = fill(0f0,d)  # initial point
        g(X) = exp.(-0.25f0 * sum(X.^2,dims=1))   # initial condition
        a(u) = u - u^3
        f(y, z, v_y, v_z, p, t) = a.(v_y) # nonlocal nonlinear part of the

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, X0, tspan, neumann_bc = [-∂, ∂] )
        # solving
        @time sol = solve(prob, 
                        alg, 
                        dt, 
                        verbose = false, 
                        abstol = 1e-5,
                        use_cuda = use_cuda,
                        maxiters = train_steps,
                        batch_size = batch_size,
                        cuda_device = cuda_device)
        u1 = sol.us[end]
        @test !isnan(u1)
        println("d = $d, u1 = $(u1)")
    end
end

# TODO: not working because of a bug in Zygote differentiation rules for adjoints
if false
    @testset "DeepSplitting - Black-Scholes-Barenblatt equation" begin
        batch_size = 20
        train_steps = 500

        tspan = (0f0, 1f0)
        dt = 25f-2  # time step

        r = 5f-2
        sigma = 4f-1

        μ(x, p, t) = 0f0 # advection coefficients
        σ(x, p, t) =  sigma * x.^2 # diffusion coefficients

        for d in [30]

            hls = d + 50 #hidden layer size

            nn = Flux.Chain(Dense(d,hls,tanh),
                            Dense(hls,hls,tanh),
                            Dense(hls,1)) # Neural network used by the scheme

            opt = ADAM(1e-3) #optimiser
            alg = DeepSplitting(nn, opt = opt )

            X0 = repeat([1.0f0, 0.5f0], div(d,2))  # initial point
            g(X) =  sum(X.^2, dims=1) # initial condition
            f(y,z,v_y,v_z,∇v_y,∇v_z, p, t) = r * (v_y .- sum(y .* ∇v_y, dims=1))

            # defining the problem
            prob = PIDEProblem(g, f, μ, σ, X0, tspan)
            # solving
            @time xs,ts,sol = solve(prob, 
                            alg, 
                            dt, 
                            verbose = true, 
                            # abstol = 1e-5,
                            use_cuda = use_cuda,
                            maxiters = train_steps,
                            batch_size=batch_size,
                            cuda_device = cuda_device)
            u1 = sol[end]
            @test !isnan(u1)
            println("d = $d, u1 = $(u1)")
        end
    end

    @testset "DeepSplitting - Hamilton Jacobi Bellman Equation" begin

        batch_size = 30
        train_steps = 500

        tspan = (0f0, 1f0)
        dt = 2f-1  # time step

        λ = 1.0f0

        μ(x, p, t) = 0f0 # advection coefficients
        σ(x, p, t) =  sqrt(2f0) # diffusion coefficients

        d = 20

        T = tspan[2]
        MC = 10^5
        W() = randn(d,1)
        u_analytical(x, t) = -(1/λ)*log(mean(exp(-λ*g(x .+ sqrt(2f0)*abs.(T-t).*W())) for _ = 1:MC))

        hls = d + 50 #hidden layer size

        nn = Flux.Chain(Dense(d,hls,tanh),
                        Dense(hls,hls,tanh),
                        Dense(hls,1)) # Neural network used by the scheme

        opt = ADAM(1e-3) #optimiser
        alg = DeepSplitting(nn, opt = opt )

        X0 = fill(0.0f0,d)  # initial point
        g(X) =  log.(5f-1 .+ 5f-1 * sum(X.^2, dims=1)) # initial condition
        f(y,z,v_y,v_z,∇v_y,∇v_z, p, t) = λ * sum(∇v_y.^2, dims=1)

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, X0, tspan)
        # solving
        @time sol = solve(prob, 
                        alg, 
                        dt, 
                        verbose = true, 
                        # abstol = 1e-5,
                        use_cuda = false,
                        maxiters = train_steps,
                        batch_size=batch_size,
                        cuda_device = cuda_device)

        u1 = sol.us[end]

        analytical_ans = u_analytical(x0, tspan[1])
        error_l2 = rel_error_l2(u1, analytical_ans)

        @test error_l2 < 2f0

        println("d = $d, u1 = $(u1)")

    end
end

@testset "DeepSplitting - Black-Scholes Equation with Default Risk" begin

    batch_size = 1000
    train_steps = (1000, 200)

    tspan = (0f0, 1f0)
    dt = 1.25f-1  # time step

    μ(x, p, t) = 0.02f0 * x # advection coefficients
    σ(x, p, t) = 0.2f0 * x # diffusion coefficients

    d = 20

    hls = d + 50 #hidden layer size

    nn = Flux.Chain(Dense(d,hls,tanh),
                    Dense(hls,hls,tanh),
                    Dense(hls,1)) # Neural network used by the scheme

    opt = ADAM()
    alg = DeepSplitting(nn, opt = opt, λs = [1e-2,1e-3] )

    X0 = fill(100f0,d)  # initial point
    g(X) =  minimum(X, dims=1) # initial condition

    δ = 2.0f0/3f0
    R = 0.02f0

    vh = 50.0f0
    vl = 70.0f0
    γh = 0.2f0
    γl = 0.02f0

    Q(u) = (u .< vh) .* γh .+ (u .>= vl) .*  γl .+ ( (u .>= vh) .* (u .< vl)) .* (((γh - γl) / (vh - vl)) * (u .- vh) .+ γh)

    µc = 0.02f0
    σc = 0.2f0

    f(y, z, v_y, v_z, p, t) = -(1f0 - δ) * Q.(v_y) .* v_y .- R * v_y

    # defining the problem
    prob = PIDEProblem(g, f, μ, σ, X0, tspan, )
    # solving
    @time sol = solve(prob, 
                    alg, 
                    dt, 
                    verbose = true, 
                    # abstol = 1e-5,
                    use_cuda = use_cuda,
                    maxiters = train_steps,
                    batch_size=batch_size,
                    cuda_device = cuda_device)

    u1 = sol.us[end]
    analytical_ans = 60.781
    error_l2 = rel_error_l2(u1, analytical_ans)
    @test error_l2 < 0.1
    println("d = $d, error_l2 = $(error_l2)")
end

###################################################
########### NON LOCAL #############################
###################################################

@testset "DeepSplitting - replicator mutator" begin
    tspan = (0f0,15f-2)
    dt = 5f-2 # time step
    μ(x, p, t) = 0f0 # advection coefficients
    σ(x, p, t) = 1f-1 #1f-1 # diffusion coefficients
    ss0 = 5f-2#std g0

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

    for d in [5]
        U = 25f-2

        ∂ = fill(U, d)
        x0_sample = UniformSampling(-∂, ∂)
        x0 = fill(0f0, d)

        batch_size = 10000
        train_steps = 2000
        K = 1

        hls = d + 50 #hidden layer size

        nn_batch = Flux.Chain(
                            # BatchNorm(d, affine = true, dim = 1),
                            Dense(d, hls, tanh),
                            # BatchNorm(hls, affine = true, dim = 1),
                            Dense(hls, hls, tanh),
                            # BatchNorm(hls, affine = true, dim = 1),
                            Dense(hls, 1, x->x^2)) # positive function

        opt = ADAM(1e-2)#optimiser
        alg = DeepSplitting(nn_batch, K=K, opt = opt, mc_sample = x0_sample)

        g(x) = Float32((2*π)^(-d/2)) * ss0^(- Float32(d) * 5f-1) * exp.(-5f-1 *sum(x .^2f0 / ss0, dims = 1)) # initial condition
        m(x) = - 5f-1 * sum(x.^2, dims=1)
        vol = prod(2*∂)
        f(y, z, v_y, v_z, p, t) =  max.(v_y, 0f0) .* (m(y) .- vol *  max.(v_z, 0f0) .* m(z)) # nonlocal nonlinear part of the

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, x0, tspan, 
                            x0_sample = x0_sample
                            )
        # solving
        sol = solve(prob, 
                    alg, 
                    dt, 
                    verbose = false, 
                    abstol = 1f-3,
                    maxiters = train_steps,
                    batch_size = batch_size,
                    use_cuda = use_cuda,
                    cuda_device = cuda_device
                    )
        xs = x0_sample(repeat(x0,1,batch_size))
        u1 = [sol.ufuns[end](x)[] for x in eachcol(xs)]
        u1_anal = [u_anal(x, tspan[end]) for x in eachcol(xs) ]
        e_l2 = mean(rel_error_l2.(u1, u1_anal))
        println("analytical sol: $(uanal(x0, tspan[end], nothing)) \napproximation: $(sol.ufuns[end](x0)[])")
        println("rel_error_l2 = ", e_l2, "\n")
        @test e_l2 < 0.1
    end
end


@testset "DeepSplitting algorithm - allen cahn non local - Neumann BC" begin
    batch_size = 2000
    train_steps = 1000
    K = 1
    tspan = (0f0, 5f-1)
    dt = 5f-2  # time step

    μ(x, p, t) = 0f0 # advection coefficients
    σ(x, p, t) = 1f-1 #1f-1 # diffusion coefficients
    
    for d in [1,2,5]
        u1s = []
        for _ in 1:2
            ∂ = fill(5f-1, d)
            hls = d + 50 #hidden layer size

            nn = Flux.Chain(Dense(d,hls,tanh),
                            Dense(hls,hls,tanh),
                            Dense(hls,1)) # Neural network used by the scheme

            opt = ADAM(1e-2) #optimiser
            alg = DeepSplitting(nn, K=K, opt = opt, mc_sample = UniformSampling(-∂, ∂) )

            x0 = fill(0f0,d)  # initial point
            g(X) = exp.(-0.25f0 * sum(X.^2,dims=1))   # initial condition
            a(u) = u - u^3
            f(y,z,v_y,v_z, p, t) = a.(v_y) .- a.(v_z) #.* Float32(π^(d/2)) * σ_sampling^d .* exp.(sum(z.^2, dims = 1) / σ_sampling^2) # nonlocal nonlinear part of the

            # defining the problem
            prob = PIDEProblem(g, f, μ, σ, x0, tspan, neumann_bc = [-∂, ∂])
            # solving
            @time sol = solve(prob, 
                            alg, 
                            dt, 
                            # verbose = true, 
                            # abstol=1e-5,
                            use_cuda = use_cuda,
                            cuda_device = cuda_device,
                            maxiters = train_steps,
                            batch_size=batch_size)
            push!(u1s, sol.us[end])
            println("d = $d, u1 = $(sol.us[end])")

        end
        e_l2 = mean(rel_error_l2.(u1s[1], u1s[2]))
        println("rel_error_l2 = ", e_l2, "\n")
        @test e_l2 < 0.1
    end
end

