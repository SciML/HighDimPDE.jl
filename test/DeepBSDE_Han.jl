using Flux, Zygote, LinearAlgebra, Statistics
println("DeepBSDE_tests")
using Test, StochasticDiffEq
using HighDimPDE

using Random
Random.seed!(100)

#relative error l2
function rel_error_l2(u, uanal)
    if abs(uanal) >= 10 * eps(eltype(uanal))
        sqrt((u - uanal)^2 / u^2)
    else # overflow
        abs(u - uanal)
    end
end

@testset "DeepBSDE_Han - one-dimensional heat equation" begin # solving at one unique point
    # one-dimensional heat equation
    x0 = [11.0f0]  # initial points
    tspan = (0.0f0, 5.0f0)
    dt = 0.5   # time step
    time_steps = div(tspan[2] - tspan[1], dt)
    d = 1      # number of dimensions
    m = 10     # number of trajectories (batch size)

    g(X) = sum(X .^ 2)   # terminal condition
    f(X, u, σᵀ∇u, p, t) = 0.0f0  # function from solved equation
    μ_f(X, p, t) = 0.0
    σ_f(X, p, t) = 1.0
    prob = ParabolicPDEProblem(μ_f, σ_f, x0, tspan, g, f)

    hls = 10 + d #hidden layer size
    opt = Flux.Optimise.Adam(0.005)  #optimizer
    #sub-neural network approximating solutions at the desired point
    u0 = Flux.Chain(Dense(d, hls, relu),
        Dense(hls, hls, relu),
        Dense(hls, 1))
    # sub-neural network approximating the spatial gradients at time point
    σᵀ∇u = [Flux.Chain(Dense(d, hls, relu),
                Dense(hls, hls, relu),
                Dense(hls, d)) for i in 1:time_steps]

    alg = DeepBSDE(u0, σᵀ∇u, opt = opt)

    sol = solve(prob,
        alg,
        verbose = true,
        abstol = 1e-8,
        maxiters = 200,
        dt = dt,
        trajectories = m)

    u_analytical(x, t) = sum(x .^ 2) .+ d * t
    analytical_sol = u_analytical(x0, tspan[end])

    error_l2 = rel_error_l2(sol.us, analytical_sol)

    println("error_l2 = ", error_l2, "\n")
    @test error_l2 < 0.1
end

@testset "DeepBSDE_Han - high-dimensional heat equation" begin # solving at one unique point
    d = 30 # number of dimensions
    x0 = fill(8.0f0, d)
    tspan = (0.0f0, 2.0f0)
    dt = 0.5
    time_steps = div(tspan[2] - tspan[1], dt)
    m = 30 # number of trajectories (batch size)

    g(X) = sum(X .^ 2)
    f(X, u, σᵀ∇u, p, t) = 0.0f0
    μ_f(X, p, t) = 0.0
    σ_f(X, p, t) = 1.0
    prob = ParabolicPDEProblem(μ_f, σ_f, x0, tspan, g, f)

    hls = 10 + d #hidden layer size
    opt = Flux.Optimise.Adam(0.005)  #optimizer
    #sub-neural network approximating solutions at the desired point
    u0 = Flux.Chain(Dense(d, hls, relu),
        Dense(hls, hls, relu),
        Dense(hls, 1))
    # sub-neural network approximating the spatial gradients at time point
    σᵀ∇u = [Flux.Chain(Dense(d, hls, relu),
                Dense(hls, hls, relu),
                Dense(hls, d)) for i in 1:time_steps]

    alg = DeepBSDE(u0, σᵀ∇u, opt = opt)

    sol = solve(prob,
        alg,
        verbose = true,
        abstol = 1e-8,
        maxiters = 150,
        dt = dt,
        trajectories = m)

    u_analytical(x, t) = sum(x .^ 2) .+ d * t
    analytical_sol = u_analytical(x0, tspan[end])
    error_l2 = rel_error_l2(sol.us, analytical_sol)

    println("error_l2 = ", error_l2, "\n")
    @test error_l2 < 1.0
end

@testset "DeepBSDE_Han - Black Scholes Barenblatt equation" begin
    d = 30 # number of dimensions
    x0 = repeat([1.0f0, 0.5f0], div(d, 2))
    tspan = (0.0f0, 1.0f0)
    dt = 0.25
    time_steps = div(tspan[2] - tspan[1], dt)
    m = 30 # number of trajectories (batch size)

    r = 0.05f0
    sigma = 0.4
    f(X, u, σᵀ∇u, p, t) = r * (u .- sum(X .* σᵀ∇u))
    g(X) = sum(X .^ 2)
    μ_f(X, p, t) = 0.0
    σ_f(X, p, t) = Diagonal(sigma * X)
    prob = ParabolicPDEProblem(μ_f, σ_f, x0, tspan, g, f)

    hls = 10 + d #hide layer size
    opt = Flux.Optimise.Adam(0.001)
    u0 = Flux.Chain(Dense(d, hls, relu),
        Dense(hls, hls, relu),
        Dense(hls, 1))
    σᵀ∇u = [Flux.Chain(Dense(d, hls, relu),
                Dense(hls, hls, relu),
                Dense(hls, hls, relu),
                Dense(hls, d)) for i in 1:time_steps]

    alg = DeepBSDE(u0, σᵀ∇u, opt = opt)

    sol = solve(prob,
        alg,
        verbose = true,
        abstol = 1e-8,
        maxiters = 150,
        dt = dt,
        trajectories = m)

    u_analytical(x, t) = exp((r + sigma^2) .* (tspan[end] .- tspan[1])) .* sum(x .^ 2)
    analytical_sol = u_analytical(x0, tspan[1])
    error_l2 = rel_error_l2(sol.us, analytical_sol)

    println("error_l2 = ", error_l2, "\n")
    @test error_l2 < 1.0
end

@testset "DeepBSDE_Han - Allen-Cahn Equation" begin
    d = 10 # number of dimensions
    x0 = fill(0.0f0, d)
    tspan = (0.3f0, 0.6f0)
    dt = 0.015 # time step
    time_steps = div(tspan[2] - tspan[1], dt)
    m = 20 # number of trajectories (batch size)

    g(X) = 1.0f0 / (2.0f0 + 0.4f0 * sum(X .^ 2))
    f(X, u, σᵀ∇u, p, t) = u .- u .^ 3
    μ_f(X, p, t) = 0.0
    σ_f(X, p, t) = 1.0
    prob = ParabolicPDEProblem(μ_f, σ_f, x0, tspan, g, f)

    hls = 10 + d #hidden layer size
    opt = Flux.Optimise.Adam(5^-4)  #optimizer
    #sub-neural network approximating solutions at the desired point
    u0 = Flux.Chain(Dense(d, hls, relu),
        Dense(hls, hls, relu),
        Dense(hls, 1))

    # sub-neural network approximating the spatial gradients at time point
    σᵀ∇u = [Flux.Chain(Dense(d, hls, relu),
                Dense(hls, hls, relu),
                Dense(hls, d)) for i in 1:time_steps]

    alg = DeepBSDE(u0, σᵀ∇u, opt = opt)

    sol = solve(prob,
        alg,
        verbose = true,
        abstol = 1e-8,
        maxiters = 150,
        dt = dt,
        trajectories = m)

    analytical_sol = 0.30879
    error_l2 = rel_error_l2(sol.us, analytical_sol)

    println("error_l2 = ", error_l2, "\n")
    @test error_l2 < 1.0
end

@testset "DeepBSDE_Han - Allen-Cahn Equation" begin
    #Hamilton Jacobi Bellman Equation
    d = 20 # number of dimensions
    x0 = fill(0.0f0, d)
    tspan = (0.0f0, 1.0f0)
    dt = 0.2
    ts = tspan[1]:dt:tspan[2]
    time_steps = length(ts) - 1
    m = 20 # number of trajectories (batch size)
    λ = 1.0f0

    g(X) = log(0.5f0 + 0.5f0 * sum(X .^ 2))
    f(X, u, σᵀ∇u, p, t) = -λ * sum(σᵀ∇u .^ 2)
    μ_f(X, p, t) = 0.0
    σ_f(X, p, t) = sqrt(2)
    prob = ParabolicPDEProblem(μ_f, σ_f, x0, tspan, g, f)

    hls = 12 + d #hidden layer size
    opt = Flux.Optimise.Adam(0.03)  #optimizer
    #sub-neural network approximating solutions at the desired point
    u0 = Flux.Chain(Dense(d, hls, relu),
        Dense(hls, hls, relu),
        Dense(hls, 1))

    # sub-neural network approximating the spatial gradients at time point
    σᵀ∇u = [Flux.Chain(Dense(d, hls, relu),
                Dense(hls, hls, relu),
                Dense(hls, hls, relu),
                Dense(hls, d)) for i in 1:time_steps]

    alg = DeepBSDE(u0, σᵀ∇u, opt = opt)

    sol = solve(prob,
        alg,
        verbose = true,
        abstol = 1e-8,
        maxiters = 200,
        dt = dt,
        trajectories = m)

    T = tspan[2]
    MC = 10^5
    W() = randn(d, 1)
    u_analytical(x,
        t) = -(1 / λ) *
             log(mean(exp(-λ * g(x .+ sqrt(2.0) * abs.(T - t) .* W()))
    for _ in 1:MC))
    analytical_sol = u_analytical(x0, tspan[1])

    error_l2 = rel_error_l2(sol.us, analytical_sol)

    println("error_l2 = ", error_l2, "\n")
    @test error_l2 < 2.0
end

@testset "DeepBSDE_Han - Nonlinear Black-Scholes Equation with Default Risk" begin
    d = 20 # number of dimensions
    x0 = fill(100.0f0, d)
    tspan = (0.0f0, 1.0f0)
    dt = 0.125 # time step
    m = 20 # number of trajectories (batch size)
    time_steps = div(tspan[2] - tspan[1], dt)

    g(X) = minimum(X)
    δ = 2.0f0 / 3
    R = 0.02f0
    function f(X, u, σᵀ∇u, p, t)
        -(1 - δ) * Q_(u) * u - R * u
    end

    vh = 50.0f0
    vl = 70.0f0
    γh = 0.2f0
    γl = 0.02f0
    function Q_(u)
        Q = 0
        if u < vh
            Q = γh
        elseif u >= vl
            Q = γl
        else  #if  u >= vh && u < vl
            Q = ((γh - γl) / (vh - vl)) * (u - vh) + γh
        end
    end

    µc = 0.02f0
    σc = 0.2f0
    μ_f(X, p, t) = µc * X #Vector d x 1
    σ_f(X, p, t) = σc * Diagonal(X) #Matrix d x d
    prob = ParabolicPDEProblem(μ_f, σ_f, x0, tspan, g, f)

    hls = 10 + d #hidden layer size
    opt = Flux.Optimise.Adam(0.008)  #optimizer
    #sub-neural network approximating solutions at the desired point
    u0 = Flux.Chain(Dense(d, hls, relu),
        Dense(hls, hls, relu),
        Dense(hls, 1))

    σᵀ∇u = [Flux.Chain(Dense(d, hls, relu),
                Dense(hls, hls, relu),
                Dense(hls, hls, relu),
                Dense(hls, d)) for i in 1:time_steps]
    alg = DeepBSDE(u0, σᵀ∇u, opt = opt)

    sol = solve(prob,
        alg,
        verbose = true,
        abstol = 1e-8,
        maxiters = 100,
        dt = dt,
        trajectories = m)

    analytical_sol = 57.3 #60.781
    error_l2 = rel_error_l2(sol.us, analytical_sol)
    println("error_l2 = ", error_l2, "\n")
    @test error_l2 < 1.0
end

# @testset "DeepBSDE_Han - Limit check on heat equation" begin
#     x0 = [10.0f0]  # initial points
#     tspan = (0.0f0,5.0f0)
#     dt = 0.5   # time step
#     time_steps = div(tspan[2]-tspan[1],dt)
#     d = 1      # number of dimensions
#     m = 10     # number of trajectories (batch size)

#     g(X) = sum(X.^2)   # terminal condition
#     f(X,u,σᵀ∇u,p,t) = 0.0  # function from solved equation
#     μ_f(X,p,t) = 0.0
#     σ_f(X,p,t) = 1.0
#     u_domain = -500:0.1:500
#     A = -2:0.01:2
#     prob = ParabolicPDEProblem(g, f, μ_f, σ_f, x0, tspan ;A = A , x0_sample = u_domain)

#     hls = 10 + d # hidden layer size
#     opt = Flux.Optimise.Adam(0.005)  # optimizer

#     u0 = Flux.Chain(Dense(d,hls,relu),
#                     Dense(hls,hls,relu),
#                     Dense(hls,1))

#     σᵀ∇u = [Flux.Chain(Dense(d,hls,relu),
#                     Dense(hls,hls,relu),
#                     Dense(hls,d)) for i in 1:time_steps]

#     alg = DeepBSDE(u0, σᵀ∇u, opt = opt)

#     sol = solve(prob, 
#                 alg, 
#                 verbose = true,
#                 abstol=1e-8,
#                 maxiters = 200, 
#                 dt=dt, 
#                 trajectories=m , 
#                 limits = true, 
#                 trajectories_upper = m,
#                 trajectories_lower = m, 
#                 maxiters_limits = 200)
#     @test sol.limits[1] < sol.us[end] < sol.limits[2] # TODO: results seem dubious and must be confirmed
# end
