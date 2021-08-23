using HighDimPDE
using Random
using Test
using Statistics

tspan = (0.0,0.5)
# using the MLP alg
μ(X,p,t) = 0.0 # advection coefficients
σ(X,p,t) = 0.1 # diffusion coefficients


anal_res = [1.398, 1.9567, 5.3554]
ds = [1,2,5]
atols = [5e-2,1e-1,2e0]

@testset "MLP algorithm - single threaded" begin
    σ_sampling = 0.1
    for i in 1:length(ds)
        d = ds[i]
        x = fill(0.,d)  # initial point
        g(X) = 2.0^(d/2)* exp(-2. * π  * sum( X.^2))   # initial condition
        m(x) = - 0.5 * sum(x.^2)
        f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = max(0.0, v_y) * (m(y) - max(0.0, v_z) * m(z) * (2.0 * π)^(d/2) * σ_sampling^d * exp(0.5 * sum(z.^2) / σ_sampling^2)) # nonlocal nonlinear part of the
        alg = MLP(M=4, K=10, L = 4, mc_sample = NormalSampling(σ_sampling) )

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, tspan, x = x)
        # solving
        @time xs,ts,sol = solve(prob, alg, verbose = false,  multithreading = false)
        @test isapprox(sol[end], anal_res[i],atol = atols[i])
        println("MLP, d = $d, u1 = $sol")
    end
end


@testset "MLP algorithm - multi threaded" begin
    σ_sampling = 0.1
    for i in 1:length(ds)
        d = ds[i]
        x = fill(0.,d)  # initial point
        g(X) = 2.0^(d/2)* exp(-2. * π  * sum( X.^2))   # initial condition
        m(x) = - 0.5 * sum(x.^2)
        f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = max(0.0, v_y) * (m(y) - max(0.0, v_z) * m(z) * (2.0 * π)^(d/2) * σ_sampling^d * exp(0.5 * sum(z.^2) / σ_sampling^2)) # nonlocal nonlinear part of the
        alg = MLP(M=4, K=10, L = 4, mc_sample = NormalSampling(σ_sampling) )

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, tspan, x = x)
        # solving
        @time xs,ts,sol = solve(prob, alg, verbose = false, multithreading=true)
        @test isapprox(sol[end], anal_res[i], atol = atols[i])
        println("MLP, d = $d, u1 = $(sol[end])")
    end
end

@testset "MLP algorithm - allen cahn reflected example" begin
    for i in 1:length(ds)
        d = ds[i]
        u_domain = repeat([-5e-1,5e-1]', d, 1)
        d = ds[i]
        x = fill(0e0,d)  # initial point
        g(X) = exp.(-0.25e0 * sum(X.^2))   # initial condition
        a(u) = u - u^3
        f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = a.(v_y) .- a.(v_z) #.* Float32(π^(d/2)) * σ_sampling^d .* exp.(sum(z.^2) / σ_sampling^2) # nonlocal nonlinear part of the
        alg = MLP(M = 4, K = 10, L = 4, mc_sample = UniformSampling(u_domain[:,1], u_domain[:,2]) )

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, tspan, x = x )
        # solving
        @time xs,ts,sol = solve(prob, alg, neumann = u_domain, verbose = false, multithreading=true)
        @test !isnan(sol[end])
        println("MLP, d = $d, u1 = $(sol[end])")
    end
end