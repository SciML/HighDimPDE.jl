using HighDimPDE
using Random
using Test
using Statistics
using Revise

#relative error l2
function rel_error_l2(u, uanal) 
    if abs(uanal) >= 10 * eps(eltype(uanal))
        sqrt((u - uanal)^2 / u^2) 
    else # overflow
        abs(u-uanal)
    end
end


# https://en.wikipedia.org/wiki/Heat_equation#Fundamental_solutions
@testset "MLP - heat equation - single thread" begin # solving at one unique point
    for d in [1, 10, 50]
        println("test for d = ", d)
        tspan = (0e0, 5e-1)
        dt = 5e-1  # time step
        μ(x, p, t) = 0e0 # advection coefficients
        σ(x, p, t) = 1e-1 # diffusion coefficients

        u_anal(x, t) = sum(x.^2) + d * σ(0., 0., 0.)^2 * t
        g(x) = sum(x.^2)

        # d = 10
        x0 = fill(2e0,d)
       
        alg = MLP()

        f(y, z, v_y, v_z, ∇v_y, ∇v_z, p, t) = 0e0 .* v_y #TODO: this fix is not nice

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, x0, tspan)
        # solving
        sol = solve(prob, alg, multithreading = false)
        u1 = sol.us[end]
        u1_anal = u_anal(x0, tspan[end])
        e_l2 = rel_error_l2(u1, u1_anal)
        println("rel_error_l2 = ", e_l2, "\n")
        @test e_l2 < 0.1
    end
end


# https://en.wikipedia.org/wiki/Heat_equation#Fundamental_solutions
@testset "MLP - heat equation - multi thread" begin # solving at one unique point
    for d in [1, 10, 50]
        println("test for d = ", d)
        tspan = (0e0, 5e-1)

        μ(x, p, t) = 0e0 # advection coefficients
        σ(x, p, t) = 1e-1 # diffusion coefficients

        u_anal(x, t) = sum(x.^2) + d * σ(0., 0., 0.)^2 * t
        g(x) = sum(x.^2)

        # d = 10
        x0 = fill(2e0,d)
       
        alg = MLP()

        f(y, z, v_y, v_z, ∇v_y, ∇v_z, p, t) = 0e0 .* v_y #TODO: this fix is not nice

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, x0, tspan)
        # solving
        sol = solve(prob, alg, multithreading = true)
        u1 = sol.us[end]
        u1_anal = u_anal(x0, tspan[end])
        e_l2 = rel_error_l2(u1, u1_anal)
        println("rel_error_l2 = ", e_l2, "\n")
        @test e_l2 < 0.1
    end
end


@testset "MLP - heat equation - single thread - Neumann BC" begin # solving at one unique point
    tspan = (0f0, 5f-1)
    μ(x, p, t) = 0f0 # advection coefficients
    σ(x, p, t) = 1f-1 #1f-1 # diffusion coefficients
    for d in [1,2,5]
        u1s = []
        for _ in 1:2
            neumann_bc = [fill(-5e-1, d), fill(5e-1, d)]
            g(x) = sum(x.^2)
            # d = 10
            x0 = fill(3e-1,d)
            alg = MLP()
            f(y, z, v_y, v_z, ∇v_y, ∇v_z, p, t) = 0e0 .* v_y #TODO: this fix is not nice

            # defining the problem
            prob = PIDEProblem(g, f, μ, σ, x0, tspan, neumann_bc = neumann_bc)
            # solving
            sol = solve(prob, alg)
            push!(u1s, sol.us[end])
            println("d = $d, u1 = $(sol.us[end])")

        end
        e_l2 = mean(rel_error_l2.(u1s[1], u1s[2]))
        println("rel_error_l2 = ", e_l2, "\n")
        @test e_l2 < 0.1
    end
end

@testset "MLP - exponential - interval" begin
    for d in [1, 3, 10]
        println("test for d = ", d)
        tspan = (0f0, 5f-1)

        μ(x, p, t) = 0f0 # advection coefficients
        σ(x, p, t) = 0f0 # diffusion coefficients
        r = 1f-1
        x = fill(0f0, d)

        u_anal(x, t) = (g(x) * exp(r * t)) 
        g(x) = sum(x.^2) .+ 2f0

        alg = MLP()

        f(y, z, v_y, v_z, ∇v_y, ∇v_z, p, t) = r * v_y #TODO: this fix is not nice

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, x, tspan)
        # solving
        sol = solve(prob, alg)
        u1 = sol.us[end]
        u1_anal = u_anal(x, tspan[end])
        e_l2 = mean(rel_error_l2.(u1, u1_anal))
        println("rel_error_l2 = ", e_l2, "\n")
        @test e_l2 < 0.1
    end
end


@testset "MLP - allen cahn" begin

    tspan = (3f-1, 6f-1)
    
    μ(x, p, t) = 0f0 # advection coefficients
    σ(x, p, t) = 1f0 # diffusion coefficients

    for d in [10]

        alg = MLP()


        X0 = fill(0f0,d)  # initial point
        g(X) =  1f0 ./ (2f0 .+ 4f-1 * sum(X.^2))   # initial condition
        a(u) = u - u^3
        f(y, z, v_y, v_z, ∇v_y, ∇v_z, p, t) = - a.(v_y)

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, X0, tspan )
        # solving
        sol = solve(prob, alg,)
        u1 = sol.us[end]
        # value coming from \cite{Beck2017a}
        e_l2 = rel_error_l2(u1, 0.30879)
        @test e_l2 < 0.5 # this is quite high as a relative error. 
        println("d = $d, rel_error_l2 = $e_l2")
    end
end


@testset "MLP - allen cahn local, Neumann BC" begin
    
    tspan = (0f0, 5f-1)
   
    μ(x, p, t) = 0f0 # advection coefficients
    σ(x, p, t) = 1f0 # diffusion coefficients

    for d in [1,2,5]
        neumann_bc = [fill(-5f-1, d), fill(5f-1, d)]

        alg = MLP()

        X0 = fill(0f0,d)  # initial point
        g(X) = exp.(-0.25f0 * sum(X.^2))   # initial condition
        a(u) = u - u^3
        f(y, z, v_y, v_z, ∇v_y, ∇v_z, p, t) = a.(v_y) # nonlocal nonlinear part of the

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, X0, tspan, neumann_bc = neumann_bc )
        # solving
        sol = solve(prob, alg, )
        u1 = sol.us[end]
        @test !isnan(u1)
        println("d = $d, u1 = $(u1)")
    end
end


@testset "MLP algorithm - Black-Scholes Equation with Default Risk" begin
    tspan = (0f0, 1f0)

    μ(x, p, t) = 0.02f0 * x # advection coefficients
    σ(x, p, t) = 0.2f0 * x # diffusion coefficients

    d = 20
    
    alg = MLP()

    X0 = fill(100f0,d)  # initial point
    g(X) =  minimum(X) # initial condition

    δ = 2.0f0/3f0
    R = 0.02f0

    vh = 50.0f0
    vl = 70.0f0
    γh = 0.2f0
    γl = 0.02f0

    Q(u) = (u .< vh) .* γh .+ (u .>= vl) .*  γl .+ ( (u .>= vh) .* (u .< vl)) .* (((γh - γl) / (vh - vl)) * (u .- vh) .+ γh)

    µc = 0.02f0
    σc = 0.2f0

    f(y, z, v_y, v_z, ∇v_y, ∇v_z, p, t) = -(1f0 - δ) * Q.(v_y) .* v_y .- R * v_y

    # defining the problem
    prob = PIDEProblem(g, f, μ, σ, X0, tspan)
    # solving
    sol = solve(prob, alg, )

    u1 = sol.us[end]

    analytical_ans = 60.781
    error_l2 = rel_error_l2(u1, analytical_ans)

    @test error_l2 < 0.1

    println("d = $d, error_l2 = $(error_l2)")

end

###################################################
########### NON LOCAL #############################
###################################################

@testset "MLP - replicator mutator" begin
    tspan = (0e0,5e-1)
    σ_sampling = 1e0
    μ(x, p, t) = 0e0 # advection coefficients
    σ(x, p, t) = 1e-1 #1f-1 # diffusion coefficients
    ss0 = 1e-2#std g0

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

    for d in [1, 2, 5]

        x = fill(0e0, d)

        alg = MLP(M=4, K=10, L = 4, mc_sample = NormalSampling(σ_sampling) )

        g(x) = (2*π)^(-d/2) * ss0^(- d * 5e-1) * exp.(-5e-1 *sum(x .^2e0 / ss0)) # initial condition
        m(x) = - 5e-1 * sum(x.^2)

        f(y, z, v_y, v_z, ∇v_y, ∇v_z, p, t) = max(0.0, v_y) * (m(y) - max(0.0, v_z) * m(z) * (2.0 * π)^(d/2) * σ_sampling^d * exp(0.5 * sum(z.^2) / σ_sampling^2)) # nonlocal nonlinear part of the

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, x, tspan)
        # solving
        sol = solve(prob, alg,)
        u1 = sol.us[end]
        u1_anal = uanal(x,tspan[2],Dict())
        e_l2 = mean(rel_error_l2.(u1, u1_anal))
        println("rel_error_l2 = ", e_l2, "\n")
        @test e_l2 < 0.1
    end
end


@testset "MLP- allen cahn non local Neumann BC" begin
    
    tspan = (0f0, 5f-1)

    μ(x, p, t) = 0f0 # advection coefficients
    σ(x, p, t) = 1f-1 #1f-1 # diffusion coefficients
    
    for d in [1,2,5]
        u1s = []
        for _ in 1:2
            neumann_bc = [fill(-5f-1, d), fill(5f-1, d)]

            alg = MLP(M=4, K=10, L = 4, mc_sample = UniformSampling(neumann_bc...) )

            x = fill(0f0,d)  # initial point
            g(X) = exp.(-0.25f0 * sum(X.^2))   # initial condition
            a(u) = u - u^3
            f(y,z,v_y,v_z,∇v_y,∇v_z, p, t) = a.(v_y) .- a.(v_z) #.* Float32(π^(d/2)) * σ_sampling^d .* exp.(sum(z.^2, dims = 1) / σ_sampling^2) # nonlocal nonlinear part of the

            # defining the problem
            prob = PIDEProblem(g, f, μ, σ, x, tspan, neumann_bc = neumann_bc)
            # solving
            sol = solve(prob, alg)
            push!(u1s, sol.us[end])
            println("d = $d, u1 = $(u1s[end])")

        end
        e_l2 = mean(rel_error_l2.(u1s[1], u1s[2]))
        println("rel_error_l2 = ", e_l2, "\n")
        @test e_l2 < 0.1
    end
end