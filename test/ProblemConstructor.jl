using HighDimPDE

@testset "PIDEs" begin
    for d in [1, 10]
        x0 = fill(8.0f0, d) # initial points
        tspan = (0.0f0, 5.0f0)

        μ_f(X, p, t) = zero(X) #Vector d x 1
        σ_f(X, p, t) = Diagonal(ones(Float32, d)) |> Matrix
        g(X) = sum(X .^ 2)   # terminal condition
        f_nonlinear(y, z, v_y, v_z, ∇v_y, ∇v_z, p, t) = 0.0f0 .* v_y

        prob = PIDEProblem(μ_f, σ_f, x0, tspan, g, f_nonlinear)

        @test prob.f == f_nonlinear
        @test prob.g == g
        @test prob.μ == μ_f
        @test prob.σ == σ_f
        @test prob.x == x0
        @test prob.u0 == g(x0)
        @test prob.p == nothing
        @test prob.tspan == tspan

        @testset "With neumann_bc" begin
            neumann_bc = [-1 * fill(5.0f-1, d), fill(5.0f-1, d)]
            prob = PIDEProblem(μ_f, σ_f, x0, tspan, g, f_nonlinear; neumann_bc)

            @test prob.f == f_nonlinear
            @test prob.g == g
            @test prob.μ == μ_f
            @test prob.σ == σ_f
            @test prob.x == x0
            @test prob.u0 == g(x0)
            @test prob.p == nothing
            @test prob.tspan == tspan
            @test prob.neumann_bc == neumann_bc
        end

        @testset "With x0_sample" begin
            x0_sample = UniformSampling(fill(-5.0f-1, d), fill(5.0f-1, d))

            prob = PIDEProblem(μ_f, σ_f, x0, tspan, g, f_nonlinear; x0_sample)

            @test prob.f == f_nonlinear
            @test prob.g == g
            @test prob.μ == μ_f
            @test prob.σ == σ_f
            @test prob.x == x0
            @test prob.u0 == g(x0)
            @test prob.p == nothing
            @test prob.tspan == tspan
            @test prob.x0_sample == x0_sample
        end

        @testset "With x0_sample and neumann_bc" begin
            x0_sample = UniformSampling(fill(-5.0f-1, d), fill(5.0f-1, d))
            neumann_bc = [-1 * fill(5.0f-1, d), fill(5.0f-1, d)]

            prob = PIDEProblem(μ_f, σ_f, x0, tspan, g, f_nonlinear; x0_sample, neumann_bc)

            @test prob.f == f_nonlinear
            @test prob.g == g
            @test prob.μ == μ_f
            @test prob.σ == σ_f
            @test prob.x == x0
            @test prob.u0 == g(x0)
            @test prob.p == nothing
            @test prob.tspan == tspan
            @test prob.x0_sample == x0_sample
            @test prob.neumann_bc == neumann_bc
        end
    end
end

@testset "Semilinear Parabolic PDEs" begin
    for d in [1, 10]
        x0 = fill(8.0f0, d) # initial points
        tspan = (0.0f0, 5.0f0)

        μ_f(X, p, t) = zero(X) #Vector d x 1
        σ_f(X, p, t) = Diagonal(ones(Float32, d)) |> Matrix
        g(X) = sum(X .^ 2)   # terminal condition
        f_semilinear(X, u, σᵀ∇u, p, t) = Float32(0.0)

        prob = ParabolicPDEProblem(μ_f, σ_f, x0, tspan; g, f = f_semilinear)

        @test prob.f == f_semilinear
        @test prob.g == g
        @test prob.μ == μ_f
        @test prob.σ == σ_f
        @test prob.x == x0
        @test prob.u0 == g(x0)
        @test prob.p == nothing
        @test prob.tspan == tspan
    end
end

@testset "Obstacle PDEs : Optimal Stopping Problems" begin
    for d in [1, 10]
        r = 0.05
        beta = 0.2
        T = 3.0
        u0 = fill(90.0, d)
        delta = 0.1
        mu(du, u, p, t) = du .= (r - delta) * u
        sigma(du, u, p, t) = du .= beta * u
        tspan = (0.0, T)
        N = 9
        dt = T / (N)
        K = 100.0
        function payoff(x, t)
            return exp(-r * t) * (max(maximum(x) - K, 0))
        end

        prob = ParabolicPDEProblem(mu, sigma, u0, tspan; payoff = payoff)

        @test prob.f == nothing
        @test prob.g == nothing
        @test prob.μ == mu
        @test prob.σ == sigma
        @test prob.x == u0
        @test prob.u0 == payoff(u0, 0.0)
        @test prob.p == nothing
        @test prob.tspan == tspan
        @test prob.kwargs.payoff == payoff
    end
end

@testset "Kolmogorov PDEs" begin
    for d in [1, 10]
        xspan = d == 1 ? (-6.0, 6.0) : [(-6.0, 6.0) for _ in 1:d]
        tspan = (0.0, 1.0)
        σ(u, p, t) = 0.5 * u
        μ(u, p, t) = 0.5 * 0.25 * u

        function g(x)
            1.77 .* x .- 0.015 .* x .^ 3
        end

        prob = ParabolicPDEProblem(μ, σ, nothing, tspan; g, xspan)

        @test prob.f == nothing
        @test prob.g == g
        @test prob.μ == μ
        @test prob.σ == σ

        if d == 1
            @test prob.u0 == g([xspan[1]])
            @test prob.x == [xspan[1]]
        else
            @test prob.u0 == g(first.(xspan))
            @test prob.x == first.(xspan)
        end
        @test prob.p == nothing
        @test prob.tspan == tspan
        @test prob.kwargs.xspan == xspan
    end
end

@testset "Kolmogorov PDEs - Parametric Family" begin
    for d in [1, 10]
        γ_mu_prototype = nothing
        γ_sigma_prototype = zeros(d, d, 1)
        γ_phi_prototype = nothing

        tspan = (0.0, 1.0)
        xspan = d == 1 ? (0.0, 3.0) : [(0.0, 3.0) for _ in 1:d]

        function g(x, p_phi)
            x .^ 2
        end

        sigma(dx, x, p_sigma, t) = dx .= p_sigma[:, :, 1]
        mu(dx, x, p_mu, t) = dx .= 0.0

        p_domain = (p_sigma = (0.0, 2.0), p_mu = nothing, p_phi = nothing)
        p_prototype = (
            p_sigma = γ_sigma_prototype,
            p_mu = γ_mu_prototype,
            p_phi = γ_phi_prototype,
        )

        prob = ParabolicPDEProblem(
            mu, sigma, nothing, tspan; g,
            xspan,
            p_domain = p_domain,
            p_prototype = p_prototype
        )

        @test prob.f == nothing
        @test prob.g == g
        @test prob.μ == mu
        @test prob.σ == sigma

        if d == 1
            @test prob.u0 == g([xspan[1]], p_prototype.p_phi)
            @test prob.x == [xspan[1]]
        else
            @test prob.u0 == g(first.(xspan), p_prototype.p_phi)
            @test prob.x == first.(xspan)
        end

        @test prob.p == nothing
        @test prob.tspan == tspan
        @test prob.kwargs.xspan == xspan
        @test prob.kwargs.p_domain == p_domain
        @test prob.kwargs.p_prototype == p_prototype
    end
end
