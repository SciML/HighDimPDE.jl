using HighDimPDE
using Random
using Test

tspan = (0.0,0.5)
σ_sampling = 0.1
# using the MLP alg
alg = MLP(M=4, K=10, L = 4 )
μ(X,p,t) = 0.0 # advection coefficients
σ(X,p,t) = 0.1 # diffusion coefficients


anal_res = [1.398, 1.9567, 5.3554]
ds = [1,2,5]
atols = [5e-2,1e-1,5e-1]

@testset "MLP algorithm - singel threaded" begin
        for i in 1:length(ds)
        d = ds[i]
        X0 = fill(0.,d)  # initial point
        g(X) = 2.0^(d/2)* exp(-2. * π  * sum( X.^2))   # initial condition
        m(x) = - 0.5 * sum(x.^2)
        f(y, z, v_y, v_z, ∇v_y, ∇v_z, p, t) = max(0.0, v_y) * (m(y) - max(0.0, v_z) * m(z) * (2.0 * π)^(d/2) * σ_sampling^d * exp(0.5 * sum(z.^2) / σ_sampling^2)) # nonlocal nonlinear part of the
        mc_sample(x) = randn(d) * σ_sampling #montecarlo samples

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, X0, tspan, 
                            # u_domain=[-1f0,1f0]
                            )
        # solving
        @time sol = solve(prob, alg,mc_sample, verbose = false)
        @test isapprox(sol, anal_res[i],atol = atols[i])
    end
end