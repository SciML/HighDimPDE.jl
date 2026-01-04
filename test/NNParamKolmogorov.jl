using Test, Flux
using StochasticDiffEq
using LinearAlgebra
using HighDimPDE
using Random
Random.seed!(100)

d = 1
m = Chain(Dense(3, 16, tanh), Dense(16, 16, tanh), Dense(16, 5, tanh), Dense(5, 1))
ensemblealg = EnsembleThreads()
γ_mu_prototype = nothing
γ_sigma_prototype = zeros(d, d, 1)
γ_phi_prototype = nothing

sdealg = EM()
tspan = (0.0, 1.0)
trajectories = 10000
function phi(x, y_phi)
    return x .^ 2
end
sigma(dx, x, γ_sigma, t) = dx .= γ_sigma[:, :, 1]
mu(dx, x, γ_mu, t) = dx .= 0.0

xspan = (0.0, 3.0)

p_domain = (p_sigma = (0.0, 2.0), p_mu = nothing, p_phi = nothing)
p_prototype = (p_sigma = γ_sigma_prototype, p_mu = γ_mu_prototype, p_phi = γ_phi_prototype)
dps = (p_sigma = 0.01, p_mu = nothing, p_phi = nothing)

dt = 0.01
dx = 0.01
opt = Flux.Optimisers.Adam(1.0e-2)

prob = ParabolicPDEProblem(
    mu,
    sigma,
    nothing,
    tspan;
    g = phi,
    xspan,
    p_domain = p_domain,
    p_prototype = p_prototype
)

sol = solve(
    prob, NNParamKolmogorov(m, opt), sdealg, verbose = true, dt = 0.01,
    abstol = 1.0e-10, dx = 0.01, trajectories = trajectories, maxiters = 1000,
    use_gpu = false, dps = dps
)

x_test = rand(xspan[1]:dx:xspan[2], d, 1, 1000)
t_test = rand(tspan[1]:dt:tspan[2], 1, 1000)
γ_sigma_test = rand(0.3:(dps.p_sigma):0.5, d, d, 1, 1000)

function analytical(x, t, y)
    return x .^ 2 .+ t .* (y .* y)
end

preds = map(
    (i) -> sol.ufuns(
        x_test[:, :, i],
        t_test[:, i],
        γ_sigma_test[:, :, :, i],
        nothing,
        nothing
    ),
    1:1000
)
y_test = map(
    (i) -> analytical(x_test[:, :, i], t_test[:, i], γ_sigma_test[:, :, :, i]),
    1:1000
)

@test Flux.mse(reduce(hcat, preds), reduce(hcat, y_test)) < 0.1
