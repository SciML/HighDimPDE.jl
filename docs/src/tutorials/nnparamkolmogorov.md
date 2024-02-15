# `NNParamKolmogorov`

## Solving Parametric Family of High Dimensional Heat Equation.

In this example we will solve the high dimensional heat equation over a range of initial values, and also over a range of thermal diffusivity.
```julia
d = 10
# models input is `d` for initial values, `d` for thermal diffusivity, and last dimension is for stopping time.
m = Chain(Dense(d + 1 + 1, 32, relu), Dense(32, 16, relu), Dense(16, 8, relu), Dense(8, 1))
ensemblealg = EnsembleThreads()
γ_mu_prototype = nothing
γ_sigma_prototype = zeros(1, 1)
γ_phi_prototype = nothing

sdealg = EM()
tspan = (0.00, 1.00)
trajectories = 100000
function phi(x, y_phi)
    sum(x .^ 2)
end
function sigma_(dx, x, γ_sigma, t)
     dx .= γ_sigma[:, :, 1]
end
mu_(dx, x, γ_mu, t) = dx .= 0.00

xspan = [(0.00, 3.00) for i in 1:d]

p_domain = (p_sigma = (0.00, 2.00), p_mu = nothing, p_phi = nothing)
p_prototype = (p_sigma = γ_sigma_prototype, p_mu = γ_mu_prototype, p_phi = γ_phi_prototype)
dps = (p_sigma = 0.1, p_mu = nothing, p_phi = nothing)

dt = 0.01
dx = 0.01
opt = Flux.Optimisers.Adam(5e-2)

prob = ParabolicPDEProblem(mu_,
    sigma_,
    nothing,
    tspan;
    g = phi,
    xspan,
    p_domain = p_domain,
    p_prototype = p_prototype)

sol = solve(prob, NNParamKolmogorov(m, opt), sdealg, verbose = true, dt = 0.01,
    abstol = 1e-10, dx = 0.1, trajectories = trajectories, maxiters = 1000,
    use_gpu = false, dps = dps)
```
Similarly we can parametrize the drift function `mu_` and the initial function `g`, and obtain a solution over all parameters and initial values.

# Inferring on the solution from `NNParamKolmogorov`:
```julia
x_test = rand(xspan[1][1]:0.1:xspan[1][2], d)
p_sigma_test = rand(p_domain.p_sigma[1]:dps.p_sigma:p_domain.p_sigma[2], 1, 1)
t_test = rand(tspan[1]:dt:tspan[2], 1, 1)
p_mu_test = nothing
p_phi_test = nothing
```
```julia
sol.ufuns(x_test, t_test, p_sigma_test, p_mu_test, p_phi_test)
```