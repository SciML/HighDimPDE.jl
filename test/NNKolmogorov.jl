using Test, Flux, StochasticDiffEq
using HighDimPDE
using Distributions

using Random
Random.seed!(100)

# For a diract delta take u0 = Normal(0 , sigma) where sigma --> 0
u0 = Normal(1.00, 1.00)
xspan = (-2.0, 6.0)
tspan = (0.0, 1.0)
σ(u, p, t) = 2.00
μ(u, p, t) = -2.00

d = 1
sdealg = EM()
g(x) = pdf(u0, x)
prob = PIDEProblem(g, μ, σ, tspan, xspan, d)
opt = Flux.ADAM(0.01)
m = Chain(Dense(1, 5, elu), Dense(5, 5, elu), Dense(5, 5, elu), Dense(5, 1))
ensemblealg = EnsembleThreads()
sol = solve(prob, NNKolmogorov(m, opt), sdealg; ensemblealg = ensemblealg, verbose = true,
    dt = 0.01,
    abstol = 1e-10, dx = 0.0001, trajectories = 100000, maxiters = 500)

## The solution is obtained taking the Fourier Transform.
analytical(xi) = pdf.(Normal(3, sqrt(1.0 + 5.00)), xi)
##Validation
xs = -5:0.00001:5
x_1 = rand(xs, 1, 1000)
err_l2 = Flux.mse(analytical(x_1), sol.ufuns(x_1))
@test err_l2 < 0.01

xspan = (-6.0, 6.0)
tspan = (0.0, 1.0)
σ(u, p, t) = 0.5 * u
μ(u, p, t) = 0.5 * 0.25 * u
d = 1
function g(x)
    1.77 .* x .- 0.015 .* x .^ 3
end

sdealg = EM()
prob = PIDEProblem(g, μ, σ, tspan, xspan, d)
opt = Flux.ADAM(0.01)
m = Chain(Dense(1, 16, elu), Dense(16, 32, elu), Dense(32, 16, elu), Dense(16, 1))
sol = solve(prob, NNKolmogorov(m, opt), sdealg, verbose = true, dt = 0.01,
    dx = 0.0001, trajectories = 1000, abstol = 1e-6, maxiters = 300)

function analytical(xi)
    y = Float64[]
    a = 1.77 * exp(0.5 * (0.5)^2 * 1.0)
    b = -0.015 * exp(0.5 * (0.5 * 3)^2 * 1.0)
    for x in xi
        y = push!(y, a * x + b * x^3)
    end
    y = reshape(y, size(xi)[1], size(xi)[2])
    return y
end
xs = -5.00:0.01:5.00
x_val = rand(xs, d, 50)
errorl2 = Flux.mse(analytical(x_val), sol.ufuns(x_val))
println("error_l2 = ", errorl2, "\n")
@test errorl2 < 0.4

##Non-Diagonal Test
μ_noise = (du, u, p, t) -> du .= 1.01u
σ_noise = function (du, u, p, t)
    du[1, 1] = 0.3u[1]
    du[1, 2] = 0.6u[1]
    du[1, 3] = 0.9u[1]
    du[1, 4] = 0.12u[2]
    du[2, 1] = 1.2u[1]
    du[2, 2] = 0.2u[2]
    du[2, 3] = 0.3u[2]
    du[2, 4] = 1.8u[2]
end
Σ = [1.0 0.3; 0.3 1.0]
uo3 = MvNormal([0.0; 0.0], Σ)
g(x) = pdf(uo3, x)

sdealg = EM()
xspan = (-10.0, 10.0)
tspan = (0.0, 1.0)
d = 2
prob = PIDEProblem(g, μ_noise, σ_noise, tspan, xspan, d; noise_rate_prototype = zeros(2, 4))
opt = Flux.ADAM(0.01)
m = Chain(Dense(d, 32, elu), Dense(32, 64, elu), Dense(64, 1))
sol = solve(prob, NNKolmogorov(m, opt), sdealg, verbose = true, dt = 0.001,
    abstol = 1e-6, dx = 0.001, trajectories = 1000, maxiters = 200)
println("Non-Diagonal test working.")
