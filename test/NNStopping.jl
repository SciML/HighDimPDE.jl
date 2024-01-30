using Test, Flux, StochasticDiffEq, LinearAlgebra
println("Optimal Stopping Time Test")
using HighDimPDE
using Statistics

using Random
Random.seed!(101)
# Bermudan Max-Call Standard Example
d = 3
r = 0.05
beta = 0.2
T = 3
u0 = fill(90.0, d)
delta = 0.1
f(du, u, p, t) = du .= (r - delta) * u
sigma(du, u, p, t) = du .= beta * u
tspan = (0.0, T)
N = 9
dt = T / (N)
K = 100.00
function g(x, t)
    return exp(-r * t) * (max(maximum(x) - K, 0))
end

prob = PIDEProblem(f, sigma, u0, tspan; payoff = g)
models = [Chain(Dense(d + 1, 32, tanh), BatchNorm(32, tanh), Dense(32, 1, sigmoid))
          for i in 1:N]
opt = Flux.Optimisers.Adam(0.01)
alg = NNStopping(models, opt)
sol = solve(prob,
    alg,
    SRIW1();
    dt = dt,
    trajectories = 1000,
    maxiters = 1000,
    verbose = true)
analytical_sol = 11.278 # Ref [1]
@test abs(analytical_sol - sol.payoff) < 0.5

# Put basket option in Dupire’s local volatility model
d = 5
r = 0.05
beta = 0.2
T = 1.0
u0 = fill(100.0, d)
delta = 0.1
f(u, p, t) = (r - delta) * u

function B(t, x)
    x * 0.6 * exp(-0.05 * sqrt(t)) *
    (1.2 - exp(-0.1 * t - 0.001 * (exp(-r * t * x - first(u0))^2)))
end
sigma(u, p, t) = B.(Ref(t), u)

tspan = (0.0f0, T)
N = 10
dt = T / (N)
K = 100.00
function g(x, t)
    return exp(-r * t) * (max(K - mean(x), 0))
end
prob = PIDEProblem(f, sigma, u0, tspan; payoff = g)
models = [Chain(Dense(d + 1, 32, tanh), BatchNorm(32, tanh), Dense(32, 1, sigmoid))
          for i in 1:N]
opt = Flux.Optimisers.Adam(0.01)
alg = NNStopping(models, opt)
sol = solve(prob,
    alg,
    SRIW1();
    dt = dt,
    trajectories = 1000,
    maxiters = 500,
    verbose = true)
analytical_sol = 6.301 # Ref [2]
@test abs(analytical_sol - sol.payoff) < 0.5

### References for Analytical Payoffs:
#=
[1] Section 4.4.1: Solving high-dimensional optimal stopping problems using deep learning (https://arxiv.org/pdf/1908.01602.pdf) 
[2] Section 4.4.3: Solving high-dimensional optimal stopping problems using deep learning (https://arxiv.org/pdf/1908.01602.pdf) 
=#
