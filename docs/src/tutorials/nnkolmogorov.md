# `NNKolmogorov`

## Solving high dimensional Rainbow European Options for a range of initial stock prices:

```julia
d = 10 # dims
T = 1/12 
sigma = 0.01 .+ 0.03.*Matrix(Diagonal(ones(d))) # volatility 
mu = 0.06 # interest rate
K = 100.0 # strike price
function μ_func(du, u, p, t)
    du .= mu*u
end

function σ_func(du, u, p, t)
    du .= sigma * u
end

tspan = (0.0, T)
# The range for initial stock price
xspan = [(98.00, 102.00) for i in 1:d]

g(x) = max(maximum(x) -K, 0)

sdealg = EM()
# provide `x0` as nothing to the problem since we are provinding a range for `x0`.
prob = ParabolicPDEProblem(μ_func, σ_func, nothing, tspan, g = g, xspan = xspan)
opt = Flux.Optimisers.Adam(0.01)
alg = NNKolmogorov(m, opt)
m = Chain(Dense(d, 16, elu), Dense(16, 32, elu), Dense(32, 16, elu), Dense(16, 1))
sol = solve(prob, alg, sdealg, verbose = true, dt = 0.01,
    dx = 0.0001, trajectories = 1000, abstol = 1e-6, maxiters = 300)
```
