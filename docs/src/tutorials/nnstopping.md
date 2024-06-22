# `NNStopping`

## Solving for optimal strategy and expected payoff of a Bermudan Max-Call option

We will calculate optimal strategy for Bermudan Max-Call option with following drift, diffusion and payoff:
```math
μ(x) =(r − δ) x, σ(x) = β diag(x1, ... , xd),\\
g(t, x) =  e^{-rt}max\lbrace max\lbrace x1, ... , xd \rbrace − K, 0\rbrace
```
We define the parameters, drift function and the diffusion function for the dynamics of the option.
```@example nnstopping
using HighDimPDE, Flux, StochasticDiffEq
d = 3 # Number of assets in the stock
r = 0.05 # interest rate
beta = 0.2 # volatility
T = 3.0 # maturity
u0 = fill(90.0, d) # initial stock value
delta = 0.1 # delta
f(du, u, p, t) = du .= (r - delta) * u # drift
sigma(du, u, p, t) = du .= beta * u # diffusion
tspan = (0.0, T)
N = 9 # discretization parameter
dt = T / (N)
K = 100.00 # strike price

# payoff function
function g(x, t)
    return exp(-r * t) * (max(maximum(x) - K, 0))
end

```
We then define a [`ParabolicPDEProblem`](@ref) with no non linear term:

```@example nnstopping
prob = ParabolicPDEProblem(f, sigma, u0, tspan; payoff = g)
```
!!! note 
    We provide the payoff function with a keyword argument `payoff` 

And now we define our models:
```@example nnstopping
models = [Chain(Dense(d + 1, 32, tanh), BatchNorm(32, tanh), Dense(32, 1, sigmoid))
          for i in 1:N]
```
!!! note 
    The number of models should be equal to the time discritization.

And finally we define our optimizer and algorithm, and call `solve`:
```@example nnstopping
opt = Flux.Optimisers.Adam(0.01)
alg = NNStopping(models, opt)

sol = solve(prob, alg, SRIW1(); dt = dt, trajectories = 1000, maxiters = 1000, verbose = true)
```
