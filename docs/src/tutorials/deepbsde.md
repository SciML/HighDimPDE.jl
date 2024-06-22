# Solving a 100-dimensional Hamilton-Jacobi-Bellman Equation with `DeepBSDE`

First, here's a fully working code for the solution of a 100-dimensional
Hamilton-Jacobi-Bellman equation that takes a few minutes on a laptop:

```@example deepbsde
using HighDimPDE
using Flux
using StochasticDiffEq
using LinearAlgebra

d = 100 # number of dimensions
x0 = fill(0.0f0, d)
tspan = (0.0f0, 1.0f0)
dt = 0.2f0
λ = 1.0f0
#
g(X) = log(0.5f0 + 0.5f0 * sum(X .^ 2))
f(X, u, σᵀ∇u, p, t) = -λ * sum(σᵀ∇u .^ 2)
μ_f(X, p, t) = zero(X)  #Vector d x 1 λ
σ_f(X, p, t) = Diagonal(sqrt(2.0f0) * ones(Float32, d)) #Matrix d x d
prob = ParabolicPDEProblem(μ_f, σ_f, x0, tspan; g, f)

hls = 10 + d #hidden layer size
opt = Flux.Optimise.Adam(0.1)  #optimizer
#sub-neural network approximating solutions at the desired point
u0 = Flux.Chain(Dense(d, hls, relu),
    Dense(hls, hls, relu),
    Dense(hls, hls, relu),
    Dense(hls, 1))
# sub-neural network approximating the spatial gradients at time point
σᵀ∇u = Flux.Chain(Dense(d + 1, hls, relu),
    Dense(hls, hls, relu),
    Dense(hls, hls, relu),
    Dense(hls, hls, relu),
    Dense(hls, d))
pdealg = DeepBSDE(u0, σᵀ∇u, opt = opt)

@time sol = solve(prob,
    pdealg,
    StochasticDiffEq.EM(),
    verbose = true,
    maxiters = 150,
    trajectories = 30,
    dt = 1.2f0,
    pabstol = 1.0f-4)

```

Now, let's explain the details!

### Hamilton-Jacobi-Bellman Equation

The Hamilton-Jacobi-Bellman equation is the solution to a stochastic optimal
control problem.

#### Symbolic Solution

Here, we choose to solve the classical Linear Quadratic Gaussian
(LQG) control problem of 100 dimensions, which is governed by the SDE

```math
d X_t = 2 \sqrt{\lambda} c_t dt + \sqrt{2}dW_t
```
where $c_t$ is a control process. The solution to the optimal control is given by a PDE of the form:

```math
\partial_t u(t,x) + \Delta_x u + \lambda \| \Nabla u (t,x)\|^2 = 0 
```

with terminating condition $g(x) = \log(1/2 + 1/2 \|x\|^2))$.

### Solving LQG Problem with Neural Net

#### Define the Problem

To get the solution above using the [`ParabolicPDEProblem`](@ref), we write:

```@example deepbsde2
using HighDimPDE
using Flux
using StochasticDiffEq
using LinearAlgebra

d = 100 # number of dimensions
X0 = fill(0.0f0,d) # initial value of stochastic control process
tspan = (0.0f0, 1.0f0)
λ = 1.0f0

g(X) = log(0.5f0 + 0.5f0*sum(X.^2))
f(X,u,σᵀ∇u,p,t) = -λ*sum(σᵀ∇u.^2)
μ_f(X,p,t) = zero(X)  #Vector d x 1 λ
σ_f(X,p,t) = Diagonal(sqrt(2.0f0)*ones(Float32,d)) #Matrix d x d
prob = ParabolicPDEProblem(μ_f, σ_f, X0, tspan; g, f)
```

#### Define the Solver Algorithm

As described in the API docs, we now need to define our `DeepBSDE` algorithm
by giving it the Flux.jl chains we want it to use for the neural networks.
`u0` needs to be a `d` dimensional -> 1 dimensional chain, while `σᵀ∇u`
needs to be `d+1` dimensional to `d` dimensions. Thus we define the following:

```@example deepbsde2
hls = 10 + d #hidden layer size
opt = Flux.Optimise.Adam(0.01)  #optimizer
#sub-neural network approximating solutions at the desired point
u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))
# sub-neural network approximating the spatial gradients at time point
σᵀ∇u = Flux.Chain(Dense(d+1,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))
pdealg = DeepBSDE(u0, σᵀ∇u, opt = opt)
```

#### Solving with Neural Nets

```@example deepbsde2
@time ans = solve(prob, pdealg, EM(), verbose=true, maxiters=100, trajectories=100, dt=0.2f0, pabstol = 1f-2)

```

Here we want to solve the underlying neural
SDE using the Euler-Maruyama SDE solver with our chosen `dt=0.2`, do at most
100 iterations of the optimizer, 100 SDE solves per loss evaluation (for averaging),
and stop if the loss ever goes below `1f-2`.

## Solving the 100-dimensional Black-Scholes-Barenblatt Equation

Black Scholes equation is a model for stock option price.
In 1973, Black and Scholes transformed their formula on option pricing and corporate liabilities into a PDE model, which is widely used in financing engineering for computing the option price over time. [1.]
In this example, we will solve a Black-Scholes-Barenblatt equation of 100 dimensions.
The Black-Scholes-Barenblatt equation is a nonlinear extension to the Black-Scholes
equation, which models uncertain volatility and interest rates derived from the
Black-Scholes equation. This model results in a nonlinear PDE whose dimension
is the number of assets in the portfolio.

To solve it using the `ParabolicPDEProblem`, we write:

```julia
d = 100 # number of dimensions
X0 = repeat([1.0f0, 0.5f0], div(d,2)) # initial value of stochastic state
tspan = (0.0f0,1.0f0)
r = 0.05f0
sigma = 0.4f0
f(X,u,σᵀ∇u,p,t) = r * (u - sum(X.*σᵀ∇u))
g(X) = sum(X.^2)
μ_f(X,p,t) = zero(X) #Vector d x 1
σ_f(X,p,t) = Diagonal(sigma*X) #Matrix d x d
prob = ParabolicPDEProblem(μ_f, σ_f, X0, tspan; g, f)
```

As described in the API docs, we now need to define our `NNPDENS` algorithm
by giving it the Flux.jl chains we want it to use for the neural networks.
`u0` needs to be a `d`-dimensional -> 1-dimensional chain, while `σᵀ∇u`
needs to be `d+1`-dimensional to `d` dimensions. Thus we define the following:

```julia
hls  = 10 + d #hide layer size
opt = Flux.Optimise.Adam(0.001)
u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))
σᵀ∇u = Flux.Chain(Dense(d+1,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))
pdealg = DeepBSDE(u0, σᵀ∇u, opt=opt)
```

And now we solve the PDE. Here, we say we want to solve the underlying neural
SDE using the Euler-Maruyama SDE solver with our chosen `dt=0.2`, do at most
150 iterations of the optimizer, 100 SDE solves per loss evaluation (for averaging),
and stop if the loss ever goes below `1f-6`.

```julia
ans = solve(prob, pdealg, EM(), verbose=true, maxiters=150, trajectories=100, dt=0.2f0)
```

## References

1. Shinde, A. S., and K. C. Takale. "Study of Black-Scholes model and its applications." Procedia Engineering 38 (2012): 270-279.
