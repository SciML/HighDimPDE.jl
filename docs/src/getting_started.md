
# Getting started
The general workflow for using `HighDimPDE.jl` is as follows:

- Define a Partial Integro Differential Equation problem
```@autodocs
Modules = [HighDimPDE]
Pages   = ["HighDimPDE.jl"]
```
- Select a solver algorithm
- Solve the problem

Let's solve the [Fisher KPP](https://en.wikipedia.org/wiki/Fisher%27s_equation) PDE in dimension 10.
## `MLP` (@ref mlp)
```math
\partial_t u = u (1 - u) + \frac{1}{2}\sigma^2\Delta_xu \tag{1}
```

```julia
using HighDimPDE

## Definition of the problem
d = 10 # dimension of the problem
tspan = (0.0,0.5) # time horizon
x0 = fill(0.,d)  # initial point
g(x) = exp(- sum(x.^2) ) # initial condition
μ(x, p, t) = 0.0 # advection coefficients
σ(x, p, t) = 0.1 # diffusion coefficients
f(x, y, v_x, v_y, ∇v_x, ∇v_y, p, t) = max(0.0, v_x) * (1 -  max(0.0, v_x)) # nonlinear part of the PDE
prob = PIDEProblem(g, f, μ, σ, x0, tspan) # defining the problem

## Definition of the algorithm
alg = MLP() # defining the algorithm. We use the Multi Level Picard algorithm

## Solving with multiple threads 
sol = solve(prob, alg, multithreading=true)
```

## `DeepSplitting`(@ref deepsplitting)
```julia
using HighDimPDE

## Definition of the problem
d = 10 # dimension of the problem
tspan = (0.0, 0.5) # time horizon
x0 = fill(0.,d)  # initial point
g(x) = exp.(- sum(x.^2, dims=1) ) # initial condition
μ(x, p, t) = 0.0 # advection coefficients
σ(x, p, t) = 0.1 # diffusion coefficients
u_domain = [-1/2, 1/2]
f(x, y, v_x, v_y, ∇v_x, ∇v_y, t) = max.(0f0, v_x) .* (1f0 .-  max.(0f0, v_x)) 
prob = PIDEProblem(g, f, μ, 
                    σ, x0, tspan, 
                    u_domain = u_domain)

## Definition of the neural network to use
using Flux # needed to define the neural network

hls = d + 50 #hidden layer size

nn = Flux.Chain(Dense(d, hls, tanh),
        Dense(hls, hls, tanh),
        Dense(hls, 1)) # neural network used by the scheme

opt = Flux.Optimiser(ExpDecay(0.1,
                0.1,
                200,
                1e-4),
                ADAM() )#optimiser

## Definition of the algorithm
alg = DeepSplitting(nn, opt = opt)

sol = solve(prob, 
            alg, 
            dt=0.1, 
            verbose = true, 
            abstol = 2e-3,
            maxiters = 1000,
            batch_size = 1000)
```
## Solving on the GPU
`DeepSplitting` can run on the GPU for (much) improved performance. To do so, just set `use_cuda = true`.

```julia
sol = solve(prob, 
            alg, 
            dt=0.1, 
            verbose = true, 
            abstol = 2e-3,
            maxiters = 1000,
            batch_size = 1000,
            use_cuda=true)
```