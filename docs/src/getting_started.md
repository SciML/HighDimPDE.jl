# Getting started

## General workflow

The general workflow for using `HighDimPDE.jl` is as follows:

- Define a Partial Integro-Differential Equation problem
- Select a solver algorithm
- Solve the problem.

## Examples

Let's illustrate that with some examples.

### MLP

#### Local PDE

Let's solve the [Fisher KPP](https://en.wikipedia.org/wiki/Fisher%27s_equation) PDE in dimension 10 with [`MLP`](@ref mlp).

```math
\partial_t u = u (1 - u) + \frac{1}{2}\sigma^2\Delta_xu \tag{1}
```

```@example MLP_local_PDE
using HighDimPDE

## Definition of the problem
d = 10 # dimension of the problem
tspan = (0.0, 0.5) # time horizon
x0 = fill(0., d)  # initial point
g(x) = exp(-sum(x.^2)) # initial condition
μ(x, p, t) = 0.0 # advection coefficients
σ(x, p, t) = 0.1 # diffusion coefficients
f(x, y, v_x, v_y, ∇v_x, ∇v_y, p, t) = max(0.0, v_x) * (1 -  max(0.0, v_x)) # nonlocal nonlinear part of the
prob = PIDEProblem(g, f, μ, σ, x0, tspan) # defining the problem

## Definition of the algorithm
alg = MLP() # defining the algorithm. We use the Multi Level Picard algorithm

## Solving with multiple threads 
sol = solve(prob, alg, multithreading = true)
```

#### Non-local PDE with Neumann boundary conditions

Let's include in the previous equation non-local competition, i.e.
```math
\partial_t u = u (1 - \int_\Omega u(t,y)dy) + \frac{1}{2}\sigma^2\Delta_xu \tag{2}
```
where $\Omega = [-1/2, 1/2]^d$, and let's assume Neumann Boundary condition on $\Omega$.

```@example MLP_non_local_PDE
using HighDimPDE

## Definition of the problem
d = 10 # dimension of the problem
tspan = (0.0,0.5) # time horizon
x0 = fill(0.,d)  # initial point
g(x) = exp( -sum(x.^2) ) # initial condition
μ(x, p, t) = 0.0 # advection coefficients
σ(x, p, t) = 0.1 # diffusion coefficients
mc_sample = UniformSampling(fill(-5f-1, d), fill(5f-1, d))
f(x, y, v_x, v_y, ∇v_x, ∇v_y, p, t) = max(0.0, v_x) * (1 -  max(0.0, v_y)) 
prob = PIDEProblem(g, f, μ, σ, x0, tspan) # defining x0_sample is sufficient to implement Neumann boundary conditions

## Definition of the algorithm
alg = MLP(mc_sample = mc_sample)

sol = solve(prob, alg, multithreading = true)
```

### DeepSplitting

Let's solve the previous equation with [`DeepSplitting`](@ref deepsplitting).

```@example DeepSplitting_non_local_PDE
using HighDimPDE
using Flux # needed to define the neural network

## Definition of the problem
d = 10 # dimension of the problem
tspan = (0.0, 0.5) # time horizon
x0 = fill(0f0, d)  # initial point
g(x) = exp.(-sum(x.^2, dims=1)) # initial condition
μ(x, p, t) = 0.0f0 # advection coefficients
σ(x, p, t) = 0.1f0 # diffusion coefficients
x0_sample = UniformSampling(fill(-5f-1, d), fill(5f-1, d))
f(x, y, v_x, v_y, ∇v_x, ∇v_y, p, t) = v_x .* (1f0 .- v_y)
prob = PIDEProblem(g, f, μ, σ, x0, tspan,
                   x0_sample = x0_sample)

## Definition of the neural network to use
hls = d + 50 #hidden layer size

nn = Flux.Chain(Dense(d, hls, tanh),
                Dense(hls, hls, tanh),
                Dense(hls, 1)) # neural network used by the scheme

opt = ADAM(1e-2)

## Definition of the algorithm
alg = DeepSplitting(nn,
                    opt = opt,
                    mc_sample = x0_sample)

sol = solve(prob, 
            alg, 
            0.1, 
            verbose = true, 
            abstol = 2e-3,
            maxiters = 1000,
            batch_size = 1000)
```

### Solving on the GPU

[`DeepSplitting`](@ref deepsplitting) can run on the GPU for (much) improved performance. To do so, just set `use_cuda = true`.

```julia
sol = solve(prob, 
            alg, 
            0.1, 
            verbose = true, 
            abstol = 2e-3,
            maxiters = 1000,
            batch_size = 1000,
            use_cuda=true)
```