# Solving the 10-dimensional Fisher-KPP equation with `MLP`

Let's solve the [Fisher KPP](https://en.wikipedia.org/wiki/Fisher%27s_equation) PDE in dimension 10 with [`MLP`](@ref mlp).

```math
\partial_t u = u (1 - u) + \frac{1}{2}\sigma^2\Delta_xu \tag{1}
```

```@example mlp
using HighDimPDE

## Definition of the problem
d = 10 # dimension of the problem
tspan = (0.0, 0.5) # time horizon
x0 = fill(0.0, d)  # initial point
g(x) = exp(-sum(x .^ 2)) # initial condition
μ(x, p, t) = 0.0 # advection coefficients
σ(x, p, t) = 0.1 # diffusion coefficients
f(x, v_x, ∇v_x, p, t) = max(0.0, v_x) * (1 - max(0.0, v_x)) # nonlocal nonlinear part of the
prob = ParabolicPDEProblem(μ, σ, x0, tspan; g, f) # defining the problem

## Definition of the algorithm
alg = MLP() # defining the algorithm. We use the Multi Level Picard algorithm

## Solving with multiple threads 
sol = solve(prob, alg, multithreading = true)
```

## Non local PDE with Neumann boundary conditions

Let's include in the previous equation non local competition, i.e.

```math
\partial_t u = u (1 - \int_\Omega u(t,y)dy) + \frac{1}{2}\sigma^2\Delta_xu \tag{2}
```

where $\Omega = [-1/2, 1/2]^d$, and let's assume Neumann Boundary condition on $\Omega$.

```@example mlp2
using HighDimPDE

## Definition of the problem
d = 10 # dimension of the problem
tspan = (0.0, 0.5) # time horizon
x0 = fill(0.0, d)  # initial point
g(x) = exp(-sum(x .^ 2)) # initial condition
μ(x, p, t) = 0.0 # advection coefficients
σ(x, p, t) = 0.1 # diffusion coefficients
mc_sample = UniformSampling(fill(-5.0f-1, d), fill(5.0f-1, d))
f(x, y, v_x, v_y, ∇v_x, ∇v_y, p, t) = max(0.0, v_x) * (1 - max(0.0, v_y))
prob = PIDEProblem(μ, σ, x0, tspan, g, f) # defining x0_sample is sufficient to implement Neumann boundary conditions

## Definition of the algorithm
alg = MLP(mc_sample = mc_sample)

sol = solve(prob, alg, multithreading = true)
```
