
# Solving the 10-dimensional Fisher-KPP equation with `DeepSplitting`
Consider the Fisher-KPP equation with non-local competition

```math
\partial_t u = u (1 - \int_\Omega u(t,y)dy) + \frac{1}{2}\sigma^2\Delta_xu \tag{1}
```

where $\Omega = [-1/2, 1/2]^d$, and let's assume Neumann Boundary condition on $\Omega$.

Let's solve Eq. (1) with the [`DeepSplitting`](@ref deepsplitting) solver. 

```@example deepsplitting
using HighDimPDE

## Definition of the problem
d = 10 # dimension of the problem
tspan = (0.0, 0.5) # time horizon
x0 = fill(0f0,d)  # initial point
g(x) = exp.(- sum(x.^2, dims=1) ) # initial condition
μ(x, p, t) = 0.0f0 # advection coefficients
σ(x, p, t) = 0.1f0 # diffusion coefficients
x0_sample = UniformSampling(fill(-5f-1, d), fill(5f-1, d))
f(x, y, v_x, v_y, ∇v_x, ∇v_y, p, t) = v_x .* (1f0 .- v_y)
prob = PIDEProblem(μ, σ, x0, tspan, g, f; x0_sample = x0_sample)

## Definition of the neural network to use
using Flux # needed to define the neural network

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

#### Solving on the GPU

`DeepSplitting` can run on the GPU for (much) improved performance. To do so, just set `use_cuda = true`.

```@example deepsplitting2
sol = solve(prob, 
            alg, 
            0.1, 
            verbose = true, 
            abstol = 2e-3,
            maxiters = 1000,
            batch_size = 1000,
            use_cuda=true)
```