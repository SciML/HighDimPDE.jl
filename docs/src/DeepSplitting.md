# [The `DeepSplitting` algorithm](@id deepsplitting)

### Problems Supported:

 1. [`PIDEProblem`](@ref)
 2. [`ParabolicPDEProblem`](@ref)

```@autodocs
Modules = [HighDimPDE]
Pages   = ["DeepSplitting.jl"]
```

The `DeepSplitting` algorithm reformulates the PDE as a stochastic learning problem.

The algorithm relies on two main ideas:

  - The approximation of the solution $u$ by a parametric function $\bf u^\theta$. This function is generally chosen as a (Feedforward) Neural Network, as it is a [universal approximator](https://en.wikipedia.org/wiki/Universal_approximation_theorem).

  - The training of $\bf u^\theta$ by simulated stochastic trajectories of particles, through the link between linear PDEs and the expected trajectory of associated Stochastic Differential Equations (SDEs), explicitly stated by the [Feynman Kac formula](https://en.wikipedia.org/wiki/Feynman%E2%80%93Kac_formula).

## The general idea 💡

Consider the PDE

```math
\partial_t u(t,x) = \mu(t, x) \nabla_x u(t,x) + \frac{1}{2} \sigma^2(t, x) \Delta_x u(t,x) + f(x, u(t,x)) \tag{1}
```

with initial conditions $u(0, x) = g(x)$, where $u \colon \R^d \to \R$.

### Local Feynman Kac formula

`DeepSplitting` solves the PDE iteratively over small time intervals by using an approximate [Feynman-Kac representation](@ref feynmankac) locally.

More specifically, considering a small time step $dt = t_{n+1} - t_n$ one has that

```math
u(t_{n+1}, X_{T - t_{n+1}}) \approx \mathbb{E} \left[ f(t, X_{T - t_{n}}, u(t_{n},X_{T - t_{n}}))(t_{n+1} - t_n) + u(t_{n}, X_{T - t_{n}}) | X_{T - t_{n+1}}\right] \tag{3}.
```

One can therefore use Monte Carlo integrations to approximate the expectations

```math
u(t_{n+1}, X_{T - t_{n+1}}) \approx \frac{1}{\text{batch\_size}}\sum_{j=1}^{\text{batch\_size}} \left[ u(t_{n}, X_{T - t_{n}}^{(j)}) + (t_{n+1} - t_n)\sum_{k=1}^{K} \big[ f(t_n, X_{T - t_{n}}^{(j)}, u(t_{n},X_{T - t_{n}}^{(j)})) \big] \right]
```

### Reformulation as a learning problem

The `DeepSplitting` algorithm approximates $u(t_{n+1}, x)$ by a parametric function ${\bf u}^\theta_n(x)$. It is advised to let this function be a neural network ${\bf u}_\theta \equiv NN_\theta$ as they are universal approximators.

For each time step $t_n$, the `DeepSplitting` algorithm

 1. Generates the particle trajectories $X^{x, (j)}$ satisfying [Eq. (2)](@ref feynmankac) over the whole interval $[0,T]$.

 2. Seeks ${\bf u}_{n+1}^{\theta}$  by minimizing the loss function

```math
L(\theta) = ||{\bf u}^\theta_{n+1}(X_{T - t_n}) - \left[ f(t, X_{T - t_{n-1}}, {\bf u}_{n-1}(X_{T - t_{n-1}}))(t_{n} - t_{n-1}) + {\bf u}_{n-1}(X_{T - t_{n-1}}) \right] ||
```

This way, the PDE approximation problem is decomposed into a sequence of separate learning problems.
In `HighDimPDE.jl` the right parameter combination $\theta$ is found by iteratively minimizing $L$ using **stochastic gradient descent**.

!!! tip
    
    To solve with `DeepSplitting`, one needs to provide to `solve`
    
      - `dt`
      - `batch_size`
      - `maxiters`: the number of iterations for minimizing the loss function
      - `abstol`: the absolute tolerance for the loss function
      - `use_cuda`: if you have a Nvidia GPU, recommended.

## Solving point-wise or on a hypercube

### Pointwise

`DeepSplitting` allows obtaining $u(t,x)$ on a single point  $x \in \Omega$ with the keyword $x$.

```julia
prob = PIDEProblem(μ, σ, x, tspan, g, f)
```

### Hypercube

Yet more generally, one wants to solve Eq. (1) on a $d$-dimensional cube $[a,b]^d$. This is offered by `HighDimPDE.jl` with the keyword `x0_sample`.

```julia
prob = PIDEProblem(μ, σ, x, tspan, g, f; x0_sample = x0_sample)
```

Internally, this is handled by assigning a random variable as the initial point of the particles, i.e.

```math
X_t^\xi = \int_0^t \mu(X_s^x)ds + \int_0^t\sigma(X_s^x)dB_s + \xi,
```

where $\xi$ a random variable uniformly distributed over $[a,b]^d$. This way, the neural network is trained on the whole interval $[a,b]^d$ instead of a single point.

## Non-local PDEs

`DeepSplitting` can solve for non-local reaction diffusion equations of the type

```math
\partial_t u = \mu(x) \nabla_x u + \frac{1}{2} \sigma^2(x) \Delta u + \int_{\Omega}f(x,y, u(t,x), u(t,y))dy
```

The non-localness is handled by a Monte Carlo integration.

```math
u(t_{n+1}, X_{T - t_{n+1}}) \approx \sum_{j=1}^{\text{batch\_size}} \left[ u(t_{n}, X_{T - t_{n}}^{(j)}) + \frac{(t_{n+1} - t_n)}{K}\sum_{k=1}^{K} \big[ f(t, X_{T - t_{n}}^{(j)}, Y_{X_{T - t_{n}}^{(j)}}^{(k)}, u(t_{n},X_{T - t_{n}}^{(j)}), u(t_{n},Y_{X_{T - t_{n}}^{(j)}}^{(k)})) \big] \right]
```

!!! tip
    

In practice, if you have a non-local model, you need to provide the sampling method and the number $K$ of MC integration through the keywords `mc_sample` and `K`.
`julia alg = DeepSplitting(nn, opt = opt, mc_sample = mc_sample, K = 1) `
`mc_sample` can be whether `UniformSampling(a, b)` or ` NormalSampling(σ_sampling, shifted)`.

## References

  - Boussange, V., Becker, S., Jentzen, A., Kuckuck, B., Pellissier, L., Deep learning approximations for non-local nonlinear PDEs with Neumann boundary conditions. [arXiv](https://arxiv.org/abs/2205.03672) (2022)
  - Beck, C., Becker, S., Cheridito, P., Jentzen, A., Neufeld, A., Deep splitting method for parabolic PDEs. [arXiv](https://arxiv.org/abs/1907.03452) (2019)
