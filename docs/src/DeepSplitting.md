# The `DeepSplitting` algorithm

```@autodocs
Modules = [HighDimPDE]
Pages   = ["DeepSplitting.jl"]
```

The `DeepSplitting` algorithm reformulates the PDE as a stochastic learning problem.

The algorithm relies on two main ideas:

- the approximation of the solution $u$ by a parametric function $\bf u^\theta$. This function is generally chosen as a (Feedforward) Neural Network, as it is [universal approximators](https://en.wikipedia.org/wiki/Universal_approximation_theorem).

- the training of $\bf u^\theta$ by simulated stochastic trajectories of particles, through the link between linear PDEs and the expected trajectory of associated Stochastic Differential Equations (SDEs), explicitly stated by the [Feynman Kac formula](https://en.wikipedia.org/wiki/Feynman–Kac_formula).

## The general idea 💡
Consider the PDE
```math
\partial_t u(t,x) = \mu(t, x) \nabla_x u(t,x) + \frac{1}{2} \sigma^2(t, x) \Delta_x u(t,x) + f(x, u(t,x)) \tag{1}
```
with initial conditions $u(0, x) = g(x)$, where $u \colon \R^d \to \R$. 

Recall that the nonlinear Feynman-Kac formula provides a solution in terms of the mean trajectory of the stochastic trajectory of particles  $X^x_t$ 
```math
u(t, x) = \int_0^t \mathbb{E} \left[ f(X^x_{t - s}, u(T-s, X^x_{t - s}))ds \right] + \mathbb{E} \left[ u(0, X^x_t) \right] \tag{2}
```
where 
```math
X_t^x = \int_0^t \mu(X_s^x)ds + \int_0^t\sigma(X_s^x)dB_s + x,
```

> The Feynman Kac formula is often expressed for terminal condition problems where $u(T,x) = g(x)$. See Ref. for the equivalence between initial condition problems $u(0,x) = g(x)$.

### Local Feynman Kac formula
`DeepSplitting` solves the PDE iteratively over small time intervals by using an approximate Feynman-Kac representation locally.

More specifically, considering a small time step $dt = t_{n+1} - t_n$ one has that
```math
u(t_{n+1}, X_{T - t_{n+1}}) \approx \mathbb{E} \left[ f(t, X_{T - t_{n}}, u(t_{n},X_{T - t_{n}}))(t_{n+1} - t_n) + u(t_{n}, X_{T - t_{n}}) | X_{T - t_{n+1}}\right] \tag{3}
```

> may be use a simple version u(t_{n+1}, X_{T - t_{n+1}}) and then talk about this tower property for conditional expectations
Notice the similarities between Eq. (1) and Eq. (3).

### Reformulation as a learning problem
The `DeepSplitting` algorithm approximates $u(t_{n+1}, x)$ by a parametric function ${\bf u}^\theta_n(x)$. It is advised to let this function be a neural network ${\bf u}_\theta \equiv NN_\theta$ as they are universal approximators.

For each time step $t_n$, the `DeepSplitting` algorithm 

1. Generates the particle trajectories $X_t^x$ over the whole interval $[0,T]$

2. Seeks ${\bf u}_{n+1}^{\theta}$  by minimising the loss function

```math
L(\theta) = ||{\bf u}^\theta_{n+1}(X_{T - t_n}) - \left[ f(t, X_{T - t_{n-1}}, {\bf u}_{n-1}(X_{T - t_{n-1}}))(t_{n} - t_{n-1}) + {\bf u}_{n-1}(X_{T - t_{n-1}}) \right] ||
```


This way the PDE approximation problem is decomposed into a sequence of separate learning problems.
In `HighDimPDE.jl` the right parameter combination $\theta$ is found by iteratively minimizing $L$ using **stochastic gradient descent**.

## Solving point-wise or on a hypercube

### Pointwise
In practice, the `DeepSplitting` allows to obtain $u(t,x)$ on a singular point. This is done exactly as described above, and in this case ...

```julia
prob = PIDEProblem(g, f, μ, σ, tspan, x = x)
```

### Hypercube
Yet more generally, one wants to solve Eq. (1) on a whole interval (hypercube). This is offered by `HighDimPDE.jl`, when you specify

```julia
prob = PIDEProblem(g, f, μ, σ, tspan, u_domain = u_domain)
```
Internally, this is handled by assigning a random variable as the initial point of the particles, i.e.
```math
X_t^\xi = \int_0^t \mu(X_s^x)ds + \int_0^t\sigma(X_s^x)dB_s + \xi,
```
where $\xi$ a random variable uniformly distributed over $[a,b]^d$. This way, the neural network is trained on the whole interval $[a,b]^d$ instead of a single point.

!!! tip
    Approximating over a hypercube demands `K` 

## Accounting for non-localness
An extension of the `DeepSplitting` method offers to solve for non-local reaction diffusion equations of the type
```math
\partial_t u = \mu(x) \nabla_x u + \frac{1}{2} \sigma^2(x) \Delta u + \int_{\Omega}f(x,y, u(t,x), u(t,y))dy
```

The non-localness is handled by a plain vanilla Monte Carlo integration.
```math
\begin{aligned}
u(t_{n+1}, X_{T - t_{n+1}}) & \approx  \mathbb{E}_X \big[ \mathbb{E}_Y \big[ f(t, X_{T - t_{n}}, Y_{X_{T - t_{n}}}, u(t_{n},X_{T - t_{n}}), u(t_{n},Y_{X_{T - t_{n}}}))(t_{n+1} - t_n) \big] \\
                            & \quad + u(t_{n}, X_{T - t_{n}}) | X_{T - t_{n+1}}\big]
\end{aligned}
```

In practice, if you have a non-local model you need to provide the sampling method for $Y$, which is to be given to the algorithm method: 

```julia
alg = DeepSplitting(nn, opt = opt, mc_sample = mc_sample
```

`mc_sample` can be whether `UniformSampling(u_domain[1], u_domain[2]))` or ` NormalSampling(σ_sampling, shifted)`.

The approximation error of [Monte Carlo integrations decreases as $1/\sqrt{N}$](https://en.wikipedia.org/wiki/Monte_Carlo_integration) with $N$ the number of samples, and is therefore not dependent on the dimensions. On the other hand, the computational complexity of traditional deterministic techniques grows exponentially in the number of dimensions. 

### References
