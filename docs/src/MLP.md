# [The `MLP` algorithm](@id mlp)

### Problems Supported:
1. [`PIDEProblem`](@ref)
2. [`ParabolicPDEProblem`](@ref)

```@autodocs
Modules = [HighDimPDE]
Pages   = ["MLP.jl"]
```

The `MLP`, for Multi-Level Picard iterations, reformulates the PDE problem as a fixed point equation through the Feynman Kac formula. 

- It relies on [Picard iterations](https://en.wikipedia.org/wiki/Picard–Lindelöf_theorem) to find the fixed point, 

- reducing the complexity of the numerical approximation of the time integral through a [multilevel Monte Carlo](https://en.wikipedia.org/wiki/Multilevel_Monte_Carlo_method) approach.

The `MLP` algorithm overcomes the curse of dimensionality, with a computational complexity that grows polynomially in the number of dimension (see [M. Hutzenthaler et al. 2020](https://arxiv.org/abs/1807.01212v3)).

!!! warning "`MLP` can only approximate the solution on a single point"
    `MLP` only works for `PIDEProblem` with `x0_sample = NoSampling()`. If you want to solve over an entire domain, you definitely want to check the `DeepSplitting` algorithm.

## The general idea 💡
Consider the PDE
```math
\partial_t u(t,x) = \mu(t, x) \nabla_x u(t,x) + \frac{1}{2} \sigma^2(t, x) \Delta_x u(t,x) + f(x, u(t,x)) \tag{1}
```
with initial conditions $u(0, x) = g(x)$, where $u \colon \R^d \to \R$. 

### Picard Iterations
The `MLP` algorithm observes that the [Feynman Kac formula](@ref feynmankac) can be viewed as a fixed point equation, i.e. $u = \phi(u)$. Introducing a sequence $(u_k)$ defined as $u_0 = g$ and 
```math
u_{l+1} = \phi(u_l),
```
the [Banach fixed-point theorem](https://en.wikipedia.org/wiki/Banach_fixed-point_theorem) ensures that the sequence converges to the true solution $u$. Such a technique is known as [Picard iterations](https://en.wikipedia.org/wiki/Picard–Lindelöf_theorem).


The time integral term is evaluated by a [Monte-Carlo integration](https://en.wikipedia.org/wiki/Monte_Carlo_integration)

```math
u_L  = \frac{1}{M}\sum_i^M \left[ f(X^{x,(i)}_{t - s_{(l, i)}}, u_{L-1}(T-s_i, X^{x,( i)}_{t - s_{(l, i)}})) + u(0, X^{x,(i)}_{t - s_{(l, i)}}) \right].
```

But the MLP uses an extra trick to lower the computational cost of the iteration. 


### Telescope sum
The `MLP` algorithm uses a telescope sum 

```math
\begin{aligned}
u_L = \phi(u_{L-1}) &= [\phi(u_{L-1}) - \phi(u_{L-2})] + [\phi(u_{L-2}) - \phi(u_{L-3})] + \dots \\
&= \sum_{l=1}^{L-1} [\phi(u_{l-1}) - \phi(u_{l-2})]
\end{aligned}
```

As $l$ grows, the term $[\phi(u_{l-1}) - \phi(u_{l-2})]$ becomes smaller, and thus demands more calculations. The `MLP` algorithm uses this fact by evaluating the integral term at level $l$ with $M^{L-l}$ samples.


!!! tip
    - `L` corresponds to the level of the approximation, i.e. $u \approx u_L$
    - `M` characterizes the number of samples for the Monte Carlo approximation of the time integral

Overall, `MLP` can be summarized by the following formula
```math
\begin{aligned}
u_L &= \sum_{l=1}^{L-1} \frac{1}{M^{L-l}}\sum_i^{M^{L-l}} \left[ f(X^{x,(l, i)}_{t - s_{(l, i)}}, u(T-s_{(l, i)}, X^{x,(l, i)}_{t - s_{(l, i)}})) + \mathbf{1}_\N(l) f(X^{x,(l, i)}_{t - s_{(l, i)}}, u(T-s_{(l, i)}, X^{x,(l, i)}_{t - s_{(l, i)}}))\right]
\\
&\qquad + \frac{1}{M^{L}}\sum_i^{M^{L}} u(0, X^{x,(l, i)}_t)\\
\end{aligned}
```
Note that the superscripts $(l, i)$ indicate the independence of the random variables $X$ across levels.

## Non-local PDEs
`MLP` can solve for non-local reaction diffusion equations of the type
```math
\partial_t u = \mu(t, x) \nabla_x u(t, x) + \frac{1}{2} \sigma^2(t, x) \Delta u(t, x) + \int_{\Omega}f(x, y, u(t,x), u(t,y))dy
```

The non-localness is handled by a Monte Carlo integration.

```math
\begin{aligned}
u_L &= \sum_{l=1}^{L-1} \frac{1}{M^{L-l}}\sum_{i=1}^{M^{L-l}} \frac{1}{K}\sum_{j=1}^{K}  \bigg[ f(X^{x,(l, i)}_{t - s_{(l, i)}}, Z^{(l,j)}, u(T-s_{(l, i)}, X^{x,(l, i)}_{t - s_{(l, i)}}), u(T-s_{l,i}, Z^{(l,j)})) + \\
&\qquad 
\mathbf{1}_\N(l) f(X^{x,(l, i)}_{t - s_{(l, i)}}, u(T-s_{(l, i)}, X^{x,(l, i)}_{t - s_{(l, i)}}))\bigg] + \frac{1}{M^{L}}\sum_i^{M^{L}} u(0, X^{x,(l, i)}_t)\\
\end{aligned}
```

!!! tip
    In practice, if you have a non-local model, you need to provide the sampling method and the number $K$ of MC integration through the keywords `mc_sample` and `K`. 
    - `K` characterizes the number of samples for the Monte Carlo approximation of the last term.
    - `mc_sample` characterizes the distribution of the `Z` variables

## References
- Boussange, V., Becker, S., Jentzen, A., Kuckuck, B., Pellissier, L., Deep learning approximations for non-local nonlinear PDEs with Neumann boundary conditions. [arXiv](https://arxiv.org/abs/2205.03672) (2022)
- Becker, S., Braunwarth, R., Hutzenthaler, M., Jentzen, A., von Wurstemberger, P., Numerical simulations for full history recursive multilevel Picard approximations for systems of high-dimensional partial differential equations. [arXiv](https://arxiv.org/abs/2005.10206) (2020)