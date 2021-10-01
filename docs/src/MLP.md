# [The `MLP` algorithm](@id mlp)

```@autodocs
Modules = [HighDimPDE]
Pages   = ["MLP.jl"]
```

The `MLP`, for Multi-Level Picard iterations, reformulates the PDE problem as a fixed point equation through the Feynman Kac formula. 

- It relies on [Picard iterations](https://en.wikipedia.org/wiki/Picard–Lindelöf_theorem) to find the fixed point, 

- reducing the complexity of the numerical approximation of the time integral through a [multilvel Monte Carlo](https://en.wikipedia.org/wiki/Multilevel_Monte_Carlo_method) approach.

The `MLP` algorithm overcomes the curse of dimensionality, with a computational complexity that grows polynomially in the number of dimension (see [M. Hutzenthaler et al. 2020](https://arxiv.org/abs/1807.01212v3)).

!!! warning "`MLP` can only solve for one point at a time"
    `MLP` works only with `PIDEProblem` defined with `x = x` option. If you want to solve over an entire domain, you definitely want to check the `DeepSplitting` algorithm.

## The general idea 💡
Consider the PDE
```math
\partial_t u(t,x) = \mu(t, x) \nabla_x u(t,x) + \frac{1}{2} \sigma^2(t, x) \Delta_x u(t,x) + f(x, u(t,x)) \tag{1}
```
with initial conditions $u(0, x) = g(x)$, where $u \colon \R^d \to \R$. 

### Picard Iterations
The `MLP` algorithm observes that the [Feynman Kac formula](@id feynmankac) can be viewed as a fixed point equation, i.e. $u = \phi(u)$. Introducing a sequence $(u_k)$ defined as $u_0 = g$ and 
```math
u_{l+1} = \phi(u_l),
```
the [Banach fixed-point theorem](https://en.wikipedia.org/wiki/Banach_fixed-point_theorem) ensures that the sequence converges to the true solution $u$. Such a technique is known as [Picard iterations](https://en.wikipedia.org/wiki/Picard–Lindelöf_theorem).


The time integral term is evaluated by a [Monte-Carlo integration](https:/en.wikipedia.org/wiki/Monte_Carlo_integration)

```math
u_L  = \frac{1}{M}\sum_i^M \mathbb{E} \left[ f(X^x_{t - s_i}, u_{L-1}(T-s_i, X^x_{t - s_i})) \right] + \mathbb{E} \left[ u(0, X^x_t) \right].
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

As $l$ grows, the term $[\phi(u_{l-1}) - \phi(u_{l-2})]$ becomes smaller - and demands more calculations. The `MLP` algorithm usses this fact by evaluating the integral term at level $l$ with $M^{L-l}$ samples.


!!! tip
    - `L` corresponds to the level of the approximation, i.e. $u \approx u_L$
    - `M` characterises the number of samples for the monte carlo approximation of the time integral

Overall, `MLP` can be summarised by the following formula
```math
\begin{aligned}
u_L &= \sum_{l=1}^{L-1} \frac{1}{M^{L-l}}\sum_i^{M^{L-l}} \left[ f(X^{x,(l, i)}_{t - s_{(l, i)}}, u(T-s_{(l, i)}, X^{x,(l, i)}_{t - s_{(l, i)}})) + \mathbf{1}_\N(l) f(X^{x,(l, i)}_{t - s_{(l, i)}}, u(T-s_{(l, i)}, X^{x,(l, i)}_{t - s_{(l, i)}}))\right]
\\
&\qquad + \frac{1}{M^{L}}\sum_i^{M^{L}} u(0, X^{x,(l, i)}_t)\\
\end{aligned}
```
Note that the superscripts $(l, i)$ indicate the independence of the random variables $l$.

## Nonlocal PDEs
`MLP` can solve for non-local reaction diffusion equations of the type
```math
\partial_t u = \mu(t, x) \nabla_x u(t, x) + \frac{1}{2} \sigma^2(t, x) \Delta u(t, x) + \int_{\Omega}f(x, y, u(t,x), u(t,y))dy
```

The non-localness is handled by a Monte Carlo integration.

```math
\begin{aligned}
u_L &= \sum_{l=1}^{L-1} \frac{1}{M^{L-l}}\sum_{i=1}^{M^{L-l}} \frac{1}{K}\sum_{j=1}^{K}  \bigg[ f(X^{x,(l, i)}_{t - s_{(l, i)}}, Z^{(l,j)}, u(T-s_{(l, i)}, X^{x,(l, i)}_{t - s_{(l, i)}}), u(T-s_{l,i}, Z^{(l,j)})) + \\
&\qquad 
\mathbf{q}_\N(l) f(X^{x,(l, i)}_{t - s_{(l, i)}}, u(T-s_{(l, i)}, X^{x,(l, i)}_{t - s_{(l, i)}}))\bigg] + \frac{1}{M^{L}}\sum_i^{M^{L}} u(0, X^{x,(l, i)}_t)\\
\end{aligned}
```

!!! tip
    - `K` characterises the number of samples for the Monte Carlo approximation of the last term.
    - `mc_sample` characterises the distribution of the `Z` variables
