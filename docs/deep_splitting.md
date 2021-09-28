# The `DeepSplitting` algorithm

The `DeepSplitting` algorithm reformulates the PDE as a stochastic learning problem.

The algorithm relies on two main ideas:

- the approximation of the solution $u$ by a parametric function $\bf u^\theta$. This function is generally chosen as a (Feedforward) Neural Network, as they are [universal approximators](https://en.wikipedia.org/wiki/Universal_approximation_theorem).

- the training of $\bf u^\theta$ by simulated stochastic trajectories of particles, through the link between linear PDEs and the expected trajectory of associated Stochastic Differential Equations (SDEs), explicitly stated by the [Feynman Kac formula](https://en.wikipedia.org/wiki/Feynman–Kac_formula).

## The general idea 💡
Consider the PDE
$$
\partial_t u = \mu(x) \nabla_x u + \frac{1}{2} \sigma^2(x) \Delta u + f(x, u(t,x))
$$
with initial conditions $u(0, x) = g(x)$, where $u \colon \R^d \to \R$ is solution we wish to approximate. 

If the function $f$ is continuous and linear in $u$, the Feynman-Kac formula provides an explicit soltution in terms of the mean trajectory of the stochastic trajectory of particles  $X^x_t$ 
$$
\begin{equation}
u(T, x) = \int_0^T \E \left[ f(X^x_{T - s})ds \right] + \E \left[ u(0, X^x_T) \right]
\end{equation}
$$
where 
$$
\begin{equation}
X_t^x = \int_0^t \mu(X_s^x)ds + \int_0^t\sigma(X_s^x)dB_s + x,
\end{equation}
$$
and $B_t$ is a [Brownian motion](https://en.wikipedia.org/wiki/Wiener_process) (see e.g. Theorem 4.1. in [Ref](https://cel.archives-ouvertes.fr/cel-00736268)).

![Brownian motion - Wikipedia](https://upload.wikimedia.org/wikipedia/commons/f/f8/Wiener_process_3d.png)


**The DeepSplitting algorithm covers more general case (tackling non-linearities), by solving the PDE iteratively over small time intervals by using an approximate Feynman-Kac representation locally.**

More specifically, considering a small time step $dt = t_{n+1} - t_n$ one has that
$$
\begin{equation}
u(t_{n+1}, X_{T - t_{n+1}}) \approx \E \left[ f(t, X_{T - t_{n}}, u(t_{n},X_{T - t_{n}}))(t_{n+1} - t_n) + u(t_{n}, X_{T - t_{n}}) | X_{T - t_{n+1}}\right]
\end{equation}
$$

> may be use a simple version u(t_{n+1}, X_{T - t_{n+1}}) and then talk about this tower property for conditional expectations
Notice the similarities between Eq. (1) and Eq. (3).

The `DeepSplitting` algorithm approximates $u(t_{n+1}, x)$ by a parametric function ${\bf u}^\theta_n(x)$. It is advised to let this function be a neural network ${\bf u}_\theta \equiv NN_\theta$ as they are universal approximators.

For each time step $t_n$, the `DeepSplitting` algorithm 

1. Generates the particle trajectories $X_t^x$ over the whole interval $[0,T]$

2. Seeks ${\bf u}_{n+1}^{\theta}$  by minimising the loss function

$$
L(\theta) = ||{\bf u}^\theta_{n+1}(X_{T - t_n}) - \left[ f(t, X_{T - t_{n-1}}, {\bf u}_{n-1}(X_{T - t_{n-1}}))(t_{n} - t_{n-1}) + {\bf u}_{n-1}(X_{T - t_{n-1}}) \right] ||
$$


This way the PDE approximation problem is decomposed into a sequence of separate learning problems.
In `HighDimPDE.jl` the right parameter combination $\theta$ is found by iteratively minimizing $L$ using **stochastic gradient descent**.

## Solving point-wise or on a hypercube

### Pointwise
In practice, the `DeepSplitting` allows to obtain $u(t,x)$ on a singular point. This is done exactly as described above, and in this case ...

```julia
prob = PIDEProblem(g, f, μ_f, σ_f, X0, tspan)
```

### Hypercube
Yet more generally, one wants to solve Eq. (1) on a whole interval (hypercube). This is offered by `HighDimPDE.jl`, when you specify

```julia
prob = PIDEProblem(g, f, μ, σ, tspan, u_domain = u_domain)
```
Internally, this is handled by 
$$
\begin{equation}
X_t^\xi = \int_0^t \mu(X_s^x)ds + \int_0^t\sigma(X_s^x)dB_s + \xi,
\end{equation}
$$
where $\xi$ a random variable uniformly distributed over $[a,b]^d$. This way, the neural network is trained on the whole interval $[a,b]^d$ instead of a single point.

## Accounting for non-localness
An extension of the `DeepSplitting` method offers to solve for non-local reaction diffusion equations of the type
$$
\partial_t u = \mu(x) \nabla_x u + \frac{1}{2} \sigma^2(x) \Delta u + \int_{\Omega}f(x,y, u(t,x), u(t,y))dy
$$

The non-localness is handled by a plain vanilla Monte Carlo integration.
$$
\begin{equation}
\begin{aligned}
u(t_{n+1}, X_{T - t_{n+1}}) & \approx  \E_X \big[ \E_Y \big[ f(t, X_{T - t_{n}}, Y_{X_{T - t_{n}}}, u(t_{n},X_{T - t_{n}}), u(t_{n},Y_{X_{T - t_{n}}}))(t_{n+1} - t_n) \big] \\
                            & \quad + u(t_{n}, X_{T - t_{n}}) | X_{T - t_{n+1}}\big]
\end{aligned}
\end{equation}
$$

In practice, if you have a non-local model you need to provide the sampling method for $Y$, which is to be given to the algorithm method: 

```julia
alg = DeepSplitting(nn,
                    opt = opt,
                    mc_sample = mc_sample
```

`mc_sample` can be whether ` = UniformSampling(u_domain[1], u_domain[2]))` or ` NormalSampling(σ_sampling, centered)`.

 choose in `HighDimPDE.jl` between two different distributions for $Y$ : Normal with 

The approximation error of [Monte Carlo integrations decreases as $1/\sqrt{N}$](https://en.wikipedia.org/wiki/Monte_Carlo_integration) with $N$ the number of samples, and is therefore not dependent on the dimensions. On the other hand, the computational complexity of traditional deterministic techniques grows exponentially in the number of dimensions. 


## Neumann Boundary conditions

`HighDimPDE.jl` can handle Eq. (1) with zero-flux boundary conditions (Neumann boundary conditions) on the domain $\Omega \in \R^d$, that is 
$$
\langle \nabla_x u(t,x) \cdot \vec{n} \rangle = 0
$$
where $\vec{n}$ denotes the outer normal vector associated to $\Omega$. 

Internally, this is done by reflecting the stochastic particles at the boundary of the domains, as [reflected brownian motion](https://en.wikipedia.org/wiki/Reflected_Brownian_motion) instead of standard brownian motions allows to account for Neumann Boundary conditions.

can be solved with the very same scheme as described above, but slightly changing reflecting the particles trajectory $X_t^x$ on the boundary of $\Omega$.


![](img/animRBM_southamerica.gif)

In `HighDimPDE.jl`, you 

<!-- 
$$
\begin{equation}
u(t, x) = \E \left[ \int_0^t f(X^x_{t})ds + u(0, X^x_t) \right]
\end{equation}
$$ -->

<!-- Intuitively, this formula is motivated by the fact that [the density of Brownian particles (motion) satisfy the diffusion equation](https://en.wikipedia.org/wiki/Brownian_motion#Einstein's_theory). -->

<!-- Simple Monte Carlo averages can be used to approximate the expectation in Eq. (1), by numerically simulating the stochastic trajectories $X_s^x$ from Eq. (2) through a standard discretization algorithm (in `HighDimPDE.jl`, [Euler Maryuyama](https://en.wikipedia.org/wiki/Euler–Maruyama_method)). -->

<!-- $$
u(t,x) = \frac{1}{N} \sum_{i=1}^N \left[u(0, X^{x, (i)}_t)  \int_0^t f(t-s, X^{x, (i)}_s)ds \right]
$$ -->

<!-- The equivalence between the average trajectory of particles and PDEs given by the Feynman-Kac formula allows to overcome the curse of dimensionality that standard numerical methods suffer from, as the approximation error of [Monte Carlo integrations decreases as $1/\sqrt{N}$](https://en.wikipedia.org/wiki/Monte_Carlo_integration) and is therefore not dependent on the dimensions. On the other hand, the computational complexity of traditional deterministic techniques grows exponentially in the number of dimensions.  -->


### Solving Eq. (1) with HighDimPDE.jl
```julia
tspan = (0f0, 5f-1)
dt = 5f-1  # time step
μ(X,p,t) = 0f0 # advection coefficients
σ(X,p,t) = 1f-1 # diffusion coefficients

u_anal(x, t) = sum(x.^2) + d * σ(0., 0., 0.)^2 * t
g(x) = sum(x.^2, dims=1)

# d = 10
x0 = fill(2f0,d)
hls = d + 10 #hidden layer size

nn = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1)) # Neural network used by the scheme

opt = ADAM(0.01) #optimiser
alg = DeepSplitting(nn, opt = opt)

f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = 0f0 .* v_y #TODO: this fix is not nice

# defining the problem
prob = PIDEProblem(g, f, μ, σ, tspan, 
                    x = x0
                    )
# solving
xs,ts,sol = solve(prob, alg, dt, 
                verbose = false, 
                use_cuda = use_cuda,
                maxiters = 1000,
                batch_size = 10000)
u1 = sol[end]
u1_anal = u_anal(x0, tspan[end])
e_l2 = rel_error_l2(u1, u1_anal)
println("rel_error_l2 = ", e_l2, "\n")
@test e_l2 < 0.1
```