---
title: 'HighDimPDE.jl: A Julia package for solving high dimensional, non-local, non-linear PDEs'
tags:
  - Julia
  - PDEs
authors:
  - name: Victor Boussange^[first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-4202-3599
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID^[co-first author] # note this makes a footnote saying 'co-first author'
    affiliation: 2
  - name: Author with no affiliation^[corresponding author]
    affiliation: 3
affiliations:
 - name: Swiss Federal Research Institute WSL, CH-8903 Birmensdorf, Switzerland
   index: 1
 - name: Landscape Ecology, Institute of Terrestrial Ecosystems, Department of Environmental System Science, ETH Zürich, CH-8092 Zürich, Switzerland
   index: 2
 - name: Independent Researcher
   index: 3
date: 23 August 2021
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary
`HighDimPDE.jl` is a Julia [@Bezanson2017] package for numerically approximating highly dimensional non-local non-linear Partial Differential Equations (PDEs). It is an open-source project hosted on GitHub and distributed under the MIT license. The package includes algorithms to solve PDEs which complexity grows polynomially in the number of dimension of the PDE. The package is designed with a user-friendly interface, provides both CPUs and GPUs support, and is integrated within the Sci-ML[@SciML] ecosystem.

# Statement of need
Non-local nonlinear Partial Differential Equations arise in a variety of scientific domains including physics, engineering, finance and biology. In biology, they are for instance used for modelling the evolution of biological populations that are phenotypically and physically structured. The dimension of the PDEs can be large, corresponding to the number of phenotypic traits and physical dimensions considered. Highly dimensional PDE's cannot be solved with standard numerical methods as their computational cost increases exponentially in the number of dimensions, a problem commonly refered as the curse of dimensionality. 

# Solving PDEs with HighDimPDE.jl
HighDimPDE.jl can solve for PDEs of the form

$$
\begin{aligned}
    (\partial_t u)(t,x) &= \int_{\Omega} f\big(t,x,{\bf x}, u(t,x),u(t,{\bf x}), ( \nabla_x u )(t,x ),( \nabla_x u )(t,{\bf x} ) \big) \, \nu_x(d{\bf x}) \\
    & \quad + \big\langle \mu(t,x), ( \nabla_x u )( t,x ) \big\rangle + \tfrac{1}{2} \text{Trace} \big(\sigma(t,x) [ \sigma(t,x) ]^* ( \text{Hess}_x u)(t, x ) \big). \tag{1}
\end{aligned}
$$
where $u \colon [0,T] \times \Omega \to \R$, $\Omega \subset \R^d$ is a function subject to initial conditions $u(0,x) = g(x)$ and with Neumann Boudary conditions.

`HighDimPDE.jl` currently proposes ? algorithms to solve Eq. (1).
## Algorithm overview

----------------------------------------------
Features  |    `DeepSplitting`   | `MLP`     |
----------|:----------------------:|:------------:
Time discretization free|   ❌ |         ✅ |
Mesh-free       | ✅ |                   ✅ |
Single point $x \in \R^d$ approximation| ✅   |  ✅ |
$d$-dimensional cube $[a,b]^d$ approximation| ✅   |          ❌ |
GPU             | ✅ |                   ❌ |
Gradient non-linearities    | ✔️|       ❌ |

✔️ : will be supported in the future

## `DeepSplitting`

The `DeepSplitting`[@Beck2019] algorithm reformulates the PDE as a stochastic learning problem.

The algorithm relies on two main ideas:

- the approximation of the solution $u$ by a parametric function $\bf u^\theta$.

- the training of $\bf u^\theta$ by simulated stochastic trajectories of particles, with [@Flux].

## `MLP`
The `MLP`[@Becker2020], for Multi-Level Picard iterations, reformulates the PDE problem as a fixed point equation through the Feynman Kac formula. 

- It relies on [Picard iterations](https://en.wikipedia.org/wiki/Picard–Lindelöf_theorem) to find the fixed point, 

- reducing the complexity of the numerical approximation of the time integral through a [multilvel Monte Carlo](https://en.wikipedia.org/wiki/Multilevel_Monte_Carlo_method) approach.

`HighDimPDE.jl` allows to use such schemes both on GPUs and CPUs with minimal coding effort.
`HighDimPDE.jl` also relies heavily on [@Flux] and
interfaces well with the [@SciML] ecoystem.

# Examples
Consider the equation
$$
$$

Subject to initial conditions
$$

$$

## `DeepSplitting`

```julia
tspan = (0f0,15f-2)
    dt = 5f-2 # time step
    μ(x, p, t) = 0f0 # advection coefficients
    σ(x, p, t) = 1f-1 #1f-1 # diffusion coefficients
    ss0 = 5f-2#std g0

    # Analytic sol
    function _SS(x, t, p)
        d = length(x)
        MM = σ(x, p, t) * ones(d)
        SSt = MM .* ((MM .* sinh.(MM *t) .+ ss0 .* cosh.( MM * t)) ./ (MM .* cosh.(MM * t ) .+ ss0 .* sinh.(MM * t)))
        return SSt
    end

    function uanal(x, t, p)
            d = length(x)
            return (2*π)^(-d/2) * prod(_SS(x, t, p) .^(-1/2)) * exp(-0.5 *sum(x .^2 ./ _SS(x, t, p)) )
    end

    sols = []
    xs = []
    for d in [5]
        U = 25f-2
        u_domain = (fill(-U, d), fill(U, d))

        batch_size = 10000
        train_steps = 2000
        K = 1

        hls = d + 50 #hidden layer size

        nn_batch = Flux.Chain(
                            # BatchNorm(d, affine = true, dim = 1),
                            Dense(d, hls, tanh),
                            # BatchNorm(hls, affine = true, dim = 1),
                            Dense(hls, hls, tanh),
                            # BatchNorm(hls, affine = true, dim = 1),
                            Dense(hls, 1, x->x^2)) # positive function

        opt = ADAM(1e-2)#optimiser
        alg = DeepSplitting(nn_batch, K=K, opt = opt, mc_sample = UniformSampling(u_domain[1], u_domain[2]) )

        g(x) = Float32((2*π)^(-d/2)) * ss0^(- Float32(d) * 5f-1) * exp.(-5f-1 *sum(x .^2f0 / ss0, dims = 1)) # initial condition
        m(x) = - 5f-1 * sum(x.^2, dims=1)
        vol = prod(u_domain[2] - u_domain[1])
        f(y, z, v_y, v_z, ∇v_y, ∇v_z, p, t) =  max.(v_y, 0f0) .* (m(y) .- vol *  max.(v_z, 0f0) .* m(z)) # nonlocal nonlinear part of the

        # defining the problem
        prob = PIDEProblem(g, f, μ, σ, tspan, 
                            u_domain = u_domain
                            )
        # solving
        xgrid,ts,sol = solve(prob, 
                        alg, 
                        dt, 
                        verbose = false, 
                        abstol = 1f-3,
                        maxiters = train_steps,
                        batch_size = batch_size,
                        use_cuda = use_cuda
                        )
        u1 = [sol[end](x)[] for x in xgrid]
        u1_anal = uanal.(xgrid, tspan[end], nothing)
        e_l2 = mean(rel_error_l2.(u1, u1_anal))
        println("rel_error_l2 = ", e_l2, "\n")
        @test e_l2 < 0.1
        push!(sols, sol[end])
    end
```
![Caption for example figure.\label{fig:hamel}](hamel_5d.pdf)
## `MLP`
Non local PDE with Neumann boundary conditions
Let's include in the previous equation non local competition and let's assume Neumann Boundary conditions, so that the domain consists in the hyper cube [-1/2, 1/2]^d.
```julia
using HighDimPDE

## Definition of the problem
d = 10 # dimension of the problem
tspan = (0.0,0.5) # time horizon
x0 = fill(0.,d)  # initial point
g(x) = exp( -sum(x.^2) ) # initial condition
μ(x, p, t) = 0.0 # advection coefficients
σ(x, p, t) = 0.1 # diffusion coefficients
u_domain = [-1/2, 1/2]
f(x, y, v_x, v_y, ∇v_x, ∇v_y, t) = max(0.0, v_x) * (1 -  max(0.0, v_y)) 
prob = PIDEProblem(g, f, μ, 
                    σ, x0, tspan, 
                    u_domain = u_domain) # defining u_domain is sufficient to implement Neumann boundary conditions

## Definition of the algorithm
alg = MLP(mc_sample = UniformSampling(u_domain[1], u_domain[2]) ) 

sol = solve(prob, alg, multithreading=true)
```

# Acknowledgements

We acknowledge contributions from ...

# References

<!-- 
# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->
