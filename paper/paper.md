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

Partial Differential Equations (PDEs) are equations that arise in a variety of models 
in physics, engineering, finance and biology.
The solution to PDEs can be defined on a domain which dimension is large,
causing problems in numerically approximating it. This problem is commonly refered
to as the curse of dimensionality. The computational cost of standard numerical methods
increases exponentially in the dimension of the solution's domain, and it is one of the
main challenge in Applied Mathematics. In the recent years, a handful of algorithms 
have been proposed that effectively overcome the curse of dimensionality.

`HighDimPDE.jl` is a Julia [@Bezanson2017] library for numerically solving high dimensional, 
non-local, non-linear PDEs.
The API for `HighDimPDE.jl` is designed to provide a user-friendly interface to using
numerical schemes, in particular for the
- DeepSplitting
- BSDE
- Multi Level Picard iterations

`HighDimPDE.jl` allows to use such schemes both on GPUs and CPUs with minimal coding effort.
`HighDimPDE.jl` also relies heavily on [@Flux] and
interfaces well with the [@SciML] ecoystem.

Equations considered

\begin{equation}
\begin{split}
(\tfrac{\partial}{\partial t}u)(t,x)
&=
\int_{\D} f\big(t,x,{\bf x}, u(t,x),u(t,{\bf x}), ( \nabla_x u )(t,x ),( \nabla_x u )(t,{\bf x} ) \big) \, \nu_x(d{\bf x}) \\
& \quad + \big\langle \mu(t,x), ( \nabla_x u )( t,x ) \big\rangle
+ \tfrac{ 1 }{ 2 }
\operatorname{Trace}\!\big(
\sigma(t,x) [ \sigma(t,x) ]^*
( \operatorname{Hess}_x u)( t,x )
\big).
\label{eq:defPDE}
\end{split}
\end{equation}

# Features
- Neumann boundary conditions
- GPU friendly
- Solving on 
    - single point $x \in \R^d$ (Deep splitting, MLP, DeepBSDE)
    - $d$-dimensional cube $[a,b]^2$ (Deep splitting)


# Examples

## MLP algorithm
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
We would like to thank Sebastian Becker, who wrote the original scripts in Python, TensorFlow and C++, and Arnulf Jentzen for the theoretical developments.

# Lorem ipsum
# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

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
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References