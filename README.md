[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/HighDimPDE/stable/)

[![codecov](https://codecov.io/gh/SciML/HighDimPDE.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SciML/HighDimPDE.jl)
[![Build Status](https://github.com/SciML/HighDimPDE.jl/workflows/CI/badge.svg)](https://github.com/SciML/HighDimPDE.jl/actions?query=workflow%3ACI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

# HighDimPDE.jl

**HighDimPDE.jl** is a Julia package to **solve Highly Dimensional non-local, non-linear PDEs** of the form

$$
\begin{aligned}
(\partial_t u)(t,x) &= \int_{\Omega} f\big(t,x,{\bf x}, u(t,x),u(t,{\bf x}), ( \nabla_x u )(t,x ),( \nabla_x u )(t,{\bf x} ) \big) d{\bf x} \\
& \quad + \big\langle \mu(t,x), ( \nabla_x u )( t,x ) \big\rangle + \tfrac{1}{2} \text{Trace} \big(\sigma(t,x) [ \sigma(t,x) ]^* ( \text{Hess}_x u)(t, x ) \big).
\end{aligned}
$$

where $u \colon [0,T] \times \Omega \to \mathbb{R}, \Omega \subseteq \mathbb{R}^{d}$ is subject to initial and boundary conditions, and where $d$ is large.

## Tutorials and Documentation

For information on using the package,
[see the stable documentation](https://docs.sciml.ai/HighDimPDE/stable/). Use the
[in-development documentation](https://docs.sciml.ai/HighDimPDE/dev/) for the version of
the documentation, which contains the unreleased features.

## Installation

Open Julia and type the following

```julia
using Pkg;
Pkg.add("HighDimPDE.jl")
```

This will download the latest version from the git repo and download all dependencies.

## Getting started

See documentation and `test` folders.

## Reference

  - Boussange, V., Becker, S., Jentzen, A., Kuckuck, B., Pellissier, L., Deep learning approximations for non-local nonlinear PDEs with Neumann boundary conditions. [arXiv](https://arxiv.org/abs/2205.03672) (2022)

<!-- - Becker, S., Braunwarth, R., Hutzenthaler, M., Jentzen, A., von Wurstemberger, P., Numerical simulations for full history recursive multilevel Picard approximations for systems of high-dimensional partial differential equations. [arXiv](https://arxiv.org/abs/2005.10206) (2020)
- Beck, C., Becker, S., Cheridito, P., Jentzen, A., Neufeld, A., Deep splitting method for parabolic PDEs. [arXiv](https://arxiv.org/abs/1907.03452) (2019)
- Han, J., Jentzen, A., E, W., Solving high-dimensional partial differential equations using deep learning. [arXiv](https://arxiv.org/abs/1707.02568) (2018) -->
<!-- ## Acknowledgements
`HighDimPDE.jl` is inspired from Sebastian Becker's scripts in Python, TensorFlow, and C++. Pr. Arnulf Jentzen largely contributed to the theoretical developments of the solver algorithms implemented. -->
