[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://vboussange.github.io/HighDimPDE.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://vboussange.github.io/HighDimPDE.jl/dev)
[![Build Status](https://github.com/vboussange/HighDimPDE.jl/workflows/CI/badge.svg)](https://github.com/vboussange/HighDimPDE.jl/actions?query=workflow%3ACI)

# HighDimPDE.jl

**HighDimPDE.jl** is a Julia package that implements solver algorithms that break down the curse of dimensionality to **solve Highly Dimensional non-linear, non-local PDEs** of the form

<div style="overflow-x: scroll;" align=center>                          
<img src="docs/src/img/equation.png" height="80"/>
</div>
<p>subject to initial and boundary conditions, where <img src="docs/src/img/function_u.png" height="20"/>.</p>

## Documentation
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://vboussange.github.io/HighDimPDE.jl/dev)

## Installation
Open Julia REPL and type the following

```julia
using Pkg;
Pkg.add("https://github.com/vboussange/HighDimPDE.jl.git")
```
This will download latest version from git repo and download all dependencies.

## Getting started
See documentation, `examples` and `test` folders.

# Related papers
- [`MLP`: Numerical simulations for full history recursive multilevel Picard approximations for systems of high-dimensional partial differential equations](https://arxiv.org/abs/2005.10206)
- [`DeepSplitting`: Deep Splitting method for parabolic PDEs](https://arxiv.org/abs/1907.03452)
- [`DeepBSDE`: Solving high-dimensional partial differential equations using deep learning](https://www.pnas.org/content/115/34/8505)

# Acknowledgements
`HighDimPDE.jl` is inspired from Sebastian Becker's scripts in Python, TensorFlow and C++. Pr. Arnulf Jentzen largely contributed to the theoretical developments of the solver algorithms implemented.
