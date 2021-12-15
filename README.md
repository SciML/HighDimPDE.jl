<!-- [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://vboussange.github.io/HighDimPDE.jl/stable) -->
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://vboussange.github.io/HighDimPDE.jl/dev)
[![Build Status](https://github.com/vboussange/HighDimPDE.jl/workflows/CI/badge.svg)](https://github.com/vboussange/HighDimPDE.jl/actions?query=workflow%3ACI)

# HighDimPDE.jl

**HighDimPDE.jl** is a Julia package to **solve Highly Dimensional PDEs**. The package implements solver algorithms that break down the curse of dimensionality, including

* the Deep Splitting scheme

* the Multi-Level Picard iterations scheme.

## Algorithm overview

----------------------------------------------
Features  |    `DeepSplitting` [1,3]   | `MLP` [1,2]     |
----------|:----------------------:|:------------:
Time discretization free|   ❌ |         ✅ |
Mesh-free       | ✅ |                   ✅ |
Single point $x \in \R^d$ approximation| ✅   |  ✅ |
$d$-dimensional cube $[a,b]^d$ approximation| ✅   |          ❌ |
GPU             | ✅ |                   ❌ |
Gradient non-linearities    | ✔️|       ❌ |
Non-local PDEs  | ✔️  | ✔️  |

✔️ : will be supported in the future

## Documentation
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://vboussange.github.io/HighDimPDE.jl/dev)

## Installation
Open Julia and type the following

```julia
using Pkg;
Pkg.add("https://github.com/vboussange/HighDimPDE.jl.git")
```
This will download latest version from git repo and download all dependencies.

## Getting started
See documentation, `examples` and `test` folders.

## References
- [1] Boussange, V., Becker, S., Jentzen, A., Pellissier, L., Deep learning approximations for non-local nonli- near PDEs with Neumann boundary conditions. _Manuscript in preparation_ (2021)
- [2] Becker, S., Braunwarth, R., Hutzenthaler, M., Jentzen, A., von Wurstemberger, P., Numerical simulations for full history recursive multilevel Picard approximations for systems of high-dimensional partial differential equations. [arXiv](https://arxiv.org/abs/2005.10206) (2020)
- [3] Beck, C., Becker, S., Cheridito, P., Jentzen, A., Neufeld, A., Deep splitting method for parabolic PDEs. [arXiv](https://arxiv.org/abs/1907.03452) (2019)

## Acknowledgements
The author thanks Sebastian Becker for fruitful discussions on the implementation of `HighDimPDE.jl`.
