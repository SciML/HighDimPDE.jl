[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://highdimpde.sciml.ai/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://highdimpde.sciml.ai/dev)
[![Build Status](https://github.com/SciML/HighDimPDE.jl/workflows/CI/badge.svg)](https://github.com/SciML/HighDimPDE.jl/actions?query=workflow%3ACI)

# HighDimPDE.jl

**HighDimPDE.jl** is a Julia package to **solve Highly Dimensional non-local, non-linear PDEs** of the form


<!-- $$
\begin{aligned}
    (\partial_t u)(t,x) &= \int_{\Omega} f\big(t,x,{\bf x}, u(t,x),u(t,{\bf x}), ( \nabla_x u )(t,x ),( \nabla_x u )(t,{\bf x} ) \big) \, d{\bf x} \\
    & \quad + \big\langle \mu(t,x), ( \nabla_x u )( t,x ) \big\rangle + \tfrac{1}{2} \text{Trace} \big(\sigma(t,x) [ \sigma(t,x) ]^* ( \text{Hess}_x u)(t, x ) \big).
\end{aligned}
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Baligned%7D%0A%20%20%20%20(%5Cpartial_t%20u)(t%2Cx)%20%26%3D%20%5Cint_%7B%5COmega%7D%20f%5Cbig(t%2Cx%2C%7B%5Cbf%20x%7D%2C%20u(t%2Cx)%2Cu(t%2C%7B%5Cbf%20x%7D)%2C%20(%20%5Cnabla_x%20u%20)(t%2Cx%20)%2C(%20%5Cnabla_x%20u%20)(t%2C%7B%5Cbf%20x%7D%20)%20%5Cbig)%20%5C%2C%20d%7B%5Cbf%20x%7D%20%5C%5C%0A%20%20%20%20%26%20%5Cquad%20%2B%20%5Cbig%5Clangle%20%5Cmu(t%2Cx)%2C%20(%20%5Cnabla_x%20u%20)(%20t%2Cx%20)%20%5Cbig%5Crangle%20%2B%20%5Ctfrac%7B1%7D%7B2%7D%20%5Ctext%7BTrace%7D%20%5Cbig(%5Csigma(t%2Cx)%20%5B%20%5Csigma(t%2Cx)%20%5D%5E*%20(%20%5Ctext%7BHess%7D_x%20u)(t%2C%20x%20)%20%5Cbig).%0A%5Cend%7Baligned%7D"></div>

where <!-- $u \colon [0,T] \times \Omega \to \R$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=u%20%5Ccolon%20%5B0%2CT%5D%20%5Ctimes%20%5COmega%20%5Cto%20%5CR">, <!-- $\Omega\subseteq \R^d$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5COmega%5Csubseteq%20%5CR%5Ed"> is subject to initial and boundary conditions, and where <!-- $d$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=d"> is large.

## Documentation
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://highdimpde.sciml.ai/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://highdimpde.sciml.ai/dev)

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
`HighDimPDE.jl` is inspired from Sebastian Becker's scripts in Python, TensorFlow and C++. Pr. Arnulf Jentzen largely contributed to the theoretical developments of the solver algorithms implemented. -->
