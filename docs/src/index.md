# HighDimPDE.jl


**HighDimPDE.jl** is a Julia package to **solve Highly Dimensional non-linear, non-local PDEs** of the forms: 

1. Partial Integro Differential Equations: 
```math
\begin{aligned}
    (\partial_t u)(t,x) &= \int_{\Omega} f\big(t,x,{\bf x}, u(t,x),u(t,{\bf x}), ( \nabla_x u )(t,x ),( \nabla_x u )(t,{\bf x} ) \big) \, d{\bf x} \\
    & \quad + \big\langle \mu(t,x), ( \nabla_x u )( t,x ) \big\rangle + \tfrac{1}{2} \text{Trace} \big(\sigma(t,x) [ \sigma(t,x) ]^* ( \text{Hess}_x u)(t, x ) \big).
\end{aligned}
```

where $u \colon [0,T] \times \Omega \to \R$, $\Omega \subseteq \R^d$ is subject to initial and boundary conditions, and where $d$ is large.

2. Parabolic Partial Differential Equations: 
```math
\begin{aligned}
    (\partial_t u)(t,x) &=  f\big(t,x, u(t,x), ( \nabla_x u )(t,x )\big) 
    + \big\langle \mu(t,x), ( \nabla_x u )( t,x ) \big\rangle + \tfrac{1}{2} \text{Trace} \big(\sigma(t,x) [ \sigma(t,x) ]^* ( \text{Hess}_x u)(t, x ) \big).
\end{aligned}
```

where $u \colon [0,T] \times \Omega \to \R$, $\Omega \subseteq \R^d$ is subject to initial and boundary conditions, and where $d$ is large.

!!! note
    The difference between the two problems is that in Partial Integro Differential Equations, the integrand is integrated over **x**, while in Parabolic Integro Differential Equations, the function `f` is just evaluated for `x`.

**HighDimPDE.jl** implements solver algorithms that break down the curse of dimensionality, including

* the [Deep Splitting scheme](@ref deepsplitting)

* the [Multi-Level Picard iterations scheme](@ref mlp)

* the Deep BSDE scheme (@ref deepbsde).


To make the most out of **HighDimPDE.jl**, we advise to first have a look at the 

* [documentation on the Feynman Kac formula](@ref feynmankac),

as all solver algorithms heavily rely on it.

## Algorithm overview

------------------------------------------------------------
Features  |    `DeepSplitting`   | `MLP`     | `DeepBSDE` |
----------|:----------------------:|:------------:|:--------:
Time discretization free|  ❌ | ✅ |   ❌ |
Mesh-free       | ✅ |   ✅ |   ✅ |
Single point $x \in \R^d$ approximation| ✅  |  ✅ | ✅ |
$d$-dimensional cube $[a,b]^d$ approximation| ✅   | ❌ | ✔️ |
GPU | ✅ |  ❌ | ✅ |      
Gradient non-linearities  | ✔️|  ❌ | ✅ |

✔️ : will be supported in the future

## Reproducibility
```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```
```@example
using Pkg # hide
Pkg.status() # hide
```
```@raw html
</details>
```
```@raw html
<details><summary>and using this machine and Julia version.</summary>
```
```@example
using InteractiveUtils # hide
versioninfo() # hide
```
```@raw html
</details>
```
```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```
```@example
using Pkg # hide
Pkg.status(;mode = PKGMODE_MANIFEST) # hide
```
```@raw html
</details>
```
```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link_manifest = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
