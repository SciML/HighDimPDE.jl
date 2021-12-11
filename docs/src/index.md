
# HighDimPDE.jl


**HighDimPDE.jl** is a Julia package to **solve Highly Dimensional non-linear, non-local PDEs** of the form

```math
\begin{aligned}
   (\partial_t u)(t,x) = & f\big(t,x, u(t,x), ( \nabla_x u )(t,x ) ) \big)  + \big\langle \mu(t,x), ( \nabla_x u )( t,x ) \big\rangle \\
    & \quad  + \tfrac{1}{2} \text{Trace} \big(\sigma(t,x) [ \sigma(t,x) ]^* ( \text{Hess}_x u)(t, x ) \big),
\end{aligned}
```

where $u \colon [0,T] \times \Omega \to \R$, $\Omega \subset \R^d$, $d$ large,
subject to initial and boundary conditions.

🚧 Work in Progress: for now, `HighDimPDE.jl` can only solve for local PDEs.

**HighDimPDE.jl** implements solver algorithms that break down the curse of dimensionality, including

* the [Deep Splitting scheme](@ref deepsplitting)

* the [Multi-Level Picard iterations scheme](@ref mlp).

To make the most out of **HighDimPDE.jl**, we advise to first have a look at the 

* [documentation on the Feynman Kac formula](@ref feynmankac),

as all solver algorithms heavily rely on it.

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