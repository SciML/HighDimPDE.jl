
# HighDimPDE.jl


**HighDimPDE.jl** is a Julia package to **solve Highly Dimensional non-linear, non-local PDEs** of the form

```math
\begin{aligned}
    (\partial_t u)(t,x) &= \int_{\Omega} f\big(t,x,{\bf x}, u(t,x),u(t,{\bf x}), ( \nabla_x u )(t,x ),( \nabla_x u )(t,{\bf x} ) \big) \, \nu_x(d{\bf x}) \\
    & \quad + \big\langle \mu(t,x), ( \nabla_x u )( t,x ) \big\rangle + \tfrac{1}{2} \text{Trace} \big(\sigma(t,x) [ \sigma(t,x) ]^* ( \text{Hess}_x u)(t, x ) \big).
\end{aligned}
```

where $u \colon \Omega \to \R$, $\Omega \subset \R^d$
subject to initial and boundary conditions.


**HighDimPDE.jl** implements schemes that break down the curse of dimensionality, including

* the [Deep BSDE scheme]()

* the [Deep Splitting scheme]()

* the [Multi-Level Picard iterations scheme]().

To make the most out of **HighDimPDE.jl**, we advise to first have a look at the 

* [documentation on the Feynman Kac formula](),

as both schemes heavily rely on it.

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