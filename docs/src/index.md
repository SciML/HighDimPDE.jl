
# HighDimPDE.jl


**HighDimPDE.jl** is a Julia package to **solve Highly Dimensional non-linear, non-local PDEs** of the form

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
-----------------------------------
time discretization free| ❌ | ✅ |
-----------------------------------
mesh-free| ✅ | ✅|
-----------------------------------
Hypercube approximation| ✅ | ❌ |
-----------------------------------
CPU | ✅ | ❌ |
-----------------------------------