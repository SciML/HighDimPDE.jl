# [The `NNKolmogorov` algorithm](@id nn_komogorov)

### Problems Supported:
1. [`ParabolicPDEProblem`](@ref)

```@autodocs
Modules = [HighDimPDE]
Pages   = ["NNKolmogorov.jl"]
```

`NNKolmogorov` obtains a 
- terminal solution for Forward Kolmogorov Equations of the form:
```math
\partial_t u(t,x) = \mu(t, x) \nabla_x u(t,x) + \frac{1}{2} \sigma^2(t, x) \Delta_x u(t,x)
```
with initial condition given by `g(x)`
- or an initial condition for Backward Kolmogorov Equations of the form:
```math
\partial_t u(t,x) = - \mu(t, x) \nabla_x u(t,x) - \frac{1}{2} \sigma^2(t, x) \Delta_x u(t,x)
```
with terminal condition given by `g(x)`

We can use the Feynman-Kac formula : 
```math
S_t^x = \int_{0}^{t}\mu(S_s^x)ds + \int_{0}^{t}\sigma(S_s^x)dB_s
```
And the solution is given by:
```math
f(T, x) = \mathbb{E}[g(S_T^x)]
```