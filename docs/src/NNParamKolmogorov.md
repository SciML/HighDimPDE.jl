# [The `NNParamKolmogorov` algorithm](@id nn_paramkolmogorov)

### Problems Supported:
1. [`ParabolicPDEProblem`](@ref)

```@autodocs
Modules = [HighDimPDE]
Pages   = ["NNParamKolmogorov.jl"]
```

`NNParamKolmogorov` obtains a 
- terminal solution for parametric families of Forward Kolmogorov Equations of the form:
```math
\partial_t u(t,x) = \mu(t, x, γ_mu) \nabla_x u(t,x) + \frac{1}{2} \sigma^2(t, x, γ_sigma) \Delta_x u(t,x)
```
with initial condition given by `g(x, γ_phi)`
- or an initial condition for parametric families of Backward Kolmogorov Equations of the form:
```math
\partial_t u(t,x) = - \mu(t, x) \nabla_x u(t,x) - \frac{1}{2} \sigma^2(t, x) \Delta_x u(t,x)
```
with terminal condition given by `g(x, γ_phi)`

We can use the Feynman-Kac formula : 
```math
S_t^x = \int_{0}^{t}\mu(S_s^x)ds + \int_{0}^{t}\sigma(S_s^x)dB_s
```
And the solution is given by:
```math
f(T, x) = \mathbb{E}[g(S_T^x, γ_phi)]
```