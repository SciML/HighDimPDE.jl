# SciMLBase Interface

`PIDEProblem` and `ParabolicPDEProblem` are out-of-place
`SciMLBase.AbstractSciMLProblem` implementations. Their state `x` and all sampled
states must use a numeric container compatible with the SciML array interface: it must
support broadcast, preserve an appropriate element type, and support the mutating array
operations required by the selected solver. The time span uses floating-point endpoints
because HighDimPDE's adaptive and stochastic solver paths perform floating-point time
arithmetic.

The condition and coefficient functions must accept the signatures documented on each
problem constructor. `PIDEProblem.f` receives both sampled states and their values and
gradients; `ParabolicPDEProblem.f` receives a state, value, gradient, parameters, and
time. Both problem types are out-of-place, so these functions return their result rather
than writing to a first output argument.

Use generic SciMLBase operations to work with the problem after construction. In
particular, import `solve` from `SciMLBase`, rather than relying on a reexport, and use
`remake` to replace problem data without reconstructing the complete problem manually.

```julia
using HighDimPDE
using LinearAlgebra: I
using SciMLBase: isinplace, remake, solve

μ(x, p, t) = zero(x)
σ(x, p, t) = I
g(x) = sum(abs2, x)
f(x, u, du, p, t) = zero(u)

prob = ParabolicPDEProblem(μ, σ, zeros(2), (0.0, 1.0); g, f)
@assert !isinplace(prob)
prob_with_parameters = remake(prob; p = [1.0])
```
