# DeepSplitting.jl

This package provides the Deep Splitting algorithm (paper in prep.), built upon DiffEqBase.jl and SciMLBase.jl.
It aims at solving non-local, non-linear Partial Differential Equations, where the solution $`u \colon \R^d \times \R^+ \to \R`$ satisfies equations of the form

```math
\begin{aligned}
  (\tfrac{\partial}{\partial t}u)(t,x)
  &=
  \int_{D} f\big(t,x,{\bf x}, u(t,x),u(t,{\bf x}), ( \nabla_x u )(t,x ),( \nabla_x u )(t,{\bf x} ) \big) \, \nu_x(d{\bf x}) \\
  & \quad + \big\langle \mu(t,x), ( \nabla_x u )( t,x ) \big\rangle + \tfrac{ 1 }{ 2 }
  \text{Trace}\!\big(
  \sigma(t,x) [ \sigma(t,x) ]^*
  ( \text{Hess}_x u)( t,x )
  \big).
\end{aligned}
```

# Acknowledgements
Sebastian Becker, who wrote the original script in tensorflow.
