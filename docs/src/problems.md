```@docs
PIDEProblem
ParabolicPDEProblem
```

!!! note
While choosing to define a PDE using `PIDEProblem`, note that the function being integrated `f` is a function of `f(x, y, v_x, v_y, ∇v_x, ∇v_y)` out of which `y` is the integrating variable and `x` is constant throughout the integration.
If a PDE has no integral and the non linear term `f` is just evaluated as `f(x, v_x, ∇v_x)` then we suggest using `ParabolicPDEProblem`
