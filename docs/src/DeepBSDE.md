# [The `DeepBSDE` algorithm](@id deepbsde)

### Problems Supported:
1. [`ParabolicPDEProblem`](@ref)

```@autodocs
Modules = [HighDimPDE]
Pages   = ["DeepBSDE.jl", "DeepBSDE_Han.jl"]
```

## The general idea 💡
The `DeepBSDE` algorithm is similar in essence to the `DeepSplitting` algorithm, with the difference that 
it uses two neural networks to approximate both the the solution and its gradient.

## References
- Han, J., Jentzen, A., E, W., Solving high-dimensional partial differential equations using deep learning. [arXiv](https://arxiv.org/abs/1707.02568) (2018)