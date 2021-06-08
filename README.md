# DeepSplitting.jl

This package provides the Deep Splitting algorithm (paper in prep.), built upon DiffEqBase.jl and SciMLBase.jl for the interface, and using Flux.jl for the solving part.
It aims at solving non-local, non-linear Partial Differential Equations, where the solution u satisfies equations of the form

<div style="overflow-x: scroll;" align=center>                          
<img src="docs/equation.png"/>
</div>

The scheme is particularly performant when the domain D is highly dimensional, as it overcomes the so called *curse of dimensionality*.

<!-- ## Installation
Open Julia in your favorite REPL and type the following

```julia
using Pkg;
Pkg.add("https://github.com/vboussange/DeepSplitting.jl.git")
```

This will download latest version from git repo and download all dependencies. -->

## Getting started
Check out the folder `examples/allen_cahn_nonlocal.jl` to see how it works.

# Acknowledgements
Sebastian Becker, who wrote the original script in TensorFlow.
