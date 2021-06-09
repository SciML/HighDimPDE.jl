# HighDimPDE.jl

This package provides the Deep Splitting and the MLP algorithms to solve for high dimensional, non-local, nonlinear PDEs (papers in prep.). 
It builds upon DiffEqBase.jl and SciMLBase.jl for the interface, and uses Flux.jl for the solving part (Deep Splitting Algorithm).
It aims at solving PDEs for which the solution u satisfies

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
Sebastian Becker, who wrote the original scripts in Python, TensorFlow and C++, and Arnulf Jentzen for the theoretical developments.
