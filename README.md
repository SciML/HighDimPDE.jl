# DeepSplitting.jl

This package provides the Deep Splitting algorithm (paper in prep.), built upon DiffEqBase.jl and SciMLBase.jl.
It aims at solving non-local, non-linear Partial Differential Equations, where the solution $u \colon \R^d \times \R^+ \to \R$ satisfies equations of the form

<div style="overflow-x: scroll;max-width:400px !important;" align=center>                          
<img src="docs/equation.png"/>
</div>

# Acknowledgements
Sebastian Becker, who wrote the original script in tensorflow.
