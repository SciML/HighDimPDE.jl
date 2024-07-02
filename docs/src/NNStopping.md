# [The `NNStopping` algorithm](@id nn_stopping)

### Problems Supported:

 1. [`ParabolicPDEProblem`](@ref)

```@autodocs
Modules = [HighDimPDE]
Pages   = ["NNStopping.jl"]
```

## The general idea 💡

Similar to DeepSplitting and DeepBSDE, NNStopping evaluates the PDE as a Stochastic Differential Equation. Consider an Obstacle PDE of the form:

```math
 max\lbrace\partial_t u(t,x) + \mu(t, x) \nabla_x u(t,x) + \frac{1}{2} \sigma^2(t, x) \Delta_x u(t,x) , g(t,x) - u(t,x)\rbrace
```

Such PDEs are commonly used as representations for the dynamics of stock prices that can be exercised before maturity, such as American Options.

Using the Feynman-Kac formula, the underlying SDE will be:

```math
dX_{t}=\mu(X,t)dt + \sigma(X,t)\ dW_{t}^{Q}
```

The payoff of the option would then be:

```math
sup\lbrace\mathbb{E}[g(X_\tau, \tau)]\rbrace
```

Where τ is the stopping (exercising) time. The goal is to retrieve both the optimal exercising strategy (τ) and the payoff.

We approximate each stopping decision with a neural network architecture, inorder to maximise the expected payoff.
